import torch
import logging
from tqdm import tqdm
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple
from transformers import Idefics2ForConditionalGeneration, AutoProcessor

import warnings

warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")

DEFAULT_IMAGE_TOKEN = "<image>"
try:
    import flash_attn

    best_fit_attn_implementation = "flash_attention_2"
except ImportError:
    best_fit_attn_implementation = "eager"


@register_model("idefics2")
class Idefics2(lmms):
    """
    Idefics2 Model for Hugging Face Transformers: https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py

    Example usage:

    accelerate launch --num_processes=8 -m lmms_eval \
        --model idefics2 \
        --model_args pretrained=HuggingFaceM4/idefics2-8b \
        --tasks mme \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_samples
    """

    def __init__(
        self,
        pretrained: str = "HuggingFaceM4/idefics2-8b",
        revision: str = "main",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "float16",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = False,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: str = "",
        use_cache: bool = True,
        do_image_splitting: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)
        self._model = Idefics2ForConditionalGeneration.from_pretrained(pretrained, revision=revision, torch_dtype=dtype, device_map=self.device_map, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation)
        self._processor = AutoProcessor.from_pretrained(pretrained, do_image_splitting=do_image_splitting, revision=revision, trust_remote_code=trust_remote_code)

        self._tokenizer = self._processor.tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with pipeline parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Idefics2 model")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visuals, doc_id, tasks, splits = zip(*chunk)
            visuals = [doc_to_visual(self.task_dict[task][split][ids]) for ids, task, split, doc_to_visual in zip(doc_id, tasks, splits, doc_to_visuals)]
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            #
            until = gen_kwargs.pop("until", None)
            image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio",  None)
            prompts = []
            for context, visual in zip(contexts, visuals):
                content = []
                if DEFAULT_IMAGE_TOKEN not in context:
                    for image in visual:
                        content.append({"type": "image"})
                content.append({"type": "text", "text": context})
                message = [{"role": "user", "content": content}]
                prompt = self._processor.apply_chat_template(message, add_generation_prompt=True)
                prompts.append(prompt)
            inputs = self._processor(text=prompts, images=visuals, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            output_ids = self.model.generate(**inputs, **gen_kwargs)
            # only retain the generated text
            for output_id, input_id in zip(output_ids, inputs["input_ids"]):
                generated_id = output_id[len(input_id) :]
                generated_text = self.tokenizer.decode(generated_id, skip_special_tokens=True)

                res.append(generated_text)
            pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
