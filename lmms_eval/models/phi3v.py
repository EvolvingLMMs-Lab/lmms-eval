import torch

from accelerate import Accelerator, DistributedType
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from typing import List, Optional, Tuple, Union

from loguru import logger as eval_logger


@register_model("phi3v")
class Phi3v(lmms):
    """
    This class implements inference for the microsoft/Phi-3-vision-128k-instruct model.
    To learn more about this model please visit the following links:
    1. https://huggingface.co/microsoft/Phi-3-vision-128k-instruct
    2. https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/
    3. https://github.com/microsoft/Phi-3CookBook

    NOTE: This class was adapted from quen_vl.py and llava_hf.py.

    Example:

    accelerate launch --num_processes=4 -m lmms_eval --model phi3v --tasks mmmu_val \
        --batch_size 1 --log_samples --log_samples_suffix phi3v_mmmu --output_path ./logs/
    """

    def __init__(
        self,
        model_id_name: str = "microsoft/Phi-3-vision-128k-instruct",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        # Setup accelerator.
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device
        # Load model.
        self._model = AutoModelForCausalLM.from_pretrained(model_id_name, device_map=device, trust_remote_code=trust_remote_code, torch_dtype=dtype)
        self._processor = AutoProcessor.from_pretrained(model_id_name, trust_remote_code=trust_remote_code)
        self._processor.tokenizer.padding_side = "left"
        self._tokenizer = self._processor.tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size_per_gpu > 1 is not supported for now."
        self.use_cache = use_cache
        if accelerator.num_processes > 1:
            distributed_type_list = [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED]
            assert accelerator.distributed_type in distributed_type_list, "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1
            self.accelerator = accelerator

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

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Not implemented for Phi3v.")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            # We assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]
            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            if isinstance(contexts, tuple):
                contexts = list(contexts)
            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    query = "" + contexts[i]
                    img_placeholder_count = 1
                    while "<image>" in query:
                        query = query.replace("<image>", f"<|image_{img_placeholder_count}|>", 1)
                        img_placeholder_count += 1
                else:
                    query = ""
                    for placeholder_id in range(len(visuals)):
                        query += f"<|image_{placeholder_id+1}|>\n"
                    query += contexts[i]
                messages = [{"role": "user", "content": query}]
                contexts[i] = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            assert len(contexts) == 1
            #
            context = contexts[0]
            input_ids = self._processor(text=context, images=visuals, return_tensors="pt").to(self._device, self.model.dtype)
            # Setting default parameters.
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            # Generate answer.
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eod_id
            generate_ids = self.model.generate(
                **input_ids,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            generate_ids = generate_ids[:, input_ids["input_ids"].shape[1] :]
            response = self._processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            res.append(response)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), response)
            pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)
        pbar.close()
        return res
