import torch

torch.backends.cuda.matmul.allow_tf32 = True


import copy
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from packaging import version
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

warnings.filterwarnings("ignore")

try:
    from mantis.models.conversation import conv_mllava_v1 as default_conv
    from mantis.models.conversation import conv_templates
    from mantis.models.mfuyu import MFuyuForCausalLM, MFuyuProcessor
    from mantis.models.mllava import LlavaForConditionalGeneration, MLlavaProcessor

except Exception as e:
    eval_logger.debug("Mantis is not installed. Please install Mantis to use this model.\nError: %s" % e)

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except Exception as e:
    eval_logger.debug("Upgrade transformers to use Mantis's idefics model.\nError: %s" % e)

# inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
# if is_flash_attn_2_available:
#     best_fit_attn_implementation = "flash_attention_2" # flash_attn has a bug that says: ERROR Error query and key must have the same dtype in generating

try:
    import flash_attn

    best_fit_attn_implementation = "flash_attention_2"
except ImportError:
    best_fit_attn_implementation = "eager"

DEFAULT_IMAGE_TOKEN = "<image>"


@register_model("mantis")
class Mantis(lmms):
    """
    Mantis Model
    This implementation is adpated from the Llava model from llava.py and the Idefics model from idefics.py
    """

    def __init__(
        self,
        pretrained: str = "TIGER-Lab/Mantis-8B-siglip-llama3",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        dtype: Optional[Union[str, torch.dtype]] = "float16",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation=best_fit_attn_implementation,
        device_map="cuda:0",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self._is_idefics = "idefics" in pretrained.lower()
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        # Here we load the "non-idefics" Mantis model.
        if not self._is_idefics:
            if "fuyu" in pretrained.lower():
                self._processor = MFuyuProcessor.from_pretrained(pretrained)
                self._model = MFuyuForCausalLM.from_pretrained(pretrained, device_map=self.device_map, attn_implementation=attn_implementation, torch_dtype=dtype)
            else:
                self._processor = MLlavaProcessor.from_pretrained(pretrained)
                self._model = LlavaForConditionalGeneration.from_pretrained(pretrained, device_map=self.device_map, attn_implementation=attn_implementation, torch_dtype=dtype)

        else:
            self._processor = AutoProcessor.from_pretrained(pretrained)
            self._model = AutoModelForVision2Seq.from_pretrained(pretrained, device_map=self.device_map, torch_dtype=dtype)
        eval_logger.info(f"Using {type(self._model)} to instantiate the Mantis model.")

        self._tokenizer = self._processor.tokenizer

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.truncate_context = truncate_context

        if accelerator.num_processes > 1:
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
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

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

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

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
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

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

            until = gen_kwargs.pop("until", None)
            image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio", None)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0

            # prompts_input = contexts[0]

            prompts = []
            for visual, context in zip(visuals, contexts):
                if self._is_idefics:
                    # Follow the idefics implementation:
                    content = []
                    if DEFAULT_IMAGE_TOKEN not in context:
                        for _ in visual:
                            content.append({"type": "image"})
                    content.append({"type": "text", "text": context})
                    message = [{"role": "user", "content": content}]
                    prompt = self._processor.apply_chat_template(message, add_generation_prompt=True)
                    prompts.append(prompt)
                else:
                    # We follow the Mantis code base: https://github.com/TIGER-AI-Lab/Mantis/blob/main/mantis/models/mllava/utils.py#L33 to make sure they are consistent
                    # Users don't need to define chat template as it is done here
                    if "llama-3" in self._model.language_model.name_or_path.lower():
                        conv = conv_templates["llama_3"]
                        terminators = [self._processor.tokenizer.eos_token_id, self._processor.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                    else:
                        conv = default_conv
                        terminators = None

                    gen_kwargs["eos_token_id"] = terminators

                    conv = conv.copy()
                    conv.append_message(conv.roles[0], context)
                    conv.append_message(conv.roles[1], "")
                    prompt = conv.get_prompt()
                    prompts.append(prompt)
            inputs = self._processor(images=visuals, text=prompts, return_tensors="pt", truncation=True)
            if "image_patches" in inputs.keys():
                inputs["image_patches"] = inputs["image_patches"][0]  # FIXME: Fuyu model would return a list instead of a pytorch tensor. This weird behavior needs fixing.
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            output_ids = self.model.generate(**inputs, **gen_kwargs)
            for output_id, input_id in zip(output_ids, inputs["input_ids"]):
                generated_id = output_id[len(input_id) :]
                generated_text = self.tokenizer.decode(generated_id, skip_special_tokens=True)

                res.append(generated_text)

            # self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Mantis")
