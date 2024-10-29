import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from sae import Sae
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

DEFAULT_IMAGE_TOKEN = "<image>"

# Default chat for llava-hf/llava-1.5 models: https://huggingface.co/collections/llava-hf/llava-15-65f762d5b6941db5c2ba07e0
VICUNA_CHAT_TEMPLATE = "{% for message in messages %}{% if loop.index0 == 0 %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {{ message['content'] }} {% elif message['role'] == 'user' %}USER: {{ message['content'] }} {% else %} ASSISTANT: {{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"

model_map = {
    "llava": LlavaForConditionalGeneration,
    "llava_next": LlavaNextForConditionalGeneration,
}


@register_model("llava_sae_hooked")
class LlavaSaeHooked(lmms):
    """
    Llava Model for Hugging Face Transformers: https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llava

    Adapted from the InstructBLIP model in lmms_eval/models/instructblip.py

    Example usage:

    accelerate launch --num_processes=8 --main_process_port 12345 -m lmms_eval \
        --model llava_hf \
        --model_args pretrained=llava-hf/llava-1.5-7b-hf \
        --tasks seedbench \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_samples
    """

    def __init__(
        self,
        pretrained: str = "llava-hf/llava-1.5-7b-hf",
        revision: str = "main",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        device_map: str = "",
        chat_template: Optional[str] = None,
        use_cache: bool = True,
        specified_eot_token_id: Optional[int] = None,
        sae_path: Optional[str] = None,
        clamped_idx: int = 0,
        clamped_value: float = 50,
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
        self.clamped_idx = (clamped_idx,)
        self.clamped_value = clamped_value

        config = AutoConfig.from_pretrained(pretrained)
        model_type = getattr(config, "model_type", "llava")
        model_type = model_map[model_type]
        self._model = model_type.from_pretrained(pretrained, revision=revision, torch_dtype=dtype, device_map=self.device_map, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation)
        if sae_path is not None:
            self.module_dict = Sae.load_many(sae_path, local=True if os.path.exists(sae_path) else False, device=self._device)
        else:
            self.module_dict = None
        self.name_to_module = {name: self.model.language_model.get_submodule(name) for name in self.module_dict.keys()}
        self.module_to_name = {v: k for k, v in self.name_to_module.items()}

        self.pretrained = pretrained
        self._image_processor = AutoProcessor.from_pretrained(pretrained, revision=revision, trust_remote_code=trust_remote_code)
        # Pad from left for batched generation: https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llava#usage-tips
        self._image_processor.tokenizer.padding_side = "left"
        self._tokenizer = self._image_processor.tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.chat_template = chat_template
        self.use_cache = use_cache
        self.specified_eot_token_id = specified_eot_token_id
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
                # self.module_dict = {k: accelerator.prepare_model(v) for k, v in self.module_dict.items()}
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

    def clamp_features_max(self, sae: Sae, feature: int, hooked_module: torch.nn.Module, k: float = 10):
        def hook(module: torch.nn.Module, _, outputs):
            # Maybe unpack tuple outputs
            if isinstance(outputs, tuple):
                unpack_outputs = list(outputs)
            else:
                unpack_outputs = list(outputs)
            latents = sae.pre_acts(unpack_outputs[0])
            # Only clamp the feature for the first forward
            if latents.shape[1] != 1:
                latents[:, :, feature] = k
            top_acts, top_indices = sae.select_topk(latents)
            sae_out = sae.decode(top_acts[0], top_indices[0]).unsqueeze(0).to(torch.float16)
            unpack_outputs[0] = sae_out
            if isinstance(outputs, tuple):
                outputs = tuple(unpack_outputs)
            else:
                outputs = unpack_outputs[0]
            return outputs

        handles = [hooked_module.register_forward_hook(hook)]

        return handles

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
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for context, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
            image_tokens = " ".join(image_tokens)
            context = f"{image_tokens}\n{context}"
            # Apply chat template
            messages = [{"role": "user", "content": context}, {"role": "assistant", "content": continuation}]
            if self.chat_template is not None:
                self.tokenizer.chat_template = self.chat_template
                prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                prompt_and_continuation = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            elif self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                prompt_and_continuation = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
                prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                prompt_and_continuation = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            formatted_contexts = [prompt]
            formatted_continuation = [prompt_and_continuation]
            model_inputs = self._image_processor(text=formatted_continuation, images=visuals).to(self._device, self.model.dtype)
            labels = model_inputs["input_ids"].clone()
            contxt_id = self._image_processor(text=formatted_contexts, return_tensors="pt")["input_ids"]
            labels[: len(contxt_id)] = -100

            if self.accelerator.is_main_process and doc_id % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id}:\n\n{formatted_contexts[0]}\n")
                eval_logger.debug(f"Prompt and continuation for doc ID {doc_id}:\n\n{formatted_continuation[0]}\n")

            with torch.inference_mode():
                outputs = self.model(**model_inputs, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = model_inputs["input_ids"][:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : model_inputs["input_ids"].shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

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
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            context = contexts[0]

            # Some benchmarks like MME do not contain image tokens, so we prepend them to the prompt.
            if DEFAULT_IMAGE_TOKEN not in context:
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                context = f"{image_tokens}\n{context}"
            # Apply chat template
            messages = [{"role": "user", "content": context}]
            if self.chat_template is not None:
                self.tokenizer.chat_template = self.chat_template
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            elif self.tokenizer.chat_template is not None:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

            inputs = self._image_processor(images=visuals, text=text, return_tensors="pt").to(self._device, self.model.dtype)

            gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            handles = []
            for name, mod in self.name_to_module.items():
                handles.extend(self.clamp_features_max(self.module_dict[name], self.clamped_idx, mod, self.clamped_value))
            try:
                cont = self.model.generate(
                    **inputs,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.specified_eot_token_id,
                )
                cont = cont[:, inputs["input_ids"].shape[-1] :]
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = ""
            finally:
                for handle in handles:
                    handle.remove()
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{text_outputs}\n")

            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAHF")
