from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Mistral3ForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)


@register_model("mistral3_vl")
class Mistral3_VL(lmms):
    """
    Mistral3-VL Model
    "https://huggingface.co/mistralai/Ministral-3-8B-Base-2512"
    """

    def __init__(
        self,
        pretrained: str = "mistralai/Ministral-3-8B-Base-2512",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        max_num_frames: int = 32,
        system_prompt: Optional[str] = None,
        temperature: float = 0.6,
        disable_thinking: bool = False,
        no_think_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            if device_map == "auto":
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                    self.device_map = "auto"
                else:
                    eval_logger.warning("CUDA is not available. Falling back to CPU for Mistral3-VL.")
                    self._device = torch.device("cpu")
                    self.device_map = "cpu"
            else:
                if device.startswith("cuda") and not torch.cuda.is_available():
                    eval_logger.warning(f"Requested device '{device}' but CUDA is not available. Falling back to CPU.")
                    device = "cpu"
                self._device = torch.device(device)
                self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if self._device.type == "cuda" else torch.float32,
            "device_map": self.device_map,
            "trust_remote_code": True,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        eval_logger.info(f"Loading Mistral3-VL model from {pretrained}")
        self._model = Mistral3ForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()

        self._processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
        self._tokenizer = self._processor.tokenizer
        self._config = self._model.config
        self._max_length = getattr(self._config, "max_position_embeddings", 2048)

        self.max_num_frames = max_num_frames
        self.system_prompt = system_prompt
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.temperature = temperature
        self.disable_thinking = disable_thinking
        self.no_think_prompt = no_think_prompt or " Respond with the final answer only. Do not output any reasoning, explanation, or <think> tags."
        self._warned_chat_template_fallback = False

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
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
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
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
        raise NotImplementedError("Loglikelihood is not supported for Mistral3-VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _prepare_prompt(self, prompt: str) -> str:
        if self.disable_thinking:
            return prompt.rstrip() + self.no_think_prompt
        return prompt

    def _build_inputs(self, prompt: str, images: List[Image.Image]):
        prompt = self._prepare_prompt(prompt)
        has_chat_template = bool(getattr(self._processor, "chat_template", None))
        image_token = getattr(self._processor, "image_token", "[IMG]")
        if len(images) <= 1:
            image_prefix = "".join([image_token for _ in images])
        else:
            image_prefix = "".join([f"Image {idx + 1}: {image_token}\n" for idx in range(len(images))])
        if image_prefix:
            text_prompt = f"<s>[INST]{image_prefix}\n{prompt}[/INST]"
        else:
            text_prompt = f"<s>[INST]{prompt}[/INST]"

        # Custom chat templates are often text-only and break on Hugging Face's
        # list-structured multimodal messages. For image inputs, use the
        # official Mistral3 `[IMG]` prompt format directly.
        if len(images) == 0 and has_chat_template:
            try:
                return self._processor.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception as e:
                if not self._warned_chat_template_fallback:
                    eval_logger.warning(f"Falling back to explicit Mistral3 prompt formatting because chat template application failed: {e}")
                    self._warned_chat_template_fallback = True

        return self._processor(
            images=images if len(images) > 0 else None,
            text=text_prompt,
            return_tensors="pt",
        )

    def _get_input_device(self) -> torch.device:
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            try:
                return torch.device(model_device)
            except (TypeError, RuntimeError, ValueError):
                pass

        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return self.device

    def _move_inputs_to_device(self, inputs):
        target_device = self._get_input_device()
        target_dtype = getattr(self.model, "dtype", None)

        moved_inputs = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved_inputs[key] = value
                continue

            if value.is_floating_point() and target_dtype is not None:
                moved_inputs[key] = value.to(device=target_device, dtype=target_dtype)
            else:
                moved_inputs[key] = value.to(target_device)
        return moved_inputs

    @staticmethod
    def _postprocess_output(output_text: str) -> str:
        output_text = parse_reasoning_model_answer(output_text)
        if "</think>" in output_text:
            output_text = output_text.split("</think>")[-1]
        return output_text.strip()

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        # Reorder requests by length
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            # Handle variable number of elements in chunk
            chunk_data = list(zip(*chunk))
            if len(chunk_data) == 6:
                contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = chunk_data
            elif len(chunk_data) == 4:
                # Some tasks may only have 4 elements
                contexts, all_gen_kwargs, doc_to_visual, doc_id = chunk_data
                task = [None] * len(contexts)
                split = [None] * len(contexts)
            else:
                eval_logger.error(f"Unexpected chunk format with {len(chunk_data)} elements")
                continue

            # Get visuals
            if task[0] is not None and split[0] is not None:
                dataset = self.task_dict[task[0]][split[0]]
                dataset_size = len(dataset)
                # Clamp out-of-bounds doc_ids (from padding) to last valid index
                clamped_doc_id = [min(ids, dataset_size - 1) for ids in doc_id]
                visuals = [doc_to_visual[0](dataset[ids]) for ids in clamped_doc_id]
            else:
                visuals = [doc_to_visual[0](ids) if callable(doc_to_visual[0]) else None for ids in doc_id]

            # Prepare inputs for Mistral3-VL
            # Format: [IMG]<prompt>
            prompts = []
            images_list = []

            for context, visual in zip(contexts, visuals):
                if isinstance(visual, list):
                    images_list.append(visual)
                else:
                    images_list.append([visual] if visual is not None else [])
                prompts.append(context)

            # Process batch
            try:
                batch_outputs = []
                for prompt, images in zip(prompts, images_list):
                    inputs = self._build_inputs(prompt, images)
                    inputs = self._move_inputs_to_device(inputs)
                    inputs.pop("token_type_ids", None)

                    # Generate
                    gen_kwargs = all_gen_kwargs[0]  # Assume same gen_kwargs for batch
                    max_new_tokens = gen_kwargs.get("max_new_tokens", 512)
                    temperature = gen_kwargs.get("temperature", self.temperature)
                    do_sample = gen_kwargs.get("do_sample", True if temperature > 0 else False)

                    with torch.no_grad():
                        outputs = self._model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature if do_sample else None,
                            top_p=0.9 if do_sample else None,
                            repetition_penalty=1.2,
                            pad_token_id=self.tokenizer.eos_token_id,
                            use_cache=self.use_cache,
                        )

                    # Decode only the newly generated tokens (exclude input tokens)
                    input_len = inputs["input_ids"].shape[-1]
                    generated_tokens = outputs[0][input_len:]
                    output_text = self._processor.decode(generated_tokens, skip_special_tokens=True)
                    output_text = self._postprocess_output(output_text)

                    batch_outputs.append(output_text)

                res.extend(batch_outputs)
                pbar.update(len(batch_outputs))

            except Exception as e:
                import traceback

                eval_logger.error(f"Error during generation: {e}")
                eval_logger.error(f"Traceback: {traceback.format_exc()}")
                res.extend([""] * len(contexts))
                pbar.update(len(contexts))

        pbar.close()
        return re_ords.get_original(res)

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Multi-round generation is not yet supported for Mistral3-VL")
