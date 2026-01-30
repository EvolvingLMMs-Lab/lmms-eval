import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Import both MoE and non-MoE model classes
try:
    from transformers import Glm4vMoeForConditionalGeneration
except ImportError:
    Glm4vMoeForConditionalGeneration = None

try:
    from transformers import Glm4vForConditionalGeneration
except ImportError:
    Glm4vForConditionalGeneration = None

if Glm4vMoeForConditionalGeneration is None and Glm4vForConditionalGeneration is None:
    eval_logger.warning("Failed to import GLM4V model classes. " "Please install transformers>=5.0.0: pip install transformers>=5.0.0")


@register_model("glm4v")
class GLM4V(lmms):
    """
    GLM-4.6V Model
    https://huggingface.co/zai-org/GLM-4.6V

    Supports:
    - GLM-4.6V (106B MoE model) - requires multiple GPUs
    - GLM-4.6V-Flash (9B model) - fits on single GPU

    Usage examples:
    - Flash model: pretrained=zai-org/GLM-4.6V-Flash
    - Full model (multi-GPU): pretrained=zai-org/GLM-4.6V
    """

    def __init__(
        self,
        pretrained: str = "zai-org/GLM-4.6V-Flash",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        max_new_tokens: int = 8192,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Determine model class based on pretrained path
        is_moe_model = "Flash" not in pretrained

        if is_moe_model:
            if Glm4vMoeForConditionalGeneration is None:
                raise ImportError("Glm4vMoeForConditionalGeneration not available. " "Please install transformers>=5.0.0: pip install transformers>=5.0.0")
            model_class = Glm4vMoeForConditionalGeneration
            eval_logger.info(f"Loading GLM-4.6V MoE model from {pretrained}")
        else:
            if Glm4vForConditionalGeneration is None:
                raise ImportError("Glm4vForConditionalGeneration not available. " "Please install transformers>=5.0.0: pip install transformers>=5.0.0")
            model_class = Glm4vForConditionalGeneration
            eval_logger.info(f"Loading GLM-4.6V Flash model from {pretrained}")

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
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "device_map": self.device_map,
            "torch_dtype": torch.bfloat16,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = model_class.from_pretrained(pretrained, **model_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens

        self._config = self.model.config
        self._max_length = 128000  # GLM-4.6V supports 128K context
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
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
        else:
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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for GLM4V")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL."""
        base64_image = image.convert("RGB")
        buffer = BytesIO()
        base64_image.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        base64_string = base64_bytes.decode("utf-8")
        return f"data:image/jpeg;base64,{base64_string}"

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = []
                if self.system_prompt:
                    message.append({"role": "system", "content": self.system_prompt})

                content = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, Image.Image):
                            content.append(
                                {
                                    "type": "image",
                                    "url": self._image_to_base64(visual),
                                }
                            )
                        elif isinstance(visual, str):
                            # URL or file path
                            content.append(
                                {
                                    "type": "image",
                                    "url": visual,
                                }
                            )

                content.append({"type": "text", "text": context})
                message.append({"role": "user", "content": content})
                batched_messages.append(message)

            # Process inputs using chat template
            inputs = self.processor.apply_chat_template(
                batched_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )

            # Remove token_type_ids if present (GLM-4.6V specific)
            inputs.pop("token_type_ids", None)

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            # Decode only the generated tokens
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for GLM4V")
