import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from transformers import Llama4ForConditionalGeneration
except ImportError:
    Llama4ForConditionalGeneration = None
    eval_logger.warning("Failed to import Llama4ForConditionalGeneration. " "Please install transformers>=4.51.0: pip install transformers>=4.51.0")


@register_model("llama4_scout")
class Llama4Scout(lmms):
    """
    Llama-4-Scout Model
    https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct

    Meta's Llama 4 Scout is a 17B active parameter (109B total) MoE model
    with native multimodal capabilities (text + images + video).

    Requirements:
    - transformers >= 4.51.0
    - GPU memory: ~218GB for bf16 (4x H100 80GB recommended)

    Usage examples:
    - pretrained=meta-llama/Llama-4-Scout-17B-16E-Instruct
    """

    def __init__(
        self,
        pretrained: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        max_new_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        max_frames_num: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if Llama4ForConditionalGeneration is None:
            raise ImportError("Llama4ForConditionalGeneration not available. " "Please install transformers>=4.51.0: pip install transformers>=4.51.0")

        # Validate attention implementation
        valid_attn_implementations = [None, "flex_attention", "sdpa", "eager"]
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

        eval_logger.info(f"Loading Llama-4-Scout model from {pretrained}")

        # Load processor/tokenizer first to get token IDs
        self.processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = self.processor.tokenizer

        # Load config and set pad_token_id from tokenizer (required by Llama4 model __init__)
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained)
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = self._tokenizer.pad_token_id
        if not hasattr(config, "eos_token_id") or config.eos_token_id is None:
            config.eos_token_id = self._tokenizer.eos_token_id
        model_kwargs["config"] = config

        self._model = Llama4ForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.max_frames_num = max_frames_num

        self._config = self.model.config
        self._max_length = 131072  # Llama-4-Scout supports up to 10M tokens, using 128K as practical limit
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP, FSDP, and DEEPSPEED are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
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
        raise NotImplementedError("Loglikelihood is not implemented for Llama4Scout")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if isinstance(video_path, str):
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

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
                        if isinstance(visual, PIL.Image.Image):
                            content.append(
                                {
                                    "type": "image",
                                    "url": self._image_to_base64(visual),
                                }
                            )
                        elif isinstance(visual, str):
                            # Video path - load frames
                            if visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                                frames = self.load_video(visual, self.max_frames_num)
                                frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
                                for frame in frames:
                                    pil_frame = to_pil_image(frame)
                                    content.append(
                                        {
                                            "type": "image",
                                            "url": self._image_to_base64(pil_frame),
                                        }
                                    )
                            else:
                                # Image URL or path
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
            # Handle case where pad_token_id is None (common for Llama models)
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id

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
        raise NotImplementedError("Multi-round generation is not implemented for Llama4Scout")
