import os
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

UNI_MOE_AVAILABLE = False
try:
    from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
    from uni_moe.model.modeling_out import GrinQwen2VLOutForConditionalGeneration
    from uni_moe.qwen_vl_utils import process_mm_info
    UNI_MOE_AVAILABLE = True
except ImportError:
    eval_logger.warning(
        "Failed to import uni_moe modules. Please install from https://github.com/HITsz-TMG/Uni-MoE:\n"
        "  git clone https://github.com/HITsz-TMG/Uni-MoE\n"
        "  export PYTHONPATH=/path/to/Uni-MoE/Uni-MoE-2:$PYTHONPATH"
    )
    Qwen2VLProcessor = None
    GrinQwen2VLOutForConditionalGeneration = None
    process_mm_info = None


@register_model("uni_moe")
class UniMoE(lmms):
    """
    Uni-MoE-2.0-Omni
    https://huggingface.co/HIT-TMG/Uni-MoE-2.0-Omni
    """

    def __init__(
        self,
        pretrained: str = "HIT-TMG/Uni-MoE-2.0-Omni",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        max_num_frames: int = 32,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if not UNI_MOE_AVAILABLE:
            raise ImportError(
                "uni_moe module is not available. Please install from https://github.com/HITsz-TMG/Uni-MoE:\n"
                "  git clone https://github.com/HITsz-TMG/Uni-MoE\n"
                "  export PYTHONPATH=/path/to/Uni-MoE/Uni-MoE-2:$PYTHONPATH"
            )

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.processor = Qwen2VLProcessor.from_pretrained(pretrained)
        self._model = GrinQwen2VLOutForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        self.processor.data_args = self._model.config

        self.max_num_frames = max_num_frames
        self._tokenizer = self.processor.tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.system_prompt = system_prompt

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
        raise NotImplementedError("Loglikelihood is not implemented for UniMoE")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

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
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            until = [self.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            messages = []
            for i, context in enumerate(contexts):
                content = []
                has_image = False
                has_audio = False

                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None

                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                        content.append({"type": "text", "text": "<audio>\n<image>\n" + context})
                        content.append({"type": "video", "video": visual})
                        has_audio = True
                        has_image = True

                    elif isinstance(visual, Image.Image):
                        content.append({"type": "text", "text": "<image>\n" + context})
                        content.append({"type": "image", "image": visual})
                        has_image = True

                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
                        text_part = "<image>\n" * len(visual) + context
                        content.append({"type": "text", "text": text_part})
                        for v in visual:
                            content.append({"type": "image", "image": v})
                        has_image = True

                    elif isinstance(visual, dict) and "array" in visual:
                        import tempfile
                        import soundfile as sf
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                            sf.write(f.name, visual["array"], visual["sampling_rate"])
                            audio_path = f.name
                        content.append({"type": "text", "text": "<audio>\n" + context})
                        content.append({"type": "audio", "audio": audio_path})
                        has_audio = True

                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, dict) for v in visual):
                        import tempfile
                        import soundfile as sf
                        text_part = "<audio>\n" * len(visual) + context
                        content.append({"type": "text", "text": text_part})
                        for v in visual:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                                sf.write(f.name, v["array"], v["sampling_rate"])
                                content.append({"type": "audio", "audio": f.name})
                        has_audio = True

                    else:
                        content.append({"type": "text", "text": context})
                else:
                    content.append({"type": "text", "text": context})

                messages.append({"role": "user", "content": content})

            texts = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts = texts.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            texts = texts.replace("<audio>", "<|audio_start|><|audio_pad|><|audio_end|>")

            image_inputs, video_inputs, audio_inputs = process_mm_info(messages)

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                audios=audio_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs["input_ids"] = inputs["input_ids"].unsqueeze(0) if inputs["input_ids"].dim() == 1 else inputs["input_ids"]
            inputs = inputs.to(device=self.model.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            try:
                output_ids = self.model.generate(
                    **inputs,
                    use_cache=self.use_cache,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    temperature=gen_kwargs["temperature"] if gen_kwargs["temperature"] > 0 else None,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                )
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                answer = ""
                res.append(answer)
                pbar.update(1)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                continue

            answers = self.processor.batch_decode(
                output_ids[:, inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )

            for ans, context in zip(answers, contexts):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
