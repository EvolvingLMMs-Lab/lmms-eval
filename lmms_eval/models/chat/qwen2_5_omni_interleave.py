"""qwen2_5_omni_interleave — Qwen2.5-Omni over interleaved doc_to_messages.

See `_interleave_base.InterleaveChatMixin` for the shared rationale and
request loop. This file only implements the Qwen-Omni-specific
messages -> output step.
"""

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat._interleave_base import (
    IMAGE_KWARGS,
    VIDEO_KWARGS,
    InterleaveChatMixin,
)
from lmms_eval.models.simple.qwen2_5_omni import Qwen2_5_Omni

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    eval_logger.warning("Failed to import qwen_omni_utils; install via `pip install qwen-omni-utils[decord]`")


def _to_qwen_messages(messages: list, image_kwargs=None, video_kwargs=None) -> list:
    """doc_to_messages blocks -> Qwen-Omni native format with size/frame caps."""
    image_kwargs = IMAGE_KWARGS if image_kwargs is None else image_kwargs
    video_kwargs = VIDEO_KWARGS if video_kwargs is None else video_kwargs
    out = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        new_content = []
        for c in content:
            t = c.get("type")
            if t == "text":
                new_content.append({"type": "text", "text": c.get("text", "")})
            elif t == "image":
                new_content.append({"type": "image", "image": c["url"], **image_kwargs})
            elif t == "video":
                new_content.append({"type": "video", "video": c["url"], **video_kwargs})
            elif t == "audio":
                new_content.append({"type": "audio", "audio": c["url"]})
            else:
                new_content.append(c)
        out.append({"role": msg["role"], "content": new_content})
    return out


@register_model("qwen2_5_omni_interleave")
class Qwen2_5_OmniInterleave(InterleaveChatMixin, Qwen2_5_Omni):
    """Qwen2.5-Omni that consumes interleaved doc_to_messages prompts."""

    def _infer_one(self, messages: list, gen_kwargs: dict) -> str:
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        hf_messages = _to_qwen_messages(messages, self.image_kwargs, self.video_kwargs)

        # Audio/video are separate content blocks (silent MELD clips).
        use_audio_in_video = False

        text = self.processor.apply_chat_template(hf_messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(hf_messages, use_audio_in_video=use_audio_in_video)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        if self.device_map == "auto":
            inputs = inputs.to("cuda").to(self.model.dtype)
        else:
            inputs = inputs.to(self.model.device).to(self.model.dtype)

        gen_kwargs.setdefault("max_new_tokens", 4096)
        gen_kwargs.setdefault("temperature", 0)
        gen_kwargs.setdefault("top_p", None)
        gen_kwargs.setdefault("num_beams", 1)

        cont = self.model.generate(
            **inputs,
            return_audio=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=gen_kwargs["temperature"] > 0,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            use_cache=self.use_cache,
            use_audio_in_video=use_audio_in_video,
            thinker_do_sample=False,
        )
        if isinstance(cont, tuple):
            cont = cont[0]
        # Decode the full sequence (NOT out_ids[len(in_ids):]) and take the
        # text after the final assistant turn. For multimodal inputs the
        # processor expands media placeholders, so input_ids length does NOT
        # align with the generated sequence prefix — trimming by it yields
        # empty strings on video inputs. This mirrors the upstream
        # XModBench/AudioBench Qwen2.5-Omni runner.
        full = self.processor.batch_decode(cont, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return full.split("assistant\n")[-1].strip()
