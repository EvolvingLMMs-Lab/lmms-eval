"""qwen3_omni_interleave — Qwen3-Omni over interleaved doc_to_messages.

See `_interleave_base.InterleaveChatMixin` for the shared rationale and
request loop. Only the Qwen3-Omni-specific inference step lives here.
"""

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat._interleave_base import InterleaveChatMixin
from lmms_eval.models.chat.qwen2_5_omni_interleave import _to_qwen_messages
from lmms_eval.models.simple.qwen3_omni import Qwen3_Omni

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    eval_logger.warning("Failed to import qwen_omni_utils; install via `pip install qwen-omni-utils[decord]`")


@register_model("qwen3_omni_interleave")
class Qwen3_OmniInterleave(InterleaveChatMixin, Qwen3_Omni):
    """Qwen3-Omni that consumes interleaved doc_to_messages prompts."""

    def _infer_one(self, messages: list, gen_kwargs: dict) -> str:
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        hf_messages = _to_qwen_messages(messages)

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
            temperature=gen_kwargs["temperature"] if gen_kwargs["temperature"] > 0 else None,
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            use_cache=self.use_cache,
            use_audio_in_video=use_audio_in_video,
            thinker_do_sample=False,
        )
        if isinstance(cont, tuple):
            cont = cont[0]
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
