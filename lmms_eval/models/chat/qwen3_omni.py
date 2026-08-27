"""qwen3_omni — Qwen3-Omni chat-style wrapper.

See `_chat_base.ChatMixin` for the shared rationale and request loop.
Only the Qwen3-Omni-specific inference step lives here.
"""

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat._chat_base import ChatMixin
from lmms_eval.models.simple.qwen3_omni import Qwen3_Omni as Qwen3_OmniSimple
from lmms_eval.protocol import ChatMessage, ChatMessages, ChatTextContent

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    eval_logger.warning("Failed to import qwen_omni_utils; install via `pip install qwen-omni-utils[decord]`")


@register_model("qwen3_omni_chat")
class Qwen3_Omni(ChatMixin, Qwen3_OmniSimple):
    """Qwen3-Omni that consumes chat-style doc_to_messages prompts."""

    def _with_system(self, chat_messages: ChatMessages) -> ChatMessages:
        system = ChatMessage(role="system", content=[ChatTextContent(text=self.system_prompt)])
        return ChatMessages(messages=[system, *chat_messages.messages])

    def _infer_one(self, chat_messages: ChatMessages, gen_kwargs: dict) -> str:
        chat_messages = self._with_system(chat_messages)
        hf_messages = chat_messages.to_hf_messages(
            image_kwargs=self.image_kwargs,
            video_kwargs=self.video_kwargs,
        )

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
        # Decode full sequence + take text after the assistant turn (see
        # qwen2_5_omni for why trimming by input_ids breaks video).
        full = self.processor.batch_decode(cont, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return full.split("assistant\n")[-1].strip()
