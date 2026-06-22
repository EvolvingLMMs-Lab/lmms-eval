"""qwen2_5_omni — Qwen2.5-Omni chat-style wrapper.

See `_chat_base.ChatMixin` for the shared rationale and request loop.
This file only implements the Qwen-Omni-specific
chat_messages -> output step.
"""

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat._chat_base import ChatMixin
from lmms_eval.models.simple.qwen2_5_omni import Qwen2_5_Omni as Qwen2_5_OmniSimple
from lmms_eval.protocol import ChatMessage, ChatMessages, ChatTextContent

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    eval_logger.warning("Failed to import qwen_omni_utils; install via `pip install qwen-omni-utils[decord]`")


@register_model("qwen2_5_omni_chat")
class Qwen2_5_Omni(ChatMixin, Qwen2_5_OmniSimple):
    """Qwen2.5-Omni that consumes chat-style doc_to_messages prompts."""

    def _with_system(self, chat_messages: ChatMessages) -> ChatMessages:
        system = ChatMessage(role="system", content=[ChatTextContent(text=self.system_prompt)])
        return ChatMessages(messages=[system, *chat_messages.messages])

    def _infer_one(self, chat_messages: ChatMessages, gen_kwargs: dict) -> str:
        chat_messages = self._with_system(chat_messages)
        hf_messages = chat_messages.to_hf_messages(
            image_kwargs=self.image_kwargs,
            video_kwargs=self.video_kwargs,
        )

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
