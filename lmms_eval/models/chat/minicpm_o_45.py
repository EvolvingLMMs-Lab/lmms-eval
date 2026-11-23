"""minicpm_o_45 — MiniCPM-o-4.5 chat-style wrapper.

See `_chat_base.ChatMixin` for the shared rationale and request loop.
MiniCPM-o-4.5's `model.chat` takes a single user message whose `content` is an
ordered list mixing PIL images, np audio arrays, per-second video chunks, and
text. This variant builds the full content list from the task's
`doc_to_messages`, so the question stem and every option keep their media +
order — e.g. XModBench-style MCQ with media in the stem and in all four
options.
"""

from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat._chat_base import ChatMixin
from lmms_eval.models.simple.minicpm_o_45 import MiniCPM_O_45 as MiniCPM_O_45Simple
from lmms_eval.protocol import ChatMessages


@register_model("minicpm_o_45_chat")
class MiniCPM_O_45Chat(ChatMixin, MiniCPM_O_45Simple):
    """MiniCPM-o-4.5 that consumes chat-style doc_to_messages prompts."""

    def _infer_one(self, chat_messages: ChatMessages, gen_kwargs: dict) -> str:
        content = []
        has_audio = False
        for msg in chat_messages.messages:
            for c in msg.content:
                if c.type == "text":
                    content.append(c.text)
                elif c.type == "image":
                    content.append(Image.open(c.url).convert("RGB"))
                elif c.type == "video":
                    try:
                        content.extend(self._omni_video_content(c.url))
                        has_audio = has_audio or self.init_audio
                    except Exception as e:
                        eval_logger.warning(f"video encode failed {c.url}: {e}")
                elif c.type == "audio":
                    try:
                        content.append(self._load_audio(c.url))
                        has_audio = True
                    except Exception as e:
                        eval_logger.warning(f"audio load failed {c.url}: {e}")

        text_items = [item for item in content if isinstance(item, str)]
        context = "\n".join(text_items) if text_items else ""
        return self._call_chat([item for item in content if not isinstance(item, str)], context, gen_kwargs, has_audio)
