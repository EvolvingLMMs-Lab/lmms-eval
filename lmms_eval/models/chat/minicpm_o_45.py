"""minicpm_o_45 — MiniCPM-o-4.5 chat-style wrapper.

See `_chat_base.ChatMixin` for the shared rationale and request loop.
MiniCPM-o-4.5's `model.chat` takes a single user message whose `content` is a
list of PIL images, np audio arrays and per-second video chunks followed by the
text. This variant reads the task's `doc_to_messages` (or the messages that
lmms-eval builds from `doc_to_visual` / `doc_to_text`): the media items keep
their message order and the text parts are joined after them.
"""

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat._chat_base import ChatMixin
from lmms_eval.models.simple.minicpm_o_45 import MiniCPM_O_45 as MiniCPM_O_45Simple
from lmms_eval.protocol import ChatMessages


@register_model("minicpm_o_45_chat")
class MiniCPM_O_45Chat(ChatMixin, MiniCPM_O_45Simple):
    """MiniCPM-o-4.5 that consumes chat-style doc_to_messages prompts."""

    def _infer_one(self, chat_messages: ChatMessages, gen_kwargs: dict) -> str:
        # Media go through the half-duplex builder, which accepts what the chat
        # path delivers: file paths, in-memory PIL images, decoded audio dicts
        # and AudioDecoder objects. A media item that cannot be loaded raises,
        # so ChatMixin reports the failure instead of answering without it.
        visuals = []
        texts = []
        for msg in chat_messages.messages:
            for c in msg.content:
                if c.type == "text":
                    texts.append(c.text)
                elif c.type in ("image", "video", "audio"):
                    visuals.append(c.url)
        content, has_audio = self.build_media_content(visuals)
        return self._call_chat(content, "\n".join(texts), gen_kwargs, has_audio)
