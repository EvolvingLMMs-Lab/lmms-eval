from typing import Dict, Tuple, List

from lmms_eval.models.chat.async_openai import AsyncOpenAIChat
from lmms_eval.api.registry import register_model
from lmms_eval.protocol import ChatMessages


@register_model("async_openai_qwen3_vl")
class AsyncOpenAIQwen3VLChat(AsyncOpenAIChat):
    is_simple = False

    def prepare_messages(self, chat_messages: ChatMessages) -> Tuple[List[Dict], Tuple]:
        video_kwargs = {"max_pixels": self.max_pixels, "min_pixels": self.min_pixels}
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
        else:
            video_kwargs["nframes"] = self.nframes
        if self.max_frames is not None:
            video_kwargs["max_frames"] = self.max_frames
        messages = chat_messages.to_qwen3_vl_openai_messages(video_kwargs)
        return messages, video_kwargs
