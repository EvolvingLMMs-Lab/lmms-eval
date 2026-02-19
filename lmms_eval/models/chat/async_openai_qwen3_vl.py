from typing import Dict, Tuple, List

from lmms_eval.models.chat.async_openai import AsyncOpenAIChat
from lmms_eval.api.registry import register_model
from lmms_eval.protocol import ChatMessages


@register_model("async_openai_qwen3_vl")
class AsyncOpenAIQwen3VLChat(AsyncOpenAIChat):
    """Async OpenAI-compatible model with Qwen3-VL message format.

    Inherits from AsyncOpenAIChat but overrides prepare_messages to use
    Qwen3-VL specific message formatting.
    """
    is_simple = False

    def prepare_messages(self, chat_messages: ChatMessages) -> Tuple[List[Dict], Tuple]:
        """Prepare Qwen3-VL compatible messages from chat messages.

        Args:
            chat_messages: The chat messages object containing user queries and media.

        Returns:
            A tuple of (messages, video_kwargs) where messages is the Qwen3-VL
            compatible message format and video_kwargs contains video processing parameters.
        """
        video_kwargs = {"max_pixels": self.max_pixels, "min_pixels": self.min_pixels}
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
        else:
            video_kwargs["nframes"] = self.nframes
        if self.max_frames is not None:
            video_kwargs["max_frames"] = self.max_frames
        messages = chat_messages.to_qwen3_vl_openai_messages(video_kwargs)
        return messages, video_kwargs
