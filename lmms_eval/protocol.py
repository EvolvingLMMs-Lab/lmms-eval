from typing import Any, Dict, List, Literal, Union

from PIL import Image
from pydantic import BaseModel


class ChatTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatImageContent(BaseModel):
    type: Literal["image"] = "image"
    url: Any

    def model_dump(self, **kwargs):
        content = super().model_dump(**kwargs)
        # Some model may need this placeholder for hf_chat_template
        content["image_url"] = "placeholder"
        return content


ChatContent = Union[ChatTextContent, ChatImageContent]


class ChatMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: List[ChatContent]


class ChatMessages(BaseModel):
    messages: List[ChatMessage]

    def extract_media(self):
        images = []
        videos = []
        audios = []

        for message in self.messages:
            for content in message.content:
                if content.type == "image":
                    images.append(content.url)

        return images, videos, audios
