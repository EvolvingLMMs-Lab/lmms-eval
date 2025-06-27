import base64
from io import BytesIO
from typing import Any, Dict, List, Literal, Union

import numpy as np
from decord import VideoReader, cpu
from PIL import Image
from pydantic import BaseModel


class ChatTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatImageContent(BaseModel):
    type: Literal["image"] = "image"
    url: Any


class ChatVideoContent(BaseModel):
    type: Literal["video"] = "video"
    url: Any


class ChatAudioContent(BaseModel):
    type: Literal["audio"] = "audio"
    url: Any


ChatContent = Union[ChatTextContent, ChatImageContent, ChatVideoContent, ChatAudioContent]


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
                elif content.type == "video":
                    videos.append(content.url)
                elif content.type == "audio":
                    audios.append(content.url)

        return images, videos, audios

    def to_hf_messages(self):
        hf_messages = []
        for message in self.messages:
            hf_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    hf_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    hf_message["content"].append({"type": "image", "image": content.url})
                elif content.type == "video":
                    hf_message["content"].append({"type": "video", "video": content.url})
                elif content.type == "audio":
                    hf_message["content"].append({"type": "audio", "audio": content.url})
            hf_messages.append(hf_message)
        return hf_messages

    def to_openai_messages(self):
        openai_messages = []
        for message in self.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    openai_message["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.encode_image(content.url)}"}})
                elif content.type == "video":
                    openai_message["content"].append({"type": "video_url", "video_url": {"url": f"data:image/png;base64,{self.encode_video(content.url)}"}})
                # TODO, audio hasn't been implemented yet
                elif content.type == "audio":
                    openai_message["content"].append({"type": "audio_url", "audio_url": {"url": content.url}})
            openai_messages.append(openai_message)
        return openai_messages

    def encode_image(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frame_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames
