import base64
from io import BytesIO
from typing import Any, Dict, List, Literal, Union

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from pydantic import BaseModel

try:
    from qwen_vl_utils import fetch_video
except ImportError:
    fetch_video = None


class ChatTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatImageContent(BaseModel):
    type: Literal["image"] = "image"
    url: Any


class ChatVideoContent(BaseModel):
    type: Literal["video"] = "video"
    url: Any
    question: str = None


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

    def to_hf_messages(self, video_kwargs: Dict[str, str] = None):
        if video_kwargs is None:
            video_kwargs = {}
        num_frames = video_kwargs.get("nframes", 32)
        hf_messages = []
        for message in self.messages:
            hf_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    hf_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    hf_message["content"].append({"type": "image", "image": content.url})
                elif content.type == "video":
                    if content.question is not None:
                        video_kwargs["question"] = content.question
                    hf_message["content"].append({"type": "video", "video": content.url, **video_kwargs})
                elif content.type == "audio":
                    hf_message["content"].append({"type": "audio", "audio": content.url})
            hf_messages.append(hf_message)
        return hf_messages

    def to_openai_messages(self, video_kwargs: Dict[str, str] = {}):
        openai_messages = []
        for message in self.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    openai_message["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.encode_image(content.url)}"}})
                elif content.type == "video":
                    if fetch_video is None:
                        raise ImportError("qwen_vl_utils is required for video processing. Please install it with: pip install qwen-vl-utils")
                    video_input = fetch_video({"type": "video", "video": content.url, **video_kwargs})
                    for frame in video_input:
                        image = Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
                        openai_message["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.encode_image(image)}"}})
                # TODO, audio hasn't been implemented yet
                elif content.type == "audio":
                    openai_message["content"].append({"type": "audio_url", "audio_url": {"url": content.url}})
            openai_messages.append(openai_message)
        return openai_messages

    def to_qwen3_vl_openai_messages(self, video_kwargs: Dict[str, str] = {}):
        openai_messages = []
        for message in self.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    openai_message["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.encode_image(content.url)}"}})
                elif content.type == "video":
                    if fetch_video is None:
                        raise ImportError("qwen_vl_utils is required for video processing. Please install it with: pip install qwen-vl-utils")
                    video_input, fps = fetch_video({"type": "video", "video": content.url, **video_kwargs}, return_video_metadata=True, return_video_sample_fps=True)
                    frames, video_metadata = video_input
                    timestamps = self._calculate_timestamps(video_metadata)
                    for frame, timestamp in zip(frames, timestamps):
                        image = Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
                        openai_message["content"].append({"type": "text", "text": f"<{timestamp:.1f} seconds>"})
                        openai_message["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.encode_image(image)}"}})
                # TODO, audio hasn't been implemented yet
                elif content.type == "audio":
                    openai_message["content"].append({"type": "audio_url", "audio_url": {"url": content.url}})
            openai_messages.append(openai_message)
        return openai_messages

    def _calculate_timestamps(self, video_metadata: Dict[str, Any]):
        indices = video_metadata["frames_indices"]
        if not isinstance(indices, list):
            indices = indices.tolist()
        fps = video_metadata["fps"]
        # Note this is a hardcode value for Qwen3-VL, should only be used for Qwen3-VL
        merge_size = 2
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / fps for idx in indices]
        # timestamps = [(timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)]
        return timestamps

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
