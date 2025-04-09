import base64
import json
import os
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

try:
    from decord import VideoReader, cpu
except ImportError:
    raise ImportError("Please install decord: pip install decord")


# Function to load video frames using Decord
def load_video(video_path, max_frames_num=8):
    """Load video frames using Decord, consistent with llama_vision.py"""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


# Function to encode image to base64
def encode_image(image, format="JPEG", quality=85, max_size=(800, 600)):
    """Encode image to base64 string with resizing and compression"""
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.copy()

    # Resize image to reduce size while maintaining aspect ratio
    img.thumbnail(max_size, Image.Resampling.LANCZOS)

    output_buffer = BytesIO()
    # Use JPEG format with quality setting for better compression
    img.save(output_buffer, format=format, quality=quality)
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str


# Specify video path
video_path = os.path.expanduser("~/.cache/huggingface/video_mmmu/Engineering/new_Computer_Science_1.mp4")

# Process the video frames
try:
    print(f"Loading video from: {video_path}")
    frames = load_video(video_path, max_frames_num=2)
    print(f"Successfully loaded {len(frames)} frames with shape {frames.shape}")

    # Convert frames to tensors, then to PIL images, then encode to base64
    frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # Convert to tensor with shape (B, C, H, W)
    frames_pil = [to_pil_image(frame) for frame in frames_tensor]

    # Encode with compression
    frames_base64 = [encode_image(frame, format="JPEG", quality=80, max_size=(640, 480)) for frame in frames_pil]

    print(f"Successfully processed {len(frames_base64)} frames to base64")

    # For the first frame, check size of base64 string
    if frames_base64:
        print(f"First frame base64 size: {len(frames_base64[0])} characters (after compression)")

    # Create the messages structure
    messages = [{"role": "user", "content": [{"type": "text", "text": "what is the topic of the video"}]}]

    # Add each frame to the message content
    for i, frame_base64 in enumerate(frames_base64):
        messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}})
        print(f"Added frame {i+1} to message (size: {len(frame_base64)} characters)")

    # Make the API request
    print("\nSending request to API with video frames...\n")
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer ",
            "Content-Type": "application/json",
        },
        data=json.dumps({"model": "meta-llama/llama-4-maverick:free", "messages": messages, "max_tokens": 1024, "temperature": 0, "top_p": 1.0}),
    )

    # Print response
    print("API response status code:", response.status_code)
    if response.status_code == 200:
        print("\nAPI RESPONSE:")
        print(response.json()["choices"][0]["message"]["content"])
    else:
        print("API error:", response.text)

except Exception as e:
    print(f"Error processing video: {str(e)}")
