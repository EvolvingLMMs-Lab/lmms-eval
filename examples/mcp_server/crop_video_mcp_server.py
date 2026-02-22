import base64
import logging
import os
from io import BytesIO
from typing import Annotated

import cv2
import torch
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent
from pydantic import Field
from qwen_vl_utils import fetch_video
from torchvision.transforms.functional import to_pil_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MCP_SERVER] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("/tmp/mcp_server_debug.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = FastMCP("Video Tools MCP Server", "0.1.0")


@app.tool(name="crop_video", description="Crop a video to a specified duration.")
def crop_video(
    video_path: Annotated[str, Field(description="Path to the video file")] = None,
    start_time: Annotated[float, Field(description="Start time in seconds")] = None,
    end_time: Annotated[float, Field(description="End time in seconds, must be > start_time")] = None,
) -> list[ImageContent]:
    """
    Crop a video to a specified duration.

    Args:
        video_path (str): Path to the video file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        str: Path to the cropped video file.
    """
    # Validate input parameters - now we control all validation logic
    logger.info(f"Validating parameters: video_path={video_path}, start_time={start_time}, end_time={end_time}")

    # Check required parameters
    if video_path is None:
        logger.error("Missing video_path parameter")
        raise ValueError("video_path parameter is required")

    if start_time is None:
        logger.error("Missing start_time parameter")
        raise ValueError("start_time parameter is required")

    if end_time is None:
        logger.error("Missing end_time parameter")
        raise ValueError("end_time parameter is required")

    # Check parameter values
    if not video_path or video_path.strip() == "":
        logger.error(f"Empty video_path parameter: '{video_path}'")
        raise ValueError("video_path cannot be empty")

    if start_time < 0:
        logger.error(f"Invalid start_time: {start_time}")
        raise ValueError(f"start_time must be non-negative, got {start_time}")

    if end_time <= start_time:
        logger.error(f"Invalid time range: start_time={start_time}, end_time={end_time}")
        raise ValueError(f"end_time ({end_time}) must be greater than start_time ({start_time})")

    # Check file existence
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # verify video duration
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    print(f"video duration: {duration:.2f}s")
    cap.release()

    # validate time range
    if start_time >= duration:
        raise ValueError(f"start_time ({start_time}s) exceeds video duration ({duration:.2f}s)")
    if end_time > duration:
        raise ValueError(f"end_time ({end_time}s) exceeds video duration ({duration:.2f}s)")

    try:
        video_ele = {
            "type": "video",
            "video": f"file://{video_path}",
            "fps": 1,  # 1fps
            "min_frames": 1,
            "max_frames": 128,
            "max_pixels": 224 * 224,
            "video_start": start_time,
            "video_end": end_time,
        }
        video_frames = fetch_video(video_ele)
        video_frames = video_frames.to(torch.uint8)
        images = [to_pil_image(frame) for frame in video_frames]
        # Encode images to base64
        image_contents = []
        for img in images:
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            image_contents.append(ImageContent(type="image", data=base64_str, mimeType="image/png"))

        return image_contents
    except Exception as e:
        raise RuntimeError(f"Failed to process video {video_path}: {str(e)}") from e


if __name__ == "__main__":
    app.run()
