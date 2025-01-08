from mimetypes import guess_type
import cv2
import numpy as np
from PIL import Image
import re
import os
from ast import literal_eval


##Image utils


def read_image(image_path):
    image = Image.open(image_path)
    # Handle the alpha channel
    if image.mode == "RGBA":
        image = _rgba_to_rgb(image)
    else:
        image = image.convert("RGB")
    return image


def _rgba_to_rgb(image):
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


## Video utils
def subsample_video(video_path, max_nframes):
    """
    Process video file and return uniformly sampled frames as PIL Images.

    Args:
        video_path (str): Path to the video file
        max_nframes (int): Maximum number of frames to return

    Returns:
        list[PIL.Image]: List of sampled frames as PIL Images
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise ValueError(f"Failed to read video from {video_path}, check data...")

    # If total frames <= max_nframes, return all frames
    if total_frames <= max_nframes:
        frame_indices = range(total_frames)
    else:
        # Calculate indices for uniform sampling
        frame_indices = np.linspace(0, total_frames - 1, max_nframes, dtype=int)

    frames = []
    for frame_idx in frame_indices:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)

    cap.release()
    return frames


def is_video_file(file_path):
    mime_type, _ = guess_type(file_path)
    if not mime_type:
        return False
    return mime_type.startswith("video")


## Handle tasks with mixed image and video inputs.
## Need to subsample video frames to multiple images


def load_media_content(media_path, max_nframes):
    # normalize media path
    if is_video_file(media_path):
        images = subsample_video(media_path, max_nframes)
    else:
        images = []
        images.append(read_image(media_path))
    return images


def process_text_and_mixed_media(doc, max_nframes, cache_dir):
    global_text, global_images = _process_text_and_mixed_media(doc["task_description"], doc["global_media"], max_nframes, cache_dir)
    example_text, example_images = _process_text_and_mixed_media(doc["example_text"], doc["example_media"], max_nframes, cache_dir)
    query_text, query_images = _process_text_and_mixed_media(doc["query_text"], doc["query_media"], max_nframes, cache_dir)
    prompt = "\n".join([global_text, example_text, query_text])
    images = global_images + example_images + query_images
    return prompt, images


def _process_text_and_mixed_media(text, media_paths, max_nframes, cache_dir):
    """
    Process the text prompt and the input medias when the media files contain
    both image and video. In this case, sample frames from the video and adjust
    the image placeholders in the text prompt accordingly.
    """
    text_chunks = re.split(r"(<image>|<video>)", text)

    if isinstance(media_paths, str):
        media_paths = literal_eval(media_paths)

    media_paths = [os.path.join(cache_dir, local_path) for local_path in media_paths]

    placeholder_count = sum(1 for chunk in text_chunks if chunk in ["<image>", "<video>"])
    if placeholder_count != len(media_paths):
        raise ValueError(f"Mismatching # placeholders ({placeholder_count}) and # media paths ({len(media_paths)}). Please check the data...")

    media_index = 0
    images = []
    texts = []
    for chunk in text_chunks:
        if chunk in ["<image>", "<video>"]:
            media_content = load_media_content(media_paths[media_index], max_nframes)
            if len(media_content) == 1:  # image
                images.extend(media_content)
                texts.append("<image>")
            else:  # video
                images.extend(media_content)
                placeholder_str = "[video start]" + "<image>" * len(media_content) + "[video end]"
                texts.append(placeholder_str)
            media_index += 1
        elif chunk.strip():
            texts.append(chunk.strip())

    fused_text = " ".join(texts)
    return fused_text, images
