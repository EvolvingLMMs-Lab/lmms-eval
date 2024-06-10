import os
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from lmms_eval.models.video_chatgpt.model import VideoChatGPTLlamaForCausalLM
from lmms_eval.models.video_chatgpt.utils import disable_torch_init
from lmms_eval.models.video_chatgpt.constants import *
import torch


def load_video(vis_path, n_clips=1, num_frm=100):
    """
    Load video frames from a video file.

    Parameters:
    vis_path (str): Path to the video file.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

    # Load video with VideoReader
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)

    # Currently, this function supports only 1 clip
    assert n_clips == 1

    # Calculate total number of frames to extract
    total_num_frm = min(total_frame_num, num_frm)
    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # Extract frames as numpy array
    img_array = vr.get_batch(frame_idx).asnumpy()
    # Set target image height and width
    target_h, target_w = 224, 224
    # If image shape is not as target, resize it
    if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

    # Reshape array to match number of clips and frames
    img_array = img_array.reshape((n_clips, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
    # Convert numpy arrays to PIL Image objects
    clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]

    return clip_imgs


def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq


def initialize_model(model_name, projection_path=None, device="cuda"):
    """
    Initializes the model with given parameters.

    Parameters:
    model_name (str): Name of the model to initialize.
    projection_path (str, optional): Path to the projection weights. Defaults to None.

    Returns:
    tuple: Model, vision tower, tokenizer, image processor, vision config, and video token length.
    """

    # Disable initial torch operations
    disable_torch_init()

    # Convert model name to user path
    model_name = os.path.expanduser(model_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    model = VideoChatGPTLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True)

    # Load image processor
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

    # Set to use start and end tokens for video
    mm_use_vid_start_end = True

    # Add tokens to tokenizer
    tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    if mm_use_vid_start_end:
        tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)

    # Resize token embeddings of the model
    model.resize_token_embeddings(len(tokenizer))

    # Load the weights from projection_path after resizing the token_embeddings
    if projection_path:
        print(f"Loading weights from {projection_path}")
        status = model.load_state_dict(torch.load(projection_path, map_location="cpu"), strict=False)
        if status.unexpected_keys:
            print(f"Unexpected Keys: {status.unexpected_keys}.\nThe Video-ChatGPT weights are not loaded correctly.")
        print(f"Weights loaded from {projection_path}")

    # Set model to evaluation mode and move to GPU
    model = model.eval()
    model = model.to(device)

    vision_tower_name = "openai/clip-vit-large-patch14"

    # Load vision tower and move to GPU
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    vision_tower = vision_tower.eval()

    # Configure vision model
    vision_config = model.get_model().vision_config
    vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]
    vision_config.use_vid_start_end = mm_use_vid_start_end
    if mm_use_vid_start_end:
        vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])

    # Set video token length
    video_token_len = 356

    return model, vision_tower, tokenizer, image_processor, video_token_len
