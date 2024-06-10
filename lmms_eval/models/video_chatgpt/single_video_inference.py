"""
How to run this file:

cd VideoChatGPT
python -m video_chatgpt.single_video_inference \
    --model-name <path of llava weights, for eg "LLaVA-7B-Lightening-v1-1"> \
    --projection_path <path of projection for eg "video-chatgpt-weights/video_chatgpt-7B.bin"> \
    --video_path <video_path>
"""

from lmms_eval.models.video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from lmms_eval.models.video_chatgpt.model.utils import KeywordsStoppingCriteria
import torch

# add new packages as below
from PIL import Image
from decord import VideoReader, cpu
from lmms_eval.models.video_chatgpt.eval.model_utils import initialize_model, load_video
import argparse
import numpy as np
import os

# Define constants
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"


def get_spatio_temporal_features_torch(features):
    """
    Computes spatio-temporal features from given features.

    Parameters:
    features (torch.Tensor): Input features to process.

    Returns:
    torch.Tensor: Spatio-temporal features.
    """

    # Extract the dimensions of the features
    t, s, c = features.shape

    # Compute temporal tokens as the mean along the time axis
    temporal_tokens = torch.mean(features, dim=1)

    # Padding size calculation
    padding_size = 100 - t

    # Pad temporal tokens if necessary
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)

    # Compute spatial tokens as the mean along the spatial axis
    spatial_tokens = torch.mean(features, dim=0)

    # Concatenate temporal and spatial tokens and cast to half precision
    concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

    return concat_tokens


def video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len):
    """
    Run inference using the Video-ChatGPT model.

    Parameters:
    sample : Initial sample
    video_frames (torch.Tensor): Video frames to process.
    question (str): The question string.
    conv_mode: Conversation mode.
    model: The pretrained Video-ChatGPT model.
    vision_tower: Vision model to extract video features.
    tokenizer: Tokenizer for the model.
    image_processor: Image processor to preprocess video frames.
    video_token_len (int): The length of video tokens.

    Returns:
    dict: Dictionary containing the model's output.
    """

    # Prepare question string for the model
    if model.get_model().vision_config.use_vid_start_end:
        qs = question + "\n" + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
    else:
        qs = question + "\n" + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len

    # Prepare conversation prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize the prompt
    inputs = tokenizer([prompt])

    # Preprocess video frames and get image tensor
    image_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]

    # Move image tensor to GPU and reduce precision to half
    image_tensor = image_tensor.half().cuda()

    # Generate video spatio-temporal features
    with torch.no_grad():
        image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
        frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # Use second to last layer as in LLaVA
    video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)

    # Move inputs to GPU
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    # Define stopping criteria for generation
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # Run model inference
    with torch.inference_mode():
        output_ids = model.generate(input_ids, video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0), do_sample=True, temperature=0.2, max_new_tokens=1024, stopping_criteria=[stopping_criteria])

    # Check if output is the same as input
    n_diff_input_output = (input_ids != output_ids[:, : input_ids.shape[1]]).sum().item()
    if n_diff_input_output > 0:
        print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")

    # Decode output tokens
    outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1] :], skip_special_tokens=True)[0]

    # Clean output string
    outputs = outputs.strip().rstrip(stop_str).strip()

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--vision_tower_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--projection_path", type=str, required=False, default="")
    parser.add_argument("--video_path", type=str, required=True, default="")
    parser.add_argument("--conv_mode", type=str, required=False, default="video-chatgpt_v1")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path)

    video_path = args.video_path

    if os.path.exists(video_path):
        video_frames = load_video(video_path)

    question = input("Enter a question to check from the video:")
    conv_mode = args.conv_mode

    try:
        # Run inference on the video and add the output to the list
        output = video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len)
        print("\n\n", output)

    except Exception as e:
        print(f"Error processing video file '{video_path}': {e}")
