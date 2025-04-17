import json
import logging
import re
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml
from huggingface_hub import hf_hub_download
from openai import RateLimitError
from PIL import Image, ImageDraw

# Suppress INFO level logs from the httpx library
logging.getLogger("httpx").setLevel(logging.WARNING)


def calculate_iou(range_1, range_2):

    start_1, end_1 = float(min(*range_1)), float(max(*range_1))
    start_2, end_2 = float(min(*range_2)), float(max(*range_2))

    intersection = max(0, min(end_1, end_2) - max(start_1, start_2))
    union = min(max(end_1, end_2) - min(start_1, start_2), end_1 - start_1 + end_2 - start_2)
    result = float(intersection) / (union + 1e-8)

    return result


def evaluate_detections(predicted_segments, gt_segments, iou_thresholds=(0.3, 0.5, 0.7, 0.9)):

    metrics = {}
    for threshold in iou_thresholds:
        metrics[str(threshold)] = {
            "gt_covered": set(),
            "pred_covered": set(),
        }
    gt_shape = gt_segments.shape[0]
    predicted_shape = predicted_segments.shape[0]

    iou_matrix = np.zeros((gt_shape, max(predicted_shape, 1)))
    for idx_g, gt_segment in enumerate(gt_segments):
        cur_max_iou = 0
        for idx_p, segment in enumerate(predicted_segments):
            sample_iou = calculate_iou(segment, gt_segment)
            iou_matrix[idx_g, idx_p] = sample_iou
            cur_max_iou = max(cur_max_iou, sample_iou)
            for threshold in iou_thresholds:
                if sample_iou > threshold:
                    metrics[str(threshold)]["pred_covered"].add(idx_p)
                    metrics[str(threshold)]["gt_covered"].add(idx_g)
    precision = []
    recall = []
    for threshold, m in metrics.items():
        pred_covered = m["pred_covered"]
        gt_covered = m["gt_covered"]
        m["precision"] = float(len(pred_covered)) / max(float(predicted_shape), 1.0)
        m["recall"] = float(len(gt_covered)) / float(gt_shape)
        precision.append(m["precision"])
        recall.append(m["recall"])

    return precision, recall, iou_matrix, metrics


def extract_delta_segments(caption):
    pattern = r"\[(\d+,\s*\d+)\]([^[]*)"
    matches = re.findall(pattern, caption)
    extracted_segments = []
    extracted_captions = []

    for match in matches:
        timestamps = [int(x) for x in match[0].replace(" ", "").split(",")]
        extracted_segments.append(timestamps)
        extracted_caption = match[1].replace(":", " ").replace("\nFrame ", "").strip()

        if extracted_caption.endswith(","):
            extracted_caption = extracted_caption[:-1]

        extracted_caption = extracted_caption.strip().replace("\n", " ").strip()
        extracted_captions.append(extracted_caption)

    if len(extracted_segments) > 0:
        extracted_segments = np.array(extracted_segments)
    else:
        return extracted_segments, extracted_captions

    return extracted_segments, extracted_captions


def chased_dp_assignment(scores):
    """dp matching from https://github.com/fujiso/SODA/blob/master/soda.py."""

    m, n = scores.shape
    dp = -np.ones((m, n))
    path = np.zeros((m, n))

    def transition(i, j):
        if dp[i, j] >= 0:
            return dp[i, j]
        elif i == 0 and j == 0:
            state = [-1, -1, scores[i, j]]
        elif i == 0:
            state = [-1, transition(i, j - 1), scores[i, j]]
        elif j == 0:
            state = [transition(i - 1, j), -1, scores[i, j]]
        else:
            state = [
                transition(i - 1, j),
                transition(i, j - 1),
                transition(i - 1, j - 1) + scores[i, j],
            ]
        dp[i, j] = np.max(state)
        path[i, j] = np.argmax(state)
        return dp[i, j]

    def get_pairs(i, j):
        p = np.where(path[i][: j + 1] == 2)[0]
        # pylint: disable=g-explicit-length-test
        if i != 0 and not len(p):
            return get_pairs(i - 1, j)
        elif i == 0 or p[-1] == 0:
            return [(i, p[-1])]
        else:
            return get_pairs(i - 1, p[-1] - 1) + [(i, p[-1])]

    n, m = scores.shape
    max_score = transition(n - 1, m - 1)
    pairs = get_pairs(n - 1, m - 1)
    return max_score, pairs


def sodac_llm_score(iou_matrix, score_matrix, predicted_captions, gt_captions, iou_thresholds=(0.0,)):
    """SODA_c with score matrix computed from LLM."""

    if not predicted_captions:
        return 0

    res = {str(index): [p] for index, p in enumerate(predicted_captions)}
    fs = [0] * len(iou_thresholds)
    gts = [{index: [x] for index in res} for x in gt_captions]
    for i, threshold in enumerate(iou_thresholds):
        iou_cur = np.copy(iou_matrix)
        iou_cur[iou_cur < threshold] = 0.0
        max_score, _ = chased_dp_assignment(iou_cur * score_matrix)
        (n_g, n_p) = iou_cur.shape
        p = max_score / n_p
        r = max_score / n_g
        fs[i] = 2 * p * r / (p + r) if p + r > 0 else 0

    mean_fs = np.mean(fs)

    return mean_fs


def get_caption_judge_prompt(gt, pred):
    sys_prompt = (
        "Your task is to score a predicted caption from a model for how similar it is to the ground truth caption, "
        "providing a single integer score between 0-10 indicating the similarity and an explanation. "
        "Focus on whether the information in the ground truth caption is present and accurately represented in the predicted caption. "
        "A score of 10 indicates that the predicted caption accurately represents all the information present in the ground truth caption. "
        "Subtract points for missing and inaccurate information, with lower scores for more significant errors. "
        "Do not penalize extra information in the predicted caption unless it contradicts the ground truth caption. "
        "Do not penalize minor differences in phrasing or word choice. "
        'Respond in the following JSON format: {"score": <int>, "explanation": "<str>"} '
        "where score is between 0-10 and explanation is a short sentence."
    )
    user_prompt = f"Please score the following predicted caption. Respond with only the JSON.\nPredicted caption: {pred}\nGround truth caption: {gt}\n\n"
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def get_sgqa_judge_prompt(question, pred, target):
    sys_prompt = (
        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
        "------"
        "##INSTRUCTIONS: "
        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
        "- Consider synonyms or paraphrases as valid matches.\n"
        "- Evaluate the correctness of the prediction compared to the answer."
    )
    user_prompt = (
        "Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {target}\n"
        f"Predicted Answer: {pred}\n\n"
        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        'For example, your response should look like this: {"pred": "yes", "score": 4.8}}.'
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def call_judge_with_retry(client, model_name, prompt, temperature=0, max_tokens=256, max_retries=5, base_delay=1):
    """Calls the OpenAI API's chat completion endpoint with exponential backoff."""
    retries = 0
    while True:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion
        except RateLimitError as e:
            retries += 1
            if retries > max_retries:
                raise Exception(f"Max retries ({max_retries}) exceeded for prompt. Error: {e}")
            delay = base_delay * (2 ** (retries - 1))
            print(f"Rate limit hit (Attempt {retries}/{max_retries}). Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            raise Exception(f"An unexpected error occurred during API call: {e}")


def decode_video(video_path: str) -> List[np.ndarray]:
    """
    Decode the video and return the RGB frames
    """
    video = cv2.VideoCapture(video_path)
    video_frames = []
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
        else:
            break
    return video_frames


def load_video(video_path, max_frames, annot_sample_rate=1):

    def uniform_sample(m, n):
        assert n <= m
        stride = (m - 1) / (n - 1) if n > 1 else 0  # Calculate the stride
        return [int(round(i * stride)) for i in range(n)]

    # Note that SAM-2 videos are either provided at 24 FPS, or at 6FPS. However the annotations are provided at 6FPS.
    # Please set annot_sample_rate = 4 if you download 24 FPS videos.
    # Please set annot_sample_rate = 1 if you download 6 FPS videos.
    # Default to 1 for 6 FPS videos.
    frames = decode_video(video_path)
    frames = frames[::annot_sample_rate]

    sample_pos = uniform_sample(len(frames), max_frames)
    all_frames = [frames[pos] for pos in sample_pos]

    return all_frames, sample_pos


def load_video_uniform(video_path, max_frames):
    def uniform_sample(m, n):
        if n >= m:
            return list(range(m))  # Return all frames if max_frames is greater than or equal to the total number of frames
        stride = (m - 1) / (n - 1) if n > 1 else 0  # Calculate the stride
        return [int(round(i * stride)) for i in range(n)]

    frames = decode_video(video_path)
    sample_pos = uniform_sample(len(frames), max_frames)
    all_frames = [frames[pos] for pos in sample_pos]

    return all_frames, sample_pos


def draw_bounding_boxes(frames, sample_pos, bbox_dict_map):
    """
    Helper function to draw bounding boxes on video frames.
    """
    assert len(frames) == len(sample_pos), f"The number of frames ({len(frames)}) must match with the number of sample positions ({len(sample_pos)})"

    frame_with_bbox = []
    for i, frame in enumerate(frames):
        pil_frame = Image.fromarray(frame)
        bbox_idx = sample_pos[i]
        if bbox_idx in bbox_dict_map:
            bbox = bbox_dict_map[bbox_idx]
            if bbox:
                draw = ImageDraw.Draw(pil_frame)
                x1, y1, x2, y2 = bbox
                draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=4)
        frame_with_bbox.append(pil_frame)
    return frame_with_bbox


def load_defualt_config():
    # Load default config parameters
    with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # Remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)

        config = yaml.safe_load("".join(safe_data))

    return config


def load_plm_stc_metadata(config):
    repo_id = config["plm_stc"]["repo_id"]
    repo_type = config["plm_stc"]["repo_type"]
    metadata_path = config["plm_stc"]["metadata_path"]
    # Download the file from the HF hub and cache it
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=metadata_path,
        repo_type=repo_type,
    )
    # Load the cached JSONL file
    with open(local_path, "r") as f:
        metadata = [json.loads(line) for line in f]
    # Convert the list of dictionaries to a dictionary
    metadata_map = {(entry["video"], entry["masklet_id"]): entry for entry in metadata}

    return metadata_map
