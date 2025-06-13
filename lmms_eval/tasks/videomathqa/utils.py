import os
import re
import cv2
import sys
import yaml
import numpy as np

from pathlib import Path
from typing import List
from collections import defaultdict
from loguru import logger as eval_logger

VIDEO_LENGTH = ["short", "medium", "long"]
CATEGORIES = ["Geometry Angle", "Geometry Area", "Geometry Length", "Chart", "Statistics", "Arithmetic", "Topology", "Graph Theory", "Counting", "Puzzle"]


def decode_video(video_path: str) -> List[np.ndarray]:
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

    frames = decode_video(video_path)
    frames = frames[::annot_sample_rate]

    sample_pos = uniform_sample(len(frames), max_frames)
    all_frames = [frames[pos] for pos in sample_pos]

    return all_frames, sample_pos


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "videomathqa_mcq.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame


def videomathqa_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, "videos", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check {doc}")
    return [video_path]


def videomathqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if len(doc["options"]) == 2:
        option_prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with the letter (A or B) of the correct option."
    else:
        option_prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with the letter (A, B, C, D or E) of the correct option."

    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = option_prompt + "\n" + question + "\n" + post_prompt
    return full_prompt


def videomathqa_doc_to_text_subtitle(doc, lmms_eval_specific_kwargs=None):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, "videos", video_path)
    subtitle_path = os.path.join(cache_dir, "subtitles", doc["videoID"] + ".srt")
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(subtitle_path):  # Denote have subtitle
        subtitle = open(subtitle_path).readlines()
    else:
        subtitle = ""
    subtitles_prompt = "This video's subtitles are listed below: \n"
    if subtitle == "":
        subtitle = "No subtitles available"
    else:
        if "gemini_api_flag" in lmms_eval_specific_kwargs:  # specific for gemini_api
            if lmms_eval_specific_kwargs["gemini_api_flag"] == "full subtitle":
                textlist = []

                subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
                frame_num = total_frame
                uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()
                subtitle_by_frame_idx = []
                for frame_idx in uniform_sampled_frames:
                    for idx, title in enumerate(subtitle_by_frame):
                        if frame_idx < title[1] and frame_idx >= title[0]:
                            subtitle_by_frame_idx.append(idx)
                subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))
                textlist = []
                for idx in subtitle_by_frame_idx:
                    raw_text = subtitle_by_frame[idx][2]
                    try:
                        textlist.append(raw_text)
                    except:
                        continue
                subtitle_text = "\n".join(textlist)
        else:
            if "frame_num" in lmms_eval_specific_kwargs:
                frame_num = lmms_eval_specific_kwargs["frame_num"]
                subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
                if frame_num == -1:
                    frame_num = total_frame
                uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

                subtitle_by_frame_idx = []
                for frame_idx in uniform_sampled_frames:
                    for idx, title in enumerate(subtitle_by_frame):
                        if frame_idx < title[1] and frame_idx >= title[0]:
                            subtitle_by_frame_idx.append(idx)
                subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

                textlist = []
                for idx in subtitle_by_frame_idx:
                    raw_text = subtitle_by_frame[idx][2]
                    try:
                        textlist.append(raw_text)
                    except:
                        continue
                subtitle_text = "\n".join(textlist)
        subtitle = subtitle_text

    if len(doc["options"]) == 2:
        option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with the letter (A or B) of the correct option."
    else:
        option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with the letter (A, B, C, D or E) of the correct option."

    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = subtitles_prompt + subtitle + "\n" + option_prompt + "\n" + question + "\n" + post_prompt
    return full_prompt


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""

    matches = re.search(r"[ABCDE]", s)
    if matches is None:
        return ""
    return matches[0]


matrices = []

for i in VIDEO_LENGTH:
    for j in CATEGORIES:
        matrices.append(f"{i}_{j}")


def videomathqa_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomathqa score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)

    category = doc["category"]
    doc["duration"] = doc["length"]
    data_dict = {"question_id": doc["question_id"], "duration": doc["duration"], "category": category, "pred_answer": pred_ans, "answer": doc["answer"]}

    return {f"videomathqa_perception_score": data_dict}


def videomathqa_mcq_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}

    for video_length in VIDEO_LENGTH:
        for category in CATEGORIES:
            key = f"{video_length}_{category}"
            category2score[key] = {"correct": 0, "answered": 0}

    for result in results:
        video_length = result["duration"]
        category = result["category"]
        key = f"{video_length}_{category}"
        category2score[key]["answered"] += 1
        category2score[key]["correct"] += result["pred_answer"] == result["answer"]

    for video_length in VIDEO_LENGTH:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if video_length in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        score = 100 * total_correct / total_answered if total_answered > 0 else 0
        eval_logger.info(f"Evaluation on Video Length: {video_length}: {score:.1f}%")

    for category in CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        score = 100 * total_correct / total_answered if total_answered > 0 else 0
        eval_logger.info(f"Evaluation on Categories: {category}: {score:.1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    overall_score = 100 * total_correct / total_answered if total_answered > 0 else 0
    eval_logger.info(f"Overall Performance: {overall_score:.1f}%")

    return 100 * total_correct / total_answered if total_answered > 0 else 0


def videomathqa_multi_binary_aggregate_results(results):

    grouped = defaultdict(list)
    for result in results:
        grouped[result["question_id"]].append(result)

    category2score = {}
    for video_length in VIDEO_LENGTH:
        for category in CATEGORIES:
            key = f"{video_length}_{category}"
            category2score[key] = {"correct": 0, "answered": 0}

    for qid, group in grouped.items():
        # Use first element to get metadata
        sample_meta = group[0]
        video_length = sample_meta["duration"]
        category = sample_meta["category"]
        key = f"{video_length}_{category}"

        all_correct = all(g["pred_answer"] == g["answer"] for g in group)
        category2score[key]["answered"] += 1
        if all_correct:
            category2score[key]["correct"] += 1

    for video_length in VIDEO_LENGTH:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if video_length in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        score = 100 * total_correct / total_answered if total_answered > 0 else 0
        eval_logger.info(f"Evaluation on Video Length: {video_length}: {score:.1f}%")

    for category in CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        score = 100 * total_correct / total_answered if total_answered > 0 else 0
        eval_logger.info(f"Evaluation on Categories: {category}: {score:.1f}%")

    total_correct = sum(v["correct"] for v in category2score.values())
    total_answered = sum(v["answered"] for v in category2score.values())
    overall_score = 100 * total_correct / total_answered if total_answered > 0 else 0
    eval_logger.info(f"Overall Performance: {overall_score:.1f}%")

    return overall_score
