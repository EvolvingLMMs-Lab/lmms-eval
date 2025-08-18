import os
import zipfile
from collections import defaultdict

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

MAX_NUM_FRAMES = 8

LEMONADE_ZIP_NAMES = [
    "videos_batch_0.zip",
    "videos_batch_1.zip",
    "videos_batch_2.zip",
    "videos_batch_3.zip",
    "videos_batch_4.zip",
]

data_dir = "./data/lemonade"


def download_and_extract_lemonade_videos(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    videos_dir = os.path.join(data_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    for zip_name in LEMONADE_ZIP_NAMES:
        print(f"Downloading {zip_name} from Hugging Face...")
        zip_path = hf_hub_download(repo_id="amathislab/LEMONADE", filename=zip_name, repo_type="dataset", cache_dir=os.path.join(data_dir, "cache"))

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(videos_dir)

    print("All videos downloaded and extracted successfully.\n")


def load_video(video_file, start_frame, end_frame, max_num_frames=MAX_NUM_FRAMES):

    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, start_frame)
    end_frame = min(end_frame, total_frames - 1)
    total_valid_frames = end_frame - start_frame + 1
    num_frames = min(max_num_frames, total_valid_frames)

    step = total_valid_frames / num_frames
    frame_indices = [int(start_frame + i * step) for i in range(num_frames)]

    frames = []
    for target_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        success, frame = cap.read()
        if not success:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb).convert("RGB")
        frames.append(pil_img)

    cap.release()
    return frames


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]

    if all(option.startswith(f"{letter}.") for option, letter in zip(options, option_letters)):
        return "\n".join(options)

    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def lemonade_doc_to_visual(doc):
    videos_dir = os.path.join(data_dir, "videos")

    if not os.path.exists(videos_dir) or len(os.listdir(videos_dir)) == 0:
        print("Videos directory is empty — downloading and extracting...\n")
        download_and_extract_lemonade_videos(data_dir)

    video_filename = doc["Clip"] + "_hololens.mp4"

    video_path = os.path.join(videos_dir, video_filename)

    if os.path.exists(video_path):
        start = int(doc["Start"])
        end = int(doc["End"])
        frames = load_video(video_path, start, end, max_num_frames=MAX_NUM_FRAMES)
    else:
        raise FileNotFoundError(f"Video file not found: {video_path}")

    return frames


def lemonade_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = "Question: " + doc["Question"]
    parsed_options = parse_options(eval(doc["Answers"]))
    choices = "Choices:\n" + parsed_options

    return f"{pre_prompt}{question}\n{choices}{post_prompt}"


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    """
    assert isinstance(options, list), f"Expected list of options, got {type(options)}: {options}"

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    if response == "API Error":
        return "API Error"

    if response == "":
        return "Empty Response"

    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    ans_with_period = False
    ans_with_colon = False
    candidates = []

    for choice in all_choices:
        if f"{choice}." in response:
            candidates.append(choice)
            ans_with_period = True
    for choice in all_choices:
        if f"{choice}:" in response:
            candidates.append(choice)
            ans_with_colon = True
    if len(candidates) == 0:
        for choice in all_choices:
            if f"({choice})" in response:
                candidates.append(choice)
                ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False
    if len(candidates) == 0:
        pred_index = "A"

    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_period:
                for can in candidates:
                    index = response.rfind(f"{can}.")
                    start_indexes.append(index)
            elif ans_with_colon:
                for can in candidates:
                    index = response.rfind(f"{can}:")
                    start_indexes.append(index)
            elif ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index


def lemonade_process_results(doc, results):
    pred = results[0]

    index2ans, all_choices = get_multi_choice_info(eval(doc["Answers"]))
    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)

    acc = {"QID": doc["QID"], "category": doc["Category"], "subcategory": doc["Subcategory"], "difficulty": doc["Difficulty"], "answer": doc["Correct Answer"], "parsed_pred": parsed_pred, "original_pred": pred}
    return {"acc": acc}


def lemonade_aggregate_results(results):
    def compute_accuracy(grouped_results):
        acc_dict = {}
        for key, samples in grouped_results.items():
            correct = sum([r["parsed_pred"] == r["answer"] for r in samples])
            total = len(samples)
            acc = round(correct / total, 5) if total > 0 else 0.0
            stderr = round(np.sqrt(acc * (1 - acc) / total), 5) if total > 0 else 0.0
            acc_dict[key] = {
                "num": total,
                "acc": acc,
                "acc_stderr": stderr,
            }
        return acc_dict

    qid_results = defaultdict(list)
    category_results = defaultdict(list)
    subcategory_results = defaultdict(list)
    difficulty_results = defaultdict(list)

    valid_results = [r for r in results if r["parsed_pred"] != "API Error"]

    for r in valid_results:
        qid_results[r["QID"]].append(r)
        category_results[r["category"]].append(r)
        subcategory_results[r["subcategory"]].append(r)
        difficulty_results[r["difficulty"]].append(r)

    qid_acc = compute_accuracy(qid_results)
    category_acc = compute_accuracy(category_results)
    subcategory_acc = compute_accuracy(subcategory_results)
    difficulty_acc = compute_accuracy(difficulty_results)

    total_correct = sum([r["parsed_pred"] == r["answer"] for r in valid_results])
    total = len(valid_results)
    overall_acc = round(total_correct / total, 5) if total > 0 else 0.0
    overall_stderr = round(np.sqrt(overall_acc * (1 - overall_acc) / total), 5) if total > 0 else 0.0

    print("\nResults:")

    print("\nAccuracy per QID:")
    for k, v in qid_acc.items():
        print(f"  {k}: {v['acc']} ± {v['acc_stderr']} ({v['num']} examples)")

    print("\nAccuracy per Category:")
    for k, v in category_acc.items():
        print(f"  {k}: {v['acc']} ± {v['acc_stderr']} ({v['num']} examples)")

    print("\nAccuracy per Subcategory:")
    for k, v in subcategory_acc.items():
        print(f"  {k}: {v['acc']} ± {v['acc_stderr']} ({v['num']} examples)")

    print("\nAccuracy per Difficulty:")
    for k, v in difficulty_acc.items():
        print(f"  {k}: {v['acc']} ± {v['acc_stderr']} ({v['num']} examples)")

    print(f"\nOverall Accuracy: {overall_acc} ± {overall_stderr} ({total} examples)")

    return overall_acc
