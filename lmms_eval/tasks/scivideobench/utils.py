import os
import random
import re
from pathlib import Path
from collections import defaultdict
import yaml
from datetime import timedelta

def convert_time_to_frames_in_question(question, total_duration_sec, total_frames):
    def parse_time(tstr):
        mm, ss = map(int, tstr.split(":"))
        return timedelta(minutes=mm, seconds=ss).total_seconds()

    # Convert total duration (float in seconds) to mm:ss string
    total_minutes = int(total_duration_sec) // 60
    total_seconds = int(total_duration_sec) % 60
    total_duration_str = f"{total_minutes:02d}:{total_seconds:02d}"
    total_sec = parse_time(total_duration_str)

    # Replace timestamp ranges like 06:15–06:36 or 6:15-6:36
    def replace_range(match):
        start_time, end_time = match.group(1), match.group(2)
        start_sec = parse_time(start_time)
        end_sec = parse_time(end_time)
        start_frame = round((start_sec / total_sec) * total_frames)
        end_frame = round((end_sec / total_sec) * total_frames)
        return f"frame {start_frame} to frame {end_frame}"

    question = re.sub(r'(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})', replace_range, question)

    # Replace standalone timestamps like 06:15
    def replace_single(match):
        time_str = match.group(1)
        time_sec = parse_time(time_str)
        frame = round((time_sec / total_sec) * total_frames)
        return f"frame {frame}"

    question = re.sub(r'(?<!frame )(\d{1,2}:\d{2})(?!\s*[-–])', replace_single, question)

    return question

def scivideobench_doc_to_visual(doc):
    with open(Path(__file__).parent / "scivideobench.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
    dataset_path = yaml.safe_load("".join(safe_data))["dataset_path"]

    video_dir = Path(dataset_path)
    video_path = video_dir / f"jove_{doc['video_id']}.mp4"


    if video_path.exists():
        return [str(video_path)]
    elif video_path.with_suffix(".MP4").exists():
        return [str(video_path.with_suffix(".MP4"))]
    elif video_path.with_suffix(".mkv").exists():
        return [str(video_path.with_suffix(".mkv"))]
    else:
        raise FileNotFoundError(f"Video not found: {video_path}")

def format_options(opts):
    if isinstance(opts, dict):
        # ensure consistent A..Z order
        keys = sorted(opts.keys())
        return "\n".join(f"{k}. {opts[k]}" for k in keys)
    elif isinstance(opts, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(opts))
    else:
        raise TypeError(f"Unsupported options type: {type(opts)}")

def scivideobench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt  = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    duration = doc["video_duration"]
    converted_question = convert_time_to_frames_in_question(doc['question'], duration, int(duration))

    options_text = format_options(doc["options"])  # ← fix here
    input_text = f"{pre_prompt}{converted_question}\n{options_text}\n{post_prompt}"
    # print("input text: ", input_text)
    return input_text


def scivideobench_doc_to_choice(doc):
    return ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def scivideobench_doc_to_target(doc):
    return ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"].index(doc["ground_truth"])

def extract_answer_letter(s):
    """
    Extract letter from answer (A, B, C, D, E, F, G, H, I, J)

    """
    s = s.strip()

    answer_prefixes = [
        "The answer is",
        "The correct answer is",
        "The best answer is",
        "Answer:",
        "Option:",
        "### Final Answer:\n$$\\boxed",
        "the final answer is"
    ]
    for prefix in answer_prefixes:
        s = s.replace(prefix, "")


    if len(s.split()) > 16 and not re.search("[ABCDEFGHIJ]", s):
        return ""


    matches = re.search(r"[ABCDEFGHIJKL]", s)
    if matches is None:
        return ""
    return matches[0]

def scivideobench_process_results(doc, results):
    pred = results[0].strip() if isinstance(results, list) else results.strip()
    pred = extract_answer_letter(pred)
    if not pred:
        pred = random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])

    gold = doc["answer"].strip()
    correct = (pred == gold)

    data_dict = {
        "id": doc["video_id"],
        "question_type": doc["question_type"],
        "category": doc.get("category", "UNKNOWN"),
        "pred_answer": pred,
        "answer": gold,
        "correct": correct,
        "raw_output": results[0] if isinstance(results, list) else results
    }

    return {
        "scivideobench_acc": data_dict
    }


def scivideobench_aggregate_results(results):
    # Buckets
    by_qtype = defaultdict(lambda: {"correct": 0, "total": 0})
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})

    total_correct = 0
    total_examples = 0

    # Accumulate
    for r in results:
        qtype = r.get("question_type", "UNKNOWN")
        cat = r.get("category", "UNKNOWN")
        is_correct = 1 if r.get("correct", False) else 0

        by_qtype[qtype]["correct"] += is_correct
        by_qtype[qtype]["total"] += 1

        by_category[cat]["correct"] += is_correct
        by_category[cat]["total"] += 1

        total_correct += is_correct
        total_examples += 1

    # Build printable summaries
    def summarize(bucket):
        out = {}
        for k, v in bucket.items():
            acc = (v["correct"] / v["total"]) if v["total"] > 0 else 0.0
            out[k] = {"num": v["total"], "acc": round(acc * 100, 2)}
        return out

    printable_qtype = summarize(by_qtype)
    printable_category = summarize(by_category)
    overall_acc = round((total_correct / total_examples) * 100, 2) if total_examples else 0.0

    # Pretty print
    print("\nSciVideoBench Evaluation Results:")
    print(f"Overall Accuracy: {overall_acc}%")

    print("\nStatistics by Question Type:")
    for k, v in printable_qtype.items():
        print(f"{k}: {v['acc']}% (samples: {v['num']})")

    print("\nStatistics by Discipline:")
    for k, v in printable_category.items():
        print(f"{k}: {v['acc']}% (samples: {v['num']})")

    return {
        "overall_acc": overall_acc,
        "by_question_type": printable_qtype,
        "by_category": printable_category
    }