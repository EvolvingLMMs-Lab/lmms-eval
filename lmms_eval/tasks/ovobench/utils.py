import glob
import json
import os
import shutil
import tarfile
from functools import lru_cache

from filelock import FileLock
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.ovobench.constant import (
    BR_PROMPT_TEMPLATE,
    CRR_PROMPT_TEMPLATE,
    REC_PROMPT_TEMPLATE,
    SSR_PROMPT_TEMPLATE,
)
from lmms_eval.tasks.ovobench.score_utils.score import (
    calculate_score_backward_realtime,
    calculate_score_forward,
)

DATASET_REPO_ID = "JoeLeelyf/OVO-Bench"
HF_HOME = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))
OVO_ROOT = os.path.join(HF_HOME, "ovobench")
CHUNKS_DIR = os.path.join(OVO_ROOT, "chunked_videos")
_LOCK_PATH = os.path.join(OVO_ROOT, ".ovobench.lock")


def _chunks_ready(d=CHUNKS_DIR):
    return os.path.isdir(d) and any(fn.endswith(".mp4") for fn in os.listdir(d))


def _concat_parts(parts, out_path):
    with open(out_path, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, length=1 << 22)


def _normalize_extracted_dir():
    """Tar may extract to a different top-level dir; rename it to CHUNKS_DIR."""
    if os.path.isdir(CHUNKS_DIR):
        return
    for name in os.listdir(OVO_ROOT):
        cand = os.path.join(OVO_ROOT, name)
        if os.path.isdir(cand) and _chunks_ready(cand):
            os.rename(cand, CHUNKS_DIR)
            return


@lru_cache(maxsize=1)
def get_chunked_videos_dir():
    """Return ``$HF_HOME/ovobench/chunked_videos``, downloading on first use.

    Skips the download if the directory already contains extracted mp4 files
    (e.g. user pre-staged them). Multi-process safe via a file lock.
    """
    from huggingface_hub import snapshot_download

    if _chunks_ready():
        return CHUNKS_DIR

    os.makedirs(OVO_ROOT, exist_ok=True)
    with FileLock(_LOCK_PATH):
        if _chunks_ready():
            return CHUNKS_DIR

        eval_logger.info(f"[ovobench] Fetching chunked video parts from {DATASET_REPO_ID} -> {HF_HOME}")
        snap = snapshot_download(
            DATASET_REPO_ID,
            repo_type="dataset",
            allow_patterns=["chunked_videos.tar.parta*"],
            cache_dir=HF_HOME,
        )
        parts = sorted(glob.glob(os.path.join(snap, "chunked_videos.tar.parta*")))
        if not parts:
            raise RuntimeError(f"[ovobench] No chunked_videos.tar.parta* found under {snap}")

        tar_path = os.path.join(OVO_ROOT, "chunked_videos.tar")
        eval_logger.info(f"[ovobench] Concatenating {len(parts)} parts -> {tar_path}")
        _concat_parts(parts, tar_path)

        eval_logger.info(f"[ovobench] Extracting -> {OVO_ROOT}")
        with tarfile.open(tar_path) as t:
            t.extractall(OVO_ROOT)
        os.remove(tar_path)
        _normalize_extracted_dir()

        assert _chunks_ready(), f"[ovobench] Extraction failed: {CHUNKS_DIR} empty"
        eval_logger.info(f"[ovobench] Ready at {CHUNKS_DIR}")
        return CHUNKS_DIR


def _resolve_data_dir(lmms_eval_specific_kwargs):
    """Honor user-provided ``data_dir`` if set, else auto-download."""
    if lmms_eval_specific_kwargs and lmms_eval_specific_kwargs.get("data_dir"):
        return lmms_eval_specific_kwargs["data_dir"]
    return get_chunked_videos_dir()


def is_forward_task(doc):
    task = doc["task"]
    return task not in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]


def get_task_type(task_name):
    backward_tasks = ["EPM", "ASI", "HLD"]
    realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
    forward_tasks = ["REC", "SSR", "CRR"]

    if task_name in backward_tasks:
        return "backward"
    elif task_name in realtime_tasks:
        return "realtime"
    elif task_name in forward_tasks:
        return "forward"
    else:
        raise ValueError(f"Unknown task name: {task_name}")


def build_prompt(doc, index):
    task = doc["task"]
    if not is_forward_task(doc):
        question = doc["question"]
        options = doc["options"]
        formatted_options = "; ".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options)) + ";"
        prompt = BR_PROMPT_TEMPLATE.format(question, formatted_options)
    else:
        if task == "REC":
            activity = doc["activity"]
            question = "How many times did they " + activity + "?"
            prompt = REC_PROMPT_TEMPLATE.format(question)
        elif task == "SSR":
            step = doc["test_info"][index]["step"]
            prompt = SSR_PROMPT_TEMPLATE.format(step)
        elif task == "CRR":
            question = doc["question"]
            prompt = CRR_PROMPT_TEMPLATE.format(question)
    return prompt


# can't be used, because multiround generation for chat models has not implemented yet
def ovo_doc_to_messages(doc, lmms_eval_specific_kwargs):
    pass


def ovo_back_real_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Convert a backward/realtime document into the flat text prompt."""
    text = build_prompt(doc, index=None)
    return text


# prepare for multiround generation
def ovo_forward_doc_to_text(doc, lmms_eval_specific_kwargs=None, previous_output=None, round_idx=None, previous_round_info=None):
    """Assemble the prompt/visual payloads for forward tasks, one round at a time."""
    if round_idx is None:
        prompt = build_prompt(doc, index=0)
        return prompt
    else:
        i = round_idx
    terminal_sign = True if i == len(doc["test_info"]) else False
    if not terminal_sign:
        prompt = build_prompt(doc, index=i)
        lmms_eval_specific_kwargs["round_idx"] = i
        visuals = ovo_doc_to_visual(doc, lmms_eval_specific_kwargs)
        lmms_eval_specific_kwargs.pop("round_idx")
        return visuals, prompt, terminal_sign, previous_output, None
    else:
        return None, None, terminal_sign, previous_output, None


def ovo_doc_to_visual(doc, lmms_eval_specific_kwargs):
    """Return the relevant video chunk path for the document/round."""
    data_dir = _resolve_data_dir(lmms_eval_specific_kwargs)
    if lmms_eval_specific_kwargs and "round_idx" in lmms_eval_specific_kwargs:
        i = lmms_eval_specific_kwargs["round_idx"]
        chunk_video_path = os.path.join(data_dir, f'{doc["id"]}_{i}.mp4')
    elif is_forward_task(doc):
        # cause of logic of sending lmms_eval_specific_kwargs to doc_to_visual at initial round in multi round generation
        chunk_video_path = os.path.join(data_dir, f'{doc["id"]}_{0}.mp4')
    else:
        chunk_video_path = os.path.join(data_dir, f'{doc["id"]}.mp4')
    assert os.path.exists(chunk_video_path), f"Video chunk path does not exists:{chunk_video_path} !"

    return [chunk_video_path]


def ovo_back_real_process_results(doc, results):
    """Normalize backward/realtime generation output into a structured record."""
    if isinstance(results, list) and isinstance(results[0], list):
        response = results[0][0].strip()
    else:
        response = results[0].strip()
    result = {"id": doc["id"], "task": doc["task"], "question": doc["question"], "response": response, "ground_truth": chr(65 + doc["gt"])}
    return {"back_real_acc": result}


def ovo_forward_process_results(doc, results):
    """Map forward task responses back onto the original document structure."""
    if isinstance(results, list) and isinstance(results[0], list):
        results = results[0]
    for i in range(len(doc["test_info"])):
        doc["test_info"][i]["response"] = results[i]
    return {"forward_acc": doc}


def ovo_back_real_acc(results, args):
    """Score backward/realtime outputs and persist summary JSON files."""
    results, scores = calculate_score_backward_realtime(results)
    task_name = get_task_type(list(scores.keys())[0])
    path = generate_submission_file(f"{task_name}_acc_results.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    scores_summary_dict = {}
    avg_scores = []
    for k, v in scores.items():
        scores_summary_dict[k] = 100 * sum(v) / len(v)
        avg_scores.append(sum(v) / len(v))

    total_avg_score = 100 * sum(avg_scores) / len(avg_scores)
    scores_summary_dict["average"] = total_avg_score
    path = generate_submission_file(f"{task_name}_acc_scores.json", args)

    print(scores_summary_dict)
    with open(path, "w") as f:
        json.dump(scores_summary_dict, f, indent=4)

    return scores_summary_dict["average"]


def ovo_forward_acc(results, args):
    """Score forward outputs, save JSON files, and return the aggregated average."""
    results, scores = calculate_score_forward(results)
    path = generate_submission_file("forward_acc_results.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    scores_summary_dict = {}
    avg_scores = []
    for k, v in scores.items():
        scores_summary_dict[k] = 100 * sum(v) / len(v)
        avg_scores.append(sum(v) / len(v))

    total_avg_score = 100 * sum(avg_scores) / len(avg_scores)
    scores_summary_dict["average"] = total_avg_score
    path = generate_submission_file("forward_acc_scores.json", args)

    print(scores_summary_dict)
    with open(path, "w") as f:
        json.dump(scores_summary_dict, f, indent=4)

    return scores_summary_dict["average"]
