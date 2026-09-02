"""HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering.

Paper: https://arxiv.org/abs/2512.14870
Data:  https://huggingface.co/datasets/DanBenAmi/HERBench

Five-way multiple-choice questions over long videos; each question requires
aggregating at least 3 distinct, temporally separated visual cues. Three
configs: 'full' (27k+ questions / 335 videos / ~161 GB of videos), 'lite'
(68 videos / ~35 GB) and 'lite_v2' (refined lite, same 68 videos).

Videos ship as 17 split archive chunks (videos/videos.tar.part.00 .. .16).
Chunks 00-03 form one complete tar with the 68 lite videos; chunks 04-16 form
a second complete tar with the remaining videos of the full set. The download
below is therefore variant-aware (lite tasks only fetch ~35 GB) and the
extraction streams straight from the chunks (no 170 GB intermediate tar).
"""

import os
import re
import tarfile
from pathlib import Path

from loguru import logger as eval_logger

REPO_ID = "DanBenAmi/HERBench"
CACHE_DIR_NAME = "herbench"
CHOICE_LETTERS = ("A", "B", "C", "D", "E")

# chunks 00-03 hold the 68 HERBench-Lite videos; the remaining chunks hold the rest
# (for the full set, the chunk list is discovered from the Hub at download time)
LITE_PART_IDS = tuple(range(0, 4))

TASK_TYPES = {
    "Temporal Shot Ordering": "temporal_shot_ordering",
    "Multi-Person Duration Reasoning": "multi_person_duration_reasoning",
    "Action Sequence Integrity Identification": "action_sequence_integrity_identification",
    "Appearance-Grounded Behavior & Interactions": "appearance_grounded_behavior_interactions",
    "Appearance-Grounded Attribute Recognition": "appearance_grounded_attribute_recognition",
    "Appearance-Grounded Localization & Trajectory": "appearance_grounded_localization_trajectory",
    "False Action Memory": "false_action_memory",
    "Scene Verification & Arrangement": "scene_verification_arrangement",
    "False Object Memory": "false_object_memory",
    "Multi-Entities Grounding and Localization": "multi_entities_grounding_localization",
    "Action Counting": "action_counting",
    "Region Localized People Counting": "region_localized_people_counting",
}


def _cache_root() -> str:
    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))
    return os.path.join(hf_home, CACHE_DIR_NAME)


def _extract_tar_parts(part_files, videos_root):
    """Stream-extract the split tar chunks without concatenating them on disk.

    ignore_zeros lets tarfile read through the end-of-archive markers between
    the two complete tar archives that make up the chunk sequence.
    """

    class _ChainedReader:
        def __init__(self, paths):
            self.paths = list(paths)
            self.idx = 0
            self.consumed = 0
            self.fp = open(self.paths[0], "rb") if self.paths else None

        def read(self, size=-1):
            if size is None or size < 0:
                raise ValueError("unbounded read not supported")
            chunks = []
            remaining = size
            while remaining > 0 and self.fp is not None:
                chunk = self.fp.read(remaining)
                if chunk:
                    chunks.append(chunk)
                    remaining -= len(chunk)
                    self.consumed += len(chunk)
                else:
                    self.fp.close()
                    self.idx += 1
                    self.fp = open(self.paths[self.idx], "rb") if self.idx < len(self.paths) else None
            return b"".join(chunks)

        def close(self):
            if self.fp is not None:
                self.fp.close()
                self.fp = None

    os.makedirs(videos_root, exist_ok=True)
    reader = _ChainedReader(part_files)
    extracted = skipped = 0
    try:
        with tarfile.open(fileobj=reader, mode="r|", ignore_zeros=True) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name.lstrip("./")
                if name.startswith("/") or ".." in name.split("/"):
                    eval_logger.warning(f"Skipping suspicious tar member: {member.name}")
                    continue
                target = os.path.join(videos_root, name)
                if os.path.exists(target) and os.path.getsize(target) == member.size:
                    skipped += 1
                    continue
                member.name = name
                tar.extract(member, path=videos_root)
                extracted += 1
    finally:
        reader.close()
    # a corrupt chunk can make the stream end early without an exception; verify
    # the whole chunk sequence was consumed before treating extraction as done
    total_bytes = sum(os.path.getsize(p) for p in part_files)
    if reader.consumed < total_bytes:
        raise RuntimeError(
            f"HERBench video extraction stopped after {reader.consumed} of {total_bytes} bytes - "
            f"an archive chunk is likely corrupted. Delete the affected videos.tar.part.* files "
            f"from the HuggingFace cache (run `huggingface-cli scan-cache` to locate it) and re-run."
        )
    eval_logger.info(f"HERBench video extraction done: {extracted} extracted, {skipped} already present.")


def _needed_video_paths(variant: str):
    """Video paths the annotations reference (lite and lite_v2 share videos)."""
    from datasets import load_dataset

    config = "full" if variant == "full" else "lite"
    return sorted(set(load_dataset(REPO_ID, config, split="test")["video_path"]))


def _ensure_videos(variant: str) -> str:
    """Download (if needed) and extract the HERBench videos for a variant.

    variant: 'lite' (chunks 00-03, ~35 GB) or 'full' (all chunks, ~161 GB).
    Returns the directory that doc['video_path'] entries are relative to.
    """
    root = _cache_root()
    marker = Path(root) / f".videos_{variant}_extracted"
    if marker.exists():
        return root

    from filelock import FileLock
    from huggingface_hub import HfApi, hf_hub_download

    os.makedirs(root, exist_ok=True)
    with FileLock(os.path.join(root, ".videos.lock")):
        if marker.exists():
            return root

        needed = _needed_video_paths(variant)
        if all(os.path.exists(os.path.join(root, p)) for p in needed):
            marker.touch()
            return root

        repo_files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
        if variant == "full":
            # take every chunk on the Hub, so chunks appended later are picked up
            part_names = sorted(f for f in repo_files if f.startswith("videos/videos.tar.part."))
        else:
            part_names = [f"videos/videos.tar.part.{i:02d}" for i in LITE_PART_IDS]
        eval_logger.info(f"Downloading HERBench videos ({variant}: {len(part_names)} archive chunks, ~{'161' if variant == 'full' else '35'} GB) from {REPO_ID}. This only happens once; downloads resume if interrupted.")
        part_files = [hf_hub_download(REPO_ID, name, repo_type="dataset") for name in part_names]
        _extract_tar_parts(part_files, os.path.join(root, "videos"))

        # Also fetch any loose video files shipped next to the archives
        # (e.g. hotfixes for videos missing from the tar chunks).
        loose = [f for f in repo_files if f.startswith("videos/") and f.lower().endswith(".mp4")]
        for repo_path in loose:
            cached = hf_hub_download(REPO_ID, repo_path, repo_type="dataset")
            _symlink_over(cached, os.path.join(root, repo_path))

        # the marker permanently short-circuits this function, so only stamp it
        # once every referenced video is actually on disk
        missing = [p for p in needed if not os.path.exists(os.path.join(root, p))]
        if missing:
            raise RuntimeError(
                f"{len(missing)} HERBench videos are missing after extraction, e.g. {missing[:3]}. "
                "If the downloaded archive chunks are corrupted, delete them from the HuggingFace "
                "cache and re-run; otherwise please report this at "
                "https://github.com/DanBenAmi/HERBench/issues."
            )
        marker.touch()
    return root


def _symlink_over(src, dst):
    """Symlink src at dst, replacing an existing or dangling link; race-safe."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        if os.path.exists(dst):
            return  # a valid file/link is already there
        os.unlink(dst)  # dangling link left by a cleaned HF cache
    try:
        os.symlink(src, dst)
    except FileExistsError:
        pass  # another rank won the race


def _doc_to_visual(doc: dict, variant: str) -> list[str]:
    # a user-provided video directory takes precedence over (and avoids) the auto-download
    for env_var in ("HERBENCH_VIDEO_DIR", "HERBENCH_ROOT"):
        override = os.getenv(env_var)
        if override:
            candidate = os.path.join(os.path.expanduser(override), doc["video_path"])
            if os.path.exists(candidate):
                return [candidate]
            eval_logger.warning(f"{env_var} is set but {candidate} does not exist; falling back to the auto-downloaded videos.")

    root = _ensure_videos(variant)
    video_path = os.path.join(root, doc["video_path"])
    if not os.path.exists(video_path):
        # self-heal: the video may exist as a loose file on the Hub
        # (e.g. a hotfix for a video missing from the tar chunks)
        try:
            from huggingface_hub import hf_hub_download

            cached = hf_hub_download(REPO_ID, doc["video_path"], repo_type="dataset")
            _symlink_over(cached, video_path)
        except Exception as e:
            eval_logger.debug(f"No loose copy of {doc['video_path']} on the Hub: {e}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"HERBench video not found: {video_path}. Delete '{_cache_root()}' (including the hidden "
                f".videos_{variant}_extracted marker) to force re-extraction, or set HERBENCH_VIDEO_DIR "
                "to a directory containing the extracted videos/ folder."
            )
    return [video_path]


def herbench_full_doc_to_visual(doc: dict) -> list[str]:
    return _doc_to_visual(doc, "full")


def herbench_lite_doc_to_visual(doc: dict) -> list[str]:
    return _doc_to_visual(doc, "lite")


def _present_letters(doc: dict) -> list[str]:
    # a handful of questions have 4 options instead of 5
    return [CHOICE_LETTERS[i] for i in range(len(doc["choices"]))]


def herbench_doc_to_text(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    # choices already carry 'A. ' style letter prefixes in the dataset
    options = "\n".join(str(choice) for choice in doc["choices"])
    post_prompt = kwargs.get("post_prompt")
    if post_prompt is None:
        # official HERBench answer instruction (dynamic letter list)
        letters = _present_letters(doc)
        letter_str = ", ".join(letters[:-1]) + f", or {letters[-1]}"
        post_prompt = f"\n\nPlease respond with only the correct answer letter ({letter_str}) without any explanations or additional text."
    return f"{pre_prompt}{doc['question']}\n\n{options}{post_prompt}"


def _doc_to_messages(doc: dict, variant: str, lmms_eval_specific_kwargs: dict | None = None) -> list[dict]:
    """Structured chat messages for chat models (video first, then the prompt)."""
    content = [{"type": "video", "url": path} for path in _doc_to_visual(doc, variant)]
    content.append({"type": "text", "text": herbench_doc_to_text(doc, lmms_eval_specific_kwargs)})
    return [{"role": "user", "content": content}]


def herbench_full_doc_to_messages(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> list[dict]:
    return _doc_to_messages(doc, "full", lmms_eval_specific_kwargs)


def herbench_lite_doc_to_messages(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> list[dict]:
    return _doc_to_messages(doc, "lite", lmms_eval_specific_kwargs)


def extract_herbench_answer(response: str, letters) -> str:
    """Extract the answer letter from a model response.

    Port of the official HERBench answer extraction
    (evaluation/model_wrappers/base_vlm.py::extract_answer_choice), with the
    shared lmms-eval MCQ extractor as a fallback for exotic answer formats.
    """
    s = str(response).strip()
    if not s:
        return ""
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Final answer:",
        "Answer:",
        "Option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "").replace(answer_prefix.lower(), "").strip()
    letter_set = "".join(letters)

    patterns = [
        rf"^([{letter_set}])[\s\.\,\)\:]",  # letter at start followed by a delimiter
        rf"^\(([{letter_set}])\)",  # letter in parentheses at start
        rf"^([{letter_set}])$",  # just the letter alone
        rf"[Aa]nswer:\s*\(?([{letter_set}])\b",  # "Answer: A"
        rf"[Cc]hoice:\s*\(?([{letter_set}])\b",  # "Choice: A"
        rf"\b([{letter_set}])\b[\.\,]",  # standalone letter followed by punctuation
    ]
    for pattern in patterns:
        match = re.search(pattern, s)
        if match:
            return match.group(1).upper()

    # standalone letter surrounded by non-letters (avoids the 'B' in 'Based', etc.)
    upper = s.upper()
    for match in re.finditer(rf"([{letter_set}])", upper):
        start, end = match.start(1), match.end(1)
        before_ok = start == 0 or not upper[start - 1].isalpha()
        after_ok = end == len(upper) or not upper[end].isalpha()
        if before_ok and after_ok:
            return match.group(1)

    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

    return extract_mcq_answer(s, choices=list(letters))


def herbench_process_results(doc, results):
    prediction = results[0] if results else ""
    letters = _present_letters(doc)
    pred_answer = extract_herbench_answer(str(prediction), letters)
    answer = str(doc["answer"]).strip().upper()
    record = {
        "question_id": doc["question_id"],
        "video_id": doc["video_id"],
        "task_type": doc["task_type"],
        "source_dataset": doc.get("source_dataset", ""),
        "pred_answer": pred_answer,
        "answer": answer,
        "score": float(pred_answer == answer),
    }
    metrics = {"herbench_overall_accuracy": record}
    slug = TASK_TYPES.get(doc["task_type"])
    if slug is not None:
        metrics[f"herbench_{slug}_accuracy"] = record
    else:
        eval_logger.warning(f"Unknown HERBench task type: {doc['task_type']}")
    return metrics


def _aggregate_accuracy(results, task_type=None):
    selected = [r for r in results if task_type is None or r["task_type"] == task_type]
    if not selected:
        return 0.0
    correct = sum(r["score"] for r in selected)
    accuracy = 100.0 * correct / len(selected)
    eval_logger.info(f"HERBench {task_type or 'overall'} accuracy: {accuracy:.2f}% ({int(correct)}/{len(selected)})")
    return accuracy


def herbench_aggregate_overall(results):
    return _aggregate_accuracy(results)


def _make_task_aggregator(task_type):
    def aggregate(results):
        return _aggregate_accuracy(results, task_type=task_type)

    return aggregate


herbench_aggregate_temporal_shot_ordering = _make_task_aggregator("Temporal Shot Ordering")
herbench_aggregate_multi_person_duration_reasoning = _make_task_aggregator("Multi-Person Duration Reasoning")
herbench_aggregate_action_sequence_integrity_identification = _make_task_aggregator("Action Sequence Integrity Identification")
herbench_aggregate_appearance_grounded_behavior_interactions = _make_task_aggregator("Appearance-Grounded Behavior & Interactions")
herbench_aggregate_appearance_grounded_attribute_recognition = _make_task_aggregator("Appearance-Grounded Attribute Recognition")
herbench_aggregate_appearance_grounded_localization_trajectory = _make_task_aggregator("Appearance-Grounded Localization & Trajectory")
herbench_aggregate_false_action_memory = _make_task_aggregator("False Action Memory")
herbench_aggregate_scene_verification_arrangement = _make_task_aggregator("Scene Verification & Arrangement")
herbench_aggregate_false_object_memory = _make_task_aggregator("False Object Memory")
herbench_aggregate_multi_entities_grounding_localization = _make_task_aggregator("Multi-Entities Grounding and Localization")
herbench_aggregate_action_counting = _make_task_aggregator("Action Counting")
herbench_aggregate_region_localized_people_counting = _make_task_aggregator("Region Localized People Counting")
