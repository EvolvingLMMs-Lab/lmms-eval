"""Video-MME-v2 task utilities.

Implements:
- Video path resolution and subtitle loading (JSONL word-level)
- Prompt construction (with/without subtitles, interleaved, reasoning)
- Answer extraction (A-H)
- Grouped non-linear scoring (relevance + logic)
- Per-level and fine-grained aggregation
"""

import ast
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from loguru import logger as eval_logger

# ---------------------------------------------------------------------------
# Config loading — read dataset_kwargs.cache_dir from our YAML
# ---------------------------------------------------------------------------
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "videommev2.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = [line for line in raw_data if "!function" not in line]
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]

# ---------------------------------------------------------------------------
# Prompts (from official evaluation script)
# ---------------------------------------------------------------------------
WO_SUB_PROMPT = "These are the frames of a video."

WITH_SUB_PROMPT = (
    "These are the frames of a video. "
    "This video's subtitles are listed below:\n{}\n"
)

WITH_SUB_PROMPT_INTERLEAVE = (
    "These are the frames of a video with corresponding subtitles "
    "shown between frames. The subtitles indicate what is being said "
    "during the time interval between adjacent frames."
)

INSTRUCT_PROMPT = (
    "Select the best answer to the following multiple-choice "
    "question based on the video. Respond with only the letter "
    "(A, B, C, D, E, F, G, or H) of the correct option."
)

THINK_PROMPT = (
    "Please perform a detailed reasoning based on the provided "
    "video frames to answer the following multiple-choice question "
    "selecting the best option from A through H and providing your "
    "final response strictly in the format: 'Final Answer: <letter>."
)

# ---------------------------------------------------------------------------
# Subtitle helpers (JSONL word-level with timestamps)
# ---------------------------------------------------------------------------



def _il_filter_noise_entries(entries):
    """Filter out non-speech tags ([Music], [Applause], ...) and dedup consecutive identical text."""
    if not entries:
        return entries
    import re as _re_il
    NOISE = _re_il.compile(r'^\s*[\(\[].*?[\)\]]\s*$')
    out = []
    last_text = None
    for ent in entries:
        t = (ent.get('text') or '').strip()
        if not t:
            continue
        if NOISE.match(t):
            continue
        if t == last_text:
            continue
        out.append(ent)
        last_text = t
    return out


def _load_subtitle_jsonl(subtitle_path: str) -> list[dict] | None:
    """Load JSONL subtitle file.

    Each line: {"text": str, "start_time": float, "end_time": float}
    """
    if not os.path.exists(subtitle_path):
        return None
    entries: list[dict] = []
    with open(subtitle_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed line (e.g. truncated subtitle file).
                continue
    return entries


def _subtitle_concat_all(entries: list[dict] | None) -> str:
    """Concatenate all subtitle words into a single string."""
    if not entries:
        return ""
    return " ".join(e["text"] for e in entries)


def _group_subtitle_segments(
    entries: list[dict] | None,
    gap_threshold: float = 0.5,
) -> list[dict]:
    """Group word-level entries into sentence-level segments."""
    if not entries:
        return []
    segments: list[dict] = []
    current_words = [entries[0]]
    for i in range(1, len(entries)):
        prev = entries[i - 1]
        curr = entries[i]
        time_gap = curr["start_time"] - prev["end_time"]
        prev_ends_sentence = prev["text"].rstrip().endswith((".", "!", "?"))
        if time_gap > gap_threshold or (
            prev_ends_sentence and time_gap > 0.1
        ):
            segments.append(
                {
                    "text": " ".join(w["text"] for w in current_words),
                    "start_time": current_words[0]["start_time"],
                    "end_time": current_words[-1]["end_time"],
                }
            )
            current_words = [curr]
        else:
            current_words.append(curr)
    if current_words:
        segments.append(
            {
                "text": " ".join(w["text"] for w in current_words),
                "start_time": current_words[0]["start_time"],
                "end_time": current_words[-1]["end_time"],
            }
        )
    return segments


def _subtitle_between_timestamps(
    entries: list[dict] | None,
    start_time: float,
    end_time: float,
) -> str:
    """Collect words whose time range overlaps [start_time, end_time)."""
    if not entries:
        return ""
    words = []
    for e in entries:
        if e["end_time"] >= start_time and e["start_time"] < end_time:
            words.append(e["text"])
    return " ".join(words)


def _segments_between_timestamps(
    segments: list[dict],
    start_time: float,
    end_time: float,
) -> list[dict]:
    """Return segments overlapping [start_time, end_time)."""
    return [
        seg
        for seg in segments
        if seg["end_time"] >= start_time and seg["start_time"] < end_time
    ]


# ---------------------------------------------------------------------------
# Video path resolution
# ---------------------------------------------------------------------------


def videommev2_doc_to_visual(doc: dict) -> list[str]:
    """Resolve local video path for a document."""
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_id = str(doc["video_id"])
    video_path = os.path.join(cache_dir, "data", video_id + ".mp4")
    if os.path.exists(video_path):
        return [video_path]
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        return [video_path.replace("mp4", "MP4")]
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        return [video_path.replace("mp4", "mkv")]
    else:
        sys.exit(f"video path: {video_path} does not exist, please check")


# ---------------------------------------------------------------------------
# doc_to_text — prompt construction
# ---------------------------------------------------------------------------


def videommev2_doc_to_text(
    doc: dict,
    lmms_eval_specific_kwargs: dict | None = None,
) -> str:
    """Build prompt without subtitles."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    reasoning = lmms_eval_specific_kwargs.get("reasoning", False)
    response_prompt = THINK_PROMPT if reasoning else INSTRUCT_PROMPT

    question = str(doc["question"])
    options = str(doc["options"])
    full_prompt = (
        f"{WO_SUB_PROMPT}\n"
        f"{response_prompt}\n"
        f"Question: {question}\n"
        f"{options}"
    )
    return full_prompt


def videommev2_doc_to_text_subtitle(
    doc: dict,
    lmms_eval_specific_kwargs: dict | None = None,
) -> str:
    """Build prompt with concatenated subtitles."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    reasoning = lmms_eval_specific_kwargs.get("reasoning", False)
    response_prompt = THINK_PROMPT if reasoning else INSTRUCT_PROMPT

    # Load subtitle
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_id = str(doc["video_id"])
    subtitle_path = os.path.join(cache_dir, "subtitle", video_id + ".jsonl")
    sub_entries = _load_subtitle_jsonl(subtitle_path)
    subtitle_text = _subtitle_concat_all(sub_entries)
    if not subtitle_text:
        subtitle_text = "No subtitles available"

    sub_prompt = WITH_SUB_PROMPT.format(subtitle_text)

    question = str(doc["question"])
    options = str(doc["options"])
    full_prompt = (
        f"{sub_prompt}"
        f"{response_prompt}\n"
        f"Question: {question}\n"
        f"{options}"
    )
    return full_prompt


# ---------------------------------------------------------------------------
# Answer extraction (A-H)
# ---------------------------------------------------------------------------


def _extract_answer(s: str) -> str:
    """Extract answer letter A-H from model response."""
    s = s.strip()
    answer_prefixes = [
        "Final Answer:",
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
    ]
    for prefix in answer_prefixes:
        s = s.replace(prefix, "")
    if len(s.split()) > 10 and not re.search("[A-H]", s):
        return ""
    matches = re.search(r"[A-H]", s)
    if matches is None:
        return ""
    return matches[0]


# ---------------------------------------------------------------------------
# Non-linear scoring (from official evaluation script)
# ---------------------------------------------------------------------------

# Relevance scoring: exponential map based on correct count in group of 4
_RELEVANCE_SCORE_MAP = {
    0: 0.0,
    1: 100.0 / 16,
    2: 100.0 * 4 / 16,
    3: 100.0 * 9 / 16,
    4: 100.0,
}


def _cal_relevance(scores: list[int]) -> tuple[float, float]:
    """Calculate relevance (non-linear) score and linear score."""
    correct_count = sum(scores)
    nonlinear = _RELEVANCE_SCORE_MAP.get(correct_count, 0.0)
    linear = correct_count * 25.0
    return nonlinear, linear


def _cal_logic(scores: list[int], group_structure: str) -> float:
    """Calculate logic score with first-error truncation."""
    group_structure_list = ast.literal_eval(group_structure)

    # Find the last consecutive correct index
    last_correct_idx = -1
    for idx, val in enumerate(scores):
        if val:
            last_correct_idx = idx
        else:
            break

    if group_structure_list == [1, 2, 3, 4]:
        score_map = {
            0: 0.0,
            1: 100.0 / 16,
            2: 100.0 * 4 / 16,
            3: 100.0 * 9 / 16,
            4: 100.0,
        }
    elif group_structure_list == [1, [2, 3], 4]:
        score_map = {
            0: 0.0,
            1: 100.0 / 12,
            2: 100.0 * 4 / 12,
            3: 100.0 * 7 / 12,
            4: 100.0,
        }
        # If first is correct and third is correct (parallel branch)
        if last_correct_idx == 0 and scores[2]:
            last_correct_idx += 1
    elif group_structure_list == [[1, 2], 3, 4]:
        score_map = {
            0: 0.0,
            1: 100.0 / 10,
            2: 100.0 * 2 / 10,
            3: 100.0 * 5 / 10,
            4: 100.0,
        }
        # If first wrong but second correct (parallel branch)
        if last_correct_idx == -1 and scores[1]:
            last_correct_idx += 1
    else:
        raise ValueError(
            f"Unknown group_structure_list: {group_structure_list}"
        )

    return score_map.get(last_correct_idx + 1, 0.0)


# ---------------------------------------------------------------------------
# process_results — called per sample, returns dict for aggregation
# ---------------------------------------------------------------------------


def videommev2_process_results(doc: dict, results: list[str]) -> dict:
    """Process a single sample's results.

    Returns dict with metric key -> data dict for aggregation.
    """
    pred = results[0]
    pred_ans = _extract_answer(pred)
    gt_ans = str(doc["answer"]).strip()

    score = 1 if pred_ans.upper() == gt_ans.upper() else 0

    data_dict = {
        "question_id": doc["question_id"],
        "video_id": doc["video_id"],
        "group_type": doc["group_type"],
        "group_structure": doc["group_structure"],
        "level": doc["level"],
        "second_head": doc["second_head"],
        "third_head": doc["third_head"],
        "pred_answer": pred_ans,
        "answer": gt_ans,
        "score": score,
    }

    return {"videommev2_score": data_dict}


# ---------------------------------------------------------------------------
# Aggregation — grouped non-linear scoring
# ---------------------------------------------------------------------------


def videommev2_aggregate_results(results: list[dict]) -> float:
    """Aggregate per-sample results into the final Video-MME-v2 score.

    Groups questions by video (4 per video), applies non-linear scoring,
    and reports per-level, per-category, and overall metrics.

    Returns the overall non-linear score.
    """
    n_total = len(results)
    n_failed = sum(1 for r in results if r["pred_answer"] == "")

    # Simple accuracy on valid predictions
    valid = [r for r in results if r["pred_answer"] != ""]
    if valid:
        simple_acc = sum(r["score"] for r in valid) / len(valid) * 100
        eval_logger.info(
            f"Video-MME-v2 | Simple accuracy (valid): {simple_acc:.2f}% "
            f"({n_total} total, {n_failed} failed extraction)"
        )

    # Group by video_id (multi-GPU gather may interleave videos)
    video_groups: dict[str, list[dict]] = {}
    for r in results:
        vid = r["video_id"]
        if vid not in video_groups:
            video_groups[vid] = []
        video_groups[vid].append(r)

    # Sort questions within each video by question_id
    groups: list[list[dict]] = []
    for vid in sorted(video_groups.keys()):
        g = sorted(video_groups[vid], key=lambda x: x["question_id"])
        groups.append(g)

    # Score each group
    level_scores: dict[str, list[float]] = {
        "1": [],
        "2": [],
        "3": [],
    }
    relevance_scores: list[float] = []
    relevance_linear_scores: list[float] = []
    logic_scores: list[float] = []
    total_scores: list[float] = []
    second_head_scores: dict[str, list[float]] = {}
    third_head_scores: dict[str, list[float]] = {}

    for group in groups:
        if len(group) < 4:
            continue

        # Use metadata from last question in group (matches official script)
        meta = group[-1]
        group_type = meta["group_type"]
        group_structure = meta["group_structure"]
        level = meta["level"]
        second_head = meta["second_head"]
        third_head = meta["third_head"]

        scores = [item["score"] for item in group]

        if group_type == "relevance":
            exp_score, linear_score = _cal_relevance(scores)
            relevance_scores.append(exp_score)
            relevance_linear_scores.append(linear_score)
        elif group_type == "logic":
            exp_score = _cal_logic(scores, group_structure)
            logic_scores.append(exp_score)
        else:
            eval_logger.warning(f"Unknown group_type: {group_type}")
            continue

        # Per-level
        if level is not None and str(level) != "None":
            level_scores[str(int(level))].append(exp_score)

        total_scores.append(exp_score)

        # Second/third head
        sh = str(second_head) if second_head else "None"
        if sh not in second_head_scores:
            second_head_scores[sh] = []
        second_head_scores[sh].append(exp_score)

        th = str(third_head) if third_head else "None"
        if th not in third_head_scores:
            third_head_scores[th] = []
        third_head_scores[th].append(exp_score)

    # Compute averages
    def _avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    overall = _avg(total_scores)

    # Log detailed results
    eval_logger.info("=" * 60)
    eval_logger.info("Video-MME-v2 Grouped Non-Linear Scoring Results")
    eval_logger.info("=" * 60)
    eval_logger.info(f"  Overall Score:          {overall:.2f}")
    for lv in ("1", "2", "3"):
        lv_avg = _avg(level_scores[lv])
        eval_logger.info(
            f"  Level {lv}:                {lv_avg:.2f} "
            f"({len(level_scores[lv])} groups)"
        )
    eval_logger.info(f"  Relevance (non-linear): {_avg(relevance_scores):.2f}")
    eval_logger.info(
        f"  Relevance (linear):     {_avg(relevance_linear_scores):.2f}"
    )
    eval_logger.info(f"  Logic:                  {_avg(logic_scores):.2f}")

    # Second head breakdown
    eval_logger.info("-" * 60)
    eval_logger.info("Second Head (capability) breakdown:")
    for k in sorted(second_head_scores.keys()):
        if k != "None":
            eval_logger.info(
                f"  {k:<45} {_avg(second_head_scores[k]):>6.2f} "
                f"({len(second_head_scores[k])} groups)"
            )

    # Third head breakdown
    eval_logger.info("-" * 60)
    eval_logger.info("Third Head (fine-grained) breakdown:")
    for k in sorted(third_head_scores.keys()):
        if k != "None":
            eval_logger.info(
                f"  {k:<45} {_avg(third_head_scores[k]):>6.2f} "
                f"({len(third_head_scores[k])} groups)"
            )

    eval_logger.info("=" * 60)

    return overall


# ---------------------------------------------------------------------------
# Interleaved-style subtitle (text-mode, fits simple adapter w/ codec offline)
# Format: groups subtitle segments with [start-end] timestamps inline so model
# sees temporal alignment without needing per-frame token interleaving.
# ---------------------------------------------------------------------------


def _subtitle_segments_with_timestamps(entries):
    """Return list of (start, end, text) sentence-level segments."""
    if not entries:
        return []
    segs = _group_subtitle_segments(entries)
    return [(float(s["start_time"]), float(s["end_time"]), str(s["text"])) for s in segs]

def videommev2_doc_to_text_interleaved(
    doc,
    lmms_eval_specific_kwargs=None,
):
    """Subtitle prompt with [start-end] timestamps inline per segment.

    Approximates videommev2_interleaved_subtitle through the simple model
    adapter (chat-side sentinel injection is not available on codec-offline
    path). Subtitle segments retain their start/end seconds so the model has
    temporal alignment information.
    """
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_id = str(doc["video_id"])
    subtitle_path = os.path.join(cache_dir, "subtitle", video_id + ".jsonl")
    sub_entries = _load_subtitle_jsonl(subtitle_path)
    segs = _subtitle_segments_with_timestamps(sub_entries)

    if segs:
        lines = [f"[{s:.2f}-{e:.2f}] {t}" for s, e, t in segs]
        sub_block = "This video's subtitles with timestamps (seconds):\n" + "\n".join(lines) + "\n\n"
    else:
        sub_block = ""

    question = str(doc["question"])
    options = str(doc["options"])
    parts = [pre_prompt, sub_block, "Question: " + question, options, post_prompt]
    return "\n".join(p for p in parts if p).strip()

# Sentinel prefix: task layer embeds subtitle JSON in a hidden text
# content item; model layer's build_ov_messages strips it and passes
# parsed subtitles to _process_video_content_with_timestamp.
from lmms_eval.tasks.videomme.utils import SUBTITLE_DATA_PREFIX


def videommev2_doc_to_messages_interleaved_subtitle(
    doc: dict,
    lmms_eval_specific_kwargs: dict | None = None,
) -> list[dict]:
    """Build messages with interleaved subtitle sentinel for model-layer injection.

    Env-var ablations:
      LMMS_IL_WORDLEVEL=1 - feed raw word entries (vs sentence-level segments)
      LMMS_IL_PREAMBLE=1  - place WITH_SUB_PROMPT_INTERLEAVE before the video
                            (vs only at end as part of question_text)
    """
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}

    videos = videommev2_doc_to_visual(doc)
    video_path = videos[0]

    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_id = str(doc["video_id"])
    subtitle_path = os.path.join(cache_dir, "subtitle", video_id + ".jsonl")
    sub_entries = _load_subtitle_jsonl(subtitle_path)

    use_wordlevel = os.environ.get("LMMS_IL_WORDLEVEL", "0") == "1"
    use_preamble = os.environ.get("LMMS_IL_PREAMBLE", "0") == "1"

    subtitle_dict: dict[str, str] = {}
    if sub_entries:
        if use_wordlevel:
            # Raw word-level: each entry stays as its own bucket so the
            # chat adapter places it at the nearest frame independently.
            for i, ent in enumerate(sub_entries):
                s = float(ent["start_time"]); e = float(ent["end_time"])
                key = f"{s:.4f}:{e:.4f}"
                if key in subtitle_dict:
                    # avoid overwrite on identical timestamps
                    key = f"{s:.4f}:{e:.4f}_{i}"
                subtitle_dict[key] = ent["text"]
        else:
            segments = _group_subtitle_segments(sub_entries)
            for seg in segments:
                key = f"{seg['start_time']:.3f}:{seg['end_time']:.3f}"
                subtitle_dict[key] = seg["text"]

    reasoning = lmms_eval_specific_kwargs.get("reasoning", False)
    response_prompt = THINK_PROMPT if reasoning else INSTRUCT_PROMPT

    question = str(doc["question"])
    options = str(doc["options"])

    use_preamble_concat = os.environ.get("LMMS_IL_PREAMBLE_CONCAT", "0") == "1"
    use_question_concat = os.environ.get("LMMS_IL_QUESTION_CONCAT", "0") == "1"
    if use_question_concat:
        # V_E: concat narrative placed AT END inside question_text — mirrors w_subtitle layout.
        full_text = _subtitle_concat_all(sub_entries) or "No subtitles available"
        sub_block = WITH_SUB_PROMPT.format(full_text)
        preamble_text = None
        question_text = sub_block + chr(10).join([response_prompt, "Question: " + question, options])
    elif use_preamble_concat:
        full_text = _subtitle_concat_all(sub_entries) or "No subtitles available"
        preamble_text = WITH_SUB_PROMPT.format(full_text)
        question_text = chr(10).join([response_prompt, "Question: " + question, options])
    elif use_preamble:
        preamble_text = WITH_SUB_PROMPT_INTERLEAVE
        question_text = chr(10).join([response_prompt, "Question: " + question, options])
    else:
        preamble_text = None
        question_text = chr(10).join([WITH_SUB_PROMPT_INTERLEAVE, response_prompt, "Question: " + question, options])

    suppress_frame_subs = os.environ.get("LMMS_IL_NO_FRAME_SUBS", "0") == "1"
    user_content: list[dict] = []
    if subtitle_dict and not suppress_frame_subs:
        user_content.append(
            {"type": "text", "text": SUBTITLE_DATA_PREFIX + json.dumps(subtitle_dict)}
        )
    if preamble_text is not None:
        user_content.append({"type": "text", "text": preamble_text})
    user_content.append({"type": "video", "url": video_path})
    user_content.append({"type": "text", "text": question_text})

    return [{"role": "user", "content": user_content}]

def videommev2_doc_to_text_simple_il(
    doc,
    lmms_eval_specific_kwargs=None,
):
    """Doc->text for simple-adapter IL: returns SUBTITLE_DATA sentinel + prompt.

    Simple adapter parses the sentinel to extract subtitle dict, then injects
    per-frame subtitles into the rewritten vision blocks (true interleaving).
    The non-sentinel prompt mirrors W_SUBTITLE so the model also gets the full
    concat narrative as fallback context.
    """
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_id = str(doc["video_id"])
    sub_path = os.path.join(cache_dir, "subtitle", video_id + ".jsonl")
    sub_entries = _load_subtitle_jsonl(sub_path)
    if os.environ.get("LMMS_IL_FILTER_NOISE", "0") == "1":
        sub_entries = _il_filter_noise_entries(sub_entries)
    segments = _group_subtitle_segments(sub_entries)

    use_wordlevel = os.environ.get("LMMS_IL_WORDLEVEL", "0") == "1"
    subtitle_dict = {}
    if use_wordlevel and sub_entries:
        for _i, _ent in enumerate(sub_entries):
            _s = float(_ent["start_time"]); _e = float(_ent["end_time"])
            _key = f"{_s:.4f}:{_e:.4f}"
            if _key in subtitle_dict:
                _key = f"{_s:.4f}:{_e:.4f}_{_i}"
            subtitle_dict[_key] = _ent["text"]
    else:
        for seg in segments:
            key = f"{seg['start_time']:.3f}:{seg['end_time']:.3f}"
            subtitle_dict[key] = seg["text"]

    full_text = _subtitle_concat_all(sub_entries) or "No subtitles available"
    use_concat_front = os.environ.get("LMMS_IL_CONCAT_FRONT", "0") == "1"
    if use_concat_front:
        # Will be moved BEFORE first vision_start by simple adapter via sentinel marker
        sub_block = ""
        front_block = "__CONCAT_FRONT__:" + WITH_SUB_PROMPT.format(full_text) + chr(10) + "__CONCAT_FRONT_END__" + chr(10)
    elif os.environ.get("LMMS_IL_NO_CONCAT_BODY", "0") == "1":
        sub_block = ""; front_block = ""
    else:
        sub_block = WITH_SUB_PROMPT.format(full_text); front_block = ""

    reasoning = lmms_eval_specific_kwargs.get("reasoning", False)
    response_prompt = THINK_PROMPT if reasoning else INSTRUCT_PROMPT
    question = str(doc["question"])
    options = str(doc["options"])

    body = sub_block + response_prompt + chr(10) + "Question: " + question + chr(10) + options
    sentinel = SUBTITLE_DATA_PREFIX + json.dumps(subtitle_dict) + chr(10)
    return sentinel + front_block + body
