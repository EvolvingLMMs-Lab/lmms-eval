"""Video-Index: 840 multiple-choice video questions from 76 public video benchmarks, one question
per video, 210 per capability group.

Tasks: `video_index` (video file, the model wrapper samples the frames), `video_index_1fps` /
`video_index_64frame` / `video_index_32frame` / `video_index_8frame` (frames of the paper protocol
passed as images) and `video_index_blind` (question and options only, four option permutations per
item). Scoring is rule based: the leading option letter, a stated answer, or the option text that
the reply repeats; a reply that names no option is wrong.

Videos are downloaded one by one from the dataset repository on first use. Set `VIDEO_INDEX_DIR`
to a local copy of the repository (the directory that holds `videos/`) to read them from disk.
"""

import os
import random
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from datasets import Dataset
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.models.model_utils.load_video import import_decord

HF_REPO_ID = "GMLRVigil/Video-Index"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ANSWER_INSTR = "Reply with ONLY the option letter (or the exact short answer if no options)."
PROMPT = "{intro}\n\nQuestion: {q}\n{opts}\n" + ANSWER_INSTR
INTRO_FRAMES = "You are given {n} frame(s) sampled from a video. Answer the question based on these frames."
INTRO_VIDEO = "You are given a video. Answer the question based on this video."
INTRO_BLIND = "You are given NO frames from the video. Answer the question from the text alone."
GROUPS = [
    ("perception", "Perception"),
    ("temporal", "Temporal"),
    ("spatial_physical", "Spatial"),
    ("reasoning_knowledge", "Reasoning"),
]
N_PERM, PERM_SEED = 4, 42

# leading option letter: "B", "(B)", "B.", "B) text", "[b]", "b: text"
_LEAD = re.compile(r"^\s*[\(\[]?([A-Ja-j])[\)\]\.,:]?(?:\s|$)")
# "answer is (B)" / "Answer: B." / "option B": the last occurrence wins
_STATED = re.compile(
    r"(?:answer|option|choice)\s*(?:is|would\s+be)?\s*[:\-]?\s*[\(\[]?([A-Ja-j])[\)\]\.,:]?(?:\s|$)",
    re.IGNORECASE,
)
_STATED_CN = re.compile(r"答案\s*(?:是|为)?\s*[:：]?\s*[\(\[]?([A-Ja-j])(?![A-Za-z])")


def permutations_for(item_id: str, k: int, n_perm: int = N_PERM, seed: int = PERM_SEED) -> list[list[int]]:
    """Option orders of the blind protocol: perm[j] is the index of the original option shown at position j."""
    rng = random.Random(f"{seed}|{item_id}")
    return [rng.sample(range(k), k) for _ in range(n_perm)]


def render_options(texts: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(texts))


def _pre(s: str) -> str:
    s = str(s)
    s = s.replace("（", "(").replace("）", ")").replace("：", ":")
    s = s.replace("。", ".").replace("，", ",")
    return s.replace("*", "").replace("#", "")


def _norm_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[‘’“”`]", "'", s)
    s = re.sub(r"[^\w\s.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(".").strip()


def _flat(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[.\-]", " ", t)).strip()


def extract_letter(reply: str | None, options: list[str]) -> str | None:
    """Option letter named by a reply, or None. Order: the leading letter, a stated answer
    ("the answer is B"), the option whose text the reply repeats. `options` are the option
    texts in the order shown to the model (at most ten)."""
    if reply is None:
        return None
    s = _pre(reply).strip()
    m = _LEAD.match(s)
    if m:
        letter = m.group(1).upper()
        # a bare "A" / "I" followed by more words is usually the article or the pronoun
        has_delim = any(ch in m.group(0) for ch in "()[].,:")
        if has_delim or letter not in ("A", "I") or len(s.split()) == 1:
            return letter
    ms = list(_STATED.finditer(s)) or list(_STATED_CN.finditer(s))
    if ms:
        return ms[-1].group(1).upper()
    ns = _norm_text(s)
    lettered = [f"{LETTERS[i]}. {t}" for i, t in enumerate(options)]
    for i, opt in enumerate(options):
        if ns and ns in (_norm_text(lettered[i]), _norm_text(opt)):
            return LETTERS[i]
    nsf = _flat(ns)
    hits = []
    for i, opt in enumerate(options):
        ot = _flat(_norm_text(opt))
        if ot and len(ot) >= 3 and f" {ot} " in f" {nsf} ":
            hits.append(i)
    if len(hits) == 1:
        return LETTERS[hits[0]]
    if re.fullmatch(r"\d{1,2}", s) and int(s) < len(options):
        return LETTERS[int(s)]
    return None


def score_reply(reply: str | None, gold: str, options: list[str]) -> float:
    """1.0 when the reply names the gold option, else 0.0. A reply that names no option is wrong."""
    letter = extract_letter(reply, options)
    return float(letter is not None and letter == str(gold).strip().upper())


def one_fps_indices(n_frames: int, native_fps: float, duration: float | None, rate: float = 1.0, cap: int = 512) -> list[int]:
    """Indices of the stored frames that are sent: the stored frame nearest to every multiple of
    1 / rate seconds of the timeline [0, duration); every stored frame when the video is stored
    below the rate; uniform thinning to at most `cap` frames."""
    if n_frames <= 0:
        return []
    native = native_fps or 2.0
    ts = np.arange(n_frames, dtype=np.float64) / native
    dur = float(duration) if duration and duration == duration else float(ts[-1] + 0.5)
    if n_frames <= dur * rate + 1:
        idx = list(range(n_frames))
    else:
        targets = np.arange(0.0, max(dur, 0.5 / rate), 1.0 / rate)
        idx = sorted(set(int(np.abs(ts - t).argmin()) for t in targets))
    if cap and len(idx) > cap:
        keep = sorted(set(int(round(x)) for x in np.linspace(0, len(idx) - 1, cap)))
        idx = [idx[i] for i in keep]
    return idx


def read_frames(reader: Any, indices: Sequence[int]) -> list[np.ndarray]:
    """Frames (numpy arrays, RGB) at the given indices of a decord VideoReader, decoded in stream order.
    Random access returns a neighbouring frame on some of the videos; the videos hold at most 1,024
    stored frames, so decoding from the first frame is affordable."""
    wanted = set(int(i) for i in indices)
    if not wanted:
        return []
    frames = {}
    reader.seek(0)
    for i in range(max(wanted) + 1):
        frame = reader.next()
        if i in wanted:
            frames[i] = frame.asnumpy()
    return [frames[int(i)] for i in indices]


# ------------------------------------------------------------------ documents
def video_index_blind_process_docs(dataset: Dataset) -> Dataset:
    """One document per (item, option permutation)."""

    def expand(batch: dict[str, list]) -> dict[str, list]:
        out = {key: [] for key in batch}
        out["perm"], out["perm_idx"] = [], []
        for i, item_id in enumerate(batch["item_id"]):
            for perm_idx, perm in enumerate(permutations_for(item_id, len(batch["options"][i]))):
                for key in batch:
                    out[key].append(batch[key][i])
                out["perm"].append(perm)
                out["perm_idx"].append(perm_idx)
        return out

    return dataset.map(expand, batched=True, remove_columns=dataset.column_names)


def _shown_options(doc: dict) -> tuple[list[str], str]:
    """Option texts in the order shown to the model and the letter of the marked answer."""
    options = [str(o) for o in doc["options"]]
    perm = list(doc.get("perm") or range(len(options)))
    return [options[j] for j in perm], LETTERS[perm.index(int(doc["answer_idx"]))]


def video_index_doc_to_target(doc: dict) -> str:
    """Letter of the marked answer in the option order shown to the model."""
    return _shown_options(doc)[1]


# --------------------------------------------------------------------- visual
def _video_file(doc: dict) -> str:
    root = os.getenv("VIDEO_INDEX_DIR")
    if root:
        path = os.path.join(os.path.expanduser(root), doc["video"])
        if os.path.exists(path):
            return path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=doc["video"])


@lru_cache(maxsize=4096)
def _frame_indices(video_file: str, duration: float | None, fps: float, max_frames: int) -> tuple[int, ...]:
    decord = import_decord()
    reader = decord.VideoReader(video_file, ctx=decord.cpu(0), num_threads=1)
    return tuple(one_fps_indices(len(reader), reader.get_avg_fps(), duration, rate=fps, cap=max_frames))


def _frame_settings(kwargs: dict | None) -> tuple[float, int, int]:
    kwargs = kwargs or {}
    return float(kwargs.get("fps", 1.0)), int(kwargs.get("max_frames", 512)), int(kwargs.get("short_side", 224))


def video_index_doc_to_visual(doc: dict) -> list[str]:
    """The video file of the item; the model wrapper samples the frames."""
    return [_video_file(doc)]


def video_index_doc_to_visual_empty(doc: dict) -> list:
    """Blind protocol: no visual input."""
    return []


def video_index_doc_to_visual_frames(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> list[Image.Image]:
    """Frames of the paper protocol: one frame per `1 / fps` seconds, at most `max_frames`, short side `short_side`."""
    fps, max_frames, short_side = _frame_settings(lmms_eval_specific_kwargs)
    video_file = _video_file(doc)
    indices = _frame_indices(video_file, doc.get("duration_s"), fps, max_frames)
    decord = import_decord()
    reader = decord.VideoReader(video_file, ctx=decord.cpu(0), num_threads=1)
    frames = []
    for frame in read_frames(reader, indices):
        image = Image.fromarray(frame).convert("RGB")
        width, height = image.size
        if short_side and min(width, height) > short_side:
            scale = short_side / min(width, height)
            image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BICUBIC)
        frames.append(image)
    return frames


# ----------------------------------------------------------------------- text
def _prompt(doc: dict, intro: str) -> str:
    return PROMPT.format(intro=intro, q=doc["question"], opts=render_options(_shown_options(doc)[0]))


def video_index_doc_to_text(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> str:
    """Prompt of the video-file task and of the blind task (`intro` is the first sentence)."""
    kwargs = lmms_eval_specific_kwargs or {}
    return _prompt(doc, kwargs.get("intro", INTRO_VIDEO))


def video_index_doc_to_text_frames(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> str:
    """Prompt of the frame tasks; the first sentence states the number of frames sent."""
    kwargs = lmms_eval_specific_kwargs or {}
    fps, max_frames, _ = _frame_settings(kwargs)
    n = len(_frame_indices(_video_file(doc), doc.get("duration_s"), fps, max_frames))
    return _prompt(doc, kwargs.get("intro", INTRO_FRAMES).format(n=n))


# -------------------------------------------------------------------- scoring
METRICS = ["video_index_acc", "video_index_perception", "video_index_temporal", "video_index_spatial", "video_index_reasoning"]


def video_index_process_results(doc: dict, results: list[str]) -> dict[str, dict]:
    """Score one reply with the rule scorer; every metric receives the same record."""
    reply = results[0] if results else ""
    options, gold = _shown_options(doc)
    letter = extract_letter(reply, options)
    record = {
        "item_id": doc["item_id"],
        "benchmark": doc["benchmark"],
        "capability_group": doc["capability_group"],
        "perm_idx": doc.get("perm_idx", 0),
        "pred": letter or "",
        "gold": gold,
        "score": score_reply(reply, gold, options),
    }
    return {metric: record for metric in METRICS}


def _accuracy(results: list[dict], group: str | None = None) -> float:
    """Accuracy in percent: the rows of one item (option permutations) are averaged first."""
    per_item = defaultdict(list)
    for record in results:
        if group is None or record["capability_group"] == group:
            per_item[record["item_id"]].append(record["score"])
    if not per_item:
        return float("nan")
    accuracy = 100.0 * float(np.mean([np.mean(scores) for scores in per_item.values()]))
    eval_logger.info(f"Video-Index {group or 'overall'}: {accuracy:.1f} ({len(per_item)} items)")
    return accuracy


def video_index_aggregate_overall(results: list[dict]) -> float:
    return _accuracy(results)


def video_index_aggregate_perception(results: list[dict]) -> float:
    return _accuracy(results, "perception")


def video_index_aggregate_temporal(results: list[dict]) -> float:
    return _accuracy(results, "temporal")


def video_index_aggregate_spatial(results: list[dict]) -> float:
    return _accuracy(results, "spatial_physical")


def video_index_aggregate_reasoning(results: list[dict]) -> float:
    return _accuracy(results, "reasoning_knowledge")
