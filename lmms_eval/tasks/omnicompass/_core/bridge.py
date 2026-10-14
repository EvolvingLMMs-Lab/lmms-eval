"""Shared OmniCompass input and scoring protocol."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import portalocker

from .extract_audio import extract_audio
from .extract_frames import decoder_versions, extract_video
from .hf_data import DatasetStore, default_cache
from .omnicompass_public_adapter import evaluate_public_predictions, render_prompt
from .omnicompass_release import format_number
from .print_frame_eval_results import is_content_policy_error
from .release_tools import rows, safe
from .review_run import record

SETTINGS = ("text", "text_image")


def source_digest(path):
    """Relative packaging imports must not change the evaluation identity."""
    if path.suffix != ".py":
        return hashlib.sha256(path.read_bytes()).hexdigest()
    tree = ast.parse(path.read_text())

    class NormalizeImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.level == 1 and node.module is None:
                return ast.Import(names=node.names)
            if node.level == 1:
                node.level = 0
            return node

    return hashlib.sha256(ast.dump(NormalizeImports().visit(tree)).encode()).hexdigest()


class Protocol:
    """Reuse the published dataset, media preparation, prompts and scorer."""

    def __init__(self, root=None, cache=None, *, nframe=32, fps=None, max_side=640, revision=None):
        self.store = DatasetStore(root, cache, revision)
        self.root = self.store.root
        self.cache = self.store.cache
        if (nframe is None) == (fps is None):
            raise ValueError("Specify exactly one of nframe and fps")
        if nframe is not None and (isinstance(nframe, bool) or int(nframe) != nframe or nframe <= 0):
            raise ValueError("nframe must be a positive integer")
        if fps is not None and (not math.isfinite(fps) or fps <= 0):
            raise ValueError("fps must be positive and finite")
        self.nframe, self.fps, self.max_side = nframe, fps, max_side
        self.questions = list(rows(self.root / "questions.jsonl"))
        self.videos = {v["video_id"]: v for v in rows(self.root / "videos.jsonl")}
        from . import omnicompass_public_adapter

        code_root = Path(omnicompass_public_adapter.__file__).parent
        sources = [self.root / name for name in ("questions.jsonl", "videos.jsonl", "media_manifest.jsonl")]
        sources.append(Path(__file__))
        sources += [
            code_root / (name + ".py")
            for name in (
                "extract_frames",
                "extract_audio",
                "omnicompass_public_adapter",
                "omnicompass_release",
                "omnicompass_frame_prompt",
                "omnicompass_av_frame_prompt",
                "omnicompass_scoring",
                "print_frame_eval_results",
                "print_modality_gain_results",
            )
        ]
        fingerprint = {
            "nframe": nframe,
            "fps": fps,
            "max_side": max_side,
            "decoder_versions": decoder_versions(),
            "source_hashes": {str(p.name): source_digest(p) for p in sources},
        }
        self.protocol_id = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_env(cls):
        fps = os.getenv("OMNICOMPASS_FPS")
        nframe = os.getenv("OMNICOMPASS_NFRAME")
        if fps and nframe:
            raise ValueError("OMNICOMPASS_FPS and OMNICOMPASS_NFRAME are mutually exclusive")
        return cls(
            os.getenv("OMNICOMPASS_DATA"),
            default_cache(),
            fps=float(fps) if fps else None,
            nframe=None if fps else int(nframe or 32),
        )

    def documents(self, track, tasks=("TG", "SG", "QA"), settings=SETTINGS, video_ids=None):
        if track not in ("VL", "AV") or not set(tasks) <= {"TG", "SG", "QA"} or not tasks:
            raise ValueError("Invalid track/tasks")
        if not settings or len(set(settings)) != len(settings) or not set(settings) <= set(SETTINGS):
            raise ValueError("Invalid settings")
        selected = [q for q in self.questions if q["track"] == track and q["task"] in tasks and (video_ids is None or q["video_id"] in video_ids)]
        if not selected:
            raise ValueError("No matching questions")
        if video_ids is not None and set(video_ids) != {q["video_id"] for q in selected}:
            raise ValueError("Unknown video or video outside selected track")
        documents = [
            {"question_id": q["question_id"], "video": q["video_id"], "track": q["track"], "task": q["task"], "setting": setting, "question_json": json.dumps(q, sort_keys=True), "answer": "", "protocol_id": self.protocol_id}
            for q in selected
            for setting in settings
        ]
        record(documents)
        return documents

    def content(self, doc):
        if doc["protocol_id"] != self.protocol_id:
            raise ValueError("Protocol mismatch; regenerate task data for this configuration")
        q = json.loads(doc["question_json"])
        # Refuse modified task/ID metadata before building a paid request.
        if any(doc[k] != q[k] for k in ("question_id", "track", "task")):
            raise ValueError("Document identity mismatch")
        video = self.videos[q["video_id"]]
        names = [video["video_path"]]
        if doc["setting"] == "text_image":
            names.append(q["reference_image"])
        self.store.ensure(names)
        path = safe(self.root, video["video_path"])
        stat = path.stat()
        directory, manifest, audio = self.prepare_media(
            q["video_id"],
            q["track"] == "AV",
            (stat.st_size, stat.st_mtime_ns, stat.st_ino),
        )
        result = []
        for frame in manifest["frames"]:
            result.append({"type": "text", "value": f"Timestamp: {format_number(frame['timestamp_sec'])} seconds."})
            result.append({"type": "image", "value": str(safe(directory, frame["path"]))})
        if q["track"] == "AV":
            result.append({"type": "audio", "value": str(audio)})
        if doc["setting"] == "text_image":
            reference = safe(self.root, q["reference_image"])
            if not reference.is_file():
                raise FileNotFoundError(reference)
            result.append({"type": "image", "value": str(reference)})
        result.append({"type": "text", "value": render_prompt(q, doc["setting"], video["duration_seconds"])})
        return result

    @lru_cache(maxsize=32)
    def prepare_media(self, video_id, use_audio, source_stat):
        """Serialize cooperating workers without changing the canonical cache."""
        path = safe(self.root, self.videos[video_id]["video_path"])
        lock_root = self.cache / "framework_locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_name = hashlib.sha256(str(path).encode()).hexdigest()
        with portalocker.Lock(str(lock_root / (lock_name + ".lock")), timeout=600):
            directory, manifest, _ = extract_video(
                path,
                self.cache / "frames",
                nframe=self.nframe,
                fps=self.fps,
                max_side=self.max_side,
            )
            audio = None
            if use_audio:
                audio, audio_manifest, _ = extract_audio(path, self.cache / "audio")
                if audio_manifest["identity"]["source_sha256"] != manifest["identity"]["source_sha256"]:
                    raise ValueError("Frame/audio source mismatch")
                if abs(audio_manifest["timeline_origin_seconds"] - manifest["video"]["timeline_origin_seconds"]) > 1e-4:
                    raise ValueError("Frame/audio clock mismatch")
            current = path.stat()
            if (current.st_size, current.st_mtime_ns, current.st_ino) != source_stat:
                raise ValueError("Video changed during preparation")
            return directory, manifest, audio


def prediction(doc, output):
    """Preserve structured status when available; never score a known API failure."""
    if isinstance(output, str):
        try:
            decoded = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            decoded = None
        if isinstance(decoded, dict) and set(decoded) == {"api_response"}:
            response = decoded["api_response"]
            body = response["body"]
            if response["status_code"] != 200:
                error = json.dumps(body.get("error", body))
                output = {
                    "raw_output": "",
                    "error": error,
                    "error_kind": "content_policy" if is_content_policy_error(error) else "transport",
                }
            else:
                choice = body["choices"][0]
                message = choice["message"]
                raw = message.get("content") or ""
                if not isinstance(raw, str):
                    raise ValueError("Non-text completion content")
                final = raw
                if "</think>" in raw:
                    final = raw.partition("</think>")[2].strip()
                elif raw.lstrip().startswith("<think>"):
                    final = ""
                output = {
                    "raw_output": final,
                    "error": None,
                    "response_meta": {
                        "finish_reason": choice.get("finish_reason"),
                        "response_id": body.get("id"),
                        "returned_model": body.get("model"),
                        "usage": body.get("usage"),
                        "reasoning_content": message.get("reasoning_content"),
                        "raw_generation": message.get("raw_generation", raw),
                    },
                }
                if message.get("refusal"):
                    output["validation_error"] = "content_policy_refusal"
    if isinstance(output, dict):
        row = dict(output)
    elif isinstance(output, str):
        if output.startswith(("Failed to obtain answer", "Error:", "[ERROR]", "[LMMS_EVAL_REQUEST_FAILED", "[LMMS_EVAL_BUDGET_EXCEEDED]")):
            row = {"raw_output": "", "error": output, "error_kind": "transport"}
        else:
            row = {"raw_output": output, "error": None}
    else:
        raise TypeError("Prediction must be a string or structured final-state record")
    return {**row, "question_id": doc["question_id"], "setting": doc["setting"]}


def evaluate(docs, outputs, settings=SETTINGS):
    """Join paired settings by ID, independent of arrival/rank ordering."""
    if len(docs) != len(outputs):
        raise ValueError("Document/prediction count mismatch")
    if len({d["protocol_id"] for d in docs}) != 1:
        raise ValueError("Mixed protocol fingerprints")
    questions = {}
    for doc in docs:
        q = json.loads(doc["question_json"])
        if q["question_id"] in questions and questions[q["question_id"]] != q:
            raise ValueError("Conflicting ground truth for paired settings")
        questions[q["question_id"]] = q
    report = evaluate_public_predictions(
        list(questions.values()),
        [prediction(d, out) for d, out in zip(docs, outputs)],
        settings,
    )
    # Framework tables must not present an incomplete transport run as final scores.
    if any(not item["complete"] for item in report["tracks"].values()):
        raise ValueError("Incomplete transport results; retry before publishing metrics")
    record(docs, report)
    return report


def viewer_question(row):
    """Losslessly restore a canonical question from an existing HF Viewer row."""
    question = {
        "question_id": row["question_id"],
        "video_id": row["video_id"],
        "track": row["track"],
        "task": row["task"],
        "question": {"text": row["question_text"], "text_image": row["question_text_image"]},
        "reference_image": row["reference_image_path"],
        "ground_truth": json.loads(row["ground_truth"]),
    }
    if row.get("given_interval") is not None:
        question["given_interval"] = row["given_interval"]
    if row.get("options") is not None:
        question["options"] = json.loads(row["options"])
    if row.get("qa_type") is not None:
        question["qa_type"] = row["qa_type"]
    return question
