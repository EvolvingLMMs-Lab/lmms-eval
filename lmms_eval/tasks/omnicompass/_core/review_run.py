"""Attempt-scoped receipts for the review launcher; scores remain unchanged."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(value, file, sort_keys=True, indent=2)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def record(docs, report=None):
    directory = os.getenv("OMNICOMPASS_REPORT_DIR")
    if not directory:
        return
    run_id = os.environ["OMNICOMPASS_RUN_ID"]
    selection = sorted(
        [{"request_id": f"{d['question_id']}::{d['setting']}", "video": d["video"], "task": d["task"], "track": d["track"], "setting": d["setting"]} for d in docs],
        key=lambda row: row["request_id"],
    )
    protocols = {d["protocol_id"] for d in docs}
    if len(protocols) != 1:
        raise ValueError("Mixed protocol identities in attempt receipt")
    name = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()[:24]
    value = {"run_id": run_id, "protocol_id": protocols.pop(), "selection": selection}
    kind = "selection" if report is None else "scores"
    if report is not None:
        value["report"] = report
    write_json(Path(directory) / f"{kind}-{name}.json", value)


def validate_attempt(directory, run_id, returncode, video_ids=None, expected_count=None):
    """Accept only this attempt's successful process and exact selected request IDs."""
    if returncode != 0:
        raise ValueError(f"Evaluation process failed with exit code {returncode}")
    directory = Path(directory)
    selections = sorted(directory.glob("selection-*.json"))
    if not selections:
        raise ValueError("No current-attempt selection receipts")
    all_ids, actual_videos, protocols, coverage = set(), set(), set(), set()
    score_names = set()
    for path in selections:
        selection = json.loads(path.read_text())
        score_path = directory / path.name.replace("selection-", "scores-", 1)
        scores = json.loads(score_path.read_text())
        if any(item["run_id"] != run_id for item in (selection, scores)):
            raise ValueError("Receipt belongs to a different attempt")
        if any(selection[key] != scores[key] for key in ("protocol_id", "selection")):
            raise ValueError("Scored inputs differ from the current selection")
        protocols.add(selection["protocol_id"])
        expected = [row["request_id"] for row in selection["selection"]]
        observed = [row["request_id"] for row in scores["report"]["parsed_predictions"]]
        if len(set(expected)) != len(expected) or len(set(observed)) != len(observed) or set(expected) != set(observed):
            raise ValueError("Missing, duplicate or unexpected result IDs")
        if not scores["report"]["tracks"] or any(not track["complete"] for track in scores["report"]["tracks"].values()):
            raise ValueError("Incomplete evaluation")
        if all_ids.intersection(expected):
            raise ValueError("Duplicate requests across task receipts")
        all_ids.update(expected)
        for row in selection["selection"]:
            if row["track"] != "VL":
                raise ValueError("The native review launcher expects the VL track")
            actual_videos.add(row["video"])
            coverage.add((row["video"], row["task"], row["setting"]))
        score_names.add(score_path.name)
    if len(protocols) != 1:
        raise ValueError("Mixed protocol identities across tasks")
    if set(p.name for p in directory.glob("scores-*.json")) != score_names:
        raise ValueError("Unexpected score receipts")
    if video_ids is not None and actual_videos != set(video_ids):
        raise ValueError("Video selection differs from launch arguments")
    expected_coverage = {(video, task, setting) for video in actual_videos for task in ("TG", "SG", "QA") for setting in ("text", "text_image")}
    if coverage != expected_coverage or len(all_ids) != len(expected_coverage):
        raise ValueError("Incomplete task/setting coverage")
    if expected_count is not None and len(all_ids) != expected_count:
        raise ValueError("Request count differs from launch arguments")
    return {"passed": True, "run_id": run_id, "requests": len(all_ids), "videos": sorted(actual_videos), "protocol_id": protocols.pop()}
