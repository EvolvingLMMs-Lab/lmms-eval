"""Score CC-OCR V2 and reproduce the official scenario, track and leaderboard aggregation.

Source: https://github.com/Eioss/CC-OCR-V2/tree/58c6f20c464ad8471e49f5faab2b126c7e39556a
Scores are normalized. The published CSV prints 100 times each result with two decimal places.
"""

import re
from collections import defaultdict

from lmms_eval.tasks.cc_ocr_v2.evaluators import text

_TRACKS = ("recognition", "parsing", "grounding", "extraction", "qa")


def score_sample(doc, prediction):
    """Return the row score and sufficient statistics for aggregation across workers."""
    item = {key: doc[key] for key in ("source_key", "track", "sub_task", "scenario")}
    track = item["track"]
    reference = doc["answer"].replace("\r\n", "\n").replace("\r", "\n")
    # The authors write raw model text and read it back in universal-newline
    # mode. Normalize the scoring copy here, keeping the framework's response intact.
    prediction = prediction.replace("\r\n", "\n").replace("\r", "\n")
    if track == "recognition":
        item.update(text.recognition(doc, reference, prediction))
    elif track == "extraction":
        item.update(text.extraction(reference, prediction))
    elif track == "qa":
        item["row_score"] = text.qa(reference, prediction)
    elif track == "grounding":
        from lmms_eval.tasks.cc_ocr_v2.evaluators import grounding

        item["row_score"] = grounding.score(doc, text.strip_fence(reference), text.strip_fence(prediction))
    elif track == "parsing":
        from lmms_eval.tasks.cc_ocr_v2.evaluators import parsing

        item["row_score"] = parsing.score(item["scenario"], reference, prediction)
    else:
        raise ValueError(f"Unknown CC-OCR V2 track: {track}")
    return item


def scenario_score(items):
    """Score one third-level scenario, including rounding in the track-specific CLI."""
    if not items:
        return None
    track = items[0]["track"]
    if track == "extraction" and not any(item["valid_prediction"] for item in items):
        return None
    scored = [item for item in items if item["row_score"] is not None]
    if not scored:
        # The QA CLI prints 0.0 when the matched pairs have no valid answers.
        return 0.0 if track == "qa" else None
    if track in ("recognition", "extraction"):
        n_matched, n_pred, n_gt = (sum(row[key] for row in scored) for key in ("matched", "predicted", "expected"))
        if track == "recognition":
            return text.recognition_f1(n_matched, n_pred, n_gt)
        return float(f"{text.extraction_f1(n_matched, n_pred, n_gt):.4f}")
    mean = sum(row["row_score"] for row in scored) / len(scored)
    return round(mean, 4) if track == "parsing" else mean


def _shell_mean(values):
    # eval_all_tasks_one_model.sh accepts plain decimal output, prints each
    # cumulative sum with 12 decimal places and the final mean with six.
    total = 0.0
    count = 0
    for value in values:
        if value is not None and re.fullmatch(r"-?\d+(?:\.\d+)?", str(value)):
            total = float(f"{total + value:.12f}")
            count += 1
    return float(f"{total / count:.6f}") if count else None


def aggregate_track(items, track):
    """Compute the selected track using its official scenario weights."""
    by_scenario = defaultdict(list)
    for item in items:
        if item["track"] == track:
            by_scenario[item["sub_task"], item["scenario"]].append(item)
    if not by_scenario:
        return None
    scores = {}
    for key in sorted(by_scenario):
        rows = sorted(by_scenario[key], key=lambda item: item["source_key"])
        scores[key] = scenario_score(rows)
    if track != "grounding":
        result = _shell_mean(scores.values())
        if track == "recognition" and result is None:
            # summarize_results.py falls back to macro F1 when every micro
            # value used exponent notation and the shell discarded the key.
            return _shell_mean(sum(r["row_score"] for r in rows) / (len(rows) + 1e-9) for rows in by_scenario.values())
        return result

    # The wrapper reads the two separately printed mean_iou values. Each is a
    # macro-average of the scenarios within that subtask. The CLI prints zero
    # for a grounding subtask with no scored rows.
    means = []
    for sub_task in ("text_grounding", "object_grounding"):
        values = [s for (sub, _), s in scores.items() if sub == sub_task and s is not None]
        means.append(sum(values) / len(values) if values else 0.0)
    return _shell_mean(means)


def aggregate_results(items):
    """Normalized leaderboard average, defined only when all five tracks have scores."""
    tracks = [aggregate_track(items, track) for track in _TRACKS]
    if any(score is None for score in tracks):
        return None
    return (sum(score * 100 for score in tracks) / len(tracks)) / 100


def aggregate_recognition(items):
    return aggregate_track(items, "recognition")


def aggregate_parsing(items):
    return aggregate_track(items, "parsing")


def aggregate_grounding(items):
    return aggregate_track(items, "grounding")


def aggregate_extraction(items):
    return aggregate_track(items, "extraction")


def aggregate_qa(items):
    return aggregate_track(items, "qa")
