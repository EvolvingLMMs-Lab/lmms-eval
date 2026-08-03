"""Physical-conflict video benchmark helpers.

The task follows VSTAT's local-data ``ConfigurableTask`` pattern while adding
the answer types required by the Numaira benchmark: binary, floating-point
numeric, single-choice, and multi-select questions.

The public QA JSONL is the source of temporal questions. A strict full run
requires an evaluation sidecar supplying canonical question types, option
mappings, and tolerances. An optional split JSONL supplies explicit video-level
conflict labels for the binary question. Empty ``annotation.events`` is
deliberately never interpreted as a negative label.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from loguru import logger as eval_logger

from lmms_eval.api.task import ConfigurableTask
from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

_CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DEFAULT_NUMERIC_TOLERANCE = 0.5
_LINEAGE_BASE_SHA256 = {
    "official": "c03c15148084754865cc472efdcd1a7617e985988f0bd48d39a7b43bb3bc6ff9",
    "human_reviewed": "55eab72049367d3670fef0651c2c0b340465e88a7e3361bd42166f4bf7da363c",
}
_REQUIRED_QUESTION_CATEGORIES = {
    "physical_conflict_existence",
    "first_conflict_start",
    "first_conflict_duration",
    "conflict_quarters",
    "max_conflict_quarter",
    "nonoverlap_total_duration",
}
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")
_OPTION_PATTERN = re.compile(
    r"(?:^|\n)\s*([A-Z])[.)：:]\s*(.*?)(?=(?:\n\s*[A-Z][.)：:]\s*)|\Z)",
    flags=re.DOTALL,
)

_annotation_root: Path | None = None

_SETUP_HINT = (
    "Set PHYSICAL_CONFLICT_QA_PATH to the QA JSON/JSONL (or provide "
    "dataset_kwargs.data_files), PHYSICAL_CONFLICT_VIDEO_ROOT to the video "
    "directory, PHYSICAL_CONFLICT_SIDECAR_PATH to the evaluation "
    "sidecar, and PHYSICAL_CONFLICT_SPLIT_PATH to the optional split JSONL "
    "containing explicit conflict/non-conflict labels. Full evaluation also "
    "requires a regenerated Task C bundle aligned to the configured Task B lineage."
)

_TYPE_ALIASES = {
    "binary": "binary",
    "boolean": "binary",
    "bool": "binary",
    "yes_no": "binary",
    "yesno": "binary",
    "existence": "binary",
    "presence": "binary",
    "numeric": "numeric",
    "number": "numeric",
    "float": "numeric",
    "integer": "numeric",
    "single_select": "mcq",
    "single_choice": "mcq",
    "multiple_choice": "mcq",
    "mcq": "mcq",
    "multi_select": "multi_select",
    "multiple_select": "multi_select",
    "multiple_selection": "multi_select",
    "multi_choice": "multi_select",
}


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _configured_path(
    env_name: str,
    dataset_kwargs: dict[str, Any],
    *config_keys: str,
) -> Path | None:
    override = os.environ.get(env_name)
    if override:
        return _resolve_path(override)

    for key in config_keys:
        value = dataset_kwargs.get(key)
        if value:
            return _resolve_path(value)
    return None


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def _lineage_name(dataset_kwargs: dict[str, Any]) -> str:
    value = os.environ.get("PHYSICAL_CONFLICT_LINEAGE") or dataset_kwargs.get("lineage") or "official"
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")
    aliases = {
        "official": "official",
        "official_canonical": "official",
        "human": "human_reviewed",
        "human_reviewed": "human_reviewed",
        "human_reviewed_v3": "human_reviewed",
    }
    if normalized not in aliases:
        choices = ", ".join(sorted(_LINEAGE_BASE_SHA256))
        raise ValueError(f"PHYSICAL_CONFLICT_LINEAGE must select one of: {choices}.")
    return aliases[normalized]


def _canonical_base_sha256(samples: Iterable[dict[str, Any]]) -> str:
    """Hash the Task B source after replacing Task C's only mutable field."""

    digest = hashlib.sha256()
    for sample in samples:
        canonical = dict(sample)
        canonical["qa_pairs"] = []
        line = json.dumps(canonical, ensure_ascii=False, separators=(",", ":")) + "\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _validate_lineage(
    samples: list[dict[str, Any]],
    *,
    lineage: str,
    expected_sha256: str,
    expected_sample_count: int,
) -> str:
    if len(samples) != expected_sample_count:
        raise ValueError(f"Task C QA contains {len(samples)} samples; the {lineage} Task B lineage " f"requires {expected_sample_count}.")

    sample_ids = [_sample_id(sample) for sample in samples]
    video_paths = [str(sample.get("video_path", "")) for sample in samples]
    if any(not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Task C QA sample IDs must be non-empty and unique.")
    if any(not value for value in video_paths) or len(set(video_paths)) != len(video_paths):
        raise ValueError("Task C QA video paths must be non-empty and unique.")

    actual_sha256 = _canonical_base_sha256(samples)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Task C QA is not aligned to the configured {lineage} Task B lineage: " f"expected base SHA-256 {expected_sha256}, got {actual_sha256}. " "Regenerate the QA main file and sidecar from the selected current canonical."
        )
    return actual_sha256


def _qa_path_from_config(dataset_kwargs: dict[str, Any]) -> Path | None:
    path = _configured_path(
        "PHYSICAL_CONFLICT_QA_PATH",
        dataset_kwargs,
        "qa_file",
        "qa_path",
    )
    if path is not None:
        return path

    data_files = dataset_kwargs.get("data_files")
    if isinstance(data_files, dict):
        value = data_files.get("test") or data_files.get("qa")
        if value is None and data_files:
            value = next(iter(data_files.values()))
        return _resolve_path(value) if value else None
    return _resolve_path(data_files) if data_files else None


def _records_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise TypeError("Physical-conflict JSON must contain objects or a list of objects.")

    for key in ("data", "items", "samples", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    if payload and all(isinstance(value, dict) for value in payload.values()):
        return [dict(value) for value in payload.values()]
    return [payload]


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing physical-conflict annotation file: {path}\n{_SETUP_HINT}")

    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number} must contain a JSON object.")
                records.append(value)
        return records

    with path.open("r", encoding="utf-8-sig") as handle:
        return _records_from_json_payload(json.load(handle))


def _first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _sample_id(record: dict[str, Any]) -> str:
    return str(_first(record, "sample_id", "video_id", "id", default=""))


def _qa_pairs(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("qa_pairs", [])
    if value in (None, []):
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ValueError(f"Sample {_sample_id(record)!r} has invalid qa_pairs; expected an object or list.")


def _sidecar_indexes(
    records: Iterable[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
]:
    by_item: dict[str, dict[str, Any]] = {}
    by_full_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    by_qa_key: dict[tuple[str, int], dict[str, Any]] = {}

    for item in records:
        item_id = str(item.get("item_id", ""))
        qa_id = str(item.get("qa_id", ""))
        sample_id = _sample_id(item)
        try:
            question_index = int(_first(item, "question_index", "index", default=0))
        except (TypeError, ValueError):
            question_index = 0
        if not item_id and not qa_id:
            raise ValueError("Every sidecar item must contain item_id or qa_id.")
        if item_id in by_item:
            raise ValueError(f"Duplicate sidecar item_id: {item_id!r}.")
        if qa_id and (qa_id, question_index) in by_qa_key:
            raise ValueError(f"Duplicate sidecar qa_id/question_index: {qa_id!r}/{question_index}.")
        if sample_id and qa_id and (sample_id, qa_id, question_index) in by_full_key:
            raise ValueError(f"Duplicate sidecar sample_id/qa_id/question_index: " f"{sample_id!r}/{qa_id!r}/{question_index}.")
        if item_id:
            by_item[item_id] = item
        if qa_id:
            by_qa_key[(qa_id, question_index)] = item
            if sample_id:
                by_full_key[(sample_id, qa_id, question_index)] = item
    return by_item, by_full_key, by_qa_key


def _find_sidecar_item(
    indexes: tuple[
        dict[str, dict[str, Any]],
        dict[tuple[str, str, int], dict[str, Any]],
        dict[tuple[str, int], dict[str, Any]],
    ],
    sample_id: str,
    qa_id: str,
    question_index: int,
) -> dict[str, Any] | None:
    by_item, by_full_key, by_qa_key = indexes
    generated_item_id = f"{qa_id}:{question_index}"
    matches = [
        candidate
        for candidate in (
            by_item.get(generated_item_id),
            by_full_key.get((sample_id, qa_id, question_index)),
            by_qa_key.get((qa_id, question_index)),
        )
        if candidate is not None
    ]
    unique_matches = {id(candidate): candidate for candidate in matches}
    if len(unique_matches) > 1:
        raise ValueError(f"Ambiguous sidecar join for {generated_item_id!r}.")
    return next(iter(unique_matches.values()), None)


def _main_qa_identities(
    samples: Iterable[dict[str, Any]],
) -> tuple[
    dict[str, tuple[str, str, int]],
    dict[tuple[str, str, int], tuple[str, str, int]],
    dict[tuple[str, int], tuple[str, str, int]],
]:
    by_item: dict[str, tuple[str, str, int]] = {}
    by_full_key: dict[tuple[str, str, int], tuple[str, str, int]] = {}
    by_qa_key: dict[tuple[str, int], tuple[str, str, int]] = {}

    for sample in samples:
        sample_id = _sample_id(sample)
        for pair_index, pair in enumerate(_qa_pairs(sample)):
            qa_id = str(pair.get("qa_id") or f"{sample_id}_qa_{pair_index + 1:03d}")
            questions = pair.get("questions")
            if isinstance(questions, str):
                questions = [questions]
            if not isinstance(questions, list):
                raise TypeError(f"QA pair {qa_id!r} must contain a questions list.")
            for question_index in range(len(questions)):
                identity = (sample_id, qa_id, question_index)
                item_id = f"{qa_id}:{question_index}"
                if item_id in by_item or (qa_id, question_index) in by_qa_key:
                    raise ValueError(f"Duplicate main QA identity: {item_id!r}.")
                by_item[item_id] = identity
                by_full_key[identity] = identity
                by_qa_key[(qa_id, question_index)] = identity
    return by_item, by_full_key, by_qa_key


def _validate_sidecar_identities(
    sidecar_records: Iterable[dict[str, Any]],
    samples: Iterable[dict[str, Any]],
) -> None:
    expected_by_item, expected_by_full_key, expected_by_qa_key = _main_qa_identities(samples)

    for item in sidecar_records:
        item_id = str(item.get("item_id", ""))
        qa_id = str(item.get("qa_id", ""))
        sample_id = _sample_id(item)
        has_explicit_index = "question_index" in item or "index" in item
        try:
            question_index = int(_first(item, "question_index", "index", default=0))
        except (TypeError, ValueError):
            question_index = 0

        candidates = {
            identity
            for identity in (
                expected_by_item.get(item_id),
                expected_by_full_key.get((sample_id, qa_id, question_index)),
                expected_by_qa_key.get((qa_id, question_index)),
            )
            if identity is not None
        }
        if not candidates:
            label = item_id or f"{qa_id}:{question_index}"
            raise ValueError(f"Unmatched sidecar item {label!r}; no corresponding main QA item exists.")
        if len(candidates) > 1:
            label = item_id or f"{qa_id}:{question_index}"
            raise ValueError(f"Sidecar item {label!r} identifies multiple main QA items.")

        expected_sample_id, expected_qa_id, expected_index = next(iter(candidates))
        expected_item_id = f"{expected_qa_id}:{expected_index}"
        if item_id and item_id != expected_item_id:
            raise ValueError(f"Sidecar item_id {item_id!r} does not match {expected_item_id!r}.")
        if qa_id and qa_id != expected_qa_id:
            raise ValueError(f"Sidecar qa_id {qa_id!r} does not match {expected_qa_id!r}.")
        if sample_id and sample_id != expected_sample_id:
            raise ValueError(f"Sidecar sample_id {sample_id!r} does not match {expected_sample_id!r}.")
        if has_explicit_index and question_index != expected_index:
            raise ValueError(f"Sidecar question_index {question_index} does not match {expected_index} " f"for {expected_qa_id!r}.")


def _extract_choices(value: Any, question: str = "") -> list[str]:
    if isinstance(value, dict):
        ordered_keys = sorted(
            value,
            key=lambda key: _CHOICE_LETTERS.index(str(key).upper()) if str(key).upper() in _CHOICE_LETTERS else len(_CHOICE_LETTERS),
        )
        return [str(value[key]).strip() for key in ordered_keys]

    if isinstance(value, list):
        choices: list[str] = []
        for item in value:
            if isinstance(item, dict):
                choices.append(str(_first(item, "text", "value", "option", "label", default="")).strip())
            else:
                choices.append(str(item).strip())
        return choices

    matches = _OPTION_PATTERN.findall(question)
    return [text.strip() for _, text in matches]


def _choice_label(value: Any, choices: list[str]) -> str:
    text = str(value).strip()
    upper = text.upper()
    if len(upper) == 1 and upper in _CHOICE_LETTERS:
        return upper
    for index, choice in enumerate(choices):
        if text.casefold() == choice.casefold():
            return _CHOICE_LETTERS[index]
    return upper


def _normalize_binary_value(value: Any) -> str | None:
    if isinstance(value, dict):
        value = _first(value, "name", "label", "value", "id")
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)) and value in (0, 1):
        return "yes" if int(value) == 1 else "no"
    if value is None:
        return None

    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
    negative = {"0", "false", "no", "negative", "non_conflict", "nonconflict", "non_fight", "nonfight", "normal"}
    positive = {"1", "true", "yes", "positive", "conflict", "physical_conflict", "fight", "fighting"}
    if normalized in negative:
        return "no"
    if normalized in positive:
        return "yes"
    return None


def _type_from_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
    if normalized in _TYPE_ALIASES:
        return _TYPE_ALIASES[normalized]
    if "quarter" in normalized:
        if any(token in normalized for token in ("which", "present", "presence", "contain", "overlap", "multi")):
            return "multi_select"
        return "mcq"
    if any(token in normalized for token in ("exist", "presence", "whether", "yes_no")):
        return "binary"
    if any(token in normalized for token in ("start", "duration", "time", "total", "numeric")):
        return "numeric"
    return None


def _infer_answer_type(
    question: str,
    target: Any,
    choices: list[str],
    explicit_types: Iterable[Any],
) -> str:
    for value in explicit_types:
        normalized = _type_from_text(value)
        if normalized:
            return normalized

    question_lower = question.casefold()
    if any(phrase in question_lower for phrase in ("是否存在", "有无冲突", "any physical conflict")):
        return "binary"
    if "quarter" in question_lower:
        if any(token in question_lower for token in ("most", "maximum", "longest", "哪个")):
            return "mcq"
        if any(token in question_lower for token in ("which quarters", "哪些", "all that")):
            return "multi_select"
    if isinstance(target, list):
        return "multi_select"
    if isinstance(target, (int, float)) and not isinstance(target, bool):
        return "numeric"
    if _normalize_binary_value(target) is not None:
        return "binary"
    compact_target = re.sub(r"[^A-Za-z]", "", str(target)).upper()
    if len(compact_target) > 1 and set(compact_target) <= set(_CHOICE_LETTERS[: max(4, len(choices))]):
        return "multi_select"
    if choices:
        return "mcq"
    try:
        float(str(target).replace(",", ""))
        return "numeric"
    except ValueError:
        return "mcq"


def _question_category(question: str, explicit_type: Any = None) -> str:
    explicit = re.sub(r"[^a-z0-9]+", "_", str(explicit_type or "").casefold()).strip("_")
    question_text = str(question).casefold()
    combined = f"{explicit} {question_text}"

    if "quarter" in combined:
        if any(token in combined for token in ("max", "most", "longest", "最多", "哪个")):
            return "max_conflict_quarter"
        return "conflict_quarters"
    if any(token in combined for token in ("nonoverlap", "non_overlap", "non-overlap", "union_duration", "总时长")):
        return "nonoverlap_total_duration"
    if "total" in combined and "duration" in combined:
        return "nonoverlap_total_duration"
    if any(token in combined for token in ("first", "第一段")) and any(token in combined for token in ("start", "begin", "何时开始")):
        return "first_conflict_start"
    if any(token in combined for token in ("first", "第一段")) and any(token in combined for token in ("duration", "how long", "last", "持续多久")):
        return "first_conflict_duration"
    if any(token in combined for token in ("exist", "whether", "any physical conflict", "是否存在", "有无冲突")):
        return "physical_conflict_existence"
    return "unknown"


def _numeric_target(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("Boolean values cannot be used as numeric temporal answers.")
    if isinstance(value, (int, float)):
        target = float(value)
        if math.isfinite(target):
            return target
        raise ValueError(f"Numeric target must be finite, got {value!r}.")
    matches = _NUMBER_PATTERN.findall(str(value))
    if not matches:
        raise ValueError(f"Cannot parse numeric target from {value!r}.")
    target = float(matches[-1].replace(",", ""))
    if not math.isfinite(target):
        raise ValueError(f"Numeric target must be finite, got {value!r}.")
    return target


def _multi_select_target(value: Any, choices: list[str]) -> str:
    allowed = set(_CHOICE_LETTERS[: max(4, len(choices))])
    if isinstance(value, list):
        values = value
    else:
        text = str(value).strip()
        compact = re.sub(r"[^A-Za-z]", "", text).upper()
        if compact and set(compact) <= allowed:
            values = list(compact)
        else:
            values = re.findall(r"\b([A-Z])\b", text.upper())
            if not values:
                values = [text]
    labels: set[str] = set()
    for item in values:
        label = _choice_label(item, choices)
        if len(label) > 1 and set(label) <= allowed:
            labels.update(label)
        elif len(label) == 1 and label in allowed:
            labels.add(label)
    if not labels:
        raise ValueError(f"Cannot parse multi-select target from {value!r}.")
    return "".join(sorted(labels, key=_CHOICE_LETTERS.index))


def _normalize_target(
    target: Any,
    *,
    answer_type: str,
    choices: list[str],
    item_id: str,
) -> tuple[str, float | None]:
    if answer_type == "numeric":
        target_value = _numeric_target(target)
        return format(target_value, ".15g"), target_value

    if answer_type == "binary":
        binary_target = _normalize_binary_value(target)
        if binary_target is None and choices:
            label = _choice_label(target, choices)
            if label in _CHOICE_LETTERS[: len(choices)]:
                binary_target = _normalize_binary_value(choices[_CHOICE_LETTERS.index(label)])
        if binary_target is None:
            raise ValueError(f"Cannot parse binary target from {target!r} for {item_id}.")
        return binary_target, None

    if answer_type == "multi_select":
        return _multi_select_target(target, choices), None

    return _choice_label(target, choices), None


def _normalize_atomic_row(
    *,
    sample: dict[str, Any],
    question: str,
    target: Any,
    qa_id: str,
    question_index: int,
    metadata: dict[str, Any],
    default_tolerance: float,
) -> dict[str, Any]:
    sample_id = _sample_id(sample) or _sample_id(metadata)
    item_id = str(metadata.get("item_id") or f"{qa_id}:{question_index}")
    source_question = str(question).strip()
    sidecar_question = metadata.get("question")
    if sidecar_question is not None and str(sidecar_question).strip() != source_question:
        raise ValueError(f"Main QA/sidecar question mismatch for {item_id!r}: " f"{source_question!r} != {str(sidecar_question).strip()!r}.")
    question = str(sidecar_question or source_question).strip()
    metadata_target = _first(metadata, "canonical_answer", "correct_answer", "ground_truth", "target", "answer")

    raw_choices = _first(metadata, "option_mapping", "options", "choices")
    choices = _extract_choices(raw_choices, question)
    answer_type = _infer_answer_type(
        question,
        target,
        choices,
        (
            metadata.get("answer_type"),
            metadata.get("question_type"),
            metadata.get("task_type"),
        ),
    )
    if answer_type not in {"numeric", "binary", "multi_select"}:
        answer_type = "mcq"

    answer_text, target_value = _normalize_target(
        target,
        answer_type=answer_type,
        choices=choices,
        item_id=item_id,
    )
    if metadata_target is not None:
        sidecar_answer, sidecar_target_value = _normalize_target(
            metadata_target,
            answer_type=answer_type,
            choices=choices,
            item_id=item_id,
        )
        if sidecar_answer != answer_text or sidecar_target_value != target_value:
            raise ValueError(f"Main QA/sidecar answer mismatch for {item_id!r}: " f"{answer_text!r} != {sidecar_answer!r}.")

    tolerance = _first(metadata, "tolerance_sec", "tolerance", "absolute_tolerance", default=default_tolerance)
    try:
        tolerance_value = float(tolerance)
    except (TypeError, ValueError):
        tolerance_value = default_tolerance

    question_type = str(_first(metadata, "question_type", "task_type", default=answer_type))
    video_time = _first(sample, "video_time_sec", default=_first(metadata, "video_time_sec"))
    try:
        video_time_sec = float(video_time) if video_time is not None else None
    except (TypeError, ValueError):
        video_time_sec = None

    return {
        "item_id": item_id,
        "sample_id": sample_id,
        "video_path": str(_first(sample, "video_path", default=_first(metadata, "video_path", default=""))),
        "question": question,
        "answer_type": answer_type,
        "answer_text": answer_text,
        "target_value": target_value,
        "choices": choices,
        "tolerance": tolerance_value,
        "question_type": question_type,
        "question_category": _question_category(question, question_type),
        "video_time_sec": video_time_sec,
    }


def _explicit_label(record: dict[str, Any]) -> str | None:
    for key in ("label", "conflict_label", "physical_conflict", "has_physical_conflict"):
        if key in record:
            normalized = _normalize_binary_value(record[key])
            if normalized is not None:
                return normalized
    return None


def _atomic_rows(
    samples: list[dict[str, Any]],
    split_records: list[dict[str, Any]],
    sidecar_records: list[dict[str, Any]],
    default_tolerance: float,
    *,
    require_sidecar: bool = False,
) -> list[dict[str, Any]]:
    all_samples = list(samples)
    indexes = _sidecar_indexes(sidecar_records)
    if require_sidecar:
        _validate_sidecar_identities(sidecar_records, all_samples)

    split_by_id = {_sample_id(record): record for record in split_records if _sample_id(record)}
    if split_by_id:
        samples = [sample for sample in samples if _sample_id(sample) in split_by_id]

    sample_by_id = {_sample_id(sample): sample for sample in samples if _sample_id(sample)}
    for sample_id, split_record in split_by_id.items():
        if sample_id not in sample_by_id:
            samples.append(split_record)
            sample_by_id[sample_id] = split_record

    rows: list[dict[str, Any]] = []
    binary_samples: set[str] = set()

    for sample in samples:
        sample_id = _sample_id(sample)
        for pair_index, pair in enumerate(_qa_pairs(sample)):
            qa_id = str(pair.get("qa_id") or f"{sample_id}_qa_{pair_index + 1:03d}")
            questions = pair.get("questions")
            answers = pair.get("answer", pair.get("answers"))
            if isinstance(questions, str):
                questions = [questions]
            if not isinstance(questions, list):
                raise TypeError(f"QA pair {qa_id!r} must contain a questions list.")
            if not isinstance(answers, list):
                answers = [answers]
            if len(questions) != len(answers):
                raise ValueError(f"QA pair {qa_id!r} has {len(questions)} questions but {len(answers)} answers.")

            for question_index, (question, target) in enumerate(zip(questions, answers)):
                expected_item_id = f"{qa_id}:{question_index}"
                sidecar_item = _find_sidecar_item(indexes, sample_id, qa_id, question_index)
                if sidecar_item is None and require_sidecar:
                    raise ValueError(f"Missing sidecar item for main QA item {expected_item_id!r}.")
                if sidecar_item is not None:
                    sidecar_item_id = str(sidecar_item.get("item_id", ""))
                    sidecar_qa_id = str(sidecar_item.get("qa_id", ""))
                    sidecar_sample_id = _sample_id(sidecar_item)
                    try:
                        sidecar_index = int(_first(sidecar_item, "question_index", "index", default=0))
                    except (TypeError, ValueError):
                        sidecar_index = 0
                    if sidecar_item_id and sidecar_item_id != expected_item_id:
                        raise ValueError(f"Sidecar item_id {sidecar_item_id!r} does not match main QA item " f"{expected_item_id!r}.")
                    if sidecar_qa_id and sidecar_qa_id != qa_id:
                        raise ValueError(f"Sidecar qa_id {sidecar_qa_id!r} does not match {qa_id!r}.")
                    if sidecar_sample_id and sidecar_sample_id != sample_id:
                        raise ValueError(f"Sidecar sample_id {sidecar_sample_id!r} does not match {sample_id!r}.")
                    if ("question_index" in sidecar_item or "index" in sidecar_item) and sidecar_index != question_index:
                        raise ValueError(f"Sidecar question_index {sidecar_index} does not match {question_index} " f"for {qa_id!r}.")

                metadata = {key: value for key, value in pair.items() if key not in {"questions", "answer", "answers"}}
                if sidecar_item is not None:
                    metadata.update(sidecar_item)
                row = _normalize_atomic_row(
                    sample=sample,
                    question=str(question),
                    target=target,
                    qa_id=qa_id,
                    question_index=question_index,
                    metadata=metadata,
                    default_tolerance=default_tolerance,
                )
                rows.append(row)
                if row["answer_type"] == "binary":
                    binary_samples.add(sample_id)

    for sample in samples:
        sample_id = _sample_id(sample)
        split_record = split_by_id.get(sample_id, {})
        label = _explicit_label(split_record) or _explicit_label(sample)
        if label is None or sample_id in binary_samples:
            continue
        merged_sample = dict(sample)
        for key, value in split_record.items():
            merged_sample.setdefault(key, value)
        rows.append(
            _normalize_atomic_row(
                sample=merged_sample,
                question="Does this video contain any physical conflict?",
                target=label,
                qa_id=f"{sample_id}_physical_conflict_exists",
                question_index=0,
                metadata={"question_type": "physical_conflict_existence", "answer_type": "binary"},
                default_tolerance=default_tolerance,
            )
        )

    return rows


def _validate_required_categories(
    rows: Iterable[dict[str, Any]],
    required_categories: Iterable[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row.get("question_category", "unknown"))
        counts[category] = counts.get(category, 0) + 1

    required = {str(category) for category in required_categories}
    missing = sorted(required - counts.keys())
    if missing:
        present = ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"
        raise ValueError(
            "The configured full physical_conflict task is incomplete. Missing required "
            f"question categories: {', '.join(missing)}. Present: {present}. "
            "Use a regenerated six-category Task C bundle, or set "
            "PHYSICAL_CONFLICT_ALLOW_PARTIAL=1 only for local smoke tests."
        )
    return counts


class PhysicalConflictTask(ConfigurableTask):
    """Load Numaira physical-conflict QA and flatten it to atomic questions."""

    def __init__(self, *args, config: dict[str, Any] | None = None, **kwargs) -> None:
        if config is not None:
            config = dict(config)
            config.pop("class", None)
        super().__init__(*args, config=config, **kwargs)

    def download(self, dataset_kwargs: dict[str, Any] | None = None) -> None:
        global _annotation_root

        kwargs = dict(dataset_kwargs or {})
        qa_path = _qa_path_from_config(kwargs)
        split_path = _configured_path(
            "PHYSICAL_CONFLICT_SPLIT_PATH",
            kwargs,
            "split_file",
            "label_file",
        )
        sidecar_path = _configured_path(
            "PHYSICAL_CONFLICT_SIDECAR_PATH",
            kwargs,
            "sidecar_file",
            "evaluation_file",
        )
        allow_partial = _as_bool(os.environ.get("PHYSICAL_CONFLICT_ALLOW_PARTIAL"), default=False)
        strict = _as_bool(kwargs.get("strict"), default=True) and not allow_partial
        require_sidecar = _as_bool(kwargs.get("require_sidecar"), default=strict) and not allow_partial

        if qa_path is None and split_path is None:
            raise FileNotFoundError(f"No physical-conflict QA or split file was configured.\n{_SETUP_HINT}")
        if strict and qa_path is None:
            raise FileNotFoundError(f"Full physical_conflict evaluation requires a QA main file.\n{_SETUP_HINT}")
        if require_sidecar and sidecar_path is None:
            raise FileNotFoundError(f"Full physical_conflict evaluation requires a QA sidecar file.\n{_SETUP_HINT}")

        samples = _read_records(qa_path) if qa_path is not None else []
        split_records = _read_records(split_path) if split_path is not None else []
        sidecar_records = _read_records(sidecar_path) if sidecar_path is not None else []
        lineage = _lineage_name(kwargs)
        lineage_sha256: str | None = None
        if strict:
            expected_sha256 = str(os.environ.get("PHYSICAL_CONFLICT_CANONICAL_SHA256") or kwargs.get("canonical_base_sha256") or _LINEAGE_BASE_SHA256[lineage])
            expected_sample_count = int(kwargs.get("expected_sample_count", 1000))
            lineage_sha256 = _validate_lineage(
                samples,
                lineage=lineage,
                expected_sha256=expected_sha256,
                expected_sample_count=expected_sample_count,
            )

        tolerance = float(kwargs.get("numeric_tolerance", _DEFAULT_NUMERIC_TOLERANCE))
        rows = _atomic_rows(
            samples,
            split_records,
            sidecar_records,
            tolerance,
            require_sidecar=require_sidecar,
        )
        if not rows:
            raise ValueError("The configured files produced no evaluation questions. The main QA file must contain " "qa_pairs, or the split file must contain explicit labels.")
        category_counts: dict[str, int] = {}
        if strict:
            required_categories = kwargs.get("required_question_categories") or _REQUIRED_QUESTION_CATEGORIES
            category_counts = _validate_required_categories(rows, required_categories)

        _annotation_root = qa_path.parent if qa_path is not None else split_path.parent
        split = self.config.test_split
        self.dataset = DatasetDict({split: Dataset.from_list(rows)})
        if self.config.process_docs is not None:
            self.dataset[split] = self.config.process_docs(self.dataset[split])
        self.dataset_no_image = self.dataset.copy()
        eval_logger.info(
            f"Loaded physical-conflict annotations ({len(rows)} atomic questions; "
            f"lineage={lineage}, lineage_sha256={lineage_sha256}, strict={strict}, "
            f"categories={category_counts}; QA={qa_path}, split={split_path}, "
            f"sidecar={sidecar_path})."
        )


def physical_conflict_process_docs(dataset: Dataset) -> Dataset:
    """Keep the hook explicit to mirror the VSTAT reference task."""

    return dataset


def _candidate_video_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("PHYSICAL_CONFLICT_VIDEO_ROOT", "PHYSICAL_CONFLICT_DATA_ROOT"):
        value = os.environ.get(env_name)
        if value:
            roots.append(_resolve_path(value))
    if _annotation_root is not None:
        roots.append(_annotation_root)
    roots.extend((Path.cwd(), Path(__file__).resolve().parents[3] / "data" / "physical_conflict"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def _resolve_video_path(video_path: str) -> Path:
    path = Path(video_path).expanduser()
    if path.is_absolute():
        return path
    for root in _candidate_video_roots():
        candidate = root / path
        if candidate.exists():
            return candidate
    roots = _candidate_video_roots()
    return (roots[0] if roots else Path.cwd()) / path


def physical_conflict_doc_to_visual(doc: dict[str, Any]) -> list[str]:
    path = _resolve_video_path(str(doc["video_path"]))
    if not path.exists():
        raise FileNotFoundError(f"Missing physical-conflict video: {path}\n{_SETUP_HINT}")
    return [str(path)]


def physical_conflict_doc_to_text(
    doc: dict[str, Any],
    lmms_eval_specific_kwargs: dict[str, Any] | None = None,
) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = str(kwargs.get("pre_prompt", ""))
    body = f"Watch the full video carefully before answering.\n\nQuestion: {doc['question']}"
    answer_type = str(doc["answer_type"])
    choices = list(doc.get("choices") or [])

    if answer_type == "numeric":
        post_prompt = kwargs.get("numeric_post_prompt", "Answer with only a single number in seconds.")
    elif answer_type == "multi_select":
        post_prompt = kwargs.get("multiselect_post_prompt", "Answer with only all applicable option letters.")
    elif answer_type == "binary" and not choices:
        post_prompt = kwargs.get("binary_post_prompt", "Answer with only Yes or No.")
    else:
        post_prompt = kwargs.get("mcq_post_prompt", "Answer with only the option letter.")
    return f"{pre_prompt}{body}\n\n{post_prompt}".strip()


def physical_conflict_doc_to_target(doc: dict[str, Any]) -> str:
    return str(doc["answer_text"])


def _extract_last_number(text: str) -> float | None:
    matches = _NUMBER_PATTERN.findall(str(text))
    if not matches:
        return None
    value = float(matches[-1].replace(",", ""))
    return value if math.isfinite(value) else None


def _numeric_parse_error_penalty(doc: dict[str, Any], target_value: float, tolerance: float) -> float:
    """Return a deterministic finite MAE penalty for an unparseable answer."""

    candidates = [1.0, abs(target_value), 2 * max(tolerance, 0.0)]
    try:
        video_time_sec = float(doc.get("video_time_sec"))
    except (TypeError, ValueError):
        video_time_sec = 0.0
    if math.isfinite(video_time_sec) and video_time_sec > 0:
        candidates.append(video_time_sec)
    return max(candidates)


def _parse_binary_prediction(prediction: str, choices: list[str]) -> str | None:
    if choices:
        allowed = list(_CHOICE_LETTERS[: len(choices)])
        label = extract_mcq_answer(prediction, choices=allowed)
        if label in allowed:
            normalized = _normalize_binary_value(choices[allowed.index(label)])
            if normalized is not None:
                return normalized
    direct = _normalize_binary_value(prediction)
    if direct is not None:
        return direct
    lowered = str(prediction).casefold()
    if re.search(r"\b(?:no|false)\b", lowered):
        return "no"
    if re.search(r"\b(?:yes|true)\b", lowered):
        return "yes"
    return None


def _parse_multi_select_prediction(prediction: str, num_choices: int) -> set[str]:
    allowed = set(_CHOICE_LETTERS[: max(4, num_choices)])
    upper = str(prediction).upper().strip()
    compact_matches = re.findall(r"(?<![A-Z])([A-Z]{1,8})(?![A-Z])", upper)
    for match in reversed(compact_matches):
        if len(match) > 1 and set(match) <= allowed:
            return set(match)
    return {match for match in re.findall(r"\b([A-Z])\b", upper) if match in allowed}


def _parse_mcq_prediction(prediction: str, choices: list[str]) -> str | None:
    allowed = list(_CHOICE_LETTERS[: max(4, len(choices))])
    return extract_mcq_answer(prediction, choices=allowed) or None


def physical_conflict_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction = str(results[0]).strip() if results else ""
    answer_type = str(doc["answer_type"])
    target = str(doc["answer_text"])
    choices = list(doc.get("choices") or [])

    if answer_type == "numeric":
        parsed = _extract_last_number(prediction)
        target_value = float(doc["target_value"])
        tolerance = float(doc.get("tolerance", _DEFAULT_NUMERIC_TOLERANCE))
        parse_error = float(parsed is None)
        absolute_error = abs(parsed - target_value) if parsed is not None else _numeric_parse_error_penalty(doc, target_value, tolerance)
        accuracy = float(absolute_error <= tolerance)
        return {
            "Numeric_Accuracy_at_0_5s": accuracy,
            "Numeric_MAE": absolute_error,
            "Numeric_Parse_Error_Rate": parse_error,
            "Overall_Accuracy": accuracy,
        }

    if answer_type == "binary":
        parsed = _parse_binary_prediction(prediction, choices)
        accuracy = float(parsed == target.casefold())
        return {"Binary_Accuracy": accuracy, "Overall_Accuracy": accuracy}

    if answer_type == "multi_select":
        predicted = _parse_multi_select_prediction(prediction, len(choices))
        expected = set(target)
        intersection = predicted & expected
        precision = len(intersection) / len(predicted) if predicted else 0.0
        recall = len(intersection) / len(expected) if expected else float(not predicted)
        exact = float(predicted == expected)
        return {
            "MultiSelect_Exact_Match": exact,
            "MultiSelect_Precision": precision,
            "MultiSelect_Recall": recall,
            "Overall_Accuracy": exact,
        }

    parsed = _parse_mcq_prediction(prediction, choices)
    accuracy = float(parsed is not None and parsed.upper() == target.upper())
    return {"MCQ_Accuracy": accuracy, "Overall_Accuracy": accuracy}


def physical_conflict_aggregate_mean(results: list[float]) -> float:
    return sum(float(result) for result in results) / len(results) if results else 0.0
