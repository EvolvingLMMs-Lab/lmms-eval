"""lmms-eval adapter for GoodVision Task C atomic v1.

The authoritative Task C main JSONL is self-contained.  This adapter validates
the frozen release before selecting the test split, exposes only the approved
model-input fields, and keeps answers and raw media paths in trusted host-only
indexes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from loguru import logger as eval_logger

from lmms_eval.api.task import ConfigurableTask
from lmms_eval.tasks.physical_conflict import runtime as _runtime

GENERATOR_VERSION = "task_c_atomic_v1"
MODEL_INPUT_FIELDS = (
    "qa_id",
    "sample_id",
    "question_type",
    "question",
    "answer_type",
    "options",
    "unit",
)
FORBIDDEN_MODEL_INPUT_FIELDS = frozenset(
    {
        "answer",
        "target_value",
        "tolerance",
        "evidence",
        "annotation",
        "video_path",
    }
)

QUESTION_ORDER = (
    "first_conflict_start_time",
    "first_conflict_duration",
    "conflict_quarter_coverage",
    "max_conflict_duration_quarter",
    "total_non_overlapping_conflict_duration",
    "conflict_presence",
)
QUESTION_TYPES = frozenset(QUESTION_ORDER)
NUMERIC_QUESTION_TYPES = frozenset(
    {
        "first_conflict_start_time",
        "first_conflict_duration",
        "total_non_overlapping_conflict_duration",
    }
)
TEMPORAL_QUESTION_TYPES = QUESTION_TYPES - {"conflict_presence"}
TEMPORAL_ELIGIBLE_DATASETS = frozenset({"ntu_cctv_fights"})
FROZEN_DATASETS = frozenset(
    {
        "ntu_cctv_fights",
        "rwf2000",
        "surveillance_camera_fight",
        "ubi_fights",
    }
)

EXPECTED_FULL_SAMPLE_COUNT = 1000
EXPECTED_FULL_QUESTION_COUNT = 2215
EXPECTED_FULL_TYPE_COUNTS = {
    "first_conflict_start_time": 250,
    "first_conflict_duration": 250,
    "conflict_quarter_coverage": 250,
    "max_conflict_duration_quarter": 215,
    "total_non_overlapping_conflict_duration": 250,
    "conflict_presence": 1000,
}
EXPECTED_TEST_SAMPLE_COUNT = 149
EXPECTED_TEST_QUESTION_COUNT = 334
EXPECTED_TEST_TYPE_COUNTS = {
    "first_conflict_start_time": 38,
    "first_conflict_duration": 38,
    "conflict_quarter_coverage": 38,
    "max_conflict_duration_quarter": 33,
    "total_non_overlapping_conflict_duration": 38,
    "conflict_presence": 149,
}

DEFAULT_MAIN_SHA256 = "2b61dc61229956181afec533fcd0e351ae3310bcee96fd0c6c2f35ece65c7879"
DEFAULT_SIDECAR_SHA256 = "5c93dab8eaa74a1bf1a1eb99fabc7eb296e0be381f8c198cae70d444ca066131"
DEFAULT_TEST_SPLIT_SHA256 = "46ebbaa4cbe39eaaed6990c6368951674d5038b9a544c1c01c40374f81da318b"

_OPTION_IDS = ("A", "B", "C", "D")
_QUESTION_ORDER_INDEX = {question_type: index for index, question_type in enumerate(QUESTION_ORDER)}
_LEGAL_RELEASE_SEQUENCES = frozenset(
    {
        ("conflict_presence",),
        (
            "first_conflict_start_time",
            "first_conflict_duration",
            "conflict_quarter_coverage",
            "total_non_overlapping_conflict_duration",
            "conflict_presence",
        ),
        QUESTION_ORDER,
    }
)

_SETUP_HINT = (
    "Set PHYSICAL_CONFLICT_QA_PATH to GoodVision's "
    "clean_samples_1000_human_reviewed_with_qa.jsonl, "
    "PHYSICAL_CONFLICT_SIDECAR_PATH to task_c_qa_evaluation_items_v1.jsonl, "
    "PHYSICAL_CONFLICT_SPLIT_PATH to splits/test.jsonl, and "
    "PHYSICAL_CONFLICT_VIDEO_ROOT to the frozen video directory."
)


class _Target:
    __slots__ = (
        "answer",
        "answer_type",
        "option_text",
        "question_type",
        "target_value",
        "tolerance",
    )

    def __init__(
        self,
        *,
        question_type: str,
        answer_type: str,
        option_text: dict[str, str],
        answer: frozenset[str],
        target_value: float | None,
        tolerance: float | None,
    ) -> None:
        self.question_type = question_type
        self.answer_type = answer_type
        self.option_text = option_text
        self.answer = answer
        self.target_value = target_value
        self.tolerance = tolerance


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _configured_path(env_name: str, dataset_kwargs: dict[str, Any], *config_keys: str) -> Path | None:
    override = os.environ.get(env_name)
    if override:
        return _resolve_path(override)
    for key in config_keys:
        value = dataset_kwargs.get(key)
        if value:
            return _resolve_path(value)
    return None


def _qa_path_from_config(dataset_kwargs: dict[str, Any]) -> Path | None:
    path = _configured_path("PHYSICAL_CONFLICT_QA_PATH", dataset_kwargs, "qa_file", "qa_path")
    if path is not None:
        return path
    data_files = dataset_kwargs.get("data_files")
    if isinstance(data_files, dict):
        value = data_files.get("test") or data_files.get("qa")
        if value is None and data_files:
            value = next(iter(data_files.values()))
        return _resolve_path(value) if value else None
    return _resolve_path(data_files) if data_files else None


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


def _read_jsonl(path: Path, *, artifact_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {artifact_name}: {path}\n{_SETUP_HINT}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{artifact_name} line {line_number} is invalid JSON: {error.msg}") from error
            if not isinstance(value, dict):
                raise TypeError(f"{artifact_name} line {line_number} must contain an object.")
            records.append(value)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_hash(path: Path, expected: str, *, artifact_name: str) -> str:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{artifact_name} SHA-256 mismatch: expected {expected}, got {observed}.")
    return observed


def _source_dataset(video_path: str, *, context: str) -> str:
    basename = video_path.replace("\\", "/").rsplit("/", 1)[-1]
    parts = basename.split("__", 2)
    if len(parts) != 3 or not parts[0].isdigit() or parts[1] not in FROZEN_DATASETS:
        raise ValueError(f"{context}.video_path does not contain a frozen dataset identifier.")
    return parts[1]


def _parse_options(raw_options: Any, *, question_type: str, context: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    if not isinstance(raw_options, list) or not raw_options:
        raise ValueError(f"{context}.options must be a nonempty array.")
    options: list[dict[str, str]] = []
    option_text: dict[str, str] = {}
    for index, option in enumerate(raw_options):
        if not isinstance(option, dict) or set(option) != {"id", "text"}:
            raise ValueError(f"{context}.options[{index}] must contain exactly id and text.")
        option_id = option["id"]
        text = option["text"]
        if not isinstance(option_id, str) or option_id not in _OPTION_IDS:
            raise ValueError(f"{context}.options[{index}].id must be A-D.")
        if not isinstance(text, str) or not text.strip() or text != text.strip():
            raise ValueError(f"{context}.options[{index}].text must be a nonempty unpadded string.")
        if option_id in option_text:
            raise ValueError(f"{context} contains duplicate option id {option_id}.")
        options.append({"id": option_id, "text": text})
        option_text[option_id] = text
    expected_ids = _OPTION_IDS[:2] if question_type == "conflict_presence" else _OPTION_IDS
    if tuple(option_text) != expected_ids:
        raise ValueError(f"{context}.options must use ordered ids {list(expected_ids)}.")
    return options, option_text


def _parse_answer(raw_answer: Any, *, answer_type: str, option_text: dict[str, str], context: str) -> frozenset[str]:
    if answer_type == "single_choice":
        if not isinstance(raw_answer, str) or raw_answer not in option_text:
            raise ValueError(f"{context}.answer must be one exact option id.")
        return frozenset({raw_answer})
    if answer_type != "multiple_choice":
        raise ValueError(f"{context}.answer_type must be single_choice or multiple_choice.")
    if not isinstance(raw_answer, list) or not raw_answer:
        raise ValueError(f"{context}.answer must be a nonempty option-id array.")
    if any(not isinstance(value, str) or value not in option_text for value in raw_answer):
        raise ValueError(f"{context}.answer contains an unknown option id.")
    if len(set(raw_answer)) != len(raw_answer):
        raise ValueError(f"{context}.answer contains duplicate option ids.")
    return frozenset(raw_answer)


def _number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{context} must be finite and nonnegative.")
    return result


def _project_main_samples(
    samples: Iterable[dict[str, Any]],
    *,
    enforce_release_contract: bool,
) -> tuple[list[dict[str, Any]], dict[str, _Target], dict[str, str], list[dict[str, Any]]]:
    model_inputs: list[dict[str, Any]] = []
    targets: dict[str, _Target] = {}
    media_index: dict[str, str] = {}
    expected_sidecar: list[dict[str, Any]] = []
    sample_sequences: dict[str, tuple[str, ...]] = {}
    sample_sources: dict[str, str] = {}
    type_counts: Counter[str] = Counter()
    records = list(samples)

    for line_number, sample in enumerate(records, start=1):
        context = f"main line {line_number}"
        sample_id = sample.get("id")
        video_path = sample.get("video_path")
        if not isinstance(sample_id, str) or not sample_id or sample_id != sample_id.strip():
            raise ValueError(f"{context}.id must be a nonempty unpadded string.")
        if sample_id in media_index:
            raise ValueError(f"Duplicate main sample id {sample_id!r}.")
        if not isinstance(video_path, str) or not video_path or video_path != video_path.strip():
            raise ValueError(f"{context}.video_path must be a nonempty unpadded string.")
        media_index[sample_id] = video_path
        sample_sources[sample_id] = _source_dataset(video_path, context=context)
        qa_pairs = sample.get("qa_pairs")
        if not isinstance(qa_pairs, list):
            raise TypeError(f"{context}.qa_pairs must be an array.")

        sequence: list[str] = []
        previous_order = -1
        for qa_index, qa in enumerate(qa_pairs):
            qa_context = f"{context}.qa_pairs[{qa_index}]"
            if not isinstance(qa, dict):
                raise TypeError(f"{qa_context} must be an object.")
            required = {"qa_id", "question_type", "question", "answer_type", "options", "answer", "evidence", "generator_version"}
            missing = required - qa.keys()
            if missing:
                raise ValueError(f"{qa_context} is missing {sorted(missing)}.")
            qa_id = qa["qa_id"]
            question_type = qa["question_type"]
            question = qa["question"]
            answer_type = qa["answer_type"]
            if not isinstance(question_type, str) or question_type not in QUESTION_TYPES:
                raise ValueError(f"{qa_context}.question_type is not part of Task C atomic v1.")
            if qa_id != f"{sample_id}__{question_type}":
                raise ValueError(f"{qa_context}.qa_id does not match sample_id and question_type.")
            if qa_id in targets:
                raise ValueError(f"Duplicate main qa_id {qa_id!r}.")
            if not isinstance(question, str) or not question.strip() or question != question.strip():
                raise ValueError(f"{qa_context}.question must be a nonempty unpadded string.")
            expected_answer_type = "multiple_choice" if question_type == "conflict_quarter_coverage" else "single_choice"
            if answer_type != expected_answer_type:
                raise ValueError(f"{qa_context}.answer_type must be {expected_answer_type!r}.")
            if qa["generator_version"] != GENERATOR_VERSION:
                raise ValueError(f"{qa_context}.generator_version must be {GENERATOR_VERSION!r}.")
            if not isinstance(qa["evidence"], list) or not qa["evidence"]:
                raise ValueError(f"{qa_context}.evidence must be a nonempty array.")

            order = _QUESTION_ORDER_INDEX[question_type]
            if order <= previous_order:
                raise ValueError(f"{context}.qa_pairs violates the frozen question order.")
            previous_order = order
            sequence.append(question_type)

            options, option_text = _parse_options(qa["options"], question_type=question_type, context=qa_context)
            answer = _parse_answer(qa["answer"], answer_type=answer_type, option_text=option_text, context=qa_context)
            target_value: float | None = None
            tolerance: float | None = None
            unit: str | None = None
            if question_type in NUMERIC_QUESTION_TYPES:
                target_value = _number(qa.get("target_value"), context=f"{qa_context}.target_value")
                tolerance = _number(qa.get("tolerance"), context=f"{qa_context}.tolerance")
                unit = qa.get("unit")
                if unit != "seconds" or tolerance != 0.5:
                    raise ValueError(f"{qa_context} must use unit='seconds' and tolerance=0.5.")
            elif {"target_value", "tolerance", "unit"} & qa.keys():
                raise ValueError(f"{qa_context} is nonnumeric and must omit numeric target fields.")

            model_input = {
                "qa_id": qa_id,
                "sample_id": sample_id,
                "question_type": question_type,
                "question": question,
                "answer_type": answer_type,
                "options": options,
                "unit": unit,
            }
            if tuple(model_input) != MODEL_INPUT_FIELDS or FORBIDDEN_MODEL_INPUT_FIELDS & model_input.keys():
                raise AssertionError("Internal model-input projection violated the Task C allowlist.")
            model_inputs.append(model_input)
            targets[qa_id] = _Target(
                question_type=question_type,
                answer_type=answer_type,
                option_text=option_text,
                answer=answer,
                target_value=target_value,
                tolerance=tolerance,
            )
            expected_sidecar.append({"item_id": qa_id, "sample_id": sample_id, "video_path": video_path, **qa})
            type_counts[question_type] += 1

        sample_sequences[sample_id] = tuple(sequence)

    if enforce_release_contract:
        if len(records) != EXPECTED_FULL_SAMPLE_COUNT:
            raise ValueError(f"Task C release must contain {EXPECTED_FULL_SAMPLE_COUNT} samples; observed {len(records)}.")
        if len(model_inputs) != EXPECTED_FULL_QUESTION_COUNT:
            raise ValueError(f"Task C release must contain {EXPECTED_FULL_QUESTION_COUNT} questions; observed {len(model_inputs)}.")
        if dict(type_counts) != EXPECTED_FULL_TYPE_COUNTS:
            raise ValueError(f"Task C release question counts mismatch: {dict(type_counts)}.")
        for sample_id, sequence in sample_sequences.items():
            if sequence not in _LEGAL_RELEASE_SEQUENCES:
                raise ValueError(f"Sample {sample_id} has an illegal Task C question sequence: {list(sequence)}.")
            source = sample_sources[sample_id]
            has_temporal = any(question_type in TEMPORAL_QUESTION_TYPES for question_type in sequence)
            if (source in TEMPORAL_ELIGIBLE_DATASETS) != has_temporal:
                raise ValueError(f"Sample {sample_id} violates the NTU-only temporal policy.")

    return model_inputs, targets, media_index, expected_sidecar


def _validate_sidecar(records: list[dict[str, Any]], expected: list[dict[str, Any]]) -> None:
    if records != expected:
        if len(records) != len(expected):
            raise ValueError(f"Task C sidecar has {len(records)} rows; expected {len(expected)}.")
        for index, (observed, wanted) in enumerate(zip(records, expected)):
            if observed != wanted:
                raise ValueError(f"Task C sidecar row {index + 1} is not the exact derivative of the main artifact.")
        raise ValueError("Task C sidecar is not the exact derivative of the main artifact.")


def _split_sample_ids(records: Iterable[dict[str, Any]], *, valid_sample_ids: set[str], enforce_release_contract: bool) -> set[str]:
    sample_ids: list[str] = []
    for line_number, record in enumerate(records, start=1):
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"split line {line_number}.id must be a nonempty string.")
        sample_ids.append(sample_id)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Task C split contains duplicate sample ids.")
    unknown = set(sample_ids) - valid_sample_ids
    if unknown:
        raise ValueError(f"Task C split contains unknown sample ids: {sorted(unknown)[:5]}.")
    if enforce_release_contract and len(sample_ids) != EXPECTED_TEST_SAMPLE_COUNT:
        raise ValueError(f"Task C test split must contain {EXPECTED_TEST_SAMPLE_COUNT} samples; observed {len(sample_ids)}.")
    return set(sample_ids)


def _project_test_rows(model_inputs: list[dict[str, Any]], split_sample_ids: set[str], *, enforce_release_contract: bool) -> list[dict[str, Any]]:
    rows = [row for row in model_inputs if not split_sample_ids or row["sample_id"] in split_sample_ids]
    if enforce_release_contract:
        counts = Counter(row["question_type"] for row in rows)
        if len(rows) != EXPECTED_TEST_QUESTION_COUNT:
            raise ValueError(f"Task C test split must project to {EXPECTED_TEST_QUESTION_COUNT} questions; observed {len(rows)}.")
        if dict(counts) != EXPECTED_TEST_TYPE_COUNTS:
            raise ValueError(f"Task C test split question counts mismatch: {dict(counts)}.")
    return rows


class PhysicalConflictTask(ConfigurableTask):
    """Load the frozen Task C release and expose its safe test projection."""

    def __init__(self, *args, config: dict[str, Any] | None = None, **kwargs) -> None:
        if config is not None:
            config = dict(config)
            config.pop("class", None)
        super().__init__(*args, config=config, **kwargs)

    def download(self, dataset_kwargs: dict[str, Any] | None = None) -> None:
        kwargs = dict(dataset_kwargs or {})
        qa_path = _qa_path_from_config(kwargs)
        sidecar_path = _configured_path("PHYSICAL_CONFLICT_SIDECAR_PATH", kwargs, "sidecar_file", "evaluation_file")
        split_path = _configured_path("PHYSICAL_CONFLICT_SPLIT_PATH", kwargs, "split_file")
        video_root = _configured_path("PHYSICAL_CONFLICT_VIDEO_ROOT", kwargs, "video_root")
        allow_partial = _as_bool(os.environ.get("PHYSICAL_CONFLICT_ALLOW_PARTIAL"), default=False)
        strict = _as_bool(kwargs.get("strict"), default=True) and not allow_partial
        require_sidecar = _as_bool(kwargs.get("require_sidecar"), default=strict) and not allow_partial

        if qa_path is None:
            raise FileNotFoundError(f"Physical-conflict evaluation requires a Task C main JSONL.\n{_SETUP_HINT}")
        if strict and split_path is None:
            raise FileNotFoundError(f"Strict physical-conflict evaluation requires the frozen test split.\n{_SETUP_HINT}")
        if require_sidecar and sidecar_path is None:
            raise FileNotFoundError(f"Strict physical-conflict evaluation requires the compatibility sidecar.\n{_SETUP_HINT}")

        main_sha: str | None = None
        sidecar_sha: str | None = None
        split_sha: str | None = None
        if strict:
            main_sha = _validate_hash(
                qa_path,
                str(os.environ.get("PHYSICAL_CONFLICT_MAIN_SHA256") or kwargs.get("expected_main_sha256") or DEFAULT_MAIN_SHA256),
                artifact_name="Task C main",
            )
            assert split_path is not None
            split_sha = _validate_hash(
                split_path,
                str(os.environ.get("PHYSICAL_CONFLICT_TEST_SPLIT_SHA256") or kwargs.get("expected_test_split_sha256") or DEFAULT_TEST_SPLIT_SHA256),
                artifact_name="Task C test split",
            )
            if require_sidecar:
                assert sidecar_path is not None
                sidecar_sha = _validate_hash(
                    sidecar_path,
                    str(os.environ.get("PHYSICAL_CONFLICT_SIDECAR_SHA256") or kwargs.get("expected_sidecar_sha256") or DEFAULT_SIDECAR_SHA256),
                    artifact_name="Task C sidecar",
                )

        samples = _read_jsonl(qa_path, artifact_name="Task C main")
        model_inputs, targets, media_index, expected_sidecar = _project_main_samples(samples, enforce_release_contract=strict)
        if sidecar_path is not None:
            sidecar = _read_jsonl(sidecar_path, artifact_name="Task C sidecar")
            _validate_sidecar(sidecar, expected_sidecar)
        split_records = _read_jsonl(split_path, artifact_name="Task C test split") if split_path is not None else []
        split_ids = _split_sample_ids(split_records, valid_sample_ids=set(media_index), enforce_release_contract=strict) if split_records else set()
        rows = _project_test_rows(model_inputs, split_ids, enforce_release_contract=strict)
        selected_qa_ids = {row["qa_id"] for row in rows}
        selected_sample_ids = {row["sample_id"] for row in rows}

        _runtime.annotation_root = qa_path.parent
        _runtime.video_root = video_root
        _runtime.targets = {qa_id: target for qa_id, target in targets.items() if qa_id in selected_qa_ids}
        _runtime.media_index = {sample_id: path for sample_id, path in media_index.items() if sample_id in selected_sample_ids}

        split = self.config.test_split
        self.dataset = DatasetDict({split: Dataset.from_list(rows)})
        if self.config.process_docs is not None:
            self.dataset[split] = self.config.process_docs(self.dataset[split])
        self.dataset_no_image = self.dataset.copy()
        eval_logger.info(f"Loaded GoodVision Task C atomic v1 ({len(selected_sample_ids)} samples, {len(rows)} questions, strict={strict}, main_sha256={main_sha}, sidecar_sha256={sidecar_sha}, split_sha256={split_sha}).")


def physical_conflict_process_docs(dataset: Dataset) -> Dataset:
    return dataset


def _candidate_video_roots() -> list[Path]:
    roots: list[Path] = []
    if _runtime.video_root is not None:
        roots.append(_runtime.video_root)
    for env_name in ("PHYSICAL_CONFLICT_VIDEO_ROOT", "PHYSICAL_CONFLICT_DATA_ROOT"):
        value = os.environ.get(env_name)
        if value:
            roots.append(_resolve_path(value))
    if _runtime.annotation_root is not None:
        roots.append(_runtime.annotation_root)
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def _resolve_video_path(sample_id: str) -> Path:
    if sample_id not in _runtime.media_index:
        raise KeyError(f"No trusted media mapping exists for sample {sample_id!r}.")
    raw_path = Path(_runtime.media_index[sample_id]).expanduser()
    if raw_path.is_absolute():
        return raw_path
    roots = _candidate_video_roots()
    for root in roots:
        candidate = root / raw_path
        if candidate.exists():
            return candidate
    return (roots[0] if roots else Path.cwd()) / raw_path


def _opaque_media_alias(path: Path, sample_id: str) -> Path:
    alias_root_value = os.environ.get("PHYSICAL_CONFLICT_OPAQUE_MEDIA_ROOT")
    if not alias_root_value:
        return path
    alias_root = _resolve_path(alias_root_value)
    alias_root.mkdir(parents=True, exist_ok=True)
    alias_path = alias_root / f"{sample_id}{path.suffix.lower()}"
    target = path.resolve()
    if alias_path.is_symlink():
        if alias_path.resolve() != target:
            raise ValueError(f"Opaque media alias {alias_path} points to the wrong target.")
    elif alias_path.exists():
        raise ValueError(f"Opaque media alias path already exists and is not a symlink: {alias_path}.")
    else:
        try:
            alias_path.symlink_to(target)
        except FileExistsError:
            if not alias_path.is_symlink() or alias_path.resolve() != target:
                raise
    return alias_path


def physical_conflict_doc_to_visual(doc: dict[str, Any]) -> list[str]:
    sample_id = str(doc["sample_id"])
    path = _resolve_video_path(sample_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing physical-conflict video for {sample_id}: {path}\n{_SETUP_HINT}")
    return [str(_opaque_media_alias(path, sample_id))]


def physical_conflict_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = str(kwargs.get("pre_prompt", ""))
    option_lines = "\n".join(f"{option['id']}. {option['text']}" for option in doc["options"])
    body = f"Watch the full video carefully before answering.\n\nQuestion: {doc['question']}\n\nOptions:\n{option_lines}"
    if doc["answer_type"] == "multiple_choice":
        post_prompt = kwargs.get("multiple_choice_post_prompt", 'Return only a JSON array of every applicable option ID, for example ["A","C"].')
    else:
        post_prompt = kwargs.get("single_choice_post_prompt", "Answer with only one option ID.")
    return f"{pre_prompt}{body}\n\n{post_prompt}".strip()


def _target_for_doc(doc: dict[str, Any]) -> _Target:
    qa_id = str(doc["qa_id"])
    if qa_id not in _runtime.targets:
        raise KeyError(f"No trusted answer exists for QA {qa_id!r}.")
    return _runtime.targets[qa_id]


def physical_conflict_doc_to_target(doc: dict[str, Any]) -> str:
    target = _target_for_doc(doc)
    ordered = sorted(target.answer, key=_OPTION_IDS.index)
    return ordered[0] if target.answer_type == "single_choice" else json.dumps(ordered, separators=(",", ":"))


def physical_conflict_normalize_prediction(doc: dict[str, Any], raw_prediction: str) -> str | list[str] | None:
    target = _target_for_doc(doc)
    text = str(raw_prediction).strip()
    if target.answer_type == "single_choice":
        return text if text in target.option_text else None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or not value or len(set(value)) != len(value):
        return None
    if any(not isinstance(option_id, str) or option_id not in target.option_text for option_id in value):
        return None
    return sorted(value, key=_OPTION_IDS.index)


def physical_conflict_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    target = _target_for_doc(doc)
    prediction = physical_conflict_normalize_prediction(doc, results[0] if results else "")
    predicted_set = frozenset({prediction}) if isinstance(prediction, str) else frozenset(prediction or [])
    exact = float(prediction is not None and predicted_set == target.answer)
    metrics: dict[str, float] = {"Overall_Exact_Accuracy": exact}

    if target.answer_type == "multiple_choice":
        intersection = predicted_set & target.answer
        precision = len(intersection) / len(predicted_set) if predicted_set else 0.0
        recall = len(intersection) / len(target.answer) if target.answer else float(not predicted_set)
        metrics.update(
            {
                "MultipleChoice_Exact_Set_Accuracy": exact,
                "MultipleChoice_Macro_Precision": precision,
                "MultipleChoice_Macro_Recall": recall,
            }
        )
        return metrics

    metrics["SingleChoice_Accuracy"] = exact
    if target.question_type == "conflict_presence":
        metrics["Binary_Accuracy"] = exact
    if target.question_type in NUMERIC_QUESTION_TYPES:
        within_tolerance = 0.0
        if isinstance(prediction, str):
            selected_value = float(target.option_text[prediction])
            assert target.target_value is not None and target.tolerance is not None
            absolute_error = abs(selected_value - target.target_value)
            metrics["Numeric_Option_MAE"] = absolute_error
            within_tolerance = float(absolute_error <= target.tolerance)
        metrics["Numeric_Accuracy_at_0_5s"] = within_tolerance
    return metrics


def physical_conflict_aggregate_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
