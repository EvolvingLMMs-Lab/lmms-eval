"""Exact function-level export of the internal evaluation implementation."""

from __future__ import annotations

from typing import Any, Sequence

CONDITION_ORDER = ("text_target", "reference_image")

CONTENT_POLICY_ERROR_MARKERS = (
    "content_policy_violation",
    "content management policy",
    "content safety system",
    "content_filter",
    "contentfilter",
    "sensitive words",
    "prohibited use policy",
    "prohibited_content",
    "prompt_blocked",
    "data_inspection_failed",
    "datainspectionfailed",
    "inappropriate content",
)

REFUSAL_VALIDATION_ERRORS = {
    "content_policy_refusal",
    "finish_reason_content_filter",
}


def is_content_policy_error(value: Any) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in CONTENT_POLICY_ERROR_MARKERS)


def has_policy_attempt(row: dict[str, Any]) -> bool:
    """True when any recorded attempt hit a provider content-policy error."""
    return any(is_content_policy_error(attempt.get("error")) for attempt in row.get("attempt_records") or [])


def is_final_policy_refusal(row: dict[str, Any]) -> bool:
    """True when the request's FINAL state is a provider content-policy refusal.

    Covers the API-exception path (error_kind / error text) and HTTP-200
    refusal text recorded as validation_error or a content_filter finish
    reason. Attempt history is deliberately NOT consulted here: a request that
    was policy-blocked on an early attempt but eventually succeeded has a
    usable answer and stays in the scoring denominator.
    """
    if row.get("error_kind") == "content_policy" or is_content_policy_error(row.get("error")):
        return True
    if str(row.get("validation_error") or "") in REFUSAL_VALIDATION_ERRORS:
        return True
    finish_reason = str((row.get("response_meta") or {}).get("finish_reason") or "")
    return finish_reason.lower() == "content_filter"


def is_terminal_policy_refusal(row: dict[str, Any]) -> bool:
    """True when the request never produced a usable answer due to policy.

    This is the final-state refusal OR a failed request (e.g. final timeout)
    whose attempt history already tripped the provider policy — such requests
    never delivered an answer and the triggering content is policy-flagged, so
    they are counted as safety-related exclusions rather than transport holes.
    """
    if is_final_policy_refusal(row):
        return True
    return row.get("error") is not None and has_policy_attempt(row)


def select_scoring_request_ids(
    raw_rows: Sequence[dict[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    """Split request IDs into (scoring, policy_excluded, transport_excluded).

    Official scoring uses successful API responses only: a row enters the
    denominator when its final state has no error and is not a final-state
    content-policy refusal. Successful responses that fail task parsing stay in
    the denominator with score 0. A request that eventually succeeded after a
    policy-flagged attempt also stays in the denominator.
    """
    scoring: set[str] = set()
    policy: set[str] = set()
    transport: set[str] = set()
    for row in raw_rows:
        request_id = str(row.get("request_id"))
        if row.get("error") is None and not is_final_policy_refusal(row):
            scoring.add(request_id)
        elif is_terminal_policy_refusal(row):
            policy.add(request_id)
        else:
            transport.add(request_id)
    return scoring, policy, transport


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def threshold_accuracy(values: Sequence[float], threshold: float) -> float | None:
    if not values:
        return None
    return sum(value >= threshold for value in values) / len(values)


def group_scores(
    parsed_rows: Sequence[dict[str, Any]],
    task: str,
    condition: str,
    scoring_ids: set[str] | None = None,
) -> list[float]:
    scores: list[float] = []
    for row in parsed_rows:
        if row.get("task") != task or row.get("target_specification") != condition:
            continue
        if scoring_ids is not None and str(row.get("request_id")) not in scoring_ids:
            continue
        score = row.get("score")
        if score is None:
            raise ValueError(f"{row.get('request_id')}: missing score")
        scores.append(float(score))
    return scores


def build_main_metrics(
    parsed_rows: Sequence[dict[str, Any]],
    scoring_ids: set[str] | None = None,
) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    selected_conditions = {str(row.get("target_specification")) for row in parsed_rows}
    unknown_conditions = selected_conditions - set(CONDITION_ORDER)
    if unknown_conditions:
        raise ValueError(f"unknown target specifications: {sorted(unknown_conditions)}")
    for condition in CONDITION_ORDER:
        if condition not in selected_conditions:
            continue
        tg_scores = group_scores(parsed_rows, "TG", condition, scoring_ids)
        sg_scores = group_scores(parsed_rows, "SG", condition, scoring_ids)
        qa_scores = group_scores(parsed_rows, "QA", condition, scoring_ids)
        tg_mean = mean(tg_scores)
        sg_mean = mean(sg_scores)
        qa_mean = mean(qa_scores)
        overall = mean([tg_mean, sg_mean, qa_mean]) if tg_mean is not None and sg_mean is not None and qa_mean is not None else None
        output[condition] = {
            "TG_A@0.3": threshold_accuracy(tg_scores, 0.3),
            "TG_A@0.5": threshold_accuracy(tg_scores, 0.5),
            "TG_A@0.7": threshold_accuracy(tg_scores, 0.7),
            "TG_m_tIoU": tg_mean,
            "SG_A@0.3": threshold_accuracy(sg_scores, 0.3),
            "SG_A@0.5": threshold_accuracy(sg_scores, 0.5),
            "SG_A@0.7": threshold_accuracy(sg_scores, 0.7),
            "SG_m_sIoU": sg_mean,
            "QA_accuracy": qa_mean,
            "Overall": overall,
        }
    return output
