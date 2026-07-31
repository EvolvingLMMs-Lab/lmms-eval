"""Task utilities and official metrics for the Multi-Crit benchmark."""

from __future__ import annotations

import itertools
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import requests
from loguru import logger as eval_logger

# The prompt text below is copied verbatim from the official Multi-Crit
# implementation at commit ``c4b0c32``. Whitespace is evaluation-significant:
# do not reflow or edit these strings without updating the parity tests.

CRITERION_OPENENDED_COMPLETENESS = """**Completeness and Coverage**  
The response should provide a thorough and well-developed answer that fully addresses the intent of the prompt. It must address the complete scope of the task, incorporating all major elements specified in the prompt, as well as relevant visual aspects and broader contextual cues. When appropriate, drawing on relevant external knowledge is encouraged to enrich the explanation.  
- Reward: The response addresses all key parts of the prompt and image, showing depth and effort in the description or explanation.  
- Penalize: The response is underdeveloped, fails to meet one or more specific requirements in the prompt, or omits important visual elements or interpretive points."""

CRITERION_OPENENDED_GROUNDEDNESS = """**Visual Grounding and Details**  
The response should demonstrate a clear and meaningful connection to the visual input. It should refer to observable elements in the image—such as objects, spatial relationships, colors, or text—and build its description or interpretation based on those elements.  
- Reward: The response explicitly references relevant visual details that are clearly visible in the image.  
- Penalize: The response fails to connect meaningfully to the image, or uses vague, generic language that lacks specific visual grounding."""

CRITERION_OPENENDED_HALLUCINATION = """**Factuality / No Hallucination**  
The response should avoid introducing any visual details, objects, relationships, or factual claims that are not present in the image or reasonably suggested by the prompt. This includes both visual hallucinations (e.g., describing elements not visible in the image) and factual inaccuracies in general knowledge.  
- Reward: The response stays grounded in the image and prompt, without inventing visual elements or making unsupported factual claims.  
- Penalize: The response introduces hallucinated visual content or inaccurate factual statements that are unsupported or misleading."""

CRITERION_OPENENDED_EXPRESSIVENESS = """**Expressiveness and Expertise**
The response should show originality or stylistic flair for open-ended tasks, and knowledge-informed articulation with precision and depth for analytical tasks. All responses must remain contextually appropriate and grounded in the visual input, while enhancing richness, nuance, and overall engagement.
- Reward: The response uses vivid language, unique phrasing, or inventive associations that enrich the interpretation, or it demonstrates professional articulation through deep and knowledge-grounded analysis.
- Penalize: The response is overly literal, flat, or dull, lacking originality, variation in expression, or in analytical contexts, fails to demonstrate professional depth or expertise."""

CRITERION_OPENENDED_CLARITY = """**Clarity and Coherence**  
The response should communicate ideas clearly and logically, with coherent structure and fluent language. This involves not only grammatical correctness, but also effective organization of information, smooth transitions, and consistent flow of ideas.  
- Reward: The response is clearly written, logically structured, and easy to follow. A brief summary at the beginning may further improve clarity.  
- Penalize: The response is difficult to follow due to unclear structure, disorganized reasoning, poor transitions, or awkward and repetitive phrasing."""

CRITERION_REASON_GROUNDEDNESS = """**Visual Grounding**  
The response should be explicitly grounded in the visual input. It must refer to salient visual elements—such as specific objects, spatial arrangements, colors, or visible text—and incorporate them meaningfully into the reasoning process. Visual references should be accurate and relevant to the task.  
- Reward: The response clearly references important visual features and integrates them into the reasoning in a precise and relevant manner.  
- Penalize: The response fails to reference relevant visual elements, or uses generic or weakly connected visual details that do not meaningfully support the reasoning."""

CRITERION_REASON_LOGIC = """**Logical Coherence and Consistency**  
The reasoning should follow a logically sound and step-by-step progression, with each step building upon the previous one. The reasoning should be internally consistent, with no contradictions, missing steps, or unjustified leaps. The final answer should naturally and justifiably emerge from the reasoning process.  
- Reward: The response presents a well-structured, internally consistent chain of reasoning that leads clearly and justifiably to the final answer.  
- Penalize: The response contains contradictions, missing steps, or disconnects between reasoning and answer. Short-cut behaviors—such as giving the final answer first with unsupported or inconsistent reasoning—should also be penalized."""

CRITERION_REASON_HALLUCINATION = """**Factual Correctness / No Hallucination**  
All claims and reasoning steps must be factually accurate and supported by the image or the prompt. The response should avoid hallucinated visual content, misidentifications, or factual inaccuracies in the reasoning process.  
- Reward: The response is free from factual errors or hallucinations and relies only on valid observations and logical inferences.  
- Penalize: The response introduces hallucinated details, misidentifications, or incorrect factual claims that compromise the reasoning."""

CRITERION_REASON_EXPLORATION = """**Reflection and Exploration**  
The reasoning should demonstrate thoughtful reflection and a willingness to explore multiple possibilities, particularly when the task is ambiguous or complex. This includes acknowledging uncertainty, considering alternative interpretations, or revising initial assumptions.  
- Reward: The response demonstrates depth through reflection, critical evaluation, or exploration of different solutions before reaching a conclusion.  
- Penalize: The response is overly rigid, superficial, or rushed, showing little to no depth of thought, reflection, or exploration of alternative possibilities."""

CRITERION_REASON_EFFICIENCY = """**Conciseness and Efficiency**  
The reasoning should be clear, focused, and efficiently communicate the steps. It should avoid redundancy, digressions, or unnecessary elaboration that dilute the argument. For straightforward tasks, over-explaining or over-analyzing should also be avoided.  
- Reward: The response is concise and well-structured, conveying reasoning steps precisely and proportionally to the task complexity.  
- Penalize: The response is verbose, repetitive, or includes irrelevant content that distracts from the reasoning. It may also overthink or over-explain simple prompts."""


PROMPT_TEMPLATE = """You are an expert in evaluating the quality of AI-generated responses according to specific evaluation criteria. Your task is to assess two responses generated by different AI assistants in reply to a user’s question about an image. The image is provided as part of the input. 

You must evaluate the responses **strictly and exclusively** based on the following evaluation criterion:

{Criterion}

Do not consider any other dimensions or criteria beyond what is specified above. 

Here are the inputs for your evaluation:
[Question]: {Question}
[Response 1]: {Response1}
[Response 2]: {Response2}

First, provide a detailed justification for your evaluation. Refer to specific elements in the responses, how they align with the evaluation criterion, and relevant visual details from the image.

On the final line, provide your final judgment on which response is better. Your judgment must be based solely on the specified criterion. Strictly follow this format:  
Response X is better.
"""


OPEN_ENDED_CRITERIA: tuple[str, ...] = (
    "completeness",
    "visual-grounding",
    "hallucination",
    "expressiveness",
    "clarity",
)

REASONING_CRITERIA: tuple[str, ...] = (
    "grounding",
    "logic",
    "hallucination",
    "exploration",
    "efficiency",
)

CRITERION_MAP: dict[str, str] = {
    "open-ended--completeness": CRITERION_OPENENDED_COMPLETENESS,
    "open-ended--visual-grounding": CRITERION_OPENENDED_GROUNDEDNESS,
    "open-ended--hallucination": CRITERION_OPENENDED_HALLUCINATION,
    "open-ended--expressiveness": CRITERION_OPENENDED_EXPRESSIVENESS,
    "open-ended--clarity": CRITERION_OPENENDED_CLARITY,
    "reasoning--grounding": CRITERION_REASON_GROUNDEDNESS,
    "reasoning--logic": CRITERION_REASON_LOGIC,
    "reasoning--hallucination": CRITERION_REASON_HALLUCINATION,
    "reasoning--exploration": CRITERION_REASON_EXPLORATION,
    "reasoning--efficiency": CRITERION_REASON_EFFICIENCY,
}


def normalize_benchmark_type(value: str) -> str:
    """Normalize a dataset split label to the official prompt-map spelling."""

    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"open-ended", "openended"}:
        return "open-ended"
    if normalized == "reasoning":
        return "reasoning"
    raise ValueError(f"Unsupported Multi-Crit benchmark type: {value!r}")


def format_prompt(
    *,
    question: str,
    response_1: str,
    response_2: str,
    benchmark_type: str,
    criterion: str,
) -> str:
    """Render the official single-criterion Multi-Crit judge prompt."""

    criterion_key = f"{normalize_benchmark_type(benchmark_type)}--{criterion}"
    if criterion_key not in CRITERION_MAP:
        raise KeyError(f"Unsupported Multi-Crit criterion: {criterion_key}")
    return PROMPT_TEMPLATE.format(
        Question=question,
        Response1=response_1,
        Response2=response_2,
        Criterion=CRITERION_MAP[criterion_key],
    )


VALID_PREFERENCES = frozenset({"model_a", "model_b"})
COMMON_METRICS: tuple[str, ...] = (
    "overall",
    "macro_avg",
    "pluralistic_acc",
    "tradeoff_sensitivity",
    "conflict_matching",
)
METRICS_BY_BENCHMARK: dict[str, tuple[str, ...]] = {
    "open-ended": COMMON_METRICS + OPEN_ENDED_CRITERIA,
    "reasoning": COMMON_METRICS + REASONING_CRITERIA,
}

DIRECT_PREFERENCE_PATTERN = re.compile(r"[\*]*response (\d+)[\*]* is better")
VERIFIER_PREFERENCE_PATTERN = re.compile(r"model_(a|b)")
VERIFIER_FAIL_PATTERN = re.compile(r"fail")

VERIFIER_PROMPT_TEMPLATE = """You are given a judgement comparing two responses (response 1 and response 2).
From this judgement, extract only the final preference: return 'model_a' if the judgement indicates response 1 is better, or 'model_b' if response 2 is better.
If the judgement does not clearly indicate either, return 'fail'.
Only output one line: 'model_a', 'model_b', or 'fail'.

Judgement:
{Judgement}
"""


def _benchmark_type_from_doc(doc: Mapping[str, Any]) -> str:
    """Return the official benchmark type encoded in a dataset row."""

    if "split" not in doc:
        raise KeyError("Multi-Crit document is missing the 'split' field")
    return normalize_benchmark_type(str(doc["split"]))


def multi_crit_doc_to_visual(doc: Mapping[str, Any]) -> list[Any]:
    """Return the embedded image as one RGB PIL image."""

    image = doc.get("image")
    if image is None or not hasattr(image, "convert"):
        raise TypeError("Multi-Crit document must contain a decoded PIL image")
    return [image.convert("RGB")]


def multi_crit_doc_to_text(
    doc: Mapping[str, Any],
    lmms_eval_specific_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Render the official criterion-specific judge prompt for one row."""

    del lmms_eval_specific_kwargs
    return format_prompt(
        question=str(doc["question"]),
        response_1=str(doc["pred_a"]),
        response_2=str(doc["pred_b"]),
        benchmark_type=_benchmark_type_from_doc(doc),
        criterion=str(doc["criterion"]),
    )


def multi_crit_doc_to_messages(
    doc: Mapping[str, Any],
    lmms_eval_specific_kwargs: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build one user message with the image before the exact prompt text."""

    image = multi_crit_doc_to_visual(doc)[0]
    text = multi_crit_doc_to_text(doc, lmms_eval_specific_kwargs)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": text},
            ],
        }
    ]


def parse_direct_preference(raw_response: str) -> str | None:
    """Parse the official final preference from only the last three lines."""

    last_three_lines = "\n".join(raw_response.strip().split("\n")[-3:])
    match = DIRECT_PREFERENCE_PATTERN.search(last_three_lines.lower())
    if match is None:
        return None
    try:
        response_id = int(match.group(1))
    except (TypeError, ValueError):
        return None
    if response_id == 1:
        return "model_a"
    if response_id == 2:
        return "model_b"
    return None


def prepare_verifier_judgement(raw_response: str) -> str:
    """Prepare exactly the last four normalized lines used by the verifier."""

    judgement = raw_response.replace("\n\n", "\n").strip()
    return "\n".join(judgement.split("\n")[-4:]).strip()


def parse_verifier_preference(raw_response: str) -> str | None:
    """Parse ``model_a``, ``model_b``, or ``fail`` from verifier output."""

    normalized = raw_response.strip().lower()
    match = VERIFIER_PREFERENCE_PATTERN.search(normalized)
    if match is not None:
        return f"model_{match.group(1)}"
    if VERIFIER_FAIL_PATTERN.search(normalized) is not None:
        return "fail"
    return None


def _verifier_model_name() -> str:
    """Return the configured verifier model, defaulting to the official model."""

    return os.getenv("MULTI_CRIT_VERIFIER_MODEL", "gpt-4o-mini")


def _verifier_api_url() -> str:
    """Resolve either a base URL or a full chat-completions endpoint."""

    base_url = os.getenv("OPENAI_API_URL") or os.getenv("OPENAI_API_BASE")
    if not base_url:
        base_url = "https://api.openai.com/v1"
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _verifier_timeout() -> float:
    """Return the official 120-second verifier timeout, with an opt-in override."""

    return float(os.getenv("MULTI_CRIT_VERIFIER_TIMEOUT", "120"))


def _safe_exception_summary(exc: Exception) -> str:
    """Describe a verifier failure without logging URLs or response bodies."""

    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status_code, int):
        return f"{type(exc).__name__} (HTTP {status_code})"
    return type(exc).__name__


def _parse_with_verifier(
    raw_response: str,
    question_id: str,
) -> tuple[str, str, str | None]:
    """Use the official LLM fallback and map any failure to ``error``."""

    judgement = prepare_verifier_judgement(raw_response)
    model_name = _verifier_model_name()
    verifier_raw_output: str | None = None
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        response = requests.post(
            _verifier_api_url(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": VERIFIER_PROMPT_TEMPLATE.format(Judgement=judgement),
                    }
                ],
                "temperature": 0,
                "max_tokens": 50,
            },
            timeout=_verifier_timeout(),
        )
        response.raise_for_status()
        payload = response.json()
        verifier_raw_output = str(payload["choices"][0]["message"]["content"]).strip()
        prediction = parse_verifier_preference(verifier_raw_output) or "error"
    except Exception as exc:  # pragma: no cover - provider failures vary
        eval_logger.error(f"Multi-Crit verifier failed for {question_id}: " f"{_safe_exception_summary(exc)}")
        prediction = "error"

    eval_logger.info(f"Multi-Crit verifier fallback for {question_id}: " f"model={model_name}, prediction={prediction}")
    return prediction, model_name, verifier_raw_output


def parse_preference(
    raw_response: str,
    question_id: str = "unknown",
) -> tuple[str, str, str | None, str | None]:
    """Run the official direct parser and verifier fallback in sequence."""

    direct_prediction = parse_direct_preference(raw_response)
    if direct_prediction is not None:
        return direct_prediction, "direct", None, None
    prediction, verifier_model, verifier_raw_output = _parse_with_verifier(
        raw_response,
        question_id,
    )
    return prediction, "verifier", verifier_model, verifier_raw_output


def _answer_matches(prediction: str, preference: str) -> bool:
    """Match answers with the official whitespace/case normalization."""

    return prediction.strip().lower() == preference.strip().lower()


def multi_crit_process_results(
    doc: Mapping[str, Any],
    results: Sequence[Any],
) -> dict[str, dict[str, Any]]:
    """Parse one generation and route its compact row to every task metric."""

    raw_response = str(results[0]).strip() if results else ""
    question_id = str(doc["question_id"])
    prediction, parse_method, verifier_model, verifier_raw_output = parse_preference(
        raw_response,
        question_id,
    )
    preference = str(doc["preference"])
    benchmark_type = _benchmark_type_from_doc(doc)
    record: dict[str, Any] = {
        "question_id": question_id,
        "prompt_id": str(doc["prompt_id"]),
        "criterion": str(doc["criterion"]),
        "preference": preference,
        "prediction": prediction,
        "answer_match": _answer_matches(prediction, preference),
        "benchmark_type": benchmark_type,
        "parse_method": parse_method,
        "verifier_model": verifier_model,
        "verifier_raw_output": verifier_raw_output,
    }
    return {metric: dict(record) for metric in METRICS_BY_BENCHMARK[benchmark_type]}


def _deduplicate_question_ids(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the first record for each question ID, matching the reference."""

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        question_id = str(record["question_id"])
        if question_id not in seen:
            seen.add(question_id)
            unique.append(dict(record))
    return unique


def _group_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Group rows by prompt and criterion, with last criterion value winning."""

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for record in _deduplicate_question_ids(records):
        prompt_id = str(record["prompt_id"])
        criterion = str(record["criterion"])
        grouped.setdefault(prompt_id, {})[criterion] = record
    return grouped


def _criteria_for_records(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return the fixed criterion order for a single-split metric input."""

    benchmark_types = {str(record["benchmark_type"]) for record in records}
    if len(benchmark_types) != 1:
        raise ValueError("Multi-Crit aggregators require exactly one benchmark type; " f"received {sorted(benchmark_types)}")
    benchmark_type = normalize_benchmark_type(benchmark_types.pop())
    return OPEN_ENDED_CRITERIA if benchmark_type == "open-ended" else REASONING_CRITERIA


def _compute_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute every official metric from criterion-row records."""

    if not records:
        return {
            "overall": None,
            "criterion": {},
            "macro_avg": None,
            "pluralistic_acc": None,
            "tradeoff_sensitivity": None,
            "conflict_matching": None,
            "tradeoff_detected": 0,
            "tradeoff_total": 0,
            "conflict_correct": 0,
            "conflict_total": 0,
        }

    criteria = _criteria_for_records(records)
    grouped = _group_records(records)
    expanded = [record for prompt in grouped.values() for record in prompt.values()]

    criterion_scores: dict[str, float | None] = {}
    for criterion in criteria:
        selected = [record for record in expanded if record["criterion"] == criterion]
        criterion_scores[criterion] = (
            float(
                round(
                    np.mean([bool(record["answer_match"]) for record in selected]) * 100,
                    2,
                )
            )
            if selected
            else None
        )

    present_criterion_scores = [score for score in criterion_scores.values() if score is not None]
    overall = (
        float(
            round(
                np.mean([bool(record["answer_match"]) for record in expanded]) * 100,
                2,
            )
        )
        if expanded
        else None
    )
    macro_avg = float(round(np.mean(present_criterion_scores), 2)) if present_criterion_scores else None
    pluralistic_acc = (
        float(
            round(
                np.mean([all(bool(record["answer_match"]) for record in prompt.values()) for prompt in grouped.values()]) * 100,
                2,
            )
        )
        if grouped
        else None
    )

    tradeoff_detected = 0
    tradeoff_total = 0
    conflict_correct = 0
    conflict_total = 0
    for prompt in grouped.values():
        conflict_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for left, right in itertools.combinations(prompt.values(), 2):
            if left["preference"] != right["preference"]:
                conflict_pairs.append((left, right))
                conflict_total += 1
                conflict_correct += int(left["prediction"] == left["preference"] and right["prediction"] == right["preference"])

        if not conflict_pairs:
            continue
        tradeoff_total += 1
        tradeoff_detected += int(any(left["prediction"] != right["prediction"] and left["prediction"] in VALID_PREFERENCES and right["prediction"] in VALID_PREFERENCES for left, right in conflict_pairs))

    tradeoff_sensitivity = (
        float(
            round(
                np.mean([1] * tradeoff_detected + [0] * (tradeoff_total - tradeoff_detected)) * 100,
                2,
            )
        )
        if tradeoff_total
        else None
    )
    conflict_matching = (
        float(
            round(
                np.mean([1] * conflict_correct + [0] * (conflict_total - conflict_correct)) * 100,
                2,
            )
        )
        if conflict_total
        else None
    )

    return {
        "overall": overall,
        "criterion": criterion_scores,
        "macro_avg": macro_avg,
        "pluralistic_acc": pluralistic_acc,
        "tradeoff_sensitivity": tradeoff_sensitivity,
        "conflict_matching": conflict_matching,
        "tradeoff_detected": tradeoff_detected,
        "tradeoff_total": tradeoff_total,
        "conflict_correct": conflict_correct,
        "conflict_total": conflict_total,
    }


def _aggregate_named_metric(records: Sequence[Mapping[str, Any]], metric: str) -> float | None:
    """Return one scalar from the shared official metric implementation."""

    summary = _compute_metrics(records)
    if metric in OPEN_ENDED_CRITERIA or metric in REASONING_CRITERIA:
        return summary["criterion"].get(metric)
    return summary[metric]


def multi_crit_aggregate_overall(results: list[dict[str, Any]]) -> float | None:
    """Aggregate micro criterion-row accuracy on a 0–100 scale."""

    direct = sum(record.get("parse_method") == "direct" for record in results)
    verifier = sum(record.get("parse_method") == "verifier" for record in results)
    errors = sum(record.get("prediction") == "error" for record in results)
    eval_logger.info(f"Multi-Crit parsing: direct={direct}, verifier={verifier}, errors={errors}")
    return _aggregate_named_metric(results, "overall")


def multi_crit_aggregate_macro_avg(results: list[dict[str, Any]]) -> float | None:
    """Aggregate the macro average of already-rounded criterion accuracies."""

    return _aggregate_named_metric(results, "macro_avg")


def multi_crit_aggregate_pluralistic_acc(
    results: list[dict[str, Any]],
) -> float | None:
    """Aggregate prompt-level all-present-criteria accuracy (PAcc)."""

    return _aggregate_named_metric(results, "pluralistic_acc")


def multi_crit_aggregate_tradeoff_sensitivity(
    results: list[dict[str, Any]],
) -> float | None:
    """Aggregate prompt-macro trade-off sensitivity over conflict prompts."""

    summary = _compute_metrics(results)
    eval_logger.info("Multi-Crit TOS: " f"{summary['tradeoff_detected']}/{summary['tradeoff_total']} conflict prompts")
    return summary["tradeoff_sensitivity"]


def multi_crit_aggregate_conflict_matching(
    results: list[dict[str, Any]],
) -> float | None:
    """Aggregate pair-micro conflict matching rate (CMR), never prompt-macro."""

    summary = _compute_metrics(results)
    eval_logger.info("Multi-Crit CMR: " f"{summary['conflict_correct']}/{summary['conflict_total']} conflict pairs")
    return summary["conflict_matching"]


def multi_crit_aggregate_completeness(
    results: list[dict[str, Any]],
) -> float | None:
    """Aggregate open-ended completeness accuracy."""

    return _aggregate_named_metric(results, "completeness")


def multi_crit_aggregate_visual_grounding(
    results: list[dict[str, Any]],
) -> float | None:
    """Aggregate open-ended visual-grounding accuracy."""

    return _aggregate_named_metric(results, "visual-grounding")


def multi_crit_aggregate_hallucination(
    results: list[dict[str, Any]],
) -> float | None:
    """Aggregate hallucination accuracy for either benchmark split."""

    return _aggregate_named_metric(results, "hallucination")


def multi_crit_aggregate_expressiveness(
    results: list[dict[str, Any]],
) -> float | None:
    """Aggregate open-ended expressiveness accuracy."""

    return _aggregate_named_metric(results, "expressiveness")


def multi_crit_aggregate_clarity(
    results: list[dict[str, Any]],
) -> float | None:
    """Aggregate open-ended clarity accuracy."""

    return _aggregate_named_metric(results, "clarity")


def multi_crit_aggregate_grounding(
    results: list[dict[str, Any]],
) -> float | None:
    """Aggregate reasoning grounding accuracy."""

    return _aggregate_named_metric(results, "grounding")


def multi_crit_aggregate_logic(results: list[dict[str, Any]]) -> float | None:
    """Aggregate reasoning logic accuracy."""

    return _aggregate_named_metric(results, "logic")


def multi_crit_aggregate_exploration(
    results: list[dict[str, Any]],
) -> float | None:
    """Aggregate reasoning exploration accuracy."""

    return _aggregate_named_metric(results, "exploration")


def multi_crit_aggregate_efficiency(
    results: list[dict[str, Any]],
) -> float | None:
    """Aggregate reasoning efficiency accuracy."""

    return _aggregate_named_metric(results, "efficiency")
