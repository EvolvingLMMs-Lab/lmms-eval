import os
import re
import threading
from typing import Any, Mapping, Sequence

from datasets import Dataset
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference
from lmms_eval.verifiers import VerificationPipeline, VerifyResult
from lmms_eval.verifiers.extractors import StripReasoningExtractor
from lmms_eval.verifiers.openai import OpenAIVerifier

Document = Mapping[str, Any]
TaskKwargs = Mapping[str, Any]
MetricRecord = dict[str, Any]

CHOICES = ("A", "B", "C", "D")

# Reasoning-type spellings exactly as they appear in VRBench_eval.jsonl.
# "Counting Porblems" is misspelled in the released annotations; matching it
# verbatim is required, so only the metric slug fixes the spelling.
REASONING_TYPE_SLUGS = {
    "Event Attribution": "event_attribution",
    "Hypothetical Reasoning": "hypothetical_reasoning",
    "Event Summarization": "event_summarization",
    "Implicit Inference": "implicit_inference",
    "Event Prediction": "event_prediction",
    "Counting Porblems": "counting_problems",
    "Logical Linkage": "logical_linkage",
}

# Judge routing, reproduced from VRBench/evaluation/run_process_eval.py.
# "Multi-element Inference" is kept for parity with the official list even
# though no released annotation carries it.
UNIQUE_ANSWER_TYPES = frozenset({"Event Attribution", "Multi-element Inference", "Implicit Inference", "Logical Linkage"})
NON_UNIQUE_ANSWER_TYPES = frozenset({"Hypothetical Reasoning", "Event Prediction"})

# The official judge truncates the narrative summary before prompting.
VIDEO_SUMMARY_CHAR_LIMIT = 10000

JUDGE_MODEL = os.getenv("MODEL_VERSION", "deepseek-chat")

# ---------------------------------------------------------------------------
# Prompts — ported verbatim from the official VRBench repository
# ---------------------------------------------------------------------------

# VRBench/inference/utils/constant.py :: MCQ_COT_PROMPT
MCQ_COT_PROMPT = """
You are a helpful video understanding assistant that answers multi-choice questions through step-by-step reasoning based on the video and its summary.

# Instructions:
1. Break down the reasoning process into clear, specific events.
2. Conclude with the best option letter(A/B/C/D) at last.

# Output Format:
<Step 1> Description of event/observation
<Step 2> Description of event/observation
...
<Answer> [Option letter]

# Multiple Choice Question
{multiple_choice_question}

# Video Summary
{video_summary}
"""

# VRBench/evaluation/model_api/prompt.py :: UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT
UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT = """
You are a reasoning process evaluation model. Given the question, your task is to compare the model's reasoning process with the correct reasoning process provided, and assess the accuracy of the model's reasoning.
Based on the number of correct steps and the overall correctness, give a score between 0 and 10, where 10 means fully correct.

Evaluation Criteria:
    1. **Step-by-Step Match**: How closely each reasoning step aligns with the ground truth process. Highest weight (40%).
    2. **Logical Integrity**: Whether the reasoning maintains valid logical progression and complete argumentation (30%).
    3. **Factual Correctness**: Absence of factual errors conflicting with established truths (20%).
    4. **Process Clarity**: Clear articulation and organization of reasoning steps (10%).

    Scoring:
    - **0-3**: Multiple missing/critical deviations from correct steps (≤30% match), broken logic, severe factual errors, or incoherent presentation.
    - **4-6**: Partial step alignment (40-60% match), basic logical structure with gaps, minor factual slips, or ambiguous explanations.
    - **7-9**: Majority steps correct (70-90% match), sound logic with minor jumps, near-perfect factual accuracy, and clear presentation.
    - **10**: Full step correspondence (100% match), flawless logic, perfect factual accuracy, and exceptionally clear reasoning flow.

Please provide the reasons for your scoring at the end.
Output Format:
<rate>the score (0-10)</rate>.
<reason>Briefly explain the reason for the score.</reason>
"""

# VRBench/evaluation/model_api/prompt.py :: NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT
NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT = """
You are a reasoning process evaluation model. Given the question, your task is to evaluate the model's reasoning process with the correct reasoning process provided, and assess the accuracy of the model's reasoning.
In addition to referring to the provided reasoning process and result, you may also assess the reasoning's validity based on the video summary. If the reasoning is logical and the result is reasonable, you can adjust the score accordingly.
Based on the correctness and reasonableness of the reasoning process, provide a score between 0 and 10, where 10 means fully correct and reasonable.

Evaluation Criteria:
    1. **Relevance and Completeness**: Evaluate whether the reasoning process adequately addresses the question and covers all essential steps, even if the approach differs from the provided standard. (40%)
    2. **Logical Consistency**: Assess the logical progression, coherence, and structural integrity of the reasoning process (30%).
    3. **Factual Accuracy**: Check for correctness and the absence of significant factual errors (20%).
    4. **Clarity and Persuasiveness**: Consider the clarity, organization, and persuasiveness of the reasoning, including the explanation of alternative valid approaches (10%).

Scoring:
    - **0-3**: The reasoning process shows significant omissions, major logical inconsistencies, or severe factual errors, resulting in an unclear and unconvincing explanation.
    - **4-6**: The reasoning process partially addresses the question with some logical or factual issues, and the explanation may be somewhat ambiguous or incomplete.
    - **7-9**: The reasoning process is largely relevant and logically consistent, with minor issues in clarity or factual details, leading to a well-argued explanation.
    - **10**: The reasoning process fully addresses the question with impeccable logic, complete factual accuracy, and is presented in a clear and highly persuasive manner.

Output Format:
<rate>the score (0-10)</rate>.
<reason>Briefly explain the reason for the score.</reason>
"""

# VRBench/evaluation/model_api/prompt.py :: UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE
UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE = """
# Question
{question}
# Model's reasoning process and Answer
{response}
# Correct reasoning step and Answer
Reasoning Step:
{procedure}
Answer:
{answer}
Please provide your rating and brief reasons.
"""

# VRBench/evaluation/model_api/prompt.py :: NON_UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE
NON_UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE = """
# Video Summary
{video_summary}
# Question
{question}
# Model's reasoning process and Answer
{response}
# Correct reasoning step and Answer
Answer:
{answer}
Reasoning Step:
{procedure}
Please provide your rating and brief reasons.
"""


# ---------------------------------------------------------------------------
# Dataset flattening
# ---------------------------------------------------------------------------


def _qa_sort_key(qa_key: str) -> tuple[int, int, str]:
    """Order the ``qa1``…``qaN`` keys numerically instead of lexicographically."""
    match = re.fullmatch(r"qa(\d+)", qa_key)
    if match:
        return (0, int(match.group(1)), "")
    return (1, 0, qa_key)


def format_reasoning_process(procedure: Any) -> str:
    """Render the annotated reasoning steps as the official ``<Step N> …`` block."""
    if isinstance(procedure, str):
        return procedure
    if not procedure:
        return ""
    steps = sorted(procedure.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 0)
    return "\n".join(f"<Step {number}> {description}" for number, description in steps)


def vrbench_process_docs(dataset: Dataset) -> Dataset:
    """Flatten one annotation record per video into one document per question.

    ``VRBench_eval.jsonl`` stores 960 videos, each carrying an ``mcq`` mapping of
    6-10 questions, for 8,243 questions in total.  lmms-eval scores one document
    per request, so the mapping is expanded here and the video-level fields are
    copied onto every question.
    """
    flattened: list[dict[str, Any]] = []
    for record in dataset:
        video_id = record["video_id"]
        video_summary = record.get("video_summary") or ""
        for qa_key in sorted(record["mcq"], key=_qa_sort_key):
            qa = record["mcq"][qa_key]
            flattened.append(
                {
                    "question_id": f"{video_id}_{qa_key}",
                    "qa_key": qa_key,
                    "video_id": video_id,
                    "video_path": record["video_path"],
                    "video_read_type": record.get("video_read_type") or "",
                    "video_summary": video_summary,
                    "question": qa["question"],
                    "options": qa["options"],
                    "answer": qa["answer"],
                    "original_answer": qa.get("original_answer") or "",
                    "reasoning_process": format_reasoning_process(qa.get("reasoning_process")),
                    "reasoning_type": qa["reasoning_type"],
                }
            )
    eval_logger.info(f"VRBench: flattened {len(dataset)} videos into {len(flattened)} questions.")
    return Dataset.from_list(flattened)


# ---------------------------------------------------------------------------
# Visual / text doc helpers
# ---------------------------------------------------------------------------


def _resolve_video(doc: Document) -> str:
    return resolve_media_reference(
        doc["video_path"],
        media_type="video",
        cache_dir="vrbench",
        env_vars=("VRBENCH_VIDEO_DIR", "VRBENCH_ROOT"),
        extra_subdirs=("videos", "videos/v001", "VRBench/videos", "VRBench/videos/v001"),
    )


def vrbench_doc_to_visual(doc: Document) -> list[str]:
    """Resolve the local video for one VRBench question."""
    video_path = _resolve_video(doc)
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"VRBench video not found: {video_path}. Download the split archives from " "https://huggingface.co/datasets/OpenGVLab/VRBench, extract them, and point " "VRBENCH_VIDEO_DIR at the directory that holds the .mp4 files."
        )
    return [video_path]


def _format_question(doc: Document) -> str:
    """Render the question and its options in the official ``dict_to_text`` layout."""
    options = doc["options"]
    keys = [key for key in CHOICES if key in options] + [key for key in sorted(options) if key not in CHOICES]
    option_prompt = "\n".join(f"{key}: {options[key]}" for key in keys)
    return f"Question: {doc['question']}\nOptions:\n{option_prompt}"


def vrbench_doc_to_text(doc: Document, lmms_eval_specific_kwargs: TaskKwargs | None = None) -> str:
    """Render the official step-by-step VRBench prompt for one question."""
    kwargs = lmms_eval_specific_kwargs or {}
    video_summary = doc.get("video_summary") or "" if kwargs.get("include_video_summary", True) else ""
    prompt = MCQ_COT_PROMPT.format(multiple_choice_question=_format_question(doc), video_summary=video_summary)
    return f"{kwargs.get('pre_prompt', '')}{prompt}{kwargs.get('post_prompt', '')}"


# ---------------------------------------------------------------------------
# Rule-based MCQ extraction — ported from
# VRBench/evaluation/calculate_scores.py :: extract_mcq_answer
# ---------------------------------------------------------------------------

_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCTUATION = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

_ANSWER_PATTERNS = [
    r"Answer[:\s]+([A-E])(?:\s|$|\.|,)",
    r"answer is[:\s]+([A-E])(?:\s|$|\.|,)",
    r"correct answer[:\s]+([A-E])(?:\s|$|\.|,)",
    r"final answer[:\s]+([A-E])(?:\s|$|\.|,)",
    r"final[:\s]+([A-E])(?:\s|$|\.|,)",
    r"therefore[:\s]+([A-E])(?:\s|$|\.|,)",
    r"conclusion[:\s]+([A-E])(?:\s|$|\.|,)",
]

_OPTION_PATTERNS = [r"([A-E])\.", r"([A-E])\)", r"([A-E]):"]


def _process_punctuation(text: str) -> str:
    """Strip punctuation the way the official scorer does before the final scan."""
    output = text
    for punctuation in _PUNCTUATION:
        if (punctuation + " " in text or " " + punctuation in text) or (re.search(_COMMA_STRIP, text) is not None):
            output = output.replace(punctuation, "")
        else:
            output = output.replace(punctuation, " ")
    return _PERIOD_STRIP.sub("", output, re.UNICODE)


def extract_mcq_answer(response_text: str) -> str | None:
    """Extract the chosen option letter from a model response.

    Faithful port of the regex cascade in
    ``VRBench/evaluation/calculate_scores.py``.  Each stage returns the *last*
    match it finds, which keeps the final answer of a long chain-of-thought
    response instead of an option discussed on the way there.

    Args:
        response_text: Raw model output.

    Returns:
        An uppercase option letter, or ``None`` when no letter can be recovered.
    """
    if not response_text:
        return None

    # 1. \boxed{A}
    boxed_match = re.search(r"\\boxed\{([A-E])\}", response_text, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).upper()

    # 2. <Answer> A
    answer_match = re.search(r"<Answer>\s*([A-E])", response_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    # 3. A line that starts with "A. <option text>"
    option_format_matches = re.findall(r"^([A-E])\.\s*(.+)$", response_text.strip(), re.IGNORECASE | re.MULTILINE)
    if option_format_matches:
        return option_format_matches[-1][0].upper()

    # 4. Explicit answer statements
    all_answer_matches: list[str] = []
    for pattern in _ANSWER_PATTERNS:
        all_answer_matches.extend(re.findall(pattern, response_text, re.IGNORECASE))
    if all_answer_matches:
        return all_answer_matches[-1].upper()

    # 5. Bare option labels
    for pattern in _OPTION_PATTERNS:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # 6. A standalone letter followed by a space
    matches = re.findall(r"(?:^|\n|\s)([A-E])\s+(?=[A-Z]|$|\n)", response_text, re.IGNORECASE | re.MULTILINE)
    if matches:
        return matches[-1].upper()

    # 7. Punctuation-stripped rescan
    processed_text = response_text.replace("\n", " ").replace("\t", " ").strip()
    processed_text = _process_punctuation(processed_text)
    processed_text = processed_text.strip("'").strip('"').strip(")").strip("(").strip().lower()
    processed_letters = re.findall(r"\b([A-E])\b", processed_text, re.IGNORECASE)
    if processed_letters:
        return processed_letters[-1].upper()

    # 8. Fallback: prefer a letter near the end of the response
    choices = re.findall(r"\b([A-E])\b", response_text)
    if choices:
        end_choices = re.findall(r"\b([A-E])\b", response_text.lower()[-50:], re.IGNORECASE)
        if end_choices:
            return end_choices[-1].upper()
        return choices[-1].upper()

    return None


# ---------------------------------------------------------------------------
# MCQ track — process_results / aggregations
# ---------------------------------------------------------------------------


def vrbench_mcq_process_results(doc: Document, results: Sequence[str]) -> dict[str, MetricRecord]:
    """Parse a prediction and emit the VRBench MCQ accuracy record."""
    prediction = results[0] if results else ""
    parsed_prediction = extract_mcq_answer(prediction)
    answer = str(doc["answer"]).strip().upper()
    reasoning_type = str(doc["reasoning_type"])
    record: MetricRecord = {
        "question_id": doc["question_id"],
        "video_id": doc["video_id"],
        "reasoning_type": reasoning_type,
        "prediction": prediction,
        "parsed_prediction": parsed_prediction,
        "answer": answer,
        "score": float(parsed_prediction == answer),
    }
    metrics: dict[str, MetricRecord] = {"vrbench_score": record}
    slug = REASONING_TYPE_SLUGS.get(reasoning_type)
    if slug is not None:
        metrics[f"vrbench_mcq_{slug}_accuracy"] = record
    return metrics


def _aggregate_mcq(results: Sequence[MetricRecord], reasoning_type: str | None = None) -> float:
    """Compute question-weighted MCQ accuracy as a percentage."""
    selected = [result for result in results if reasoning_type is None or result["reasoning_type"] == reasoning_type]
    if not selected:
        return 0.0
    correct = sum(result["score"] for result in selected)
    accuracy = 100.0 * correct / len(selected)
    eval_logger.info(f"VRBench MCQ {reasoning_type or 'overall'} accuracy: {accuracy:.2f}% ({int(correct)}/{len(selected)})")
    return accuracy


def vrbench_mcq_aggregate_results(results: Sequence[MetricRecord]) -> float:
    """Aggregate overall VRBench MCQ accuracy."""
    return _aggregate_mcq(results)


def vrbench_mcq_aggregate_event_attribution(results: Sequence[MetricRecord]) -> float:
    """Aggregate MCQ accuracy for the Event Attribution questions."""
    return _aggregate_mcq(results, "Event Attribution")


def vrbench_mcq_aggregate_hypothetical_reasoning(results: Sequence[MetricRecord]) -> float:
    """Aggregate MCQ accuracy for the Hypothetical Reasoning questions."""
    return _aggregate_mcq(results, "Hypothetical Reasoning")


def vrbench_mcq_aggregate_event_summarization(results: Sequence[MetricRecord]) -> float:
    """Aggregate MCQ accuracy for the Event Summarization questions."""
    return _aggregate_mcq(results, "Event Summarization")


def vrbench_mcq_aggregate_implicit_inference(results: Sequence[MetricRecord]) -> float:
    """Aggregate MCQ accuracy for the Implicit Inference questions."""
    return _aggregate_mcq(results, "Implicit Inference")


def vrbench_mcq_aggregate_event_prediction(results: Sequence[MetricRecord]) -> float:
    """Aggregate MCQ accuracy for the Event Prediction questions."""
    return _aggregate_mcq(results, "Event Prediction")


def vrbench_mcq_aggregate_counting_problems(results: Sequence[MetricRecord]) -> float:
    """Aggregate MCQ accuracy for the counting questions."""
    return _aggregate_mcq(results, "Counting Porblems")


def vrbench_mcq_aggregate_logical_linkage(results: Sequence[MetricRecord]) -> float:
    """Aggregate MCQ accuracy for the Logical Linkage questions."""
    return _aggregate_mcq(results, "Logical Linkage")


# ---------------------------------------------------------------------------
# Process track — LLM judge
# ---------------------------------------------------------------------------

_RATE_PATTERN = re.compile(r"<rate>(.*?)</rate>", re.DOTALL)
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_rate_response(text: str) -> VerifyResult:
    """Parse the ``<rate>0-10</rate>`` block returned by the VRBench judge."""
    rate = 0.0
    parsed = False
    match = _RATE_PATTERN.search(text or "")
    if match:
        numbers = _NUMBER_PATTERN.findall(match.group(1))
        if numbers:
            rate = max(0.0, min(10.0, float(numbers[0])))
            parsed = True
    return VerifyResult(score=rate / 10.0, is_correct=rate >= 5.0, raw_output=text, metadata={"rate": rate, "rate_parsed": parsed})


def _unique_judge_prompt(question: str, prediction: str, ground_truth: str, **kwargs: Any) -> str:
    """Build the judge prompt for reasoning types that have a unique answer."""
    human_prompt = UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE.format(question=question, response=prediction, procedure=kwargs.get("procedure", ""), answer=ground_truth)
    return f"{UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT}{human_prompt}"


def _non_unique_judge_prompt(question: str, prediction: str, ground_truth: str, **kwargs: Any) -> str:
    """Build the judge prompt for reasoning types that accept several answers."""
    human_prompt = NON_UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE.format(
        video_summary=kwargs.get("video_summary", ""),
        question=question,
        response=prediction,
        procedure=kwargs.get("procedure", ""),
        answer=ground_truth,
    )
    return f"{NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT}{human_prompt}"


_pipelines: dict[str, VerificationPipeline] = {}
_pipeline_lock = threading.Lock()


def _get_pipeline(route: str) -> VerificationPipeline:
    """Return the lazily built judge pipeline for ``unique`` or ``non_unique``."""
    pipeline = _pipelines.get(route)
    if pipeline is None:
        with _pipeline_lock:
            pipeline = _pipelines.get(route)
            if pipeline is None:  # double-check
                pipeline = VerificationPipeline(
                    extractors=[StripReasoningExtractor()],
                    verifier=OpenAIVerifier(
                        model=JUDGE_MODEL,
                        custom_prompt=_unique_judge_prompt if route == "unique" else _non_unique_judge_prompt,
                        response_parser=_parse_rate_response,
                        max_retries=3,
                        retry_delay=2.0,
                        max_tokens=1024,
                    ),
                )
                _pipelines[route] = pipeline
    return pipeline


def vrbench_process_process_results(doc: Document, results: Sequence[str]) -> dict[str, MetricRecord]:
    """Score one reasoning trace with the VRBench process judge.

    Event Summarization and any reasoning type outside the official routing
    table are left unjudged, matching ``run_process_eval.py``.  Their records
    are still emitted so the sample log stays complete, but the aggregation
    skips them.
    """
    prediction = results[0] if results else ""
    reasoning_type = str(doc["reasoning_type"])
    record: MetricRecord = {
        "question_id": doc["question_id"],
        "video_id": doc["video_id"],
        "reasoning_type": reasoning_type,
        "prediction": prediction,
        "judged": False,
        "score": 0.0,
        "judge_output": "",
    }

    if reasoning_type in UNIQUE_ANSWER_TYPES:
        route = "unique"
        judge_kwargs: dict[str, Any] = {"procedure": doc["reasoning_process"]}
    elif reasoning_type in NON_UNIQUE_ANSWER_TYPES:
        route = "non_unique"
        judge_kwargs = {"procedure": doc["reasoning_process"], "video_summary": (doc.get("video_summary") or "")[:VIDEO_SUMMARY_CHAR_LIMIT]}
    else:
        return {"vrbench_score": record}

    result = _get_pipeline(route)(question=doc["question"], prediction=prediction, ground_truth=doc["original_answer"], **judge_kwargs)
    if result.metadata.get("judge_failed"):
        eval_logger.warning(f"VRBench process judge failed for {record['question_id']}; the question is excluded.")
        return {"vrbench_score": record}

    record["judged"] = True
    record["score"] = float(result.metadata.get("rate", 0.0))
    record["judge_output"] = result.raw_output

    metrics: dict[str, MetricRecord] = {"vrbench_score": record}
    slug = REASONING_TYPE_SLUGS.get(reasoning_type)
    if slug is not None:
        metrics[f"vrbench_process_{slug}_score"] = record
    return metrics


def _aggregate_process(results: Sequence[MetricRecord], reasoning_type: str | None = None) -> float:
    """Average the judged 0-10 process scores and rescale them to 0-100."""
    selected = [result for result in results if result["judged"] and (reasoning_type is None or result["reasoning_type"] == reasoning_type)]
    if not selected:
        return 0.0
    mean_rate = sum(result["score"] for result in selected) / len(selected)
    eval_logger.info(f"VRBench process {reasoning_type or 'overall'} score: {mean_rate:.2f}/10 ({len(selected)} judged)")
    return 10.0 * mean_rate


def vrbench_process_aggregate_results(results: Sequence[MetricRecord]) -> float:
    """Aggregate the overall VRBench process score."""
    return _aggregate_process(results)


def vrbench_process_aggregate_event_attribution(results: Sequence[MetricRecord]) -> float:
    """Aggregate the process score for the Event Attribution questions."""
    return _aggregate_process(results, "Event Attribution")


def vrbench_process_aggregate_hypothetical_reasoning(results: Sequence[MetricRecord]) -> float:
    """Aggregate the process score for the Hypothetical Reasoning questions."""
    return _aggregate_process(results, "Hypothetical Reasoning")


def vrbench_process_aggregate_implicit_inference(results: Sequence[MetricRecord]) -> float:
    """Aggregate the process score for the Implicit Inference questions."""
    return _aggregate_process(results, "Implicit Inference")


def vrbench_process_aggregate_event_prediction(results: Sequence[MetricRecord]) -> float:
    """Aggregate the process score for the Event Prediction questions."""
    return _aggregate_process(results, "Event Prediction")


def vrbench_process_aggregate_logical_linkage(results: Sequence[MetricRecord]) -> float:
    """Aggregate the process score for the Logical Linkage questions."""
    return _aggregate_process(results, "Logical Linkage")
