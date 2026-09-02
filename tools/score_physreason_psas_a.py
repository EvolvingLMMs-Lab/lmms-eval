"""Score lmms-eval PhysReason samples with the paper's PSAS-A protocol.

PSAS-A uses an LLM twice for every sub-question: first to extract the model's
answer from its complete response, and then to judge semantic equivalence with
the reference answer. Per-problem scores weight sub-questions by the number of
annotated solution steps. The final score is the mean across problems.

The output JSONL is append-only and resumable. Paid inference is never started
unless ``--confirm-paid-inference`` is supplied explicitly.

Example:
    uv run python tools/score_physreason_psas_a.py \
      --samples outputs/.../samples_physreason.jsonl \
      --provider deepinfra \
      --model deepseek-ai/DeepSeek-V3 \
      --confirm-paid-inference
"""

import argparse
import json
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import InferenceClient

EXTRACTION_SYSTEM_PROMPT = "You are a professional answer extraction assistant. Please only return the extracted answer without adding any additional explanations."
JUDGE_SYSTEM_PROMPT = "You are a professional mathematical problem answer evaluation assistant"

_thread_local = threading.local()


def _response_text(response) -> str:
    content = response.choices[0].message.content
    return content.strip() if content else ""


def _usage(response) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def _merge_usage(*usages: dict[str, int]) -> dict[str, int]:
    return {key: sum(usage.get(key, 0) for usage in usages) for key in ("prompt_tokens", "completion_tokens")}


def _get_client(model: str, provider: str, timeout: float) -> InferenceClient:
    key = (model, provider, timeout)
    if getattr(_thread_local, "client_key", None) != key:
        _thread_local.client = InferenceClient(model=model, provider=provider, timeout=timeout)
        _thread_local.client_key = key
    return _thread_local.client


def _chat_with_retry(
    *,
    model: str,
    provider: str,
    timeout: float,
    retries: int,
    system_prompt: str,
    user_prompt: str,
):
    last_error = None
    for attempt in range(retries):
        try:
            client = _get_client(model, provider, timeout)
            return client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as error:  # Network/provider failures are retried.
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Judge request failed after {retries} attempts: {last_error}") from last_error


def _step_weights(problem: dict) -> list[int]:
    counts = Counter(step["sub_question"] for step in problem["explanation_steps"])
    weights = [counts.get(f"sub_question_{index}", 0) for index in range(1, len(problem["sub_questions"]) + 1)]
    if any(weight == 0 for weight in weights):
        raise ValueError(f"Every PhysReason sub-question must have at least one annotated solution step: {weights}")
    return weights


def _score_sub_question(job: dict, args: argparse.Namespace) -> dict:
    sub_question_number = job["sub_question_index"] + 1
    extraction_prompt = f"""Please extract the answer for the specific question from the following output text.
Specific question:
{job["question"]}

Output text:
{job["prediction"]}

Please return the answer directly without any explanation or additional text. The answer is usually after 'sub_question_{sub_question_number}_answer:'."""

    extraction_response = _chat_with_retry(
        model=args.model,
        provider=args.provider,
        timeout=args.timeout,
        retries=args.retries,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=extraction_prompt,
    )
    extracted_answer = _response_text(extraction_response)
    extraction_usage = _usage(extraction_response)

    judge_response_text = ""
    judge_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    correct = False
    if extracted_answer:
        judge_prompt = f"""Based on the following information, please determine whether the two answers are semantically equivalent:
Specific question:
{job["question"]}
Actual answer:
{extracted_answer}
Expected answer:
{job["expected_answer"]}
Please only answer "true" or "false" to indicate whether these two answers express the same meaning. When evaluating, please consider whether mathematical expressions, units, and other details are equivalent."""
        judge_response = _chat_with_retry(
            model=args.model,
            provider=args.provider,
            timeout=args.timeout,
            retries=args.retries,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=judge_prompt,
        )
        judge_response_text = _response_text(judge_response)
        judge_usage = _usage(judge_response)
        # Match the authors' released evaluator exactly.
        correct = judge_response_text.lower() == "true"

    return {
        "doc_id": job["doc_id"],
        "problem_id": job["problem_id"],
        "difficulty": job["difficulty"],
        "sub_question_index": job["sub_question_index"],
        "question": job["question"],
        "expected_answer": job["expected_answer"],
        "extracted_answer": extracted_answer,
        "judge_response": judge_response_text,
        "correct": correct,
        "weight": job["weight"],
        "judge_model": args.model,
        "judge_provider": args.provider,
        "usage": _merge_usage(extraction_usage, judge_usage),
    }


def _load_existing(path: Path, *, model: str, provider: str) -> dict[tuple[int, int], dict]:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("judge_model") != model or record.get("judge_provider") != provider:
                raise ValueError(f"{path}:{line_number} was produced by a different judge model/provider")
            records[(int(record["doc_id"]), int(record["sub_question_index"]))] = record
    return records


def _build_jobs(samples_path: Path) -> tuple[list[dict], dict[int, dict]]:
    dataset = load_dataset("lmms-lab-eval/PhysReason", "full", split="test").select_columns(["problem_id", "difficulty", "sub_questions", "answers", "explanation_steps"])
    samples = {}
    with samples_path.open(encoding="utf-8") as file:
        for line in file:
            sample = json.loads(line)
            doc_id = int(sample["doc_id"])
            if doc_id in samples:
                raise ValueError(f"Duplicate PhysReason doc_id in samples: {doc_id}")
            samples[doc_id] = sample

    if set(samples) != set(range(len(dataset))):
        missing = sorted(set(range(len(dataset))) - set(samples))
        extra = sorted(set(samples) - set(range(len(dataset))))
        raise ValueError(f"Samples must contain the complete PhysReason split; missing={missing[:10]}, extra={extra[:10]}")

    jobs = []
    problems = {}
    for doc_id in range(len(dataset)):
        problem = dataset[doc_id]
        sample = samples[doc_id]
        prediction = sample["filtered_resps"]
        if isinstance(prediction, list):
            prediction = prediction[0] if prediction else ""
        weights = _step_weights(problem)
        problems[doc_id] = {"problem_id": problem["problem_id"], "difficulty": problem["difficulty"], "weights": weights}
        for sub_question_index, (question, expected_answer, weight) in enumerate(zip(problem["sub_questions"], problem["answers"], weights, strict=True)):
            jobs.append(
                {
                    "doc_id": doc_id,
                    "problem_id": problem["problem_id"],
                    "difficulty": problem["difficulty"],
                    "sub_question_index": sub_question_index,
                    "question": question,
                    "expected_answer": expected_answer,
                    "prediction": prediction,
                    "weight": weight,
                }
            )
    return jobs, problems


def _summarize(records: dict[tuple[int, int], dict], problems: dict[int, dict], expected_records: int) -> dict:
    scores_by_difficulty = defaultdict(list)
    problem_scores = {}
    for doc_id, problem in problems.items():
        weighted_correct = 0
        total_weight = sum(problem["weights"])
        complete = True
        for sub_question_index, weight in enumerate(problem["weights"]):
            record = records.get((doc_id, sub_question_index))
            if record is None:
                complete = False
                break
            weighted_correct += weight * int(record["correct"])
        if complete:
            score = weighted_correct / total_weight
            problem_scores[doc_id] = score
            scores_by_difficulty[problem["difficulty"]].append(score)

    difficulty_scores = {difficulty: 100 * sum(scores) / len(scores) if scores else None for difficulty, scores in sorted(scores_by_difficulty.items())}
    usage = {key: sum(record.get("usage", {}).get(key, 0) for record in records.values()) for key in ("prompt_tokens", "completion_tokens")}
    return {
        "complete": len(records) == expected_records and len(problem_scores) == len(problems),
        "records": len(records),
        "expected_records": expected_records,
        "problems_scored": len(problem_scores),
        "expected_problems": len(problems),
        "psas_a": 100 * sum(problem_scores.values()) / len(problem_scores) if problem_scores else None,
        "by_difficulty": difficulty_scores,
        "usage": usage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True, help="Complete lmms-eval PhysReason samples JSONL")
    parser.add_argument("--output", type=Path, help="Resumable per-sub-question JSONL (defaults beside samples)")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3", help="Paper scoring model")
    parser.add_argument("--provider", default="deepinfra", help="Hugging Face Inference Provider")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Run only this many pending sub-questions (for a paid smoke test)")
    parser.add_argument(
        "--confirm-paid-inference",
        action="store_true",
        help="Required acknowledgment that remote judge calls can incur charges",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.retries < 1:
        raise ValueError("--workers and --retries must be positive")
    output_path = args.output or args.samples.with_name(f"{args.samples.stem}_psas_a.jsonl")
    summary_path = output_path.with_suffix(".summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jobs, problems = _build_jobs(args.samples)
    records = _load_existing(output_path, model=args.model, provider=args.provider)
    pending = [job for job in jobs if (job["doc_id"], job["sub_question_index"]) not in records]
    if args.limit is not None:
        pending = pending[: args.limit]

    if pending and not args.confirm_paid_inference:
        raise SystemExit(f"Refusing to start {len(pending)} paid judge jobs without --confirm-paid-inference. Existing cached jobs: {len(records)}/{len(jobs)}")

    completed = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_score_sub_question, job, args): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            try:
                record = future.result()
            except Exception as error:
                failures.append((job, error))
                print(f"FAILED {job['problem_id']} sub_question_{job['sub_question_index'] + 1}: {error}")
                continue
            key = (record["doc_id"], record["sub_question_index"])
            records[key] = record
            with output_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                print(f"Scored {completed}/{len(pending)} pending sub-questions ({len(records)}/{len(jobs)} total)")

    summary = _summarize(records, problems, len(jobs))
    summary.update({"judge_model": args.model, "judge_provider": args.provider, "failed_jobs": len(failures)})
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
