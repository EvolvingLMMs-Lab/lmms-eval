from typing import Any, Dict, List, Optional


def _coerce_score(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        nested = value.get("score")
        if isinstance(nested, (int, float)):
            return float(nested)
    return None


def _extract_score(sample: Dict[str, Any], score_key: str) -> Optional[float]:
    primary = _coerce_score(sample.get(score_key))
    if primary is not None:
        return primary

    fallback_keys = ["score", "acc", "accuracy", "exact_match"]
    fallback_keys.extend([k for k in sample.keys() if isinstance(k, str) and k.endswith("_score")])
    for key in fallback_keys:
        score = _coerce_score(sample.get(key))
        if score is not None:
            return score

    return None


def _summarize_task_samples(samples: List[Dict[str, Any]], score_key: str) -> Dict[str, Any]:
    total_input_tokens = 0.0
    total_output_tokens = 0.0
    total_score = 0.0
    docs_with_tokens = 0

    for sample in samples:
        token_entries = sample.get("token_counts")
        sample_input = 0.0
        sample_output = 0.0
        has_token_data = False

        if isinstance(token_entries, list):
            for entry in token_entries:
                if not isinstance(entry, dict):
                    continue
                sample_input += float(entry.get("input_tokens") or 0)
                sample_output += float(entry.get("output_tokens") or 0)
                has_token_data = True

        if has_token_data:
            docs_with_tokens += 1
            total_input_tokens += sample_input
            total_output_tokens += sample_output

        score = _extract_score(sample, score_key)
        if score is not None and score > 0:
            total_score += score

    total_tokens = total_input_tokens + total_output_tokens
    avg_output = total_output_tokens / docs_with_tokens if docs_with_tokens > 0 else 0.0
    tokens_per_correct = (total_output_tokens / total_score) if total_score > 0 else None

    return {
        "docs": float(len(samples)),
        "docs_with_token_counts": float(docs_with_tokens),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_correct_score": total_score,
        "avg_output_tokens_per_sample": avg_output,
        "tokens_per_correct_answer": tokens_per_correct,
    }


def build_efficiency_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    samples_by_task = results.get("samples")
    if not isinstance(samples_by_task, dict) or not samples_by_task:
        return {}

    by_task: Dict[str, Any] = {}
    overall: Dict[str, Any] = {
        "docs": 0.0,
        "docs_with_token_counts": 0.0,
        "total_input_tokens": 0.0,
        "total_output_tokens": 0.0,
        "total_tokens": 0.0,
        "total_correct_score": 0.0,
    }

    configs_obj = results.get("configs")
    configs = configs_obj if isinstance(configs_obj, dict) else {}

    for task_name, task_samples in samples_by_task.items():
        if not isinstance(task_samples, list):
            continue

        task_config_obj = configs.get(task_name)
        task_config = task_config_obj if isinstance(task_config_obj, dict) else {}
        score_key = str(task_config.get("score_key") or "score")

        task_summary = _summarize_task_samples(task_samples, score_key)
        by_task[task_name] = task_summary

        overall["docs"] += task_summary["docs"]
        overall["docs_with_token_counts"] += task_summary["docs_with_token_counts"]
        overall["total_input_tokens"] += task_summary["total_input_tokens"]
        overall["total_output_tokens"] += task_summary["total_output_tokens"]
        overall["total_tokens"] += task_summary["total_tokens"]
        overall["total_correct_score"] += task_summary["total_correct_score"]

    docs_with_tokens = overall["docs_with_token_counts"]
    overall["avg_output_tokens_per_sample"] = (overall["total_output_tokens"] / docs_with_tokens) if docs_with_tokens > 0 else 0.0
    overall["tokens_per_correct_answer"] = (overall["total_output_tokens"] / overall["total_correct_score"]) if overall["total_correct_score"] > 0 else None

    return {
        "by_task": by_task,
        "overall": overall,
    }
