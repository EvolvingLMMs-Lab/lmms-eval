"""
ROVER Evaluator

Main evaluation logic for:
- reasoning_process: Quality of reasoning text (think_output)
- reasoning_visual: Quality of generated images
- reasoning_alignment: Alignment between reasoning text and generated images
"""

import json
import re
import logging
import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from PIL import Image

from .api import call_gpt4o_with_images
from .prompts import (
    get_reasoning_process_prompt,
    get_reasoning_visual_prompt,
    get_reasoning_alignment_prompt,
)


# Available metrics
METRICS = ["reasoning_process", "reasoning_visual", "reasoning_alignment"]


@dataclass
class EvaluationSample:
    """A single sample for evaluation."""
    sample_id: str
    original_image: Union[str, Image.Image]  # Path or PIL Image
    generated_image: Union[str, Image.Image]  # Path or PIL Image
    target_image: Optional[Union[str, Image.Image]] = None  # Optional target reference
    prompt: str = ""  # Task instruction
    dimension: str = ""  # Knowledge domain
    reasoning_type: str = "temporal"  # Type of reasoning
    keywords: str = ""  # Domain concepts
    target_description: str = ""  # Expected outcomes
    think_output: str = ""  # Model's reasoning text


def extract_score_and_reason(
    response: str,
    score_key: str,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Extract score and reasoning from GPT response.

    Args:
        response: GPT response text
        score_key: Key to look for (e.g., "reasoning_process_score")

    Returns:
        Tuple of (score, reasoning)
    """
    if not response or not response.strip():
        return None, None

    response = response.strip()
    score, reason = None, None

    # Strategy 1: Direct JSON parsing
    try:
        data = json.loads(response)
        score = data.get(score_key)
        reason = data.get("reasoning")
        if score is not None:
            score_int = int(score)
            if 1 <= score_int <= 5:
                return score_int, reason
    except Exception:
        pass

    # Strategy 2: Handle double braces
    try:
        if response.startswith("{{") and response.endswith("}}"):
            clean_response = response[1:-1]
            data = json.loads(clean_response)
            score = data.get(score_key)
            reason = data.get("reasoning")
            if score is not None:
                return int(score), reason
    except Exception:
        pass

    # Strategy 3: Extract JSON from text
    try:
        json_pattern = r"\{[^{}]*\"" + re.escape(score_key) + r"\"[^{}]*\}"
        json_match = re.search(json_pattern, response, re.IGNORECASE)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            score = data.get(score_key)
            reason = data.get("reasoning")
            if score is not None:
                return int(score), reason
    except Exception:
        pass

    # Strategy 4: Regex fallback
    patterns = [
        rf"{score_key}\s*[:：]?\s*([1-5])",
        r"([1-5])\s*/\s*5",
        r"([1-5])\s+out\s+of\s+5",
        r"score\s*[:：]?\s*([1-5])",
    ]

    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            try:
                score = int(m.group(1))
                if 1 <= score <= 5:
                    break
            except (ValueError, TypeError):
                pass

    # Extract reasoning
    reason_patterns = [
        r"reasoning\s*[:：]\s*(.+)",
        r"explanation\s*[:：]\s*(.+)",
        r"analysis\s*[:：]\s*(.+)",
    ]

    for pat in reason_patterns:
        reason_match = re.search(pat, response, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()
            if reason.startswith('"') and reason.endswith('"'):
                reason = reason[1:-1]
            break

    return score, reason


def evaluate_reasoning_process(
    sample: EvaluationSample,
    max_retries: int = 3,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Evaluate reasoning process (think_output quality).

    Args:
        sample: Evaluation sample
        max_retries: Number of retry attempts

    Returns:
        Tuple of (score, reasoning)
    """
    prompt_text = get_reasoning_process_prompt(
        reasoning_type=sample.reasoning_type,
        prompt=sample.prompt,
        dimension=sample.dimension,
        keywords=sample.keywords,
        target_description=sample.target_description,
        think_output=sample.think_output,
    )

    for attempt in range(max_retries):
        response = call_gpt4o_with_images(
            prompt=prompt_text,
            images=[sample.original_image],
        )
        if response:
            score, reason = extract_score_and_reason(response, "reasoning_process_score")
            if score is not None:
                return score, reason
        logging.warning(f"reasoning_process attempt {attempt + 1} failed, retrying...")

    return None, None


def evaluate_reasoning_visual(
    sample: EvaluationSample,
    max_retries: int = 3,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Evaluate reasoning visual (generated image quality).

    Args:
        sample: Evaluation sample
        max_retries: Number of retry attempts

    Returns:
        Tuple of (score, reasoning)
    """
    prompt_text = get_reasoning_visual_prompt(
        reasoning_type=sample.reasoning_type,
        prompt=sample.prompt,
        dimension=sample.dimension,
        keywords=sample.keywords,
        target_description=sample.target_description,
    )

    # Build image list
    images = [sample.original_image, sample.generated_image]
    if sample.target_image is not None:
        images.append(sample.target_image)

    for attempt in range(max_retries):
        response = call_gpt4o_with_images(
            prompt=prompt_text,
            images=images,
        )
        if response:
            score, reason = extract_score_and_reason(response, "reasoning_visual_score")
            if score is not None:
                return score, reason
        logging.warning(f"reasoning_visual attempt {attempt + 1} failed, retrying...")

    return None, None


def evaluate_reasoning_alignment(
    sample: EvaluationSample,
    max_retries: int = 3,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Evaluate reasoning alignment (process-visual consistency).

    Args:
        sample: Evaluation sample
        max_retries: Number of retry attempts

    Returns:
        Tuple of (score, reasoning)
    """
    prompt_text = get_reasoning_alignment_prompt(
        prompt=sample.prompt,
        think_output=sample.think_output,
    )

    for attempt in range(max_retries):
        response = call_gpt4o_with_images(
            prompt=prompt_text,
            images=[sample.original_image, sample.generated_image],
        )
        if response:
            score, reason = extract_score_and_reason(response, "reasoning_alignment_score")
            if score is not None:
                return score, reason
        logging.warning(f"reasoning_alignment attempt {attempt + 1} failed, retrying...")

    return None, None


def evaluate_single_sample(
    sample: EvaluationSample,
    metrics: List[str] = None,
    max_retries: int = 3,
) -> Dict:
    """
    Evaluate a single sample on specified metrics.

    Args:
        sample: Evaluation sample
        metrics: List of metrics to evaluate (default: all)
        max_retries: Number of retry attempts per metric

    Returns:
        Dict with scores and reasoning for each metric
    """
    metrics = metrics or METRICS
    results = {"sample_id": sample.sample_id}

    for metric in metrics:
        try:
            if metric == "reasoning_process":
                score, reason = evaluate_reasoning_process(sample, max_retries)
                results["reasoning_process_score"] = score
                results["reasoning_process_reasoning"] = reason

            elif metric == "reasoning_visual":
                score, reason = evaluate_reasoning_visual(sample, max_retries)
                results["reasoning_visual_score"] = score
                results["reasoning_visual_reasoning"] = reason

            elif metric == "reasoning_alignment":
                score, reason = evaluate_reasoning_alignment(sample, max_retries)
                results["reasoning_alignment_score"] = score
                results["reasoning_alignment_reasoning"] = reason

        except Exception as e:
            logging.error(f"Error evaluating {metric} for {sample.sample_id}: {e}")
            results[f"{metric}_score"] = None
            results[f"{metric}_reasoning"] = f"Error: {str(e)}"

    return results


def evaluate_batch(
    samples: List[EvaluationSample],
    metrics: List[str] = None,
    num_workers: int = 10,
    max_retries: int = 3,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """
    Evaluate a batch of samples in parallel.

    Args:
        samples: List of evaluation samples
        metrics: List of metrics to evaluate (default: all)
        num_workers: Number of parallel workers
        max_retries: Number of retry attempts per metric
        output_path: Optional path to save results as JSONL

    Returns:
        List of result dicts
    """
    metrics = metrics or METRICS
    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(evaluate_single_sample, sample, metrics, max_retries): sample
            for sample in samples
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                result = future.result()
                results.append(result)

                # Save incrementally if output path specified
                if output_path:
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

            except Exception as e:
                sample = futures[future]
                logging.error(f"Error processing sample {sample.sample_id}: {e}")
                results.append({
                    "sample_id": sample.sample_id,
                    "error": str(e),
                })

    return results


class ROVEREvaluator:
    """
    ROVER Evaluator class for evaluating Interleaved Reasoning and Reasoning Alignment.

    Example usage:
        evaluator = ROVEREvaluator(metrics=["reasoning_process", "reasoning_visual", "reasoning_alignment"])

        # Single sample
        result = evaluator.evaluate(
            sample_id="sample_001",
            original_image="path/to/original.png",
            generated_image="path/to/generated.png",
            prompt="Show what this plant looks like after 3 months",
            think_output="The plant will grow taller...",
            reasoning_type="temporal",
            dimension="science",
        )

        # Batch evaluation
        results = evaluator.evaluate_batch(samples, num_workers=10)
    """

    def __init__(
        self,
        metrics: List[str] = None,
        max_retries: int = 3,
    ):
        """
        Initialize ROVER Evaluator.

        Args:
            metrics: List of metrics to evaluate (default: all)
            max_retries: Number of retry attempts per metric
        """
        self.metrics = metrics or METRICS
        self.max_retries = max_retries

    def evaluate(
        self,
        sample_id: str,
        original_image: Union[str, Image.Image],
        generated_image: Union[str, Image.Image],
        prompt: str,
        think_output: str = "",
        reasoning_type: str = "temporal",
        dimension: str = "",
        keywords: str = "",
        target_description: str = "",
        target_image: Optional[Union[str, Image.Image]] = None,
    ) -> Dict:
        """
        Evaluate a single sample.

        Args:
            sample_id: Unique identifier for the sample
            original_image: Original input image (path or PIL Image)
            generated_image: Generated output image (path or PIL Image)
            prompt: Task instruction
            think_output: Model's reasoning text
            reasoning_type: Type of reasoning (temporal/spatial/causal/etc.)
            dimension: Knowledge domain (science/humanity/common_sense/logic)
            keywords: Domain concepts
            target_description: Expected visual outcomes
            target_image: Optional reference image

        Returns:
            Dict with scores and reasoning for each metric
        """
        sample = EvaluationSample(
            sample_id=sample_id,
            original_image=original_image,
            generated_image=generated_image,
            target_image=target_image,
            prompt=prompt,
            dimension=dimension,
            reasoning_type=reasoning_type,
            keywords=keywords,
            target_description=target_description,
            think_output=think_output,
        )

        return evaluate_single_sample(sample, self.metrics, self.max_retries)

    def evaluate_batch(
        self,
        samples: List[EvaluationSample],
        num_workers: int = 10,
        output_path: Optional[str] = None,
    ) -> List[Dict]:
        """
        Evaluate a batch of samples in parallel.

        Args:
            samples: List of EvaluationSample objects
            num_workers: Number of parallel workers
            output_path: Optional path to save results as JSONL

        Returns:
            List of result dicts
        """
        return evaluate_batch(
            samples,
            self.metrics,
            num_workers,
            self.max_retries,
            output_path,
        )

    def compute_average_scores(self, results: List[Dict]) -> Dict:
        """
        Compute average scores across all results.

        Args:
            results: List of evaluation results

        Returns:
            Dict with average scores for each metric
        """
        averages = {}

        for metric in self.metrics:
            score_key = f"{metric}_score"
            scores = [r[score_key] for r in results if r.get(score_key) is not None]
            if scores:
                averages[f"avg_{metric}"] = sum(scores) / len(scores)
                averages[f"count_{metric}"] = len(scores)
            else:
                averages[f"avg_{metric}"] = None
                averages[f"count_{metric}"] = 0

        return averages
