"""
Visual CoT Evaluator for ROVER

Evaluates Visual Chain-of-Thought tasks based on JSON metadata format.
Two main metrics:
- RA (Reasoning-to-Visual Alignment): Evaluates if generated images match generation_prompt
- AL (Answer-to-Visual Alignment): Evaluates if answer is consistent with generated images
"""

import json
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
from PIL import Image

from .api import call_gpt4o_with_images
from .visual_cot_prompts import (
    get_reasoning_visual_prompt,
    get_answer_visual_alignment_prompt,
)


def load_metadata_json(json_path: str) -> Dict:
    """
    Load metadata JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dict with metadata
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_score_and_reason(response: str, score_key: str) -> tuple:
    """Extract score and reasoning from GPT response."""
    if not response:
        return None, None
    
    import re
    
    # Try JSON parsing first
    try:
        data = json.loads(response.strip())
        score = data.get(score_key)
        reason = data.get("reasoning", "")
        if score and 1 <= int(score) <= 5:
            return int(score), reason
    except:
        pass
    
    # Try regex extraction
    patterns = [
        rf'"{score_key}":\s*(\d+)',
        rf'{score_key}:\s*(\d+)',
        rf'Score:\s*(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score, response
    
    return None, None


def evaluate_reasoning_visual_alignment(
    doc_id: str,
    task: str,
    original_image: Union[str, Image.Image],
    generated_images: List[Union[str, Image.Image]],
    generation_prompt: str,
    question: str,
    task_category: Optional[str] = None,
    max_retries: int = 3,
) -> Dict:
    """
    Evaluate Reasoning-to-Visual Alignment (RA).
    
    Checks if generated images match the generation_prompt.
    
    Args:
        doc_id: Document ID
        task: Task name
        original_image: Original question image
        generated_images: List of generated visualization images
        generation_prompt: Prompt used to generate images
        question: Original question text
        task_category: Optional task category for customized prompts
        max_retries: Max retry attempts
        
    Returns:
        Dict with score and reasoning
    """
    prompt_text = get_reasoning_visual_prompt(
        generation_prompt=generation_prompt,
        question=question,
        task=task,
        task_category=task_category,
    )
    
    # Build image list: original + all generated
    images = [original_image] + generated_images
    
    for attempt in range(max_retries):
        response = call_gpt4o_with_images(
            prompt=prompt_text,
            images=images,
        )
        if response:
            score, reason = extract_score_and_reason(response, "reasoning_visual_score")
            if score is not None:
                return {
                    "doc_id": doc_id,
                    "task": task,
                    "metric": "reasoning_visual_alignment",
                    "score": score,
                    "reasoning": reason,
                }
        logging.warning(f"RA evaluation attempt {attempt + 1} failed, retrying...")
    
    return {
        "doc_id": doc_id,
        "task": task,
        "metric": "reasoning_visual_alignment",
        "score": None,
        "reasoning": "Evaluation failed after max retries",
    }


def evaluate_answer_visual_alignment(
    doc_id: str,
    task: str,
    original_image: Union[str, Image.Image],
    generated_images: List[Union[str, Image.Image]],
    answer: str,
    question: str,
    task_category: Optional[str] = None,
    max_retries: int = 3,
) -> Dict:
    """
    Evaluate Answer-to-Visual Alignment (AL).
    
    Checks if the answer is consistent with generated images and question.
    
    Args:
        doc_id: Document ID
        task: Task name
        original_image: Original question image
        generated_images: List of generated visualization images
        answer: Final answer text
        question: Original question text
        task_category: Optional task category for customized prompts
        max_retries: Max retry attempts
        
    Returns:
        Dict with score and reasoning
    """
    prompt_text = get_answer_visual_alignment_prompt(
        answer=answer,
        question=question,
        task=task,
        task_category=task_category,
    )
    
    # Build image list: original + all generated
    images = [original_image] + generated_images
    
    for attempt in range(max_retries):
        response = call_gpt4o_with_images(
            prompt=prompt_text,
            images=images,
        )
        if response:
            score, reason = extract_score_and_reason(response, "answer_visual_score")
            if score is not None:
                return {
                    "doc_id": doc_id,
                    "task": task,
                    "metric": "answer_visual_alignment",
                    "score": score,
                    "reasoning": reason,
                }
        logging.warning(f"AL evaluation attempt {attempt + 1} failed, retrying...")
    
    return {
        "doc_id": doc_id,
        "task": task,
        "metric": "answer_visual_alignment",
        "score": None,
        "reasoning": "Evaluation failed after max retries",
    }


def evaluate_from_json(
    json_path: str,
    original_image: Union[str, Image.Image],
    metrics: List[str] = None,
    task_category: Optional[str] = None,
    max_retries: int = 3,
) -> Dict:
    """
    Evaluate Visual CoT from JSON metadata file.
    
    Args:
        json_path: Path to JSON metadata file
        original_image: Path to original question image
        metrics: List of metrics to evaluate ["ra", "al"] (default: both)
        task_category: Optional task category for customized prompts
        max_retries: Max retry attempts
        
    Returns:
        Dict with all evaluation results
    """
    # Load metadata
    metadata = load_metadata_json(json_path)
    
    doc_id = str(metadata.get("doc_id", "unknown"))
    task = metadata.get("task", "unknown")
    generation_prompt = metadata.get("generation_prompt", "")
    generated_images = metadata.get("generated_images", [])
    question = metadata.get("question", "")
    answer = metadata.get("stage2_answer", "")
    
    # Default to both metrics
    if metrics is None:
        metrics = ["ra", "al"]
    
    results = {
        "doc_id": doc_id,
        "task": task,
        "json_path": json_path,
    }
    
    # Evaluate RA (Reasoning-to-Visual Alignment)
    if "ra" in metrics:
        ra_result = evaluate_reasoning_visual_alignment(
            doc_id=doc_id,
            task=task,
            original_image=original_image,
            generated_images=generated_images,
            generation_prompt=generation_prompt,
            question=question,
            task_category=task_category,
            max_retries=max_retries,
        )
        results["ra_score"] = ra_result["score"]
        results["ra_reasoning"] = ra_result["reasoning"]
    
    # Evaluate AL (Answer-to-Visual Alignment)
    if "al" in metrics:
        al_result = evaluate_answer_visual_alignment(
            doc_id=doc_id,
            task=task,
            original_image=original_image,
            generated_images=generated_images,
            answer=answer,
            question=question,
            task_category=task_category,
            max_retries=max_retries,
        )
        results["al_score"] = al_result["score"]
        results["al_reasoning"] = al_result["reasoning"]
    
    return results


def evaluate_batch_from_jsons(
    json_paths: List[str],
    original_images: List[Union[str, Image.Image]],
    metrics: List[str] = None,
    task_category: Optional[str] = None,
    max_retries: int = 3,
    max_workers: int = 10,
) -> List[Dict]:
    """
    Evaluate multiple Visual CoT samples from JSON files.
    
    Args:
        json_paths: List of JSON file paths
        original_images: List of original image paths (same order as json_paths)
        metrics: List of metrics to evaluate ["ra", "al"]
        task_category: Optional task category for customized prompts
        max_retries: Max retry attempts
        max_workers: Max parallel workers
        
    Returns:
        List of evaluation results
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    if len(json_paths) != len(original_images):
        raise ValueError("json_paths and original_images must have same length")
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_from_json,
                json_path,
                original_image,
                metrics,
                task_category,
                max_retries,
            ): (json_path, original_image)
            for json_path, original_image in zip(json_paths, original_images)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                json_path, _ = futures[future]
                logging.error(f"Error evaluating {json_path}: {e}")
                results.append({
                    "json_path": json_path,
                    "error": str(e),
                })
    
    return results


class VisualCoTEvaluator:
    """Visual CoT Evaluator for ROVER."""
    
    def __init__(
        self,
        metrics: List[str] = None,
        task_category: Optional[str] = None,
        max_retries: int = 3,
    ):
        """
        Initialize Visual CoT Evaluator.
        
        Args:
            metrics: List of metrics ["ra", "al"] (default: both)
            task_category: Task category for customized prompts
            max_retries: Max retry attempts per metric
        """
        self.metrics = metrics or ["ra", "al"]
        self.task_category = task_category
        self.max_retries = max_retries
    
    def evaluate_from_json(
        self,
        json_path: str,
        original_image: Union[str, Image.Image],
        task_category: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate single sample from JSON.
        
        Args:
            json_path: Path to JSON metadata file
            original_image: Original question image
            task_category: Override task category
            
        Returns:
            Evaluation results
        """
        cat = task_category if task_category is not None else self.task_category
        return evaluate_from_json(
            json_path=json_path,
            original_image=original_image,
            metrics=self.metrics,
            task_category=cat,
            max_retries=self.max_retries,
        )
    
    def evaluate_batch(
        self,
        json_paths: List[str],
        original_images: List[Union[str, Image.Image]],
        max_workers: int = 10,
    ) -> List[Dict]:
        """
        Evaluate multiple samples from JSON files.
        
        Args:
            json_paths: List of JSON file paths
            original_images: List of original image paths
            max_workers: Max parallel workers
            
        Returns:
            List of evaluation results
        """
        return evaluate_batch_from_jsons(
            json_paths=json_paths,
            original_images=original_images,
            metrics=self.metrics,
            task_category=self.task_category,
            max_retries=self.max_retries,
            max_workers=max_workers,
        )
