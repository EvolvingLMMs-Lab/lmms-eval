"""
UEval (Unified Evaluation) Utils
支持多模态生成评估 + Rubric 评估
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from google import genai
from loguru import logger as eval_logger
from PIL import Image

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION", "gemini-2.5-pro")
if not GOOGLE_API_KEY:
    eval_logger.warning("GOOGLE_API_KEY has not been set")
else:
    eval_logger.info("GOOGLE_API_KEY has been set")
TEXT_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score how well the model's text answer satisfies the rubric.

# Conversation
Question: <<question>>
Text Answer: <<text_answer>>

# Rubric Item
<<rubric_item>>

# Instructions
Return a JSON object with two fields: "explanation" (string) and "criteria_met" (boolean or "not sure").
- Explain briefly why the text answer does or does not satisfy the rubric.
- Set "criteria_met" to true only if the rubric is fully satisfied. Use false if any requirement is missing or incorrect. If there is not enough information, return "not sure".
Return only the JSON object (no extra narration).
""".strip()

IMAGE_TEMPLATE_OPEN = """
You are evaluating whether the generated image (considered together with the accompanying text answer) satisfies the rubric.

# Conversation
Question: <<question>>
Text Answer: <<text_answer>>

# Rubric Item
<<rubric_item>>

# Instructions
You are given the question, the model's text answer, and the generated image(s). Judge whether the visual content (and its alignment with the text) meets the rubric.
Return a JSON object with "explanation" and "criteria_met". 
- Explain briefly why the image (considered together with the accompanying text answer) does or does not meet the rubric.
- Set "criteria_met" to true only if the rubric is completely satisfied; false otherwise. Use "not sure" if the evidence is insufficient.
- One important clarification regarding the requirement is that each image must include a visual depiction of the described action — it cannot rely solely on text rendered within the image as a substitute for visual content. For example, if the rubric says “Each image must directly correspond to a single, sequential step outlined in the text answer,” then the image must visually represent the action described in the text (e.g., showing the motion, object, or scene), rather than merely displaying textual labels or written descriptions inside the image.
Return only the JSON object.
- One important exception to the above point is that when the criterion is used to evaluate the consistency between an image and its corresponding text step, the image does not need to depict all actions or details mentioned in that step to meet the criterion.
For example, if the criterion states, “Each image must visually represent the primary action described in its corresponding numbered step in the text,” then an image that clearly shows the main action—such as turning the oven dial to preheat—would still satisfy the criterion, even if the step also includes secondary actions (like preparing the baking tray or measuring ingredients).
The key point is that the image should accurately represent the primary action of the step, rather than all of its described details.
""".strip()

IMAGE_TEMPLATE_CLOSED = """
You are evaluating whether the generated image satisfies an image-focused rubric.

# Question
<<question>>

# Rubric Item
<<rubric_item>>

# Instructions
You are given the question and the generated image(s). Judge whether the image meets the rubric. Return a JSON object with "explanation" and "criteria_met". 
- Explain briefly why the image does or does not meet the rubric.
- Set "criteria_met" to true only if the rubric is completely satisfied; false otherwise. Use "not sure" if the evidence is insufficient.
- One important clarification regarding the image requirement is that each image must include a visual depiction of the described action — it cannot rely solely on text rendered within the image as a substitute for visual content.
Return only the JSON object. If any image consists purely of text with no visual content, it should be judged as false directly.
""".strip()


def ueval_doc_to_visual(doc):
    return []


def ueval_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["prompt"]
    return f"{question}"


def ueval_doc_to_target(doc):
    return doc["text_ref"]


def parse_gemini_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON from Gemini response with better error handling."""
    start = response_text.find("{")
    end = response_text.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError(f"Could not locate JSON object in response:\n{response_text}")
    json_str = response_text[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common escape issues
        eval_logger.debug(f"JSON parsing error: {e}, attempting to fix")
        try:
            # Fix unescaped backslashes in strings
            fixed_json = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", json_str)
            return json.loads(fixed_json)
        except Exception:
            raise ValueError(f"Failed to parse JSON. Original error: {e}\n" f"JSON (first 500 chars): {json_str[:500]}")


def send_to_gemini(prompt: str, image_paths: Optional[Sequence[str]], retries: int = 5) -> Optional[Dict[str, Any]]:
    """
    Send prompt and images to Gemini API for evaluation

    Args:
        prompt: Evaluation prompt
        image_paths: List of image file paths to include
        retries: Number of retry attempts

    Returns:
        Parsed JSON response from Gemini
    """
    contents: List[Any] = [prompt]
    if image_paths:
        for rel_path in image_paths:
            img_path = Path(rel_path)
            if not img_path.exists():
                eval_logger.warning(f"Image not found: {img_path}")
                continue
            image = Image.open(img_path)
            contents.append(image)

    for attempt in range(retries):
        try:
            client = genai.Client(api_key=GOOGLE_API_KEY)
            response = client.models.generate_content(
                model=GOOGLE_EVAL_MODEL_NAME,
                contents=contents,
            )
            parsed = parse_gemini_response(response.text)
            return parsed
        except Exception as e:
            eval_logger.error(f"Gemini API error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                eval_logger.error(f"All {retries} attempts failed: {e}")
                return ""
    return None


def normalize_criteria_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        lowered = cleaned.lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
        return cleaned
    return value


def get_eval(doc, model_text, model_images):
    """
    Evaluate model output against rubrics using Gemini judge

    Args:
        doc: Document containing question, rubrics, etc.
        model_text: Generated text answer
        model_images: List of generated image paths

    Returns:
        List of evaluation results for each rubric
    """
    question = doc["prompt"]
    question_type = doc.get("question_type", "open")

    all_results: List[Dict[str, Any]] = []

    # Evaluate text rubrics
    for rubric in doc.get("text_rubrics", []):
        template = TEXT_TEMPLATE
        prompt = template.replace("<<question>>", question).replace("<<text_answer>>", model_text or "").replace("<<rubric_item>>", rubric.get("criterion", ""))

        parsed = send_to_gemini(prompt, image_paths=None)
        print(parsed)
        if parsed:
            criteria_met = normalize_criteria_value(parsed.get("criteria_met"))
            all_results.append(
                {
                    "criterion": rubric.get("criterion", ""),
                    "criteria_met": criteria_met,
                    "explanation": parsed.get("explanation", ""),
                    "rubric_tags": rubric.get("tags", []),
                    "type": "text",
                    "raw_response": str(parsed),
                }
            )

    # Evaluate image rubrics
    for rubric in doc.get("image_rubrics", []):
        # Choose template based on question type
        if question_type == "closed":
            template = IMAGE_TEMPLATE_CLOSED
            prompt = template.replace("<<question>>", question).replace("<<rubric_item>>", rubric.get("criterion", ""))
        else:  # open type
            template = IMAGE_TEMPLATE_OPEN
            prompt = template.replace("<<question>>", question).replace("<<text_answer>>", model_text or "").replace("<<rubric_item>>", rubric.get("criterion", ""))

        parsed = send_to_gemini(prompt, image_paths=model_images)
        print(parsed)
        if parsed:
            criteria_met = normalize_criteria_value(parsed.get("criteria_met"))
            all_results.append(
                {
                    "criterion": rubric.get("criterion", ""),
                    "criteria_met": criteria_met,
                    "explanation": parsed.get("explanation", ""),
                    "rubric_tags": rubric.get("tags", []),
                    "type": "image",
                    "raw_response": str(parsed),
                }
            )

    return all_results


def ueval_process_results(doc, results):
    """
    Process model predictions and evaluate against rubrics

    Args:
        doc: Document containing question, rubrics, ground truth
        results: Model predictions [{"text": "...", "images": [...]}]

    Returns:
        Dict with metrics: text_score, image_score, overall_score, domain
    """
    pred = results[0]
    try:
        pred = json.loads(pred)
    except json.JSONDecodeError:
        eval_logger.warning(f"Failed to parse prediction JSON: {pred}")
        pred = {"text": "", "images": []}

    model_text = pred.get("text", "")
    model_images = pred.get("images", [])

    # Get evaluation results from Gemini judge
    eval_results = get_eval(doc, model_text, model_images)

    # Calculate scores
    text_results = [r for r in eval_results if r["type"] == "text"]
    image_results = [r for r in eval_results if r["type"] == "image"]

    text_met = sum(1 for r in text_results if r["criteria_met"] is True)
    text_total = len(text_results)
    text_score = text_met / text_total if text_total > 0 else 0.0

    image_met = sum(1 for r in image_results if r["criteria_met"] is True)
    image_total = len(image_results)
    image_score = image_met / image_total if image_total > 0 else 0.0

    overall_score = (text_score + image_score) / 2

    # Extract domain from task_type field
    domain = doc.get("task_type", "unknown")

    eval_logger.info(f"[{domain}] Sample {doc.get('id', 'N/A')}: " f"Text={text_met}/{text_total}, " f"Image={image_met}/{image_total}, " f"Overall={overall_score:.4f}")

    return {
        "text_score": {"score": text_score, "domain": domain, "id": doc.get("id", "N/A")},
        "image_score": {"score": image_score, "domain": domain, "id": doc.get("id", "N/A")},
        "overall_score": {"score": overall_score, "domain": domain, "id": doc.get("id", "N/A")},
    }


def _compute_domain_breakdown(results: List[Dict], metric_key: str) -> Dict:
    """
    Helper function to compute per-domain statistics

    Args:
        results: List of dicts with 'score' and 'domain' fields
        metric_key: Name of the metric (for logging only)

    Returns:
        Dict with macro_avg and per_domain breakdown
    """
    from collections import defaultdict

    domain_scores = defaultdict(list)

    for result in results:
        domain = result.get("domain", "unknown")
        score = result.get("score", 0.0)
        domain_scores[domain].append(score)

    # Compute per-domain averages
    per_domain = {}
    domain_avgs = []

    for domain, scores in domain_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0.0
        per_domain[domain] = {
            "average": avg_score,
            "count": len(scores),
        }
        domain_avgs.append(avg_score)

    # Compute macro average (average of domain averages)
    macro_avg = sum(domain_avgs) / len(domain_avgs) if domain_avgs else 0.0

    eval_logger.info(f"\n{metric_key.upper()} Per-Domain Breakdown:")
    for domain, stats in sorted(per_domain.items()):
        eval_logger.info(f"  {domain}: {stats['average']:.4f} (n={stats['count']})")
    eval_logger.info(f"  Macro Average: {macro_avg:.4f}")

    return {
        "macro_avg": macro_avg,
        "per_domain": per_domain,
    }


def ueval_text_score_aggregation(results):
    """Aggregate text scores with per-domain breakdown"""
    eval_logger.info(f"Text scores: {results}")
    breakdown = _compute_domain_breakdown(results, "text_score")
    return breakdown["macro_avg"]


def ueval_image_score_aggregation(results):
    """Aggregate image scores with per-domain breakdown"""
    eval_logger.info(f"Image scores: {results}")
    breakdown = _compute_domain_breakdown(results, "image_score")
    return breakdown["macro_avg"]


def ueval_aggregation(results):
    """Aggregate overall scores with per-domain breakdown"""
    print(results)
    eval_logger.info(f"Overall scores: {results}")
    breakdown = _compute_domain_breakdown(results, "overall_score")
    return breakdown["macro_avg"]
