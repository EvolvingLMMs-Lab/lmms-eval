"""
CaptionQA evaluation utilities for lmms-eval.

CaptionQA evaluates how well image captions preserve information for downstream QA tasks.
The evaluation works by:
1. Generating captions for images using the evaluated model
2. Using Qwen2.5-72B-Instruct as judge to answer questions based on the generated captions
3. Computing accuracy and score based on the answers

Usage:
    python -m lmms_eval \\
        --model qwen2_5_vl \\
        --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \\
        --tasks captionqa \\
        --batch_size 1 \\
        --launcher_args "name=sglang,model=Qwen/Qwen2.5-72B-Instruct,tp=2" \\
        --output_path ./logs/captionqa_results

    The --launcher_args option uses lmms-eval's internal eval_server_launcher to:
    1. Clean up GPU memory after caption generation (lm.clean())
    2. Launch the judge server (SGLang) as a subprocess
    3. Set OPENAI_BASE_URL for the judge API

    This allows running the full evaluation in a single command.

Requirements:
    - SGLang installed for judge server
    - At least 2 GPUs with ~80GB VRAM each (for the 72B judge model with tp=2)

Paper: https://arxiv.org/abs/2511.21025
Dataset: https://huggingface.co/datasets/Borise/CaptionQA
"""

import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests
from datasets import Dataset
from loguru import logger as eval_logger

# Constants
LETTER_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CANNOT_ANSWER_TEXT = "Cannot answer from the caption"

# Fixed seed for reproducibility (matching original CaptionQA implementation)
SHUFFLE_SEED = 0


# Judge configuration
# The judge model and server URL are configured via the launcher:
#   --launcher_args "name=sglang,model=Qwen/Qwen2.5-72B-Instruct,tp=2"
# The launcher sets OPENAI_BASE_URL automatically.
JUDGE_MODEL = os.getenv("CAPTIONQA_JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# System prompt for QA evaluation (matching original)
QA_SYSTEM_PROMPT = "You are given a caption describing an image, and a question about the image. Answer with a SINGLE LETTER (A, B, C, ...), no explanation."


def _is_yesno_question(question_text: str, choices: List[str]) -> bool:
    """Check if question is a yes/no question."""
    choice_texts = [str(c).strip().lower() for c in choices]
    has_yes = any("yes" in choice for choice in choice_texts)
    has_no = any("no" in choice for choice in choice_texts)
    if has_yes and has_no:
        return True
    question_lower = question_text.strip().lower()
    yesno_starters = ["is ", "are ", "was ", "were ", "do ", "does ", "did ", "have ", "has ", "had ", "can ", "could ", "will ", "would ", "should ", "shall ", "may ", "might ", "must "]
    return any(question_lower.startswith(s) for s in yesno_starters)


def _compute_all_shuffle_permutations() -> Dict[tuple, List[int]]:
    """
    Compute all shuffle permutations matching the original CaptionQA implementation.

    The original implementation uses a single sequential RNG (seed=0) that advances through
    ALL questions in the "all" split order. Each shuffle depends on all previous shuffles.

    Returns:
        Dictionary mapping (image_id, q_idx) -> permutation list
    """
    from datasets import load_dataset

    eval_logger.info("Computing shuffle permutations from 'all' split (matching original RNG order)...")
    all_dataset = load_dataset("Borise/CaptionQA", split="all")

    # Use the exact same RNG setup as the original
    rng = random.Random(SHUFFLE_SEED)
    shuffle_cache: Dict[tuple, List[int]] = {}

    for entry in all_dataset:
        image_id = str(entry.get("id", "unknown"))
        questions = entry.get("questions", [])

        if not questions:
            # Single question format
            if "question" in entry:
                cat = entry.get("category", [])
                if isinstance(cat, list):
                    cat = cat[0] if cat else ""
                questions = [{"question": entry["question"], "choices": entry.get("choices", []), "answer": entry.get("answer"), "category": cat}]

        for q_idx, q in enumerate(questions):
            question_text = q.get("question", "")
            choices = q.get("choices", [])
            answer = q.get("answer")

            # Skip invalid questions (matching original logic)
            if not choices or len(choices) < 2:
                continue

            # Check if ground truth can be found (matching original logic)
            gt_found = False
            if isinstance(answer, str):
                for choice in choices:
                    if answer.strip() == str(choice).strip():
                        gt_found = True
                        break

            if not gt_found:
                continue

            # Add "cannot answer" option for non-yes/no questions (matching original)
            if _is_yesno_question(question_text, choices):
                choices_with_option = choices
            else:
                choices_with_option = choices + [CANNOT_ANSWER_TEXT]

            # Create permutation and shuffle with the sequential RNG
            n_opts = len(choices_with_option)
            perm = list(range(n_opts))
            rng.shuffle(perm)  # This advances the RNG state

            # Store the permutation
            shuffle_cache[(image_id, q_idx)] = perm

    eval_logger.info(f"Computed {len(shuffle_cache)} shuffle permutations")
    return shuffle_cache


def captionqa_process_docs(dataset: Dataset) -> Dataset:
    """
    Process the dataset to add precomputed shuffle permutations to each document.

    This is called during dataset loading (before any parallel processing) to ensure
    all documents have consistent shuffle permutations that match the original
    CaptionQA implementation.

    The original implementation uses a single sequential RNG (seed=0) that advances
    through ALL questions in the "all" split order. To reproduce exact shuffles,
    we load the "all" split, compute all permutations, then add them to each document.

    Args:
        dataset: The dataset split being processed

    Returns:
        Dataset with 'shuffle_perms' field added to each document
    """
    # Compute all permutations from the "all" split (matches original RNG order)
    all_perms = _compute_all_shuffle_permutations()

    def _add_shuffle_perms(example):
        """Add shuffle permutations to a single document."""
        image_id = str(example.get("id", "unknown"))
        questions = example.get("questions", [])

        if not questions:
            # Single question format
            if "question" in example:
                questions = [{"question": example["question"], "choices": example.get("choices", [])}]

        # Build shuffle_perms dict: q_idx -> permutation (as JSON string for HF datasets)
        shuffle_perms = {}
        for q_idx in range(len(questions)):
            perm = all_perms.get((image_id, q_idx))
            if perm is not None:
                shuffle_perms[str(q_idx)] = perm

        # Store as JSON string (HF datasets doesn't support nested dicts directly)
        example["shuffle_perms"] = json.dumps(shuffle_perms)
        return example

    # Process with num_proc=1 to ensure deterministic order
    dataset = dataset.map(_add_shuffle_perms, num_proc=1)
    return dataset


def get_shuffle_permutation(doc: Dict, q_idx: int, n_choices: int) -> List[int]:
    """
    Get the shuffle permutation for a question.

    First checks if the document has precomputed shuffle_perms (from process_docs).
    Falls back to hash-based seed if not available.

    Args:
        doc: The document containing shuffle_perms field
        q_idx: Question index within the image
        n_choices: Number of choices (including "Cannot answer" if added)

    Returns:
        A permutation list that matches the original implementation's shuffle
    """
    # Check for precomputed permutation in document
    shuffle_perms_str = doc.get("shuffle_perms", "{}")
    try:
        shuffle_perms = json.loads(shuffle_perms_str) if isinstance(shuffle_perms_str, str) else shuffle_perms_str
        cached_perm = shuffle_perms.get(str(q_idx))
        if cached_perm is not None:
            if len(cached_perm) == n_choices:
                return list(cached_perm)
            else:
                eval_logger.debug(f"Permutation length mismatch for q_idx={q_idx}, using fallback")
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback to hash-based seed
    image_id = str(doc.get("id", "unknown"))
    question_seed = hash((image_id, q_idx, SHUFFLE_SEED)) % (2**32)
    rng = random.Random(question_seed)
    perm = list(range(n_choices))
    rng.shuffle(perm)
    return perm


# ---------- Document Processing Functions ----------


def captionqa_doc_to_visual(doc):
    """Extract visual content from document."""
    images = doc.get("images", [])
    if not images:
        # Try single image field
        if "image" in doc:
            return [doc["image"].convert("RGB")]
        return []

    # Convert all images to RGB
    return [img.convert("RGB") if hasattr(img, "convert") else img for img in images]


def captionqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Generate the prompt for caption generation."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    caption_prompt = lmms_eval_specific_kwargs.get("caption_prompt", "Describe this image in detail.")

    return f"{pre_prompt}{caption_prompt}{post_prompt}"


# ---------- Helper Functions ----------


def extract_letter(answer_text: str, num_options: int) -> Optional[str]:
    """Extract answer letter from model output."""
    if not answer_text:
        return None

    # If response contains </think>, extract letter from text after it
    if "</think>" in answer_text:
        after_think = answer_text.split("</think>", 1)[1]
        answer_text = after_think

    if "Answer: " in answer_text:
        after_answer = answer_text.split("Answer: ", 1)[1]
        answer_text = after_answer

    if "\n" in answer_text:
        after_n = answer_text.split("\n", 1)[1]
        answer_text = after_n

    m = re.search(r"\b([A-Z])\b", answer_text.upper())
    if m:
        letter = m.group(1)
        idx = LETTER_ALPH.find(letter)
        if 0 <= idx < max(1, num_options):
            return letter
    m = re.search(r"\b([1-9][0-9]?)\b", answer_text)
    if m:
        k = int(m.group(1))
        if 1 <= k <= max(1, num_options):
            return LETTER_ALPH[k - 1]
    return None


def normalize_gt_letter(choices: List[str], answer: str) -> Optional[str]:
    """Extract ground truth answer letter from question."""
    if not choices or not isinstance(answer, str):
        return None

    for i, choice in enumerate(choices):
        if answer.strip() == str(choice).strip():
            return LETTER_ALPH[i]

    return None


def add_cannot_answer_option(question_text: str, choices: List[str]) -> List[str]:
    """Add 'cannot answer from the caption' option to non-yes/no questions."""
    if _is_yesno_question(question_text, choices):
        return choices
    return choices + [CANNOT_ANSWER_TEXT]


def build_caption_qa_prompt(caption: str, question: str, choices: List[str]) -> str:
    """Build prompt for QA with caption."""
    lines = [f"{LETTER_ALPH[i]}. {choice}" for i, choice in enumerate(choices)]

    prompt = f"""Caption:
{caption}

Question:
{question}

Options:
{chr(10).join(lines)}

Answer:"""

    return prompt


def call_llm_judge(prompt: str) -> str:
    """
    Call the LLM judge for a single prompt using direct HTTP request.

    The judge server is started by lmms-eval's eval_server_launcher.
    Usage: --launcher_args "name=sglang,model=Qwen/Qwen2.5-72B-Instruct,tp=2"

    The launcher sets OPENAI_API_URL to http://localhost:8000/v1 automatically.
    """
    # Get API URL from environment (set by the launcher)
    base_url = os.environ.get("OPENAI_API_URL", "http://localhost:8000/v1").rstrip("/")

    # Build the correct endpoint URL
    if base_url.endswith("/v1"):
        api_url = f"{base_url}/chat/completions"
    else:
        api_url = f"{base_url}/v1/chat/completions"

    try:
        resp = requests.post(api_url, json={"model": JUDGE_MODEL, "messages": [{"role": "system", "content": QA_SYSTEM_PROMPT}, {"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 4}, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        eval_logger.warning(f"Judge API error: {e}")
        return ""


# ---------- Main Processing Functions ----------


def captionqa_process_results(doc, results):
    """
    Process results for a single document.

    Calls judge directly for each question (server is initialized once and reused).

    Args:
        doc: A document from the CaptionQA dataset
        results: [caption] - The generated caption from the model

    Returns:
        Dictionary with evaluated results for each question
    """
    caption = results[0] if results else ""
    image_id = doc.get("id", "unknown")

    # Get questions from the document
    questions = doc.get("questions", [])
    if not questions:
        # Single question format
        if "question" in doc:
            cat = doc.get("category", [])
            if isinstance(cat, list):
                cat = cat[0] if cat else ""
            questions = [{"question": doc["question"], "choices": doc.get("choices", []), "answer": doc.get("answer"), "category": cat}]

    question_results = []

    for q_idx, q in enumerate(questions):
        question_text = q.get("question", "")
        choices = q.get("choices", [])
        answer = q.get("answer")
        category = q.get("category", [])
        if isinstance(category, list):
            category = category[0] if category else ""

        if not choices or len(choices) < 2:
            continue

        # Get original ground truth
        gt_letter_orig = normalize_gt_letter(choices, answer)
        if gt_letter_orig is None:
            continue
        gt_idx = LETTER_ALPH.index(gt_letter_orig)

        # Add "cannot answer" option for non-yes/no questions
        choices_with_option = add_cannot_answer_option(question_text, choices)

        # Get shuffle permutation matching original CaptionQA implementation
        n_opts = len(choices_with_option)
        perm = get_shuffle_permutation(doc, q_idx, n_opts)

        # Create shuffled choices
        shuffled_opts = [choices_with_option[i] for i in perm]

        # Build the prompt and call judge
        prompt = build_caption_qa_prompt(caption, question_text, shuffled_opts)
        response = call_llm_judge(prompt)

        # Parse response and compute score
        letter = extract_letter(response, n_opts)
        is_correct = False
        is_cannot_answer = False
        score = 0.0

        if letter is not None:
            shuf_idx = LETTER_ALPH.find(letter)
            if 0 <= shuf_idx < len(perm):
                orig_idx = perm[shuf_idx]

                if orig_idx == n_opts - 1 and n_opts > len(choices):
                    is_cannot_answer = True
                    score = (1.0 / len(choices)) + 0.05
                elif orig_idx == gt_idx:
                    is_correct = True
                    score = 1.0

        question_results.append(
            {
                "image_id": image_id,
                "question_idx": q_idx,
                "category": category,
                "is_correct": is_correct,
                "is_cannot_answer": is_cannot_answer,
                "score": score,
            }
        )

    return {
        "captionqa_score": question_results,
        "captionqa_accuracy": question_results,
        "captionqa_cannot_answer_rate": question_results,
    }


# ---------- Aggregation Functions ----------


def captionqa_aggregate_score(results):
    """Aggregate CaptionQA score across all questions."""
    all_scores = []
    category_scores = defaultdict(list)

    for result_list in results:
        if not isinstance(result_list, list):
            result_list = [result_list]
        for result in result_list:
            score = result.get("score", 0.0)
            all_scores.append(score)
            category = result.get("category", "unknown")
            if category:
                category_scores[category].append(score)

    if not all_scores:
        return 0.0

    avg_score = sum(all_scores) / len(all_scores)

    # Log category-level scores
    eval_logger.info("=" * 60)
    eval_logger.info("CaptionQA Score by Category:")
    for category, scores in sorted(category_scores.items()):
        cat_avg = sum(scores) / len(scores) if scores else 0.0
        eval_logger.info(f"  {category}: {cat_avg:.4f} ({len(scores)} questions)")
    eval_logger.info(f"Overall Score: {avg_score:.4f} ({len(all_scores)} questions)")
    eval_logger.info("=" * 60)

    return round(avg_score, 4)


def captionqa_aggregate_accuracy(results):
    """Aggregate CaptionQA accuracy across all questions."""
    total_correct = 0
    total_questions = 0
    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    for result_list in results:
        if not isinstance(result_list, list):
            result_list = [result_list]
        for result in result_list:
            total_questions += 1
            category = result.get("category", "unknown")
            category_total[category] += 1
            if result.get("is_correct", False):
                total_correct += 1
                category_correct[category] += 1

    if total_questions == 0:
        return 0.0

    accuracy = total_correct / total_questions

    # Log category-level accuracy
    eval_logger.info("=" * 60)
    eval_logger.info("CaptionQA Accuracy by Category:")
    for category in sorted(category_total.keys()):
        cat_acc = category_correct[category] / category_total[category] if category_total[category] else 0.0
        eval_logger.info(f"  {category}: {cat_acc:.2%} ({category_correct[category]}/{category_total[category]})")
    eval_logger.info(f"Overall Accuracy: {accuracy:.2%} ({total_correct}/{total_questions})")
    eval_logger.info("=" * 60)

    return round(accuracy, 4)


def captionqa_aggregate_cannot_answer(results):
    """Aggregate 'cannot answer' rate across all questions."""
    total_cannot_answer = 0
    total_questions = 0

    for result_list in results:
        if not isinstance(result_list, list):
            result_list = [result_list]
        for result in result_list:
            total_questions += 1
            if result.get("is_cannot_answer", False):
                total_cannot_answer += 1

    if total_questions == 0:
        return 0.0

    rate = total_cannot_answer / total_questions
    eval_logger.info(f"'Cannot answer' rate: {rate:.2%} ({total_cannot_answer}/{total_questions})")

    return round(rate, 4)
