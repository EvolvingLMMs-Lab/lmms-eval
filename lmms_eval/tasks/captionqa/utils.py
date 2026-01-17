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
        --output_path ./logs/captionqa_results

Requirements:
    - At least 2 GPUs with ~80GB VRAM each (for the 72B judge model with TP=2)
    - vLLM installed for efficient judge inference

Optional Environment Variables:
    CAPTIONQA_JUDGE_MODEL: Model name (default: Qwen/Qwen2.5-72B-Instruct)
    CAPTIONQA_JUDGE_BACKEND: "vllm" or "api" (default: vllm)
    CAPTIONQA_JUDGE_TP_SIZE: Tensor parallel size for vLLM (default: 2)
    CAPTIONQA_JUDGE_GPU_UTIL: GPU memory utilization for judge (default: 0.8)

Paper: https://arxiv.org/abs/2511.21025
Dataset: https://huggingface.co/datasets/Borise/CaptionQA
"""

import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional

from loguru import logger as eval_logger

# Constants
LETTER_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CANNOT_ANSWER_TEXT = "Cannot answer from the caption"

# Fixed seed for reproducibility (matching original CaptionQA implementation)
SHUFFLE_SEED = 0

# Global shuffle permutation cache: (image_id, q_idx) -> permutation list
# This is needed to match the original implementation's sequential RNG behavior.
# The original uses a single sequential RNG (seed=0) that advances through ALL questions
# in the "all" split order. To reproduce exact shuffles, we precompute all permutations.
_global_shuffle_cache: Dict[tuple, List[int]] = {}
_global_shuffle_initialized = False


# Judge configuration from environment
# The 72B judge model requires at least 2 GPUs with tensor parallelism
JUDGE_BACKEND = os.getenv("CAPTIONQA_JUDGE_BACKEND", "vllm")  # "vllm" or "api"
JUDGE_MODEL = os.getenv("CAPTIONQA_JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct")
JUDGE_TP_SIZE = int(
    os.getenv("CAPTIONQA_JUDGE_TP_SIZE", "2")
)  # Default 2 for 72B model
JUDGE_GPU_MEMORY_UTILIZATION = float(
    os.getenv("CAPTIONQA_JUDGE_GPU_UTIL", "0.8")
)  # GPU memory for judge

# System prompt for QA evaluation (matching original)
QA_SYSTEM_PROMPT = "You are given a caption describing an image, and a question about the image. Answer with a SINGLE LETTER (A, B, C, ...), no explanation."

# Global judge client (lazy initialization)
_judge_client = None


def _is_yesno_question_for_init(question_text: str, choices: List[str]) -> bool:
    """Check if question is a yes/no question (for initialization)."""
    choice_texts = [str(c).strip().lower() for c in choices]
    has_yes = any("yes" in choice for choice in choice_texts)
    has_no = any("no" in choice for choice in choice_texts)
    if has_yes and has_no:
        return True
    question_lower = question_text.strip().lower()
    yesno_starters = [
        "is ",
        "are ",
        "was ",
        "were ",
        "do ",
        "does ",
        "did ",
        "have ",
        "has ",
        "had ",
        "can ",
        "could ",
        "will ",
        "would ",
        "should ",
        "shall ",
        "may ",
        "might ",
        "must ",
    ]
    return any(question_lower.startswith(s) for s in yesno_starters)


def _initialize_global_shuffle_cache():
    """
    Initialize the global shuffle permutation cache to match the original CaptionQA implementation.

    The original implementation uses a single sequential RNG (seed=0) that advances through
    ALL questions in the "all" split order. Each shuffle depends on all previous shuffles.

    This function loads the "all" split, processes questions in the exact same order as the
    original, and stores the permutation for each (image_id, q_idx) pair.
    """
    global _global_shuffle_cache, _global_shuffle_initialized

    if _global_shuffle_initialized:
        return

    try:
        from datasets import load_dataset

        eval_logger.info(
            "Initializing global shuffle cache from 'all' split (matching original RNG order)..."
        )
        dataset = load_dataset("Borise/CaptionQA", split="all")

        # Use the exact same RNG setup as the original
        rng = random.Random(SHUFFLE_SEED)

        question_count = 0
        for entry in dataset:
            image_id = str(entry.get("id", "unknown"))
            questions = entry.get("questions", [])

            if not questions:
                # Single question format
                if "question" in entry:
                    cat = entry.get("category", [])
                    if isinstance(cat, list):
                        cat = cat[0] if cat else ""
                    questions = [
                        {
                            "question": entry["question"],
                            "choices": entry.get("choices", []),
                            "answer": entry.get("answer"),
                            "category": cat,
                        }
                    ]

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
                if _is_yesno_question_for_init(question_text, choices):
                    choices_with_option = choices
                else:
                    choices_with_option = choices + [CANNOT_ANSWER_TEXT]

                # Create permutation and shuffle with the sequential RNG
                n_opts = len(choices_with_option)
                perm = list(range(n_opts))
                rng.shuffle(perm)  # This advances the RNG state

                # Store the permutation
                _global_shuffle_cache[(image_id, q_idx)] = perm.copy()
                question_count += 1

        _global_shuffle_initialized = True
        eval_logger.info(
            f"Global shuffle cache initialized: {question_count} questions cached"
        )

    except Exception as e:
        eval_logger.warning(f"Failed to initialize global shuffle cache: {e}")
        eval_logger.warning("Falling back to per-question hash-based seeds")
        _global_shuffle_initialized = True  # Prevent repeated attempts


def get_shuffle_permutation(image_id: str, q_idx: int, n_choices: int) -> List[int]:
    """
    Get the shuffle permutation for a question that matches the original CaptionQA implementation.

    The original uses a single sequential RNG (seed=0) that advances through all questions.
    This function looks up the precomputed permutation from the cache.

    Args:
        image_id: The image identifier
        q_idx: Question index within the image
        n_choices: Number of choices (including "Cannot answer" if added)

    Returns:
        A permutation list that matches the original implementation's shuffle
    """
    global _global_shuffle_cache, _global_shuffle_initialized

    # Initialize cache if not done
    if not _global_shuffle_initialized:
        _initialize_global_shuffle_cache()

    # Look up cached permutation
    cached_perm = _global_shuffle_cache.get((image_id, q_idx))

    if cached_perm is not None:
        # Verify length matches (should always match if dataset is consistent)
        if len(cached_perm) == n_choices:
            return cached_perm.copy()
        else:
            eval_logger.warning(
                f"Cached permutation length mismatch for ({image_id}, {q_idx}): "
                f"cached={len(cached_perm)}, expected={n_choices}. Using fallback."
            )

    # Fallback to hash-based seed if not in cache
    eval_logger.debug(
        f"Question ({image_id}, {q_idx}) not in shuffle cache, using hash-based seed"
    )
    question_seed = hash((image_id, q_idx, SHUFFLE_SEED)) % (2**32)
    rng = random.Random(question_seed)
    perm = list(range(n_choices))
    rng.shuffle(perm)
    return perm


def _get_vllm_judge():
    """Initialize vLLM judge client (lazy loading)."""
    global _judge_client
    if _judge_client is not None:
        return _judge_client

    try:
        import vllm
        from transformers import AutoTokenizer

        eval_logger.info(
            f"Initializing vLLM judge with model: {JUDGE_MODEL}, tp_size: {JUDGE_TP_SIZE}"
        )

        tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, trust_remote_code=True)
        llm = vllm.LLM(
            model=JUDGE_MODEL,
            tensor_parallel_size=JUDGE_TP_SIZE,
            trust_remote_code=True,
            gpu_memory_utilization=JUDGE_GPU_MEMORY_UTILIZATION,
        )

        _judge_client = {"tokenizer": tokenizer, "llm": llm, "type": "vllm"}
        eval_logger.info("vLLM judge initialized successfully")
        return _judge_client

    except ImportError as e:
        eval_logger.warning(f"vLLM not available: {e}. Falling back to API backend.")
        return _get_api_judge()
    except Exception as e:
        error_msg = str(e)
        if "memory" in error_msg.lower() or "oom" in error_msg.lower():
            eval_logger.error(f"Failed to initialize vLLM judge due to GPU memory: {e}")
            eval_logger.error(
                "Try reducing CAPTIONQA_JUDGE_GPU_UTIL or increasing CAPTIONQA_JUDGE_TP_SIZE"
            )
        else:
            eval_logger.error(
                f"Failed to initialize vLLM judge: {e}. Falling back to API backend."
            )
        return _get_api_judge()


def _get_api_judge():
    """Initialize API-based judge client."""
    global _judge_client
    if _judge_client is not None and _judge_client.get("type") == "api":
        return _judge_client

    from lmms_eval.llm_judge import ServerConfig, get_server

    API_TYPE = os.getenv("API_TYPE", "openai")

    eval_logger.info(
        f"Initializing API judge with model: {JUDGE_MODEL}, API type: {API_TYPE}"
    )

    server_config = ServerConfig(
        model_name=JUDGE_MODEL,
        system_prompt=QA_SYSTEM_PROMPT,
        max_tokens=4,
        temperature=0.0,
    )
    server = get_server(server_name=API_TYPE, config=server_config)

    _judge_client = {"server": server, "config": server_config, "type": "api"}
    return _judge_client


def _get_judge():
    """Get the appropriate judge client based on configuration."""
    if JUDGE_BACKEND == "vllm":
        return _get_vllm_judge()
    else:
        return _get_api_judge()


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
    caption_prompt = lmms_eval_specific_kwargs.get(
        "caption_prompt", "Describe this image in detail."
    )

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


def is_yesno_question(question_text: str, choices: List[str]) -> bool:
    """Check if question is a yes/no question."""
    choice_texts = [str(c).strip().lower() for c in choices]

    has_yes = any("yes" in choice for choice in choice_texts)
    has_no = any("no" in choice for choice in choice_texts)

    if has_yes and has_no:
        return True

    question_lower = question_text.strip().lower()
    yesno_starters = [
        "is ",
        "are ",
        "was ",
        "were ",
        "do ",
        "does ",
        "did ",
        "have ",
        "has ",
        "had ",
        "can ",
        "could ",
        "will ",
        "would ",
        "should ",
        "shall ",
        "may ",
        "might ",
        "must ",
    ]

    for starter in yesno_starters:
        if question_lower.startswith(starter):
            return True

    return False


def add_cannot_answer_option(question_text: str, choices: List[str]) -> List[str]:
    """Add 'cannot answer from the caption' option to non-yes/no questions."""
    if is_yesno_question(question_text, choices):
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


def _build_chat_prompt(prompt: str, tokenizer) -> str:
    """Build chat prompt using tokenizer's chat template."""
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except TypeError:
        # Fallback for different tokenizer formats
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def call_llm_judge(prompt: str) -> str:
    """Call the LLM judge to answer a question."""
    try:
        judge = _get_judge()

        if judge["type"] == "vllm":
            # Use vLLM directly
            import vllm

            tokenizer = judge["tokenizer"]
            llm = judge["llm"]

            # Build prompt with chat template
            full_prompt = _build_chat_prompt(prompt, tokenizer)

            # Generate
            sampling_params = vllm.SamplingParams(
                temperature=0.0,
                max_tokens=4,
                n=1,
            )

            outputs = llm.generate([full_prompt], sampling_params, use_tqdm=False)

            if outputs and outputs[0].outputs:
                return outputs[0].outputs[0].text.strip()
            return ""

        else:
            # Use API backend
            from lmms_eval.llm_judge import Request

            server = judge["server"]
            config = judge["config"]

            request = Request(
                messages=[{"role": "user", "content": prompt}],
                question=prompt,
                config=config,
            )

            response = server.evaluate(request)
            return response.content if response.success else ""

    except Exception as e:
        eval_logger.error(f"LLM judge error: {e}")
        return ""


# ---------- Main Processing Functions ----------


def captionqa_process_results(doc, results):
    """
    Process results for a single document.

    Args:
        doc: A document from the CaptionQA dataset containing:
            - id: Image identifier
            - questions: List of questions with choices and answers
        results: [caption] - The generated caption from the model

    Returns:
        Dictionary with metrics for each question

    Note:
        Judge evaluation is deferred to the aggregation phase, allowing the
        caption model to be unloaded first to free GPU memory for the judge.
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
            questions = [
                {
                    "question": doc["question"],
                    "choices": doc.get("choices", []),
                    "answer": doc.get("answer"),
                    "category": cat,
                }
            ]

    question_results = []

    # Store results for judge evaluation (runs during aggregation after caption model unloads)
    for q_idx, q in enumerate(questions):
        question_text = q.get("question", "")
        choices = q.get("choices", [])
        answer = q.get("answer")
        category = q.get("category", [])
        if isinstance(category, list):
            category = category[0] if category else ""

        if not choices or len(choices) < 2:
            continue

        question_results.append(
            {
                "image_id": image_id,
                "question_idx": q_idx,
                "question": question_text,
                "choices": choices,
                "answer": answer,
                "category": category,
                "caption": caption,
                # Placeholder values - computed by judge during aggregation
                "is_correct": False,
                "is_cannot_answer": False,
                "score": 0.0,
                "model_answer": None,
                "pending_judge": True,
            }
        )

    return {
        "captionqa_score": question_results,
        "captionqa_accuracy": question_results,
        "captionqa_cannot_answer_rate": question_results,
    }


def evaluate_single_question(
    caption: str,
    image_id: str,
    q_idx: int,
    question_text: str,
    choices: List[str],
    answer: str,
) -> Dict:
    """
    Evaluate a single question using the LLM judge.

    Args:
        caption: The generated caption
        image_id: Image identifier
        q_idx: Question index
        question_text: The question text
        choices: List of answer choices
        answer: Ground truth answer

    Returns:
        Dictionary with evaluation results
    """
    # Get original ground truth
    gt_letter_orig = normalize_gt_letter(choices, answer)
    if gt_letter_orig is None:
        return None
    gt_idx_orig = LETTER_ALPH.index(gt_letter_orig)

    # Add "cannot answer" option for non-yes/no questions
    choices_with_option = add_cannot_answer_option(question_text, choices)

    # Get shuffle permutation matching original CaptionQA implementation
    n_opts = len(choices_with_option)
    perm = get_shuffle_permutation(image_id, q_idx, n_opts)

    # Create shuffled choices
    shuffled_opts = [choices_with_option[i] for i in perm]

    # Build prompt and get LLM judge response
    prompt = build_caption_qa_prompt(caption, question_text, shuffled_opts)
    response = call_llm_judge(prompt)

    # Parse and score the response
    letter = extract_letter(response, n_opts)
    is_correct = False
    is_cannot_answer = False
    model_answer_text = None
    score = 0.0

    n_original_choices = len(choices)

    if letter is not None:
        shuf_idx = LETTER_ALPH.find(letter)
        if 0 <= shuf_idx < len(perm):
            orig_idx = perm[shuf_idx]

            if orig_idx < len(choices_with_option):
                model_answer_text = str(choices_with_option[orig_idx])

                if model_answer_text == CANNOT_ANSWER_TEXT:
                    is_cannot_answer = True
                    score = (1.0 / n_original_choices) + 0.05
                elif orig_idx == gt_idx_orig:
                    is_correct = True
                    score = 1.0
                else:
                    score = 0.0

    return {
        "is_correct": is_correct,
        "is_cannot_answer": is_cannot_answer,
        "score": round(score, 4),
        "model_answer": model_answer_text,
        "judge_response": response,
    }


# ---------- Deferred Evaluation ----------

_deferred_eval_done = False
_evaluated_results_cache = {}


def _cleanup_gpu_memory():
    """
    Attempt to clean up GPU memory before loading the judge model.

    This is called before deferred evaluation to try to free memory
    that may still be held by the main model.
    """
    import gc

    eval_logger.info("Cleaning up GPU memory before loading judge model...")

    try:
        import torch

        if torch.cuda.is_available():
            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Get memory info
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                eval_logger.info(
                    f"  GPU {i}: {free / 1024**3:.2f} GB free / {total / 1024**3:.2f} GB total"
                )
    except Exception as e:
        eval_logger.debug(f"GPU cleanup info: {e}")

    # Force garbage collection
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _run_deferred_evaluation(results):
    """
    Run judge evaluation on pending questions.

    Called during aggregation after the caption model is unloaded.
    Uses subprocess isolation to ensure GPU memory is available for the 72B judge.
    """
    global _deferred_eval_done, _evaluated_results_cache

    # Only run once
    if _deferred_eval_done:
        return _evaluated_results_cache

    # Check if we have pending evaluations
    pending_count = 0
    for result_list in results:
        if not isinstance(result_list, list):
            result_list = [result_list]
        for result in result_list:
            if result.get("pending_judge", False):
                pending_count += 1

    if pending_count == 0:
        _deferred_eval_done = True
        return {}

    eval_logger.info(f"\n{'=' * 60}")
    eval_logger.info(
        f"Running deferred judge evaluation on {pending_count} questions..."
    )
    eval_logger.info("Using subprocess isolation for GPU memory.")
    eval_logger.info(f"{'=' * 60}\n")

    # Collect all pending questions
    pending_items = []
    for result_list in results:
        if not isinstance(result_list, list):
            result_list = [result_list]
        for result in result_list:
            if result.get("pending_judge", False):
                pending_items.append(result)

    # Save pending items to temp file
    import pickle
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
        pickle.dump(pending_items, f)
        input_path = f.name

    output_path = input_path.replace(".pkl", "_results.pkl")

    # Run judge in subprocess using the helper script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    helper_script = os.path.join(script_dir, "_judge_subprocess.py")

    try:
        import sys

        cmd = [sys.executable, helper_script, input_path, output_path]
        eval_logger.info(f"Launching subprocess: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=False)

        if result.returncode == 0 and os.path.exists(output_path):
            with open(output_path, "rb") as f:
                _evaluated_results_cache = pickle.load(f)
        else:
            eval_logger.error(f"Subprocess failed with code {result.returncode}")
            _evaluated_results_cache = {}
    finally:
        # Cleanup temp files
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

    _deferred_eval_done = True
    eval_logger.info(
        f"\nDeferred evaluation complete. Evaluated {len(_evaluated_results_cache)} questions.\n"
    )

    return _evaluated_results_cache


def _get_evaluated_result(result):
    """Get the evaluated result, either from cache or the original result."""
    if not result.get("pending_judge", False):
        return result

    cache_key = (result["image_id"], result["question_idx"])
    if cache_key in _evaluated_results_cache:
        # Merge cached evaluation with original result
        cached = _evaluated_results_cache[cache_key]
        return {
            **result,
            "is_correct": cached["is_correct"],
            "is_cannot_answer": cached["is_cannot_answer"],
            "score": cached["score"],
            "model_answer": cached["model_answer"],
            "pending_judge": False,
        }

    # Fallback to original (unevaluated) result
    return result


# ---------- Aggregation Functions ----------


def captionqa_aggregate_score(results):
    """Aggregate CaptionQA score across all questions."""
    # Run judge evaluation (deferred until aggregation phase)
    _run_deferred_evaluation(results)

    all_scores = []
    category_scores = defaultdict(list)

    for result_list in results:
        if not isinstance(result_list, list):
            result_list = [result_list]
        for result in result_list:
            # Get evaluated result (from cache if deferred)
            result = _get_evaluated_result(result)
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
    # Run judge evaluation (may already be done by score aggregation)
    _run_deferred_evaluation(results)

    total_correct = 0
    total_questions = 0
    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    for result_list in results:
        if not isinstance(result_list, list):
            result_list = [result_list]
        for result in result_list:
            # Get evaluated result (from cache if deferred)
            result = _get_evaluated_result(result)
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
        cat_acc = (
            category_correct[category] / category_total[category]
            if category_total[category]
            else 0.0
        )
        eval_logger.info(
            f"  {category}: {cat_acc:.2%} ({category_correct[category]}/{category_total[category]})"
        )
    eval_logger.info(
        f"Overall Accuracy: {accuracy:.2%} ({total_correct}/{total_questions})"
    )
    eval_logger.info("=" * 60)

    return round(accuracy, 4)


def captionqa_aggregate_cannot_answer(results):
    """Aggregate 'cannot answer' rate across all questions."""
    # Run judge evaluation (may already be done)
    _run_deferred_evaluation(results)

    total_cannot_answer = 0
    total_questions = 0

    for result_list in results:
        if not isinstance(result_list, list):
            result_list = [result_list]
        for result in result_list:
            # Get evaluated result (from cache if deferred)
            result = _get_evaluated_result(result)
            total_questions += 1
            if result.get("is_cannot_answer", False):
                total_cannot_answer += 1

    if total_questions == 0:
        return 0.0

    rate = total_cannot_answer / total_questions
    eval_logger.info(
        f"'Cannot answer' rate: {rate:.2%} ({total_cannot_answer}/{total_questions})"
    )

    return round(rate, 4)
