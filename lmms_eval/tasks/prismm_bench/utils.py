"""
Utility functions for PRISMM-Bench evaluation task.

PRISMM-Bench is a dataset for evaluating LMMs on detecting inconsistencies
in scientific papers. The task presents parts of a scientific paper (both
text and images) and asks the model to identify what the inconsistency is
from multiple choice options.
"""

# ============================================================================
# Task Default Functions
# ============================================================================


def prismm_doc_to_visual(doc):
    """
    Extract all images from the 'parts' field of the document.

    Args:
        doc: A document from the PRISMM-Bench dataset

    Returns:
        List of PIL Images in RGB format
    """
    visuals = []
    parts = doc.get("parts", {})

    # Extract images from parts
    if isinstance(parts, dict):
        content_images = parts.get("content_image", [])
        for img in content_images:
            if img is not None:
                visuals.append(img.convert("RGB"))

    return visuals


def prismm_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Construct the question text with all text parts and multiple choice options.

    Args:
        doc: A document from the PRISMM-Bench dataset
        lmms_eval_specific_kwargs: Model-specific configuration

    Returns:
        Formatted question string with context and choices
    """
    # Get the task_default question and choices
    task_default = doc.get("task_default", {})
    question = task_default.get(
        "question", "What is the inconsistency in these parts of a scientific paper?"
    )
    choices = task_default.get("choices", [])

    # Extract text content from parts
    parts = doc.get("parts", {})
    text_parts = []

    if isinstance(parts, dict):
        content_texts = parts.get("content_text", [])
        part_types = parts.get("type", [])

        for i, text in enumerate(content_texts):
            if text:
                part_type = part_types[i] if i < len(part_types) else "unknown"
                text_parts.append(f"[{part_type.upper()}]: {text}")

    # Combine all text parts
    context = "\n\n".join(text_parts) if text_parts else ""

    # Get the letter mapping from the dataset (to handle debiasing)
    letters = task_default.get("letters", [])

    # If letters are not provided, fall back to A, B, C, D
    if not letters or len(letters) != len(choices):
        letters = [chr(ord("A") + i) for i in range(len(choices))]

    # Format the choices with their corresponding letters
    choices_str = "\n".join(
        [f"{letter}. {choice}" for letter, choice in zip(letters, choices)]
    )

    # Get pre and post prompts
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {
            "pre_prompt": "",
            "post_prompt": "\nAnswer with the option's letter from the given choices directly.",
        }

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "\nAnswer with the option's letter from the given choices directly.",
    )

    # Construct the full prompt
    if context:
        full_text = (
            f"{pre_prompt}Context:\n{context}\n\n{question}\n{choices_str}{post_prompt}"
        )
    else:
        full_text = f"{pre_prompt}{question}\n{choices_str}{post_prompt}"

    return full_text


def prismm_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """
    Convert document to message format for interleaved text-image content.

    This function creates a messages array that properly interleaves the images
    from the 'parts' with the text content.

    Args:
        doc: A document from the PRISMM-Bench dataset
        lmms_eval_specific_kwargs: Model-specific configuration

    Returns:
        List of message dictionaries in the format expected by the model
    """
    # Get the question text
    question_text = prismm_doc_to_text(doc, lmms_eval_specific_kwargs)

    # Get all visuals
    visuals = prismm_doc_to_visual(doc)

    # Create the message structure
    messages = [{"role": "user", "content": []}]

    # Add all images first, then the text
    # This is a simple approach - you could make it more sophisticated
    # by interleaving images with their corresponding text parts
    for img in visuals:
        messages[0]["content"].append({"type": "image", "url": img})

    # Add the question text
    messages[0]["content"].append({"type": "text", "text": question_text})

    return messages


def prismm_process_results(doc, results):
    """
    Process the model results and compare with the ground truth answer.

    Args:
        doc: A document from the PRISMM-Bench dataset
        results: List of model predictions

    Returns:
        Dictionary with the exact_match score
    """
    pred = results[0].strip()
    task_default = doc.get("task_default", {})
    target = task_default.get("answer", "").strip().upper()

    # Normalize the prediction
    pred_upper = pred.upper()

    # Check for exact match with the answer letter
    if pred_upper == target:
        return {"exact_match": 1.0}

    # Check if the answer starts with the letter followed by a period or other punctuation
    # Pattern: "A. Some text" or "A) Some text" or just "A"
    if len(pred_upper) >= 1:
        first_char = pred_upper[0]
        if first_char == target:
            return {"exact_match": 1.0}

    return {"exact_match": 0.0}


# ============================================================================
# Task Edit Functions
# ============================================================================


def prismm_edit_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Construct the question text for the task_edit task.
    Similar to task_default but uses the task_edit field.

    Args:
        doc: A document from the PRISMM-Bench dataset
        lmms_eval_specific_kwargs: Model-specific configuration

    Returns:
        Formatted question string with context and choices
    """
    # Get the task_edit question and choices
    task_edit = doc.get("task_edit", {})
    question = task_edit.get(
        "question",
        "What action needs to be taken to resolve the inconsistency in these parts of a scientific paper?",
    )
    choices = task_edit.get("choices", [])

    # Extract text content from parts
    parts = doc.get("parts", {})
    text_parts = []

    if isinstance(parts, dict):
        content_texts = parts.get("content_text", [])
        part_types = parts.get("type", [])

        for i, text in enumerate(content_texts):
            if text:
                part_type = part_types[i] if i < len(part_types) else "unknown"
                text_parts.append(f"[{part_type.upper()}]: {text}")

    # Combine all text parts
    context = "\n\n".join(text_parts) if text_parts else ""

    # Get the letter mapping from the dataset (to handle debiasing)
    letters = task_edit.get("letters", [])

    # If letters are not provided, fall back to A, B, C, D
    if not letters or len(letters) != len(choices):
        letters = [chr(ord("A") + i) for i in range(len(choices))]

    # Format the choices with their corresponding letters
    choices_str = "\n".join(
        [f"{letter}. {choice}" for letter, choice in zip(letters, choices)]
    )

    # Get pre and post prompts
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {
            "pre_prompt": "",
            "post_prompt": "\nAnswer with the option's letter from the given choices directly.",
        }

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "\nAnswer with the option's letter from the given choices directly.",
    )

    # Construct the full prompt
    if context:
        full_text = (
            f"{pre_prompt}Context:\n{context}\n\n{question}\n{choices_str}{post_prompt}"
        )
    else:
        full_text = f"{pre_prompt}{question}\n{choices_str}{post_prompt}"

    return full_text


def prismm_edit_doc_to_target(doc):
    """Get the target answer for task_edit."""
    task_edit = doc.get("task_edit", {})
    return task_edit.get("answer", "")


def prismm_edit_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """
    Convert document to message format for task_edit.

    Args:
        doc: A document from the PRISMM-Bench dataset
        lmms_eval_specific_kwargs: Model-specific configuration

    Returns:
        List of message dictionaries in the format expected by the model
    """
    # Get the question text
    question_text = prismm_edit_doc_to_text(doc, lmms_eval_specific_kwargs)

    # Get all visuals (same as task_default)
    visuals = prismm_doc_to_visual(doc)

    # Create the message structure
    messages = [{"role": "user", "content": []}]

    # Add all images first, then the text
    for img in visuals:
        messages[0]["content"].append({"type": "image", "url": img})

    # Add the question text
    messages[0]["content"].append({"type": "text", "text": question_text})

    return messages


def prismm_edit_process_results(doc, results):
    """
    Process the model results for task_edit.

    Args:
        doc: A document from the PRISMM-Bench dataset
        results: List of model predictions

    Returns:
        Dictionary with the exact_match score
    """
    pred = results[0].strip()
    task_edit = doc.get("task_edit", {})
    target = task_edit.get("answer", "").strip().upper()

    # Normalize the prediction
    pred_upper = pred.upper()

    # Check for exact match with the answer letter
    if pred_upper == target:
        return {"exact_match": 1.0}

    # Check if the answer starts with the letter
    if len(pred_upper) >= 1:
        first_char = pred_upper[0]
        if first_char == target:
            return {"exact_match": 1.0}

    return {"exact_match": 0.0}


# ============================================================================
# Task Pair Match Functions
# ============================================================================


def prismm_pair_match_filter_docs(dataset):
    """
    Filter the dataset to only include samples where pair_match task is available.

    Args:
        dataset: HuggingFace dataset

    Returns:
        Filtered dataset with only available pair_match samples
    """
    return dataset.filter(lambda x: x["task_pair_match"]["is_available"])


def prismm_pair_match_doc_to_visual(doc):
    """
    Extract visuals for the pair_match task.
    Returns query image (if image type) followed by all candidate images.

    Args:
        doc: A document from the PRISMM-Bench dataset

    Returns:
        List of PIL Images in RGB format
    """
    visuals = []
    task = doc.get("task_pair_match", {})

    # Add query image if it's an image query
    if task.get("query_type") == "image" and task.get("query_image") is not None:
        visuals.append(task["query_image"].convert("RGB"))

    # Add all candidate images
    candidates = task.get("candidates", [])
    for candidate in candidates:
        if candidate is not None:
            visuals.append(candidate.convert("RGB"))

    return visuals


def prismm_pair_match_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Construct the question text for the pair_match task.

    Args:
        doc: A document from the PRISMM-Bench dataset
        lmms_eval_specific_kwargs: Model-specific configuration

    Returns:
        Formatted question string
    """
    task = doc.get("task_pair_match", {})

    # Get pre and post prompts
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {
            "pre_prompt": "",
            "post_prompt": "\nAnswer with the option's letter from the given choices directly.",
        }

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "\nAnswer with the option's letter from the given choices directly.",
    )

    # Build query part
    if task.get("query_type") == "text":
        query = f"Query (text):\n{task.get('query_text', '')}"
    else:
        query = "Query: <image>"

    # Build question
    question = "You are provided with a part of a scientific paper. The combination with one of the other parts within the same paper results in an inconsistency. Pick the letter of the correct answer option."

    # Build options with letters
    letters = task.get("letters", [])
    if not letters:
        # Fallback to A, B, C, D
        num_candidates = len(task.get("candidates", []))
        letters = [chr(ord("A") + i) for i in range(num_candidates)]

    options = "\n".join([f"{letter}. <image>" for letter in letters])

    return f"{pre_prompt}{query}\n\n{question}\n\n{options}{post_prompt}"


def prismm_pair_match_doc_to_target(doc):
    """Get the target answer for task_pair_match."""
    task = doc.get("task_pair_match", {})
    return task.get("answer", "")


def prismm_pair_match_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """
    Convert document to message format for task_pair_match with proper interleaving.

    Args:
        doc: A document from the PRISMM-Bench dataset
        lmms_eval_specific_kwargs: Model-specific configuration

    Returns:
        List of message dictionaries in the format expected by the model
    """
    task = doc.get("task_pair_match", {})
    messages = [{"role": "user", "content": []}]

    # Get pre and post prompts
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {
            "pre_prompt": "",
            "post_prompt": "\nAnswer with the option's letter from the given choices directly.",
        }

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "\nAnswer with the option's letter from the given choices directly.",
    )

    # Add pre-prompt if any
    if pre_prompt:
        messages[0]["content"].append({"type": "text", "text": pre_prompt})

    # Add query
    if task.get("query_type") == "text":
        messages[0]["content"].append(
            {"type": "text", "text": f"Query:\n{task.get('query_text', '')}"}
        )
    else:
        messages[0]["content"].append({"type": "text", "text": "Query:"})
        if task.get("query_image") is not None:
            messages[0]["content"].append(
                {"type": "image", "url": task["query_image"].convert("RGB")}
            )

    # Add question text
    messages[0]["content"].append(
        {
            "type": "text",
            "text": "\n\nYou are provided with a part of a scientific paper. The combination with one of the other parts within the same paper results in an inconsistency. Pick the letter of the correct answer option.\n\n",
        }
    )

    # Add candidates with letters
    letters = task.get("letters", [])
    candidates = task.get("candidates", [])

    if not letters:
        # Fallback to A, B, C, D
        letters = [chr(ord("A") + i) for i in range(len(candidates))]

    for letter, candidate in zip(letters, candidates):
        messages[0]["content"].append({"type": "text", "text": f"{letter}. "})
        if candidate is not None:
            messages[0]["content"].append(
                {"type": "image", "url": candidate.convert("RGB")}
            )
        messages[0]["content"].append({"type": "text", "text": "\n"})

    # Add post-prompt
    messages[0]["content"].append({"type": "text", "text": post_prompt})

    return messages


def prismm_pair_match_process_results(doc, results):
    """
    Process the model results for task_pair_match.

    Args:
        doc: A document from the PRISMM-Bench dataset
        results: List of model predictions

    Returns:
        Dictionary with the exact_match score
    """
    pred = results[0].strip()
    task = doc.get("task_pair_match", {})
    target = task.get("answer", "").strip().upper()

    # Normalize the prediction
    pred_upper = pred.upper()

    # Check for exact match with the answer letter
    if pred_upper == target:
        return {"exact_match": 1.0}

    # Check if the answer starts with the letter
    if len(pred_upper) >= 1:
        first_char = pred_upper[0]
        if first_char == target:
            return {"exact_match": 1.0}

    return {"exact_match": 0.0}
