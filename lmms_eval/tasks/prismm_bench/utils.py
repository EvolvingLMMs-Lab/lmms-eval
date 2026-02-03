from matplotlib.pylab import choice

SYSTEM_PROMPT = "Answer with only the letter of the correct answer (A, B, C, or D), do not output anything else."


def prismm_doc_to_visual(doc):
    """Extract all images from the 'parts' field of the document."""
    visuals = []
    parts = doc.get("parts", {})

    if isinstance(parts, dict):
        content_images = parts.get("content_image", [])
        for img in content_images:
            if img is not None:
                visuals.append(img.convert("RGB"))

    return visuals


def prismm_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Construct the question text with all text parts and multiple choice options."""
    task_identification = doc.get("task_identification", {})
    question = task_identification.get(
        "question", "What is the inconsistency in these parts of a scientific paper?"
    )
    choices = task_identification.get("choices", [])

    parts = doc.get("parts", {})
    text_parts = []

    if isinstance(parts, dict):
        content_texts = parts.get("content_text", [])
        part_types = parts.get("type", [])

        for i, text in enumerate(content_texts):
            if text:
                part_type = part_types[i] if i < len(part_types) else "unknown"
                text_parts.append(text)

    context = "\n\n".join(text_parts) if text_parts else ""

    letters = task_identification.get("letters", [])
    if not letters or len(letters) != len(choices):
        letters = [chr(ord("A") + i) for i in range(len(choices))]

    choices_str = "\n".join(choices)

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {
            "pre_prompt": "",
            "post_prompt": "\nOutput only a single letter corresponding to the correct answer. Do not output any explanation or text of the question again.",
        }

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "\nOutput only a single letter corresponding to the correct answer. Do not output any explanation or text of the question again.",
    )

    if context:
        full_text = f"{pre_prompt}\n{context}\n\n{question}\n{choices_str}{post_prompt}"
    else:
        full_text = f"{pre_prompt}{question}\n{choices_str}{post_prompt}"

    return full_text


def prismm_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """Convert document to message format for interleaved text-image content."""
    task_identification = doc.get("task_identification", {})
    question = task_identification.get(
        "question", "What is the inconsistency in these parts of a scientific paper?"
    )
    choices = task_identification.get("choices", [])

    parts = doc.get("parts", {})
    text_parts = []

    if isinstance(parts, dict):
        content_texts = parts.get("content_text", [])
        part_types = parts.get("type", [])

        for i, text in enumerate(content_texts):
            if text:
                part_type = part_types[i] if i < len(part_types) else "unknown"
                text_parts.append(text)

    context = "\n\n".join(text_parts) if text_parts else ""

    letters = task_identification.get("letters", [])
    if not letters or len(letters) != len(choices):
        letters = [chr(ord("A") + i) for i in range(len(choices))]

    letter_choice_pairs = list(zip(letters, choices))
    letter_choice_pairs.sort(key=lambda x: x[0])

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    visuals = prismm_doc_to_visual(doc)

    system_messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    ]
    messages = [{"role": "user", "content": []}]

    for img in visuals:
        messages[0]["content"].append({"type": "image", "url": img})

    if pre_prompt:
        messages[0]["content"].append({"type": "text", "text": pre_prompt})

    if context:
        messages[0]["content"].append({"type": "text", "text": context})

    messages[0]["content"].append({"type": "text", "text": question})

    for letter, choice in letter_choice_pairs:
        messages[0]["content"].append({"type": "text", "text": choice})

    if post_prompt:
        messages[0]["content"].append({"type": "text", "text": post_prompt})

    messages = system_messages + messages
    return messages


def prismm_process_results(doc, results):
    """Process the model results and compare with the ground truth answer."""
    pred = results[0].strip()
    task_identification = doc.get("task_identification", {})
    target = task_identification.get("answer", "").strip().upper()

    pred_upper = pred.upper()

    if pred_upper == target:
        return {"exact_match": 1.0}

    if len(pred_upper) >= 1 and pred_upper[0] == target:
        return {"exact_match": 1.0}

    return {"exact_match": 0.0}


def prismm_edit_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Construct the question text for the task_remedy task."""
    task_remedy = doc.get("task_remedy", {})
    question = task_remedy.get(
        "question",
        "What action needs to be taken to resolve the inconsistency in these parts of a scientific paper?",
    )
    choices = task_remedy.get("choices", [])

    parts = doc.get("parts", {})
    text_parts = []

    if isinstance(parts, dict):
        content_texts = parts.get("content_text", [])
        part_types = parts.get("type", [])

        for i, text in enumerate(content_texts):
            if text:
                part_type = part_types[i] if i < len(part_types) else "unknown"
                text_parts.append(text)

    context = "\n\n".join(text_parts) if text_parts else ""

    letters = task_remedy.get("letters", [])
    if not letters or len(letters) != len(choices):
        letters = [chr(ord("A") + i) for i in range(len(choices))]

    choices_str = "\n".join(choices)

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    if context:
        full_text = f"{pre_prompt}\n{context}\n\n{question}\n{choices_str}{post_prompt}"
    else:
        full_text = f"{pre_prompt}{question}\n{choices_str}{post_prompt}"

    return full_text


def prismm_edit_doc_to_target(doc):
    """Get the target answer for task_remedy."""
    task_remedy = doc.get("task_remedy", {})
    return task_remedy.get("answer", "")


def prismm_edit_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """Convert document to message format for task_remedy."""
    task_remedy = doc.get("task_remedy", {})
    question = task_remedy.get(
        "question",
        "What action needs to be taken to resolve the inconsistency in these parts of a scientific paper?",
    )
    choices = task_remedy.get("choices", [])

    parts = doc.get("parts", {})
    text_parts = []

    if isinstance(parts, dict):
        content_texts = parts.get("content_text", [])
        part_types = parts.get("type", [])

        for i, text in enumerate(content_texts):
            if text:
                part_type = part_types[i] if i < len(part_types) else "unknown"
                text_parts.append(text)

    context = "\n\n".join(text_parts) if text_parts else ""

    letters = task_remedy.get("letters", [])
    if not letters or len(letters) != len(choices):
        letters = [chr(ord("A") + i) for i in range(len(choices))]

    letter_choice_pairs = list(zip(letters, choices))
    letter_choice_pairs.sort(key=lambda x: x[0])

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    visuals = prismm_doc_to_visual(doc)

    system_messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    ]
    messages = [{"role": "user", "content": []}]

    for img in visuals:
        messages[0]["content"].append({"type": "image", "url": img})

    if pre_prompt:
        messages[0]["content"].append({"type": "text", "text": pre_prompt})

    if context:
        messages[0]["content"].append({"type": "text", "text": context})

    messages[0]["content"].append({"type": "text", "text": question})

    for letter, choice in letter_choice_pairs:
        messages[0]["content"].append({"type": "text", "text": f"{letter}) {choice}"})

    if post_prompt:
        messages[0]["content"].append({"type": "text", "text": post_prompt})

    messages = system_messages + messages
    return messages


def prismm_edit_process_results(doc, results):
    """Process the model results for task_remedy."""
    pred = results[0].strip()
    task_remedy = doc.get("task_remedy", {})
    target = task_remedy.get("answer", "").strip().upper()

    pred_upper = pred.upper()

    if pred_upper == target:
        return {"exact_match": 1.0}

    if len(pred_upper) >= 1 and pred_upper[0] == target:
        return {"exact_match": 1.0}

    return {"exact_match": 0.0}


def prismm_pair_match_filter_docs(dataset):
    """Filter the dataset to only include samples where pair_match task is available."""
    return dataset.filter(lambda x: x["task_pair_match"]["is_available"])


def prismm_pair_match_doc_to_visual(doc):
    """Extract visuals for the pair_match task."""
    visuals = []
    task = doc.get("task_pair_match", {})

    if task.get("query_type") == "image" and task.get("query_image") is not None:
        visuals.append(task["query_image"].convert("RGB"))

    candidates = task.get("candidates", [])
    for candidate in candidates:
        if candidate is not None:
            visuals.append(candidate.convert("RGB"))

    return visuals


def prismm_pair_match_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Construct the question text for the pair_match task."""
    task = doc.get("task_pair_match", {})

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    intro = "You are provided with a part of a scientific paper:"
    if task.get("query_type") == "text":
        query = task.get("query_text", "")
    else:
        query = "<image>"

    question = "The combination with one of the other parts within the same paper results in an inconsistency. Pick the letter of the correct answer option."

    letters = task.get("letters", [])
    if not letters:
        num_candidates = len(task.get("candidates", []))
        letters = [chr(ord("A") + i) for i in range(num_candidates)]

    options = "\n".join([f"{letter}) <image>" for letter in letters])

    return f"{pre_prompt}{intro}\n{query}\n{question}\n{options}{post_prompt}"


def prismm_pair_match_doc_to_target(doc):
    """Get the target answer for task_pair_match."""
    task = doc.get("task_pair_match", {})
    return task.get("answer", "")


def prismm_pair_match_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """Convert document to message format for task_pair_match with proper interleaving."""
    task = doc.get("task_pair_match", {})

    system_messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    ]
    messages = [{"role": "user", "content": []}]

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    if pre_prompt:
        messages[0]["content"].append({"type": "text", "text": pre_prompt})

    messages[0]["content"].append(
        {
            "type": "text",
            "text": "You are provided with a part of a scientific paper:\n",
        }
    )

    if task.get("query_type") == "text":
        messages[0]["content"].append(
            {"type": "text", "text": task.get("query_text", "")}
        )
    else:
        if task.get("query_image") is not None:
            messages[0]["content"].append(
                {"type": "image", "url": task["query_image"].convert("RGB")}
            )

    messages[0]["content"].append(
        {
            "type": "text",
            "text": "\nThe combination with one of the other parts within the same paper results in an inconsistency. Pick the letter of the correct answer option.\n",
        }
    )

    letters = task.get("letters", [])
    candidates = task.get("candidates", [])

    if not letters:
        letters = [chr(ord("A") + i) for i in range(len(candidates))]

    for letter, candidate in zip(letters, candidates):
        messages[0]["content"].append({"type": "text", "text": f"{letter}) "})
        if candidate is not None:
            messages[0]["content"].append(
                {"type": "image", "url": candidate.convert("RGB")}
            )
        messages[0]["content"].append({"type": "text", "text": "\n"})

    messages[0]["content"].append({"type": "text", "text": post_prompt})

    messages = system_messages + messages
    return messages


def prismm_pair_match_process_results(doc, results):
    """Process the model results for task_pair_match."""
    pred = results[0].strip()
    task = doc.get("task_pair_match", {})
    target = task.get("answer", "").strip("'.\" )").upper()

    pred_upper = pred.upper()

    if pred_upper == target:
        return {"exact_match": 1.0}

    if len(pred_upper) >= 1 and pred_upper[0] == target:
        return {"exact_match": 1.0}

    return {"exact_match": 0.0}
