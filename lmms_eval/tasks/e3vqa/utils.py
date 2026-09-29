from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer


SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "You are provided with two visual inputs in sequence, each captured from a different perspective:\n"
    "1. The view from the camera worn by the user ('I').\n"
    "2. The view captured by an external camera observing the user ('I').\n\n"
    "The first image shows what the user ('I') sees from their perspective.\n"
    "The user's ('My') full body cannot be visible; you may only see parts of their body, "
    "like their hand, foot, or arm, or in some cases, none of the user's body at all.\n\n"
    "The second image shows both the user and the environment from a third-person perspective "
    "with a broad view.\n"
    "The user's ('My') full body is visible, but due to the fixed viewpoint, some parts may not be visible.\n\n"
    "These two images capture the same event at the same time.\n"
    "Your task is to analyze both images along with the question and provide the most accurate "
    "response based on the visual information from both perspectives.\n"
)


def e3vqa_filter_egoexo4d(dataset):
    return dataset.filter(lambda x: x["source"] == "Ego-Exo4D")


def e3vqa_filter_lemma(dataset):
    return dataset.filter(lambda x: x["source"] == "LEMMA")


def e3vqa_doc_to_visual(doc):
    return [
        doc["ego"].convert("RGB"),
        doc["exo"].convert("RGB"),
    ]


def _build_question_prompt(doc):
    formatted_options = "\n".join(
        f"{chr(65 + i)}) {option}"
        for i, option in enumerate(doc["options"])
    )

    return (
        f"Question:\n"
        f"{doc['question']}\n\n"
        f"Choices:\n"
        f"{formatted_options}\n\n"
        f"Only one option is correct.\n"
        f"Present the answer in the form X).\n\n"
    )


def e3vqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # Legacy/simple model fallback.
    # The system instruction is flattened into the text prompt.
    return SYSTEM_PROMPT + "\n" + _build_question_prompt(doc)


def e3vqa_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    # Chat-model interface. Preserve the original order:
    # first ego image, then exo image, then question.
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": doc["ego"].convert("RGB"),
                },
                {
                    "type": "image",
                    "url": doc["exo"].convert("RGB"),
                },
                {
                    "type": "text",
                    "text": _build_question_prompt(doc),
                },
            ],
        },
    ]


def e3vqa_doc_to_target(doc):
    gold_answer = str(doc["answer"]).strip().lower()

    for i, option in enumerate(doc["options"]):
        if gold_answer == str(option).strip().lower():
            return chr(65 + i)

    raise ValueError(
        f"Answer '{doc['answer']}' is not found in options for sample {doc['id']}."
    )


def _score_sample(doc, results):
    prediction = results[0] if results else ""

    pred_answer = extract_mcq_answer(
        prediction,
        choices=["A", "B", "C", "D"],
    )

    gold_answer = e3vqa_doc_to_target(doc)

    return float(pred_answer == gold_answer)


def e3vqa_process_results(doc, results):
    """
    Used by source-specific tasks:
      - e3vqa_egoexo4d
      - e3vqa_lemma

    Returns overall accuracy and the corresponding
    category-perspective metric for this sample.
    """
    score = _score_sample(doc, results)

    category = doc["category"].lower()
    perspective = doc["perspective"].lower()

    subset_metric = f"{category}_{perspective}_acc"

    return {
        "overall_acc": score,
        subset_metric: score,
    }


def e3vqa_process_results_full(doc, results):
    """
    Used by the full E3VQA task.

    In addition to overall and category-perspective accuracy,
    returns the corresponding source-specific metric.
    """
    metrics = e3vqa_process_results(doc, results)

    score = metrics["overall_acc"]

    if doc["source"] == "Ego-Exo4D":
        metrics["egoexo4d_acc"] = score
    elif doc["source"] == "LEMMA":
        metrics["lemma_acc"] = score
    else:
        raise ValueError(
            f"Unknown source '{doc['source']}' for sample {doc['id']}."
        )

    return metrics