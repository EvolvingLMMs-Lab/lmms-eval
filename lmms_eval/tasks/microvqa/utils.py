"""MicroVQA evaluation protocol.

Source: jmhb0/microvqa@3b52dc7131c3a285c33654856b349d9073e3604b.
"""

import re

PROMPT_TEMPLATE = """\
The following is a multiple choice question (with answers).
Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.

{{QUESTION}}

Options:
{{CHOICES}}
"""

ANSWER_PATTERN = r"answer is \*?\*?\(?([0-9])\)?\*?\*?"


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = PROMPT_TEMPLATE.replace("{{QUESTION}}", doc["question"])
    choices = "".join(f"  ({index + 1}): {choice}\n" for index, choice in enumerate(doc["choices"]))
    return prompt.replace("{{CHOICES}}", choices)


def doc_to_visual(doc):
    images = doc["images_list"]
    assert images, "MicroVQA owner protocol requires at least one image"
    assert all(image is not None for image in images), "MicroVQA sample contains a null image"
    return [image.convert("RGB") for image in images]


def doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    content = [{"type": "text", "text": doc_to_text(doc)}]
    content.extend({"type": "image", "url": image} for image in doc_to_visual(doc))
    return [{"role": "user", "content": content}]


def doc_to_target(doc):
    return doc["correct_index"]


def process_results(doc, results):
    assert len(results) == 1, f"Expected one response, got {len(results)}"
    match = re.search(ANSWER_PATTERN, results[0])
    prediction = int(match.group(1)) - 1 if match is not None else -1
    return {"accuracy": float(prediction == doc["correct_index"])}
