import base64
import io
import re

from datasets import Dataset
from PIL import Image

from lmms_eval.filters.extraction import ExtendedRegexFilter


def ai2d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if lmms_eval_specific_kwargs["prompt_format"] == "mcq":
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"
    elif lmms_eval_specific_kwargs["prompt_format"] == "qa":
        options = "\n".join(choices)
        return f"{pre_prompt}{question}{options}{post_prompt}"
    elif lmms_eval_specific_kwargs["prompt_format"] == "mcq_xcomposer":
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = " ".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        return f"{pre_prompt}{question}\nContext: N/A\n{choices_str}{post_prompt}"
    else:
        raise ValueError(f"Unknown prompt format: {lmms_eval_specific_kwargs['prompt_format']}")


def ai2d_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def ai2d_vlmeval_process_docs(dataset: Dataset) -> Dataset:
    """Adapt AI2D_TEST TSV rows without losing their per-question images."""
    images = {}
    for row in dataset:
        index = str(row["index"])
        if index in images:
            raise ValueError(f"Duplicate AI2D_TEST index: {index}")
        images[index] = row["image"]

    def convert(row):
        value = row["image"]
        seen = set()
        # TSV images may reference another row; resolve before limit/sharding.
        while isinstance(value, str) and value.isdecimal():
            if value in seen or value not in images:
                raise ValueError(f"Invalid AI2D_TEST image reference: {value}")
            seen.add(value)
            value = images[value]
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing AI2D_TEST image at index {row['index']}")
        if row["answer"] not in ("A", "B", "C", "D"):
            raise ValueError(f"Invalid AI2D_TEST answer at index {row['index']}: {row['answer']!r}")
        options = [row[letter] for letter in "ABCD"]
        if any(not isinstance(option, str) or not option for option in options):
            raise ValueError(f"Missing AI2D_TEST option at index {row['index']}")
        return {"image": value, "options": options, "answer": str("ABCD".index(row["answer"]))}

    return dataset.map(convert)


def ai2d_vlmeval_doc_to_visual(doc: dict) -> list[Image.Image]:
    """Decode the image selected by the pinned VLMEvalKit dataset."""
    with Image.open(io.BytesIO(base64.b64decode(doc["image"], validate=True))) as image:
        return [image.convert("RGB")]


def ai2d_doc_to_target(doc, model_specific_target_kwargs):
    if model_specific_target_kwargs == "mcq":
        len_choices = len(doc["options"])
        options = [chr(ord("A") + i) for i in range(len_choices)]
        return options[int(doc["answer"])]
    elif model_specific_target_kwargs == "qa":
        return doc["options"][int(doc["answer"])]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            # Regex to directly extract the option letter from the model response
            option_letter_regex = re.compile(r"^\s*([A-Z])\.")

            # Process each response
            filtered = []
            for resp in r:
                # Try to match the option letter at the start of the response
                match = option_letter_regex.match(resp)
                if match:
                    # If a match is found, append the matched letter
                    filtered.append(match.group(1))
                else:
                    # If no match, return the original response
                    filtered.append(resp)

            # Assuming we need the first response that matches or the original response
            filtered_resps.append(filtered[0])

        return filtered_resps
