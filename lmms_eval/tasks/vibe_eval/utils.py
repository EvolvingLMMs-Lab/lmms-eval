from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

import yaml
import re
import os
from copy import deepcopy

try:
    from reka import ChatMessage
    from reka.client import Reka
except ImportError:
    eval_logger.warning("Reka is not installed, please install it by `pip install reka-api`")

REKA_API_KEY = os.getenv("REKA_API_KEY", "YOUR_API_KEY")

with open(Path(__file__).parent / "vibe_eval.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

EVALUATOR_NAME = config["metadata"]["evaluator"]

_PROMPT_WITH_IMAGE = """\
[Question]
{prompt}

[Assistant Response]
{generation}

[Ground Truth Response]
{reference}

[System]
Rate whether the assistant response correctly matches the ground truth, in regards to the image above.
The rating should be 1-5, where 1 is incorrect and 5 is correct.
Your response should be in the format:
Explanation: (your explanation)
Rating: (int)"""

_PROMPT_WITH_NO_IMAGE = """\
[Question]
{prompt}

[Assistant Response]
{generation}

[Ground Truth Response]
{reference}

[System]
Rate whether the assistant response correctly matches the ground truth, it's about an image shared by the user.
The rating should be 1-5, where 1 is incorrect and 5 is correct.
Your response should be in the format:
Explanation: (your explanation)
Rating: (int)"""


@dataclass
class Example:
    """An example loaded from vibe-eval, stored as jsonl in the repo."""

    example_id: str
    category: str
    prompt: str
    reference: str
    media_filename: str
    media_url: str

    # The fields below are not stored in the dataset, but are populated by this script.
    generation: Optional[str] = None
    score: Optional[int] = None
    evaluator_explanation: Optional[str] = None


class Evaluator(Enum):
    # Use Reka Core (including image input).
    REKA_CORE = "reka-core"

    # Use Reka Core, only using text input.
    REKA_CORE_TEXT = "reka-core-text"


def make_evaluator_prompt(example: Example, include_image: bool) -> str:
    return (_PROMPT_WITH_IMAGE if include_image else _PROMPT_WITH_NO_IMAGE).format(
        prompt=example.prompt,
        reference=example.reference,
        generation=example.generation,
    )


def evaluate(example: Example, evaluator: Evaluator) -> Example:
    """Evaluates the generation and populates the score and explanation fields."""
    include_image = evaluator == Evaluator.REKA_CORE
    evaluator_prompt = make_evaluator_prompt(example, include_image=include_image)
    client = Reka(api_key=REKA_API_KEY)
    content = [
        {"type": "text", "text": evaluator_prompt},
    ]
    if include_image:
        content.append({"type": "image_url", "image_url": example.media_url})
    evaluator_response = client.chat.create(
        messages=[
            ChatMessage(
                content=content,
                role="user",
            )
        ],
        model="reka-core",
        temperature=0.4,
        max_tokens=1024,
    )
    evaluator_response = evaluator_response.responses[0].message.content
    # evaluator_response = reka.chat(
    # human=evaluator_prompt,
    # media_url=example.media_url if include_image else None,
    # temperature=0.4,
    # model_name="reka-core-20240415",
    # request_output_len=1024,
    # )["text"]
    re_match = re.search(r"Rating:\s*([1-5])", evaluator_response)
    if re_match is None:
        example.score = 0
        example.evaluator_explanation = evaluator_response
        return example
    example.score = int(re_match.group(1))
    example.evaluator_explanation = evaluator_response
    return example


def vibe_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vibe_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["prompt"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def vibe_process_results(doc, results):
    example_id = doc["example_id"]
    category = doc["category"]
    prompt = doc["prompt"]
    reference = doc["reference"]
    media_filename = doc["media_url"]
    media_url = doc["media_url"]
    generation = results[0]
    example = Example(example_id=example_id, category=category, prompt=prompt, reference=reference, media_filename=media_filename, media_url=media_url, generation=generation)

    evaluator = Evaluator.REKA_CORE if EVALUATOR_NAME == "reka-core" else Evaluator.REKA_CORE_TEXT

    example = evaluate(example, evaluator=evaluator)
    data_dict = {
        "score": example.score,
        "evaluator_explanation": example.evaluator_explanation,
        "prompt": example.prompt,
        "generation": example.generation,
        "media_url": example.media_url,
        "category": example.category,
    }

    return {
        "hard": deepcopy(data_dict),
        "normal": deepcopy(data_dict),
        "all": deepcopy(data_dict),
    }


def _mean(scores: List[int]) -> float:
    """Scale from 1-5 to 0-100 and compute means."""
    return sum(25 * (score - 1) for score in scores) / len(scores)


def vibe_aggregation_results(results, category):
    score = []
    for res in results:
        if category in res["category"] or category == "all":
            score.append(res["score"])

    aggregate_scores = _mean(score)
    return aggregate_scores


def vibe_aggregation_results_normal(results):
    return vibe_aggregation_results(results, "normal")


def vibe_aggregation_results_hard(results):
    return vibe_aggregation_results(results, "hard")


def vibe_aggregation_results_all(results):
    return vibe_aggregation_results(results, "all")
