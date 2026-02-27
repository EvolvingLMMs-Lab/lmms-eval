import base64
import io
import re

from PIL import Image

from lmms_eval.tasks._task_utils.reasoning_utils import (
    make_reasoning_doc_to_messages,
    make_reasoning_process_results,
)

TASKS = [
    "Reasoning",
    "Perception",
]

SUBTASKS = [
    "Monitoring",
    "Autonomous_Driving",
    "OCR with Complex Context",
    "Diagram and Table",
    "Remote Sensing",
]


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def mme_realworld_doc_to_visual(doc):
    img = decode_base64_to_image(doc["bytes"])
    return [img.convert("RGB")]


def mme_realworld_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    option_prompt = "The choices are listed below:\n" + "\n".join(doc["multi-choice options"]) + "\n"

    question += " " + option_prompt + "Select best answer to above multiple-choice question based on image."
    return question


def mme_realworld_cn_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    option_prompt = "选项如下所示:\n" + "\n".join(doc["multi-choice options"]) + "\n"

    question += " " + option_prompt + "根据图像选择上述多项选择题的最佳答案。"
    return question


mme_realworld_reasoning_doc_to_messages = make_reasoning_doc_to_messages(mme_realworld_doc_to_visual, mme_realworld_doc_to_text)
mme_realworld_cn_reasoning_doc_to_messages = make_reasoning_doc_to_messages(mme_realworld_doc_to_visual, mme_realworld_cn_doc_to_text)


def extract_characters_regex(s, choices=["(A)", "(B)", "(C)", "(D)", "(E)"]):
    if type(s) is dict:
        s = ""
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""
    matches = re.search(r"[ABCDE]", s)
    if matches is None:
        for choice in choices:
            if s.lower() in choice.lower():
                return choice[1]
        return ""
    return matches[0]


def get_correct_answer(sample):
    sample["multi-choice options"] = [option.replace("（", "(").replace("）", ")") for option in sample["multi-choice options"]]
    correct_answer = next(option.split(") ")[1] for option in sample["multi-choice options"] if option.startswith(f"({sample['answer']})"))
    return correct_answer


mme_realworld_reasoning_process_results = make_reasoning_process_results("mmerealworld", mme_realworld_doc_to_text)
mme_realworld_cn_reasoning_process_results = make_reasoning_process_results("mmerealworld_cn", mme_realworld_cn_doc_to_text)
