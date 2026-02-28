import base64
import io

from PIL import Image

from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

EVAL_AI_PROCESSOR = EvalAIAnswerProcessor()


def simplevqa_doc_to_visual(doc):
    image = doc["image"]
    if isinstance(image, Image.Image):
        return [image.convert("RGB")]

    image_bytes = base64.b64decode(image)
    decoded_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return [decoded_image]


def simplevqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["question"].strip()
    return f"{pre_prompt}{question}{post_prompt}"


def simplevqa_process_results(doc, result):
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    prediction = EVAL_AI_PROCESSOR(result[0])
    reference = EVAL_AI_PROCESSOR(doc["answer"])
    return {"exact_match": float(prediction == reference)}
