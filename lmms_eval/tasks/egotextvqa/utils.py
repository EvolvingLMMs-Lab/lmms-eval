"""EgoTextVQA-Indoor task utilities.

Annotations come from ``nv-njb/EgoTextVQA-Indoor``. Videos are *not*
redistributed; users must obtain them from
https://github.com/zhangyuanhan-ai/EgoTextVQA.

The video directory is resolved in this order:

1. ``EGOTEXTVQA_VIDEO_DIR`` environment variable, if set.
2. ``~/.cache/lmms_eval/egotextvqa/videos/`` (default).

EgoTextVQA's metric is a GPT-based semantic-match judge. Configure with:

- ``JUDGE_OPENAI_API_KEY`` (required for the judge)
- ``JUDGE_OPENAI_API_BASE`` (optional, for non-OpenAI endpoints)
- ``EGOTEXTVQA_GPT_MODEL`` (default: ``gpt-4o-mini``)

When the judge cannot be reached, the metric falls back to substring
matching so the task still runs.
"""

import ast
import os
from pathlib import Path

from loguru import logger as eval_logger

GPT_EVAL_MODEL = os.environ.get("EGOTEXTVQA_GPT_MODEL", "gpt-4o-mini")
GPT_API_BASE = os.environ.get("JUDGE_OPENAI_API_BASE", None)
GPT_API_KEY = os.environ.get("JUDGE_OPENAI_API_KEY", None)


def _video_dir() -> str:
    override = os.environ.get("EGOTEXTVQA_VIDEO_DIR")
    if override:
        return override
    return str(Path.home() / ".cache" / "lmms_eval" / "egotextvqa" / "videos")


def egotextvqa_doc_to_visual(doc):
    video_path = os.path.join(_video_dir(), f"{doc['video_id']}.mp4")
    if not os.path.exists(video_path):
        eval_logger.warning(
            f"Video not found: {video_path}. Set EGOTEXTVQA_VIDEO_DIR or place "
            f"videos under ~/.cache/lmms_eval/egotextvqa/videos/."
        )
    return [video_path]


def egotextvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    prompt = (
        "You are a person in the situation shown in the following consecutive images from a video. \n"
        "You can answer questions that humans ask to help them make decisions. "
        "Now you are observing your surroundings and answering questions based on the current situation. "
        "Understanding the scene text around you is important for answering questions. "
        "Answer the questions in the first-person perspective. "
        "Answer the question as detailed as possible, covering all relevant aspects and providing comprehensive context."
        f"\n\nQuestion: {doc['question']}"
    )
    return f"{pre_prompt}{prompt}{post_prompt}"


def _gpt_evaluate(question, ground_truth, prediction):
    """Use GPT to evaluate semantic match between prediction and ground truth.

    Falls back to substring matching when the judge can't be reached.
    """
    if not GPT_API_KEY:
        eval_logger.warning(
            "JUDGE_OPENAI_API_KEY not set; falling back to substring matching."
        )
    else:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=GPT_API_KEY, base_url=GPT_API_BASE)
            system_prompt = (
                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the meaningful match between the predicted answer and the correct answer. "
                "Please note that not only matches of noun phrases between answers, but also matches of prepositional phrases. "
                'For example, "at the car wash on your right" does not exactly match "car wash". '
                '"at the gas station beside the sign \'gas sale\'" does not exactly match "gas station".\n'
                "- Consider synonyms or paraphrases as valid matches. "
                "Note that the predicted answer must be consistent with the string type of the correct answer, which may include phone numbers, email addresses, numbers, dates, etc. "
                'For example, the string types of "www.usps.com" and "visit their website" are inconsistent, '
                'and the string types of "9849041316" and "advertiser\'s contact number" are inconsistent.\n'
                "- Evaluate the correctness of the prediction compared to the answer."
            )
            user_prompt = (
                "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {ground_truth}\n"
                f"Predicted Answer: {prediction}\n\n"
                "Provide your eval_code only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. "
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'pred': 'yes', 'score': 5}, {'pred': 'no', 'score': 1}."
            )
            response = client.chat.completions.create(
                model=GPT_EVAL_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=50,
            )
            eval_dict = ast.literal_eval(response.choices[0].message.content.strip())
            return {
                "is_correct": int(eval_dict.get("pred", "no").lower() == "yes"),
                "score": eval_dict.get("score", 0),
            }
        except Exception as e:
            eval_logger.warning(f"GPT evaluation error: {e}, falling back to substring matching")

    pred_lower = prediction.lower().strip()
    gt_lower = ground_truth.lower().strip()
    is_match = pred_lower == gt_lower or pred_lower in gt_lower or gt_lower in pred_lower
    return {"is_correct": int(is_match), "score": 5 if is_match else 0}


def egotextvqa_process_results(doc, results):
    pred = results[0].strip()
    ground_truth = str(doc["correct_answer"]).replace("\n", "").lower()
    prediction = pred.replace("\n", "").lower()
    eval_result = _gpt_evaluate(doc["question"], ground_truth, prediction)
    return {
        "egotextvqa_accuracy": eval_result["is_correct"],
        "egotextvqa_score": eval_result["score"],
    }


def egotextvqa_aggregate_accuracy(results):
    return sum(results) / len(results) if results else 0


def egotextvqa_aggregate_score(results):
    return sum(results) / len(results) if results else 0
