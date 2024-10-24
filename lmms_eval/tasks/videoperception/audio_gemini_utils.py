import json
import logging

import pandas as pd

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.videoperception.utils import (
    videoperception_doc_to_answer as original_doc_to_answer,
)
from lmms_eval.tasks.videoperception.utils import (
    videoperception_doc_to_text as original_doc_to_text,
)

logger = logging.getLogger(__name__)

template = """\
[System]

You are an assistant that helps with question evaluation. I will provide you with a video, along with a pair of questions and answers. Your task is to assess whether the question requires audio information from the video to be answered, or if it can be answered purely through visual information. You need to provide a complete and detailed reason explaining why the question does or does not require audio from the video.

[Question]

{question}

[Answer]

{answer}

[Standard]

The standard for determining whether audio is necessary is: if a question does not require audio, then I should be able to turn off the video's sound and still be able to infer the correct answer entirely from the visual information.

[Output Format]

Your answer must strictly follow the JSON format below:

{{
    "reason": "This question requires audio information from the video to be answered because...",
    "use_audio": true
}}

"use_audio" should be set to "true" if the question requires audio information from the video to be answered, and "false" otherwise.

Please note that you should output only the JSON code, with no additional information.\
"""


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = original_doc_to_text(doc, lmms_eval_specific_kwargs)
    answer = original_doc_to_answer(doc)
    return template.format(question=question, answer=answer)


def process_results(doc, results: str):
    results = results[0].strip()
    tmp_results = results
    try:
        if results.startswith("```json") and results.endswith("```"):
            results = results[7:-3]
        results = json.loads(results)
        results["use_audio"] = 1 if results["use_audio"] else 0
    except Exception as e:
        results = {
            "reason": f"Failed to parse the results. Error: {e}\nResults: {tmp_results}",
            "use_audio": -1,
        }
    results["id"] = doc["id"]
    return {"audio": results}


def aggregate_results(results, args):
    path = generate_submission_file("video_mmmu_audio.xlsx", args)
    logger.info(f"Saving results to {path}")
    df = pd.DataFrame(results)
    df.to_excel(path, index=False)
    use_audios = df["use_audio"].value_counts()
    print("Use audio counts:")
    print(use_audios)
    return use_audios[1] if 1 in use_audios else 0
