import os

import pandas as pd
from openai import OpenAI
from PIL import Image

from lmms_eval.tasks.plm_videobench.eval_utils import *

# Load default config parameters
config = load_defualt_config()

# Load video paths
video_base_dir = config["plm_sgqa"]["video_base_dir"]
assert video_base_dir is not None, f"video_base_dir is not set. Please double check if you have downloaded the videos and set the correct path in _default_template_yaml."

# Load the number of video frames
num_video_frames = config["plm_sgqa"]["num_video_frames"]
assert num_video_frames is not None, f"num_video_frames must not be None."

# Initialize LLM Judge for RCap Evaluation
base_url = config["llm_judge"]["base_url"]
api_key = config["llm_judge"]["api_key"]
client = OpenAI(api_key=api_key, base_url=base_url)
llm_judge_name = config["llm_judge"]["model"]


def plm_sgqa_doc_to_visual(doc):
    video_id = doc["video"]
    video_path = os.path.join(video_base_dir, video_id)
    video_frames, _ = load_video_uniform(video_path, num_video_frames)
    video_frames = [Image.fromarray(frame) for frame in video_frames]

    return [video_frames]


def plm_sgqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    assert lmms_eval_specific_kwargs and "prompt" in lmms_eval_specific_kwargs, "'prompt' must be specified in lmms_eval_specific_kwargs for the 'plm_sgqa' task."

    prompt = lmms_eval_specific_kwargs["prompt"].format(question=doc["question"], answer="{answer}")
    return prompt


def plm_sgqa_process_results(doc, results):
    pred = results[0]

    results_dict = {kk: vv for kk, vv in doc.items() if not kk == "metadata"}
    results_dict["prediction"] = pred

    llm_prompt = get_sgqa_judge_prompt(question=doc["question"], pred=pred, target=doc["answer"])
    completion = call_judge_with_retry(client, model_name=llm_judge_name, prompt=llm_prompt)
    llm_response = completion.choices[0].message.content

    try:
        judgement = json.loads(llm_response)
    except:
        if "yes" in llm_response or "Yes" in llm_response:
            judgement = {"pred": "yes", "reason": "parse_error"}
        else:
            judgement = {"pred": "no", "reason": "parse_error"}

    results_dict.update(
        {
            "success": judgement["pred"] == "yes",
        }
    )

    return {"plm_sgqa_scores": results_dict}


def plm_sgqa_aggregate_results(results):
    df_res = pd.DataFrame(results)

    # -- add the onevsall accuracy for all q_kinds
    success = df_res.success.mean().item()

    printable_results = {"success": success, "num_instances": len(results)}

    return printable_results
