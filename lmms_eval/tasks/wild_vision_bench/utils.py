import base64
import json
import os
import re
import time
from collections import defaultdict
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import yaml
from loguru import logger as eval_logger
from scipy import stats

NUM_SECONDS_TO_SLEEP = 5


with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

GPT_EVAL_MODEL_NAME = config["metadata"]["judge_model"]
BASELINE_MODEL_NAME = config["metadata"]["baseline_model"]

API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }

system_prompt = """\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]".\
"""

prompt_template = "<|User Prompt|>\n{question_1}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>"


def get_chat_response(base64_image, prompt, max_retries=5, wait_time=10):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64, {base64_image}"}},
                ],
            },
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            # print(response_data)
            return response_data["choices"][0]["message"]["content"], GPT_EVAL_MODEL_NAME
        except requests.exceptions.RequestException as e:
            print(f"Request failed on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                print(f"Failed to get response after {max_retries} attempts")
                return "", GPT_EVAL_MODEL_NAME
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            return "", GPT_EVAL_MODEL_NAME


def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_score(judgement, pattern, pairwise=True):
    matches = pattern.findall(judgement)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None, True
    elif len(set(matches)) == 1:
        if pairwise:
            return matches[0].strip("\n"), False
        return int(matches[0])
    else:
        return None, False


def wild_vision_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def wild_vision_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["instruction"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def wild_vision_doc_to_target(doc):
    return doc[BASELINE_MODEL_NAME]


def wild_vision_process_results(doc, results):
    pred = results[0]
    user_prompt = prompt_template.format(question_1=doc["instruction"], answer_1=doc[BASELINE_MODEL_NAME], answer_2=pred)
    base64_image = image_to_base64(doc["image"])
    resps, gpt_name = get_chat_response(base64_image, user_prompt)
    score, _ = get_score(resps, pattern=re.compile("\[\[([AB<>=]+)\]\]"))

    if score is None:
        score = resps

    if "A>B" in score:
        raw_score = -1
        winner = "model_a"
        judgement = "Worse"  # Baseline better
    elif "A>>B" in score:
        raw_score = -2
        winner = "model_a"
        judgement = "Worse++"
    elif "A=B" in score:
        raw_score = 0
        winner = "tie"
        judgement = "Tie"
    elif "B>A" in score:
        raw_score = 1
        winner = "model_b"
        judgement = "Better"
    elif "B>>A" in score:
        raw_score = 2
        winner = "model_b"
        judgement = "Better++"
    else:
        raw_score = 0
        winner = "tie"
        judgement = "Unclear"

    return {
        "raw_scores": {
            "final_score": raw_score,
        },
        "elo_scores": {
            "question": doc["instruction"],
            "model_a": BASELINE_MODEL_NAME,
            "model_b": "evaluation_model",
            "winner": winner,
            "gpt_resps": resps,
            "model_resps": pred,
            "judgement": judgement,
        },
        "win_rates": {
            "question": doc["instruction"],
            "model_a": BASELINE_MODEL_NAME,
            "model_b": "evaluation_model",
            "winner": winner,
        },
        "judgements_better": {
            "judgement": judgement,
        },
        "judgements_better_plus": {
            "judgement": judgement,
        },
        "judgements_worse": {
            "judgement": judgement,
        },
        "judgements_worse_plus": {
            "judgement": judgement,
        },
        "judgements_tie": {
            "judgement": judgement,
        },
        "judgements_unclear": {
            "judgement": judgement,
        },
    }


import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def prepare_elo_data(results):
    battles = []
    for result in results:
        battles.append({"model_a": result["model_a"], "model_b": result["model_b"], "winner": result["winner"]})
    return pd.DataFrame(battles)


def compute_mle_elo(df, baseline, SCALE=400, BASE=10, INIT_RATING=1000):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
    try:
        lr.fit(X, Y)
        elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    except ValueError as e:
        eval_logger.warning(f"Error in LogisticRegression: {e}")
        eval_logger.warning("Falling back to default ELO scores")
        elo_scores = np.full(p, INIT_RATING)

    # set anchor as gpt-4-0314 = 1000
    if baseline in models.index:
        elo_scores += 1000 - elo_scores[models[baseline]]

    # Create a DataFrame with "model" and "score" columns
    elo_df = pd.DataFrame({"model": models.index, "score": elo_scores})

    return elo_df.sort_values("score", ascending=False)


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {a: [wins[a][b] if a != b else np.nan for b in names] for a in names}

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T


def get_win_rate_column(df, column, baseline):
    to_dict = df.set_index("model")[column].to_dict()
    win_rate_table = predict_win_rate(to_dict)
    return win_rate_table[baseline].fillna(0.5).apply(lambda x: round(x * 100, 2))


def wild_vision_aggregation_raw_scores(results):
    total_score = 0
    for result in results:
        total_score += result["final_score"]
    return total_score / len(results)


def wild_vision_aggregation_elo_scores(results):
    battles = prepare_elo_data(results)
    elo_ratings = compute_mle_elo(battles, BASELINE_MODEL_NAME)
    elo_score = get_win_rate_column(elo_ratings, "score", BASELINE_MODEL_NAME)
    return elo_score["evaluation_model"]


def wild_vision_aggregation_win_rates(results):
    battles = prepare_elo_data(results)
    win_rates = battles.groupby("model_b").apply(lambda x: (x["winner"] == "model_b").mean()).to_dict()
    win_rates[BASELINE_MODEL_NAME] = battles.groupby("model_a").apply(lambda x: (x["winner"] == "model_a").mean()).get(BASELINE_MODEL_NAME, 0)
    return win_rates["evaluation_model"] * 100


def wild_vision_aggregation_judgements_better(results):
    judgements = pd.DataFrame(results)["judgement"].value_counts(normalize=True).to_dict()
    return judgements["Better"] * 100 if "Better" in judgements else 0


def wild_vision_aggregation_judgements_better_plus(results):
    judgements = pd.DataFrame(results)["judgement"].value_counts(normalize=True).to_dict()
    return judgements["Better++"] * 100 if "Better++" in judgements else 0


def wild_vision_aggregation_judgements_worse(results):
    judgements = pd.DataFrame(results)["judgement"].value_counts(normalize=True).to_dict()
    return judgements["Worse"] * 100 if "Worse" in judgements else 0


def wild_vision_aggregation_judgements_worse_plus(results):
    judgements = pd.DataFrame(results)["judgement"].value_counts(normalize=True).to_dict()
    return judgements["Worse++"] * 100 if "Worse++" in judgements else 0


def wild_vision_aggregation_judgements_tie(results):
    judgements = pd.DataFrame(results)["judgement"].value_counts(normalize=True).to_dict()
    return judgements["Tie"] * 100 if "Tie" in judgements else 0


def wild_vision_aggregation_judgements_unclear(results):
    judgements = pd.DataFrame(results)["judgement"].value_counts(normalize=True).to_dict()
    return judgements["Unclear"] * 100 if "Unclear" in judgements else 0
