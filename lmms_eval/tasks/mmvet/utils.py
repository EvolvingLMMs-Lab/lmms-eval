import os
import time
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.llm_judge import Request, ServerConfig, get_server

with open(Path(__file__).parent / "mmvet.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

API_TYPE = os.getenv("API_TYPE", "openai")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o-2024-08-06")

# Initialize the judge server
server_config = ServerConfig(
    model_name=MODEL_VERSION,
    temperature=0.0,
    max_tokens=128
)
server = get_server(server_name=API_TYPE, config=server_config)
MM_VET_PROMPT = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score.
gpt_query_prompt | Ground truth | Prediction | Correctness
--- | --- | --- | ---
What is x in the equation? | -1 <AND> -5 | x = 3 | 0.0
What is x in the equation? | -1 <AND> -5 | x = -1 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 or 5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -1 or x = -5 | 1.0
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0
"""


def get_chat_response(prompt, model=MODEL_VERSION, temperature=0.0, max_tokens=128, patience=3, sleep_time=5):
    # Update server config with specific parameters for this request
    custom_config = ServerConfig(
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    while patience > 0:
        patience -= 1
        try:
            # Create a Request object for the unified judge API
            request = Request(
                messages=[{"role": "user", "content": prompt}],
                config=custom_config
            )
            
            # Use the unified judge API
            response = server.evaluate(request)
            
            content = response.content.strip() if response.content else ""
            if content != "":
                return content, response.model_used

        except Exception as e:
            eval_logger.error(f"Error: {e}")
            if "Rate limit" in str(e):
                eval_logger.info("Sleeping due to rate limit...")
                time.sleep(sleep_time)
            eval_logger.info(f"Retrying...Patience left: {patience}")

    return "", ""


def mmvet_doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]


def mmvet_process_results(doc, results):
    # get pred and ground truth here
    pred = results[0]
    question = doc["question"]
    answer = doc["answer"]
    gpt_query_prompt = f"{MM_VET_PROMPT}\n{question} | {answer.replace('<AND>', ' <AND> ').replace('<OR>', ' <OR> ')} | {pred} |"
    grade_sample_run_complete = False
    temperature = 0.0

    while not grade_sample_run_complete:
        content, model_name = get_chat_response(
            gpt_query_prompt,
            temperature=temperature,
        )
        if content:
            try:
                content = content.split(" ")[0].strip()
                score = float(content)
                if 0.0 <= score <= 1.0:
                    grade_sample_run_complete = True
            except ValueError:
                time.sleep(5)
                temperature += 0.5
                eval_logger.info(f"Sleep 5 secs, {doc['question_id']} try again with increased temperature {temperature}.")
                content, model_name = get_chat_response(
                    gpt_query_prompt,
                    temperature=temperature,
                )
                if temperature >= 2:  # Assuming a max temperature threshold
                    score = 0.0
                    grade_sample_run_complete = True
                    eval_logger.info(f"Reach to max trials, {doc['question_id']} failed to get a score.")
        else:
            score = 0.0
            grade_sample_run_complete = True
            eval_logger.info(f"{doc['question_id']} failed to get a score.")

    return {
        f"gpt_eval_score": {
            "question_id": doc["question_id"],
            "question": doc["question"],
            "gt_answer": doc["answer"],
            "capabilities": doc["capability"],
            "pred_answer": pred,
            "score": score,
            "eval_model": model_name,
        }
    }


cap_columns = pd.DataFrame(["rec", "ocr", "know", "gen", "spat", "math"])
cap_details_columns = pd.DataFrame(
    [
        "rec_gen_know",
        "rec",
        "spat_ocr",
        "spat_math_ocr",
        "rec_spat",
        "ocr",
        "math_ocr",
        "rec_know",
        "rec_spat_gen_ocr",
        "rec_gen_ocr_know",
        "rec_spat_ocr",
        "rec_ocr",
        "know_spat_ocr",
        "rec_spat_know",
        "spat_gen_ocr",
        "rec_spat_math_ocr",
    ]
)


def mmvet_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    # Calculate the overall score
    overall_score = sum([result["score"] for result in results]) / len(results) * 100
    eval_logger.info(f"Overall Score: {overall_score:.2f}")

    # Initialize dictionaries to store scores for each capability and detail
    cap_scores = {cap: 0 for cap in cap_columns.squeeze().tolist()}
    cap_details_scores = {detail: 0 for detail in cap_details_columns.squeeze().tolist()}

    # Count the number of results for each capability and detail
    cap_counts = {cap: 0 for cap in cap_scores}
    cap_details_counts = {detail: 0 for detail in cap_details_scores}

    # Aggregate scores for each capability and detail
    for result in results:
        for cap in cap_scores:
            if cap in result["capabilities"]:
                cap_scores[cap] += result["score"]
                cap_counts[cap] += 1
        for detail in cap_details_scores:
            detail_set = set(detail.split("_"))
            result_detail_set = set(result["capabilities"].split(","))
            if detail_set == result_detail_set:
                cap_details_scores[detail] += result["score"]
                cap_details_counts[detail] += 1

    # Calculate the average score for each capability
    for cap in cap_scores:
        if cap_counts[cap] > 0:
            cap_scores[cap] = cap_scores[cap] / cap_counts[cap] * 100
        eval_logger.info(f"Score for {cap}: {cap_scores[cap]:.2f}")

    # Calculate the average score for each detailed capability
    for detail in cap_details_scores:
        if cap_details_counts[detail] > 0:
            cap_details_scores[detail] = cap_details_scores[detail] / cap_details_counts[detail] * 100
        eval_logger.info(f"Score for {detail}: {cap_details_scores[detail]:.2f}")

    return overall_score


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        return doc["question"]
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre_prompt}{question}{post_prompt}"
