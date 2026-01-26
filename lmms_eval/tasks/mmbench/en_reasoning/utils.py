import os

from lmms_eval.tasks._task_utils.reasoning_utils import compute_score
from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator

GPT_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")
API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
else:
    API_URL = "YOUR_API_URL"
    API_KEY = "YOUR_API_KEY"


mmbench_evaluator = MMBench_Evaluator(sys_prompt="", API_KEY=API_KEY, API_URL=API_URL, model_version=GPT_EVAL_MODEL_NAME)

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


def mmbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_candidate = ["A", "B", "C", "D", "E"]
    options_prompt, options_dict = mmbench_evaluator.create_options_prompt(doc, option_candidate)

    query_prompt = f"{doc['hint']} {doc['question']} {options_prompt}" if str(doc["hint"]) != "nan" and doc["hint"] else f"{doc['question']} {options_prompt}"

    return query_prompt


def mmbench_doc_to_visual(doc):
    num_image = int(os.environ.get("NUM_IMAGE", "1"))

    if num_image == 1:
        return [doc["image"].convert("RGB")]
    elif num_image == 2:
        return [doc["image"].convert("RGB"), doc["image"].convert("RGB")]
    else:
        raise ValueError(f"num_image must be 1 or 2, got {num_image}")


def mmbench_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = mmbench_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = mmbench_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    for visual in visuals:
        messages[0]["content"].append({"type": "image", "url": visual})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def mmbench_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = mmbench_doc_to_text(doc, None)
    ground_truth = doc["answer"]
    extra_info = {"question": question}
    for pred in results:
        score_dict = compute_score(data_source="mmbench_en", solution_str=pred.strip(), ground_truth=ground_truth, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}
