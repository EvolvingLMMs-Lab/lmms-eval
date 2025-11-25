import base64
import json
import os
import re
import sys
from io import BytesIO
from typing import Any, Dict, List

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)


def doc_to_visual(sample: dict) -> list:
    visual_list = []

    if "image" in sample:
        img_val = sample.get("image")
        if img_val:
            if img_val.startswith("data:image"):
                img_val = re.sub(r"^data:image/[^;]+;base64,", "", img_val)
            img = Image.open(BytesIO(base64.b64decode(img_val)))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            visual_list.append(img)

    question = sample.get("question", "")
    image_tag_nums = re.findall(r"<image_(\d+)>", question)
    for num in image_tag_nums:
        img_col = f"image_{num}"
        img_val = sample.get(img_col)
        if img_val:
            if img_val.startswith("data:image"):
                img_val = re.sub(r"^data:image/[^;]+;base64,", "", img_val)
            img = Image.open(BytesIO(base64.b64decode(img_val)))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            visual_list.append(img)

    return visual_list


def pil_to_base64_url(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"


def doc_to_text(sample: Dict[str, Any], lmms_kwargs: Dict[str, Any] = None) -> str:
    pre_prompt = lmms_kwargs.get("pre_prompt", "") if lmms_kwargs else ""
    post_prompt = lmms_kwargs.get("post_prompt", "") if lmms_kwargs else ""
    question = str(sample.get("question", "")).strip()

    options = sample.get("options", [])
    if isinstance(options, dict):
        options = list(options.values())
    elif not isinstance(options, list):
        options = [str(options)]

    options_text = ""
    if options:
        letters = ["A", "B", "C", "D"]
        options_text = "\n".join(f"{letters[i]}: {opt}" for i, opt in enumerate(options) if i < len(letters))

    return f"{pre_prompt}\n{question}\n{options_text}\n{post_prompt}".strip()


def doc_to_messages(sample: Dict[str, Any], lmms_kwargs: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    text_content = doc_to_text(sample, lmms_kwargs)
    image_list = doc_to_visual(sample)

    content = [{"type": "text", "text": text_content}]

    for img in image_list:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        content.append({"type": "image", "url": f"data:image/png;base64,{img_b64}"})

    return [{"role": "user", "content": content}]


def process_results(sample, outputs, *args, **kwargs):
    target = sample.get("answer", "").strip()
    return {"target": target, "sample_id": sample["id"]}


def mmesci_agg(results: List[Dict[str, Any]]) -> Dict[str, float]:

    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = os.getenv("PORT", "8001")
    TIMEOUT = os.getenv("TIMEOUT", "600")

    sys_prompt_of_judger = (
        "You are a strict and impartial judge. "
        "Based on the original question, the standard answer, and the AI assistant's response provided by the user, "
        "determine whether the AI assistant's response is correct. "
        "If there is any difference in meaning between the AI's response and the standard answer, reply with 'incorrect'. "
        "If the meanings are the same, reply with 'correct'. "
        "Important: Do not answer the original question, and do not provide reasoning or explanation. Only respond with 'correct' or 'incorrect'."
    )

    api_key = os.environ.get("OPENAI_API_KEY", "sk-local")
    client = OpenAI(api_key=api_key, base_url=f"http://{HOST}:{PORT}/v1")

    judged_samples = []

    with open(results, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Judging samples"):
        sample = json.loads(line)
        sample_id = sample.get("sample_id")
        question = sample.get("input")  # or "question"
        standard_answer = sample.get("target", "").strip()
        ai_respond = sample.get("filtered_resps", [""])[0].strip()

        judge_prompt = f"""## Original Question: {question}

    ## Standard Answer: {standard_answer}

    ## AI Assistant's Response: {ai_respond}

    ## NOTE: Do not answer the original question, and do not provide reasoning or explanation. Only respond with 'correct' or 'incorrect'.

    ## Your respond:
    """

        try:
            messages = []
            if sys_prompt_of_judger:
                messages.append({"role": "system", "content": sys_prompt_of_judger})
            messages.append({"role": "user", "content": judge_prompt})
            resp = client.chat.completions.create(
                model=os.getenv("MODEL_NAME"),
                messages=messages,
                temperature=0.0,
                max_tokens=8,
                timeout=TIMEOUT,
            )
            judge_result = resp.choices[0].message.content.strip()
            if judge_result not in ["correct", "incorrect"]:
                judge_result = "error"
        except Exception as e:
            print(f"[ERROR] sample_id={sample_id} failed: {e}")
            judge_result = "error"

        judged_samples.append({"sample_id": sample_id, "judge": judge_result, "target": standard_answer, "filtered_resps": ai_respond})

    valid_samples = [x for x in judged_samples if x["judge"] in ["correct", "incorrect"]]
    total = len(valid_samples)
    correct = sum(1 for x in valid_samples if x["judge"] == "correct")
    accuracy = correct / total * 100 if total > 0 else 0.0

    print(f"[INFO] Judging complete.")
    print(f"[INFO] Total valid samples: {total}")
    print(f"[INFO] Correct: {correct}")
    print(f"[INFO] Accuracy: {accuracy:.2f}%")

    total = len(results)
    if total == 0:
        return {"accuracy": 0.0}

    correct = sum(1 for r in results if r["prediction"] == r["target"])
    return {
        "accuracy": round(correct / total, 4),
        "total_samples": total,
        "correct_samples": correct,
    }
