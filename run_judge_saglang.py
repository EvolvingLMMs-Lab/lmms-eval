import sys
import os
import json
from tqdm import tqdm
from openai import OpenAI
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
from lmms_eval.llm_judge.launcher.sglang import SGLangLauncher

HOST = "127.0.0.1"
PORT = 8001
MODEL_PATH = "Qwen2___5-VL-3B-Instruct"
MEM_FRACTION = 0.6
TP = 1
TIMEOUT = 600

INPUT_FILE = "lmms-eval/logs/jiangdan__Qwen2___5-VL-3B-Instruct/20250929_223245_samples_mme_sci_image.jsonl"
OUTPUT_FILE = "lmms-eval/logs/judge_output/outputs_judged_image.jsonl"

sys_prompt_of_judger = (
    "You are a strict and impartial judge. "
    "Based on the original question, the standard answer, and the AI assistant's response provided by the user, "
    "determine whether the AI assistant's response is correct. "
    "If there is any difference in meaning between the AI's response and the standard answer, reply with 'incorrect'. "
    "If the meanings are the same, reply with 'correct'. "
    "Important: Do not answer the original question, and do not provide reasoning or explanation. Only respond with 'correct' or 'incorrect'."
)

launcher = SGLangLauncher(
    host=HOST,
    port=PORT,
    model=MODEL_PATH,
    mem_fraction_static=MEM_FRACTION,
    tp=TP,
    timeout=TIMEOUT,
    enable_torch_compile=False,
    enable_cuda_graph=False,
    log_level="warning", 
    log_level_http="warning",
)
launcher.launch()

api_key = os.environ.get("OPENAI_API_KEY", "sk-local")
client = OpenAI(api_key=api_key, base_url=f"http://{HOST}:{PORT}/v1")

judged_samples = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
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
            model=MODEL_PATH,
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

    judged_samples.append({
        "sample_id": sample_id,
        "judge": judge_result,
        "target": standard_answer,
        "filtered_resps": ai_respond
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in judged_samples:   
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

valid_samples = [x for x in judged_samples if x["judge"] in ["correct", "incorrect"]]
total = len(valid_samples)
correct = sum(1 for x in valid_samples if x["judge"] == "correct")
accuracy = correct / total * 100 if total > 0 else 0.0

print(f"[INFO] Judging complete.")
print(f"[INFO] Total valid samples: {total}")
print(f"[INFO] Correct: {correct}")
print(f"[INFO] Accuracy: {accuracy:.2f}%")
