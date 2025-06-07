import os
import ast
import json
import argparse
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

system_prompt = """
You are a intelligent assistant for grading math question solutions. You will be given:
- A mathematical question (question) with multiple-choice options (options).
- A list of numbered ground truth steps (gt_steps) showing the correct reasoning to solve a math problem.
- A answer (answer) that is the correct final solution to the question.
- A model prediction (prediction) that includes the steps the model followed and possibly the final answer.

TASK: Compare the prediction to gt_steps and assign a score out of 10 using the rubric below. You must reward both matching logic and valid alternative reasoning. Avoid overly strict step-by-step comparison, instead focus if the model follows a coherent and plausible mathematical approach.

---

### Scoring Rubric:

#### 1. Relative Step Matching (Main Criterion)
- Count the total number of ground truth steps: N
- Evaluate how many predicted steps correctly align with gt_steps in terms of mathematical logic, reasoning, or computations. 
- Score = (matching steps / N) × 10, rounded to nearest whole number. 
- Remember: A step MATCHES if it serves the same mathematical purpose, even if phrased or ordered differently.

#### 2. Correct Final Answer via Different Reasoning
- If the model's final answer is correct, and the steps are logically valid (even if they differ from gt_steps), assign a full score of 10.
- Ignore number of matching steps in this case unless the reasoning is clearly flawed or incoherent.
- Carefully analyze if predicted alternative reasoning includes incorrect observations that contradict parts of the ground truth. Reduce the score proportionally even if the final answer is correct.
- Remember: Reward even PARTIALLY correct reasoning if the steps are accurate, meaningful and follow a valid alternative path to the correct answer.

#### 3. **Implicit or Inferred Steps**
- Do NOT penalize if early steps are skipped, but later logic clearly depends on them.
- E.g., if a model does not mention "identify the chart," but proceeds to use values from that chart correctly, assume the step was completed implicitly.
- Remember: ALWAYS check for implied steps before reducing the score. Credit should be given when logic shows the step was likely understood.

#### 4. **Ignore Superficial Differences**  
- Do NOT deduct score for formatting, using a different notation or variable names, or additional clarifications.
- Remember: FOCUS on the underlying mathematical meaning rather than literal step-by-step matching.

---

**Output Format: Score Dict (strict):**  
SCORE_CARD: {"matched_steps": "X/N", "final_answer_correct": <0 or 1>, "critique": "<2–3 sentence summary>", "score": <0–10>}

Be strict when awarding credit. Do NOT be lenient. Carefully evaluate how far the model's reasoning aligns with the ground truth steps before assigning a score.
"""

response_format = """SCORE_CARD: {"matched_steps": "X/N", "final_answer_correct": <0 or 1>, "critique": "<2–3 sentence summary>", "score": <0–10>}"""


def get_user_prompt(question, options, gt_steps, gt_answer, prediction):
    user_prompt = f"""
INSTRUCTIONS: {system_prompt}

QUESTION: {question}

OPTIONS: {options}

GROUND TRUTH STEPS:
{json.dumps(gt_steps, indent=2)}

CORRECT FINAL ANSWER:
{gt_answer}

PREDICTION:
{prediction}

Now evaluate this prediction and return the response in the specified format: {response_format}

"""
    return user_prompt


def safe_parse_response(reply):
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(reply)
            return json.loads(json.dumps(parsed))
        except Exception as e:
            print("Failed to parse GPT response:", e)
            return None


def prepare_input(sample, matched):
    qid = sample["question_id"]
    question = sample["question"]
    gt_answer = sample["answer"]
    gt_steps = sample["steps"]
    options = sample["options"]
    prediction = matched["resps"][0][0]
    input = {"qid": qid, "category": sample["category"], "question": question, "options": options, "gt_answer": gt_answer, "gt_steps": gt_steps, "prediction": prediction}
    return input


def prepare_batch_prompts(batch):
    prompts = []
    for sample in batch:
        prompt = get_user_prompt(sample["question"], sample["options"], sample["gt_steps"], sample["gt_answer"], sample["prediction"])
        prompts.append(prompt)
    return prompts


def compute_score(gt_data, res_data, res_file, tokenizer, llm, sampling_params, bs=64):
    batch = []
    scored_samples = []
    for sample in tqdm(gt_data, desc="Assigning scores with Qwen3"):

        qid = sample["question_id"]
        matched = [res for res in res_data if res["doc"]["question_id"] == qid]
        if not matched:
            print(f"Sample with qid {qid} not found in results.")
            continue
        input_sample = prepare_input(sample, matched[0])
        batch.append(input_sample)

        if len(batch) == bs:
            batch_prompt = prepare_batch_prompts(batch)
            messages = [{"role": "user", "content": p} for p in batch_prompt]
            texts = [tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True, enable_thinking=True) for msg in messages]
            outputs = llm.generate(texts, sampling_params=sampling_params)

            for input_sample, output in zip(batch, outputs):
                score_dict = {"matched_steps": "0/0", "final_answer_correct": 0, "critique": "Error", "score": 0}
                raw_reply = ""
                try:
                    raw_reply = output.outputs[0].text.strip()
                    parsed_reply = raw_reply.split("SCORE_CARD: ")[-1]
                    result = safe_parse_response(parsed_reply)
                    if result is not None:
                        score_dict = {"matched_steps": result.get("matched_steps", ""), "final_answer_correct": result.get("final_answer_correct", 0), "critique": result.get("critique", "").strip(), "score": int(result.get("score", 0))}
                        raw_reply = None

                except Exception as e:
                    print("Scoring error:", e)

                input_sample["score"] = score_dict["score"]
                input_sample["score_dict"] = score_dict
                input_sample["score_reply"] = raw_reply
                scored_samples.append(input_sample)

            batch = []

    # Save scored samples
    output_file = res_file.replace(".jsonl", "_step_scored_samples_qwen3_batch_think.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in scored_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved scored samples to {output_file}")

    # Compute mean score
    scores = [sample["score"] for sample in scored_samples]
    mean_score = sum(scores) / len(scores)
    print(f"Step evaluation score: {mean_score}")


def main():
    parser = argparse.ArgumentParser(description="Compute step evaluation score for CoT evaluation.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B", help="Path to the model or model identifier (default: Qwen/Qwen3-4B)")
    parser.add_argument("--gt_file", type=str, required=True, help="Path to ground truth file (Parquet format)")
    parser.add_argument("--res_file", type=str, required=True, help="Path to results file (JSONL format)")

    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=32768)

    # Load ground truth data from parquet
    if not os.path.exists(args.gt_file):
        print(f"Ground truth file {args.gt_file} does not exist.")
        return
    gt_df = pd.read_parquet(args.gt_file)
    gt_data = gt_df.to_dict(orient="records")

    # Load result data from jsonl
    if not os.path.exists(args.res_file):
        print(f"Result file {args.res_file} does not exist.")
        return
    print(f"Processing {args.res_file} ...")
    res_data = []
    with open(args.res_file, "r", encoding="utf-8") as f:
        for line in f:
            res_data.append(json.loads(line))

    # Compute score
    compute_score(gt_data, res_data, args.res_file, tokenizer, llm, sampling_params, bs=8)


if __name__ == "__main__":
    main()
