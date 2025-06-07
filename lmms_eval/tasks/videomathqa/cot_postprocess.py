import os
import re
import sys
import json
import random
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from videomathqa.utils import (extract_characters_regex,
                            videomathqa_process_results,
                            videomathqa_mcq_aggregate_results,
                            videomathqa_multi_binary_aggregate_results)


mcq_prompt = (
    "Given the original multiple-choice options and a model-generated answer containing reasoning and a final answer, identify the option that best matches the final answer and return only the corresponding letter (A, B, C, D, or E)."
)
mbin_prommpt = "Given the original binary options and a model-generated answer containing reasoning and a final answer, identify the option that best matches the final answer and return only the corresponding letter (A or B)."


def extract_choice_vllm(llm, sampling_params, tokenizer, model_prompt, mcq=True):
    if mcq:
        prompt_type = mcq_prompt
    else:
        prompt_type = mbin_prommpt
    chat_prompt = [
        {
            "role": "user",
            "content": f"""{prompt_type}:

Text:
{model_prompt}

Only return the letter A, B, C, D, or E. If none is found, return "None".""",
        }
    ]
    text = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    output = llm.generate([text], sampling_params=sampling_params)
    reply = output[0].outputs[0].text.strip().upper()
    if mcq:
        if re.fullmatch(r"[A-E]", reply):
            return reply
    else:
        if re.fullmatch(r"[A-B]", reply):
            return reply
    return None


def refine_samples_vllm(llm, sampling_params, tokenizer, sample_jsonl, output_jsonl, mcq=True):
    raw_samples = []
    with open(sample_jsonl, "r") as f:
        for line in f:
            raw_samples.append(json.loads(line))
    print(f"Loaded {len(raw_samples)} samples from {sample_jsonl}")

    updated_samples = []
    for sample in tqdm(raw_samples, desc="Postprocessing samples with Qwen"):
        options = sample["doc"]["options"]
        raw_pred = sample["resps"][0][0]
        input_text = f"The options are: {options}\n\n The model response is: {raw_pred}"
        try:
            choice = extract_choice_vllm(llm, sampling_params, tokenizer, input_text, mcq)
        except Exception as e:
            choice = None
        if choice is None:
            answer = sample["target"]
            if mcq:
                options = ["A", "B", "C", "D", "E"]
            else:
                options = ["A", "B"]
            options.remove(answer)
            random.shuffle(options)
            choice = options[0]
        sample["resps"][0][0] = choice
        updated_samples.append(sample)

    with open(output_jsonl, "w") as f:
        for sample in updated_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved {len(updated_samples)} updated samples to {output_jsonl}")
    return updated_samples


def postprocess_jsonl(llm, sampling_params, tokenizer, sample_jsonl, output_jsonl):
    if "mcq" in sample_jsonl:
        mcq = True
    elif "mbin" in sample_jsonl:
        mcq = False

    updated_samples = refine_samples_vllm(llm, sampling_params, tokenizer, sample_jsonl, output_jsonl, mcq)

    print(f"Computing score ...")
    processed = []
    for item in tqdm(updated_samples, desc="Computing scores..."):
        pred_raw = item["resps"][0][0] if isinstance(item["resps"][0], list) else item["resps"][0]
        pred_clean = extract_characters_regex(pred_raw)
        item["filtered_resps"] = [pred_clean]
        result = videomathqa_process_results(item["doc"], [pred_clean])
        processed.append(result["videomathqa_perception_score"])

    if mcq:
        final_score = videomathqa_mcq_aggregate_results(processed)
    else:
        final_score = videomathqa_multi_binary_aggregate_results(processed)
    print(f"Final Postprocessed VideoMathQA Score: {final_score:.2f}")
    print(f"Saved {len(updated_samples)} updated samples to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Postprocess a CoT predictions using the Qwen model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the postprocessed output JSONL file.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B", help="Path to the pretrained Qwen model (default: Qwen3-4B).")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Input file '{args.input_file}' does not exist.")
        return

    if os.path.exists(args.output_file):
        print(f"Output file '{args.output_file}' already exists. Skipping.")
        return

    print("Loading Qwen-3 model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=16)

    print(f"Processing {args.input_file} ...")
    postprocess_jsonl(llm, sampling_params, tokenizer, args.input_file, args.output_file)
    print(f"Saved postprocessed output to {args.output_file}")


if __name__ == "__main__":
    main()
