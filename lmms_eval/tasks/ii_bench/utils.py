import json
import logging
import re
from collections import Counter
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

PROMPT = """Question: {}
(A) {}
(B) {}
(C) {}
(D) {}
(E) {}
(F) {}"""

def ii_bench_doc_to_text(doc, model_specific_prompt_kwargs):
    question = PROMPT.format(doc["question"], doc["option1"], doc["option2"], doc["option3"], doc["option4"], doc["option5"], doc["option6"])
    pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    post_prompt = model_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def ii_bench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def extract_option_labels(text, options=None):
    if isinstance(text, dict):
        return 'error'
    pattern = r"\(([A-F])\)"
    matches = re.findall(pattern, text)
    
    if not matches:
        pattern = r"\b([A-F])\b"
        matches = re.findall(pattern, text)
    
    if matches:
        counter = Counter(matches)
        most_common = counter.most_common()
        max_count = most_common[0][1]
        candidates = [item for item in most_common if item[1] == max_count]
        return candidates[-1][0]
    else:
        if options:
            counter = Counter()
            for i, option in enumerate(options, start=1):
                label = chr(64 + i)
                option_stripped = option.strip()
                if option_stripped in text:
                    counter[label] += 1
                elif text in option:
                    counter[label] += 1
            if counter:
                most_common = counter.most_common()
                max_count = most_common[0][1]
                candidates = [item for item in most_common if item[1] == max_count]
                return candidates[-1][0]
    return None


def ii_bench_process_results(doc, results):
    response = results[0]
    predict = extract_option_labels(response, [doc["option1"], doc["option2"], doc["option3"], doc["option4"], doc["option5"], doc["option6"]])
    return {"submission": {"id": doc["id"], "predict_answer": predict, "response": response}}


def ii_bench_aggregate_submissions(results, args):
    file = generate_submission_file("ii_bench_test_for_submission.json", args)
    with open(file, "w") as f:
        json.dump(results, f, indent=4)
    logging.getLogger("lmms-eval").info(f"Results saved to {file}")
