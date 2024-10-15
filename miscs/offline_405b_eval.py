import argparse
import json
import os
from pathlib import Path

import jsonlines
from tqdm import tqdm

from lmms_eval.api.metrics import mean_stderr, sambajudge

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", type=str, required=True)
    args = parser.parse_args()

    if os.getenv("SAMBAKEY") is None:
        raise ValueError("SAMBAKEY is not set in environment!")

    result_json_path = Path(args.result_json)
    parent_dir = result_json_path.parent.absolute()
    sambajudge_completions_path = parent_dir / f"{result_json_path.stem}_405bjudge_completions.jsonl"
    sambajudge_results_path = parent_dir / f"{result_json_path.stem}_405bjudge_results.json"

    completions_handle = jsonlines.open(sambajudge_completions_path, "w", flush=True)

    with open(args.result_json) as f:
        result_json = json.load(f)

    samba_judge_logs = []
    correct = []

    logs = result_json["logs"]
    for log in tqdm(logs, dynamic_ncols=True, desc=f"Judging answers", total=len(logs)):

        reference_answers = [log["target"]]
        predictions = log["filtered_resps"]
        question_key = "question"
        if question_key not in log["doc"]:
            question_key = "Question"
            assert question_key in log["doc"], f'"question" or "Question" not found in log keys: {list(log["doc"].keys())}'
        question_text = log["doc"][question_key]

        assert len(predictions) == 1, f'Found a response number of predictions != 1 at doc ID = {log["doc_id"]}'

        samba_judge_response_dict = sambajudge(reference_answers, predictions, question_text)
        correct_or_not = samba_judge_response_dict["sambajudge"]
        responses_this_log = samba_judge_response_dict["samba_for_log"]

        judge_result_this_log = {"sambajudge": correct_or_not, "samba_for_log": responses_this_log, **log}
        completions_handle.write(judge_result_this_log)
        samba_judge_logs.append(judge_result_this_log)
        correct.append(correct_or_not)

    completions_handle.close()
    total_correct = sum(correct) / len(correct)
    sambajudge_json = {
        "sambajudge": total_correct,
        "sambajudge_stderr": mean_stderr(correct),
        "logs": samba_judge_logs,
    }
    with open(sambajudge_results_path, "w") as f:
        json.dump(sambajudge_json, f)
    print("Done!")
    print(f"Original results path = {str(result_json_path.absolute())}")
    print(f"sambajudge: {total_correct}")
    print(f"sambajudge_stderr: {mean_stderr(correct)}")
