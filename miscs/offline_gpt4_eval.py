import argparse
import json
import os
import jsonlines
from tqdm import tqdm
from pathlib import Path

from lmms_eval.api.metrics import gpt4judge, mean_stderr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-json', type=str, required=True)
    args = parser.parse_args()

    if os.getenv('AZURE_API_KEY') is None:
        raise ValueError('AZURE_API_KEY is not set in environment!')
    if os.getenv('AZURE_ENDPOINT') is None:
        raise ValueError('AZURE_ENDPOINT is not set in environment!')

    result_json_path = Path(args.result_json)
    parent_dir = result_json_path.parent.absolute()
    gpt4_completions_path = parent_dir / f'{result_json_path.stem}_gpt4judge_completions.jsonl'
    gpt4_results_path = parent_dir / f'{result_json_path.stem}_gpt4judge_results.json'

    completions_handle = jsonlines.open(gpt4_completions_path, 'w', flush=True)

    with open(args.result_json) as f:
        result_json = json.load(f)

    gpt4_judge_logs = []
    correct = []

    logs = result_json['logs']
    for log in tqdm(logs, dynamic_ncols=True, desc=f'Judging answers', total=len(logs)):

        reference_answers = [log['target']]
        predictions = log['filtered_resps']
        question_text = log['doc']['question']

        assert len(predictions) == 1, f'Found a response number of predictions != 1 at doc ID = {log["doc_id"]}'

        gpt4_judge_response_dict = gpt4judge(reference_answers, predictions, question_text)
        correct_or_not = gpt4_judge_response_dict['gpt4judge']
        responses_this_log = gpt4_judge_response_dict['gpt4_for_log']

        judge_result_this_log = {
            'gpt4judge': correct_or_not,
            'gpt4_for_log': responses_this_log,
            **log
        }
        completions_handle.write(judge_result_this_log)
        gpt4_judge_logs.append(judge_result_this_log)
        correct.append(correct_or_not)

    completions_handle.close()
    total_correct = sum(correct) / len(correct)
    gpt4judge_json = {
        'gpt4judge': total_correct,
        'gpt4judge_stderr': mean_stderr(correct),
        'logs': gpt4_judge_logs,
    }
    with open(gpt4_results_path, 'w') as f:
        json.dump(gpt4judge_json, f)
    print('Done!')
    print(f'Original results path = {str(result_json_path.absolute())}')
    print(f'gpt4judge: {total_correct}')
    print(f'gpt4judge_stderr: {mean_stderr(correct)}')
