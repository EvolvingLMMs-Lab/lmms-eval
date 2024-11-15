# code from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/scripts/regression.py
import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

from lmms_eval import utils
from lmms_eval.api.registry import ALL_TASKS

model_types = ["llava_onevision"]
vision_models = [
    "lmms-lab/llava-onevision-qwen2-0.5b-ov",
]

single_image_tasks = ["ocrbench", "mmmu_val", "ai2d"]
multi_image_tasks = ["muirbench"]
video_tasks = ["videomme"]
# choice_tasks = []
# perplexity_tasks = []
# generation_tasks = []
task_names = single_image_tasks + multi_image_tasks + video_tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches", default=[])
    parser.add_argument("--models", default=vision_models)
    parser.add_argument("--tasks", default=task_names)
    parser.add_argument("--acc_norm", type=bool, default=False)
    parser.add_argument("--perplexity", default=None)
    # TODO: implement num_fewshot and limit per task, e.g. task1:5,task2:1:100,task3::1000
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=float, default=8)
    # TODO: implement hf-auto to pick between causal and seq2seq models so we don't need this
    parser.add_argument("--model", default="llava_onevision")
    # Use whatever is faster here
    parser.add_argument("--model_args", default="conv_template=qwen_1_5,model_name=llava_qwen")
    parser.add_argument("--batch_size", default="1")
    return parser.parse_args()


def eval_models(args, branch=None):
    if branch is not None:
        if os.system(f"git checkout {branch}") != 0:
            return {}, 0

    branch = branch or initial_branch

    start_time = time.time()

    results = {}

    for indx, model in enumerate(args.models):
        model_type = model_types[indx]
        model_args = f"pretrained={model},{args.model_args}"
        tasks = args.tasks
        batch_size = args.batch_size
        output_path = f"logs/regression_test/{int(start_time)}-{branch.replace('/', '_')}"

        original_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        command = (
            f"python3 -m accelerate.commands.launch --main_process_port=12580 --num_processes=8 lmms_eval --model {model_type} --model_args {model_args} --tasks {','.join(tasks)} "
            f"--num_fewshot {args.num_fewshot}{'' if args.limit is None else f' --limit {args.limit}'} "
            f"--batch_size {batch_size} --output_path {output_path}"
        )

        print(f"{'=' * 80}\nEvaluating {model} on {', '.join(tasks)} at {branch} with:\n\n{command}\n{'=' * 80}")

        ret = os.system(command)
        os.chdir(original_dir)

        json_file_path = find_json_file(output_path)

        if json_file_path and ret == 0:
            with open(json_file_path, encoding="utf-8") as f:
                results[model] = json.load(f)
        else:
            results[model] = {"results": {}}

    end_time = time.time()

    return results, end_time - start_time


def extract_value(args, results, model, task, err=False):
    if model not in results:
        return 0
    results = results[model]["results"]
    if task not in results:
        return 0
    results = results[task]
    if task == "ai2d":
        return results["exact_match,flexible-extract"]
    elif task == "mmmu_val":
        return results["mmmu_acc,none"]
    elif task == "ocrbench":
        return results["ocrbench_accuracy,none"]
    elif task == "videomme":
        return results["videomme_perception_score,none"]
    elif task == "muirbench":
        return results["muirbench_score_overall,flexible-extract"]
    return 0


def format_value(args, results, model, task):
    val = 100 * extract_value(args, results, model, task)
    err = 100 * extract_value(args, results, model, task, err=True)
    return f"{val:.2f}{f' ± {err:.2f}' if err != 0 else ''}"


def format_diff(args, results1, results2, model, task):
    val1 = 100 * extract_value(args, results1, model, task)
    val2 = 100 * extract_value(args, results2, model, task)
    diff = val2 - val1
    return f"**+{diff:.2f}**" if diff > 0 else f"{diff:.2f}"


def find_json_file(base_path):
    pattern = os.path.join(base_path, "**", "*_results.json")
    json_files = glob.glob(pattern, recursive=True)
    return json_files[0] if json_files else None


def main():
    args = parse_args()

    args.branches = args.branches.split(",") if isinstance(args.branches, str) else args.branches
    args.models = args.models.split(",") if isinstance(args.models, str) else args.models
    args.tasks = ALL_TASKS if args.tasks == "all_tasks" else utils.pattern_match(args.tasks.split(","), ALL_TASKS) if isinstance(args.tasks, str) else args.tasks

    global initial_branch
    initial_branch = subprocess.check_output("git branch --show-current", shell=True).decode("ascii").strip()

    # TODO: implement proper timing for each task
    # TODO: reduce IO by sharing tasks between models?

    results, runtime = eval_models(args)
    print(results, runtime)

    runs = []
    for branch in args.branches:
        runs.append((branch, *eval_models(args, branch)))

    os.system(f"git checkout {initial_branch}")

    print("")
    print(f"|task|{'|'.join(map(lambda model: Path(model).name, args.models))}|")
    print(f"|--|{'--|' * len(args.models)}")
    for task in args.tasks:
        print(f"|{task} ({initial_branch})|{'|'.join(map(lambda model: format_value(args, results, model, task), args.models))}|")
        for branch, branch_results, branch_runtime in runs:
            print(f"|{task} ({branch})|{'|'.join(map(lambda model: format_value(args, branch_results, model, task), args.models))}|")
            print(f"|{task} (diff)|{'|'.join(map(lambda model: format_diff(args, results, branch_results, model, task), args.models))}|")

    print("")
    print("|branch|runtime|%|")
    print("|--|--|--|")
    print(f"|{initial_branch}|{runtime:.1f}s|100%|")
    for branch, _, branch_runtime in runs:
        print(f"|{branch}|{branch_runtime:.1f}s|{100 * branch_runtime / runtime:.2f}%|")


if __name__ == "__main__":
    main()
