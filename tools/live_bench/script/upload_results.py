import argparse
import json
import os
from typing import Dict, List, Union

import datasets
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm import tqdm

EPS = 1e-6

RESULT_FEATURES = {
    "id": datasets.Value("int32"),
    "images": datasets.Sequence(datasets.Image()),
    "question": datasets.Value("string"),
    "ground_truth": datasets.Value("string"),
    "criteria": datasets.Value("string"),
    "subtask": datasets.Value("string"),
    "response": datasets.Value("string"),
    "score": datasets.Value("float32"),
    "reason": datasets.Value("string"),
}

SUBTASKS = ["Concrete Recognition", "Analytical Questions", "Evaluative Questions", "Divergent Thinking", "Real-world Assistance"]


def load_images(config) -> Dict[int, List[Image.Image]]:
    dataset = datasets.load_dataset(config["dataset_path"], config["dataset_name"], split=config["test_split"])
    images = {}
    for data in tqdm(dataset, desc="Loading images"):
        images[data["id"]] = data["images"]
    return images


def get_hf_results(results, detailed_results):
    live_bench_images = load_images(results["configs"]["live_bench"])

    def load_results():
        for result in tqdm(detailed_results["logs"], desc="Loading results"):
            doc = result["doc"]
            res = {}
            res["id"] = doc["id"]
            res["images"] = live_bench_images[doc["id"]]
            res["question"] = doc["question"]
            res["ground_truth"] = doc["answer"]
            res["criteria"] = doc["criteria"]
            res["subtask"] = doc["subtask"]
            res["response"] = result["filtered_resps"][0]
            res["score"] = result["gpt4_eval_score"]["rating"]
            res["reason"] = result["gpt4_eval_score"]["explanation"]
            yield res

    result_dataset = Dataset.from_generator(load_results, features=datasets.Features(RESULT_FEATURES))

    return result_dataset


def preview_results(results, heading: str):
    HEADING = "=" * 15 + " " + heading + " " + "=" * 15
    ENDING = "=" * len(HEADING)
    print(HEADING)
    print(results)
    print(ENDING)


def calculate_score(results: Dataset):
    results = results.to_pandas()

    sum_score, count = 0, 0
    score = {}
    for subtask in SUBTASKS:
        score[subtask] = []
    for index, result in tqdm(results.iterrows(), total=len(results), desc="Calculating score"):
        if result["score"] == -1:
            continue
        sum_score += result["score"] / 10
        count += 1
        subtask = result["subtask"]
        if subtask not in SUBTASKS:
            subtask = "Further Insights"
        score[result["subtask"]].append(result["score"] / 10)
    res = [(subtask, len(score[subtask]), np.mean(score[subtask]) * 100) for subtask in SUBTASKS]
    res.append(("Total", count, sum_score / count * 100))
    res = pd.DataFrame(res, columns=["Subtask", "Count", "Score"])

    return res


def get_results(folder):
    detailed_file = os.path.join(folder, "live_bench.json")
    results_file = os.path.join(folder, "results.json")

    with open(results_file, "r") as f:
        results = json.load(f)

    assert "live_bench" in results["configs"], "No live_bench config found in results.json"
    final_score = results["results"]["live_bench"]["gpt4_eval_score,none"]
    model_configs = results["model_configs"]
    version = results["configs"]["live_bench"]["metadata"]["version"]

    assert model_configs["limit"] is None, "Model limit is not None, please check if the model is tested on the full dataset"

    with open(detailed_file, "r") as f:
        detailed_results = json.load(f)

    hf_results = get_hf_results(results, detailed_results)
    preview_results(hf_results.to_pandas().iloc[0], "Detailed Results")
    score = calculate_score(hf_results)
    preview_results(score, "Final Score")

    assert (
        abs(score[score["Subtask"] == "Total"]["Score"] - final_score) <= EPS
    ).all(), f"Final score does not match the calculated score, the score calculated by the script is {score[score['Subtask'] == 'Total']['Score'].values[0]} and the final score in the log is {final_score}."

    return hf_results, score, version


def upload_results(
    hf_results: Dataset,
    score: pd.DataFrame,
    model_name,
    dataset_version,
    log_folder="logs",
):
    hf_results.push_to_hub(
        "lmms-lab/LiveBenchDetailedResults",
        config_name=dataset_version,
        split=model_name.replace("-", "_"),
    )
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    score_path = os.path.abspath(os.path.join(log_folder, f"{dataset_version}_{model_name}.csv"))
    score.to_csv(score_path, index=False)
    print(f"Results saved to {score_path}")
    score_dict = {item["Subtask"]: item["Score"] for index, item in score.iterrows()}
    score_dict["Model Name"] = model_name
    try:
        hf_score = datasets.load_dataset("lmms-lab/LiveBenchResults", dataset_version, split="test")
    except:
        hf_score = Dataset.from_dict({subtask: [] for subtask in ["Model Name", "Total"] + SUBTASKS})
    hf_score = hf_score.add_item(score_dict)
    df_score = pd.DataFrame(hf_score)
    df_score = df_score.drop_duplicates(subset=["Model Name"], keep="last")
    df_score = df_score[["Model Name", "Total"] + SUBTASKS]
    hf_score = Dataset.from_pandas(df_score)
    hf_score.push_to_hub("lmms-lab/LiveBenchResults", dataset_version, split="test")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--folder", "-f", type=str, required=True, help="Results folder")
    argparse.add_argument("--name", "-m", type=str, required=True, help="Model name")
    argparse.add_argument("--log_folder", "-l", type=str, default="logs", help="Log folder")
    argparse.add_argument("--force", "-F", action="store_true", help="Force upload")
    args = argparse.parse_args()
    hf_results, score, version = get_results(args.folder)
    print(f"Results will be uploaded with model name {args.name} and model version {version}")
    if args.force is False:
        print("Are you sure you want to upload the results? (y/n)", end=" ")
        while True:
            choice = input().lower()
            if choice == "y":
                break
            elif choice == "n":
                exit()
            else:
                print("Invalid choice, please enter 'y' or 'n'")
    upload_results(hf_results, score, args.name, version, args.log_folder)
