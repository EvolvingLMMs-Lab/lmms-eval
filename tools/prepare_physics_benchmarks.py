"""Build and publish normalized datasets for the physics benchmark tasks.

The upstream repositories mix JSON annotations with ZIP archives or nested
problem directories. This script converts the annotations to stable parquet
schemas while preserving the original media files and licenses.

Examples:
    uv run python tools/prepare_physics_benchmarks.py --dataset all
    uv run python tools/prepare_physics_benchmarks.py --dataset physbench --push
"""

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlencode

import datasets
from huggingface_hub import HfApi, hf_hub_download, metadata_update

TARGET_ORG = "lmms-lab-eval"

PHYSBENCH_SOURCE = "USC-PSI-Lab/PhysBench"
PHYSGAME_SOURCE = "PhysGame/GameBench"
PHYSICS_RW_SOURCE = "pengyz/Physics-RW"
PHYSREASON_SOURCE = "zhibei1204/PhysReason"

SOURCE_REVISIONS = {
    PHYSBENCH_SOURCE: "478fd93da8ec8d6f5252b9586b1fa10f335c5a95",
    PHYSGAME_SOURCE: "fd5c649f02a876b7b32061cba08b8622e12e25eb",
    PHYSREASON_SOURCE: "b63b7fa6e6bc99563038cafb1e136b060da2f91d",
}
PHYSBENCH_ANSWER_REVISION = "a7eadc4da554e20163303ec0c1e076a35a9f52e2"
PHYSBENCH_ANSWER_URL = f"https://raw.githubusercontent.com/USC-GVL/PhysBench/{PHYSBENCH_ANSWER_REVISION}/eval/physbench/test_answer.json"

PHYSICS_RW_DOMAINS = ["Electromagnetism", "Mechanics", "Optics", "Thermodynamics"]
PHYSICS_RW_REVISION = "79c52f89d7113f8b71b4da422630a5b214a79ab4"
PHYSICS_RW_PAPER_COUNTS = {
    "Mechanics": 716,
    "Thermodynamics": 138,
    "Electromagnetism": 152,
    "Optics": 129,
}
PHYSICS_RW_MODELSCOPE_ENDPOINT = f"https://www.modelscope.cn/api/v1/datasets/{PHYSICS_RW_SOURCE}/repo"
PHYSREASON_ARCHIVES = {
    "full": ("PhysReason-full.zip", "PhysReason_full"),
    "mini": ("PhysReason-mini.zip", "PhysReason-mini"),
}
PHYSREASON_MAX_IMAGES = 5

DATASET_METADATA = {
    "physbench": {
        "repo": "PhysBench",
        "pretty_name": "PhysBench",
        "license": "apache-2.0",
        "source": "https://huggingface.co/datasets/USC-PSI-Lab/PhysBench",
        "description": "Normalized PhysBench annotations with the official public answers and media archives.",
    },
    "physgame": {
        "repo": "PhysGame",
        "pretty_name": "PhysGame / GameBench",
        "license": "apache-2.0",
        "source": "https://huggingface.co/datasets/PhysGame/GameBench",
        "description": "Normalized 880-question PhysGame benchmark with the official gameplay video archive.",
    },
    "physics_rw": {
        "repo": "Physics-RW",
        "pretty_name": "Physics-RW",
        "license": "cc-by-nc-4.0",
        "source": "https://www.modelscope.cn/datasets/pengyz/Physics-RW",
        "description": "Normalized 1,135-example English Physics-RW classification benchmark and official videos.",
    },
    "physreason": {
        "repo": "PhysReason",
        "pretty_name": "PhysReason",
        "license": "mit",
        "source": "https://huggingface.co/datasets/zhibei1204/PhysReason",
        "description": "Normalized full and mini PhysReason configurations with Viewer-compatible image columns.",
    },
}


def _read_json_url(url: str):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _modelscope_url(relative_path: str) -> str:
    query = urlencode({"Revision": PHYSICS_RW_REVISION, "FilePath": relative_path})
    return f"{PHYSICS_RW_MODELSCOPE_ENDPOINT}?{query}"


def _download_url(url: str, destination: Path, *, retries: int = 5) -> None:
    """Download a large source file atomically, retrying transient CDN errors."""
    if destination.is_file() and destination.stat().st_size > 0:
        head_request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "lmms-eval-dataset-preparer"})
        try:
            with urllib.request.urlopen(head_request, timeout=30) as response:
                expected_size = int(response.headers.get("Content-Length", 0))
                expected_sha256 = response.headers.get("X-Linked-Etag", "").strip('"').lower()
            size_matches = expected_size and destination.stat().st_size == expected_size
            hash_matches = not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or _sha256(destination) == expected_sha256
            if size_matches and hash_matches:
                return
        except (OSError, TimeoutError, urllib.error.URLError, http.client.IncompleteRead):
            # A fresh, validated GET below is safer than trusting a cached file
            # when the CDN metadata request itself is unavailable.
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(f"{destination.name}.partial")
    request = urllib.request.Request(url, headers={"User-Agent": "lmms-eval-dataset-preparer"})

    for attempt in range(1, retries + 1):
        partial_path.unlink(missing_ok=True)
        try:
            with urllib.request.urlopen(request, timeout=300) as response, open(partial_path, "wb") as output:
                expected_size = int(response.headers.get("Content-Length", 0))
                expected_sha256 = response.headers.get("X-Linked-Etag", "").strip('"').lower()
                shutil.copyfileobj(response, output, length=1024 * 1024)
            downloaded_size = partial_path.stat().st_size
            if downloaded_size == 0:
                raise OSError(f"Downloaded an empty file from {url}")
            if expected_size and downloaded_size != expected_size:
                raise OSError(f"Incomplete download from {url}: expected {expected_size} bytes, got {downloaded_size}")
            if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) and _sha256(partial_path) != expected_sha256:
                raise OSError(f"Checksum mismatch for {url}")
            partial_path.replace(destination)
            return
        except (OSError, TimeoutError, urllib.error.URLError, http.client.IncompleteRead):
            partial_path.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(2**attempt)


def _sha256(path: Path) -> str:
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def build_physbench() -> datasets.DatasetDict:
    annotation_path = hf_hub_download(PHYSBENCH_SOURCE, "test.json", repo_type="dataset", revision=SOURCE_REVISIONS[PHYSBENCH_SOURCE])
    with open(annotation_path, encoding="utf-8") as file:
        annotations = json.load(file)

    answer_rows = _read_json_url(PHYSBENCH_ANSWER_URL)
    answers = {int(row["idx"]): row for row in answer_rows}

    split_rows = {"val": [], "test": []}
    full_rows = []
    for annotation in annotations:
        idx = int(annotation["idx"])
        split = annotation["split"]
        if split not in split_rows:
            continue

        answer_info = answers.get(idx, {})
        row = {
            "idx": idx,
            "split": split,
            "question": str(annotation["question"]),
            "file_name": [str(name) for name in annotation.get("file_name", [])],
            "answer": str(answer_info.get("answer", "")),
            "task_type": str(answer_info.get("task_type", "")),
            "sub_type": str(answer_info.get("sub_type", "")),
            "ability_type": str(answer_info.get("ability_type", "")),
        }
        split_rows[split].append(row)
        full_rows.append(row)

    assert len(split_rows["val"]) == 200
    assert len(split_rows["test"]) == 9802
    assert len(full_rows) == 10002
    assert sum(not row["answer"] for rows in split_rows.values() for row in rows) == 17
    assert all(row["file_name"] for rows in split_rows.values() for row in rows)

    features = datasets.Features(
        {
            "idx": datasets.Value("int64"),
            "split": datasets.Value("string"),
            "question": datasets.Value("string"),
            "file_name": datasets.Sequence(datasets.Value("string")),
            "answer": datasets.Value("string"),
            "task_type": datasets.Value("string"),
            "sub_type": datasets.Value("string"),
            "ability_type": datasets.Value("string"),
        }
    )
    return datasets.DatasetDict(
        {
            "full": datasets.Dataset.from_list(full_rows, features=features),
            **{split: datasets.Dataset.from_list(rows, features=features) for split, rows in split_rows.items()},
        }
    )


def build_physgame() -> datasets.DatasetDict:
    annotation_path = hf_hub_download(PHYSGAME_SOURCE, "PhysGame_880_annotation.json", repo_type="dataset", revision=SOURCE_REVISIONS[PHYSGAME_SOURCE])
    with open(annotation_path, encoding="utf-8") as file:
        annotations = json.load(file)

    rows = []
    for annotation in annotations:
        question_id = str(annotation["question_id"])
        options = annotation["options"]
        rows.append(
            {
                "question_id": question_id,
                "question": str(annotation["question"]),
                "options": {letter: str(options[letter]) for letter in "ABCD"},
                "answer": str(annotation["answer"]).upper(),
                "class_anno": str(annotation["class_anno"]),
                "subclass_anno": str(annotation["subclass_anno"]),
                "video_path": f"PhysGame-Benchmark/{question_id}.mp4",
            }
        )

    assert len(rows) == 880
    assert len({row["question_id"] for row in rows}) == 880
    assert all(row["answer"] in "ABCD" for row in rows)

    features = datasets.Features(
        {
            "question_id": datasets.Value("string"),
            "question": datasets.Value("string"),
            "options": {letter: datasets.Value("string") for letter in "ABCD"},
            "answer": datasets.Value("string"),
            "class_anno": datasets.Value("string"),
            "subclass_anno": datasets.Value("string"),
            "video_path": datasets.Value("string"),
        }
    )
    return datasets.DatasetDict({"test": datasets.Dataset.from_list(rows, features=features)})


def build_physics_rw() -> datasets.DatasetDict:
    rows = []
    for domain in PHYSICS_RW_DOMAINS:
        relative_path = f"Physics-RW/{domain}/classification/classification_en.json"
        annotations = _read_json_url(_modelscope_url(relative_path))
        paper_count = PHYSICS_RW_PAPER_COUNTS[domain]
        if len(annotations) < paper_count:
            raise ValueError(f"Physics-RW {domain} has {len(annotations)} rows; expected at least {paper_count}")

        # The ModelScope Mechanics file contains 39 later rows whose labels are
        # still Chinese. Table 2 of the paper defines the English benchmark as
        # the first 716 Mechanics rows, for 1,135 examples across four domains.
        for annotation in annotations[:paper_count]:
            source_idx = int(annotation["idx"])
            source_video_path = str(annotation["video_path"])
            label = str(annotation.get("answer", annotation.get("label", ""))).lower()
            rows.append(
                {
                    "id": f"{domain.lower()}-{source_idx}",
                    "source_idx": source_idx,
                    "domain": domain,
                    "instruction": str(annotation["instruction"]),
                    "label": label,
                    "video_path": f"media/{domain}/classification/{source_video_path}",
                }
            )

    assert len(rows) == sum(PHYSICS_RW_PAPER_COUNTS.values()) == 1135
    assert len({row["id"] for row in rows}) == 1135
    assert all(row["label"] in {"yes", "no"} for row in rows)

    features = datasets.Features(
        {
            "id": datasets.Value("string"),
            "source_idx": datasets.Value("int64"),
            "domain": datasets.Value("string"),
            "instruction": datasets.Value("string"),
            "label": datasets.Value("string"),
            "video_path": datasets.Value("string"),
        }
    )
    return datasets.DatasetDict({"test": datasets.Dataset.from_list(rows, features=features)})


def _numbered_key(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", value)
    return (int(match.group(1)) if match else 0, value)


def _normalize_explanation_steps(problem: dict) -> list[dict]:
    normalized = []
    explanation_steps = problem.get("explanation_steps", {})
    if not isinstance(explanation_steps, dict):
        return normalized

    for sub_question, steps in sorted(explanation_steps.items(), key=lambda item: _numbered_key(item[0])):
        if not isinstance(steps, dict):
            continue
        for step, explanation in sorted(steps.items(), key=lambda item: _numbered_key(item[0])):
            normalized.append(
                {
                    "sub_question": str(sub_question),
                    "step": str(step),
                    "explanation": str(explanation),
                }
            )
    return normalized


def _normalize_step_analysis(problem: dict) -> list[dict]:
    # One full-set problem uses the misspelled source key ``steps _analysis``.
    step_analysis = problem.get("steps_analysis", problem.get("steps _analysis", {}))
    if not isinstance(step_analysis, dict):
        return []

    normalized = []
    for step, analysis in sorted(step_analysis.items(), key=lambda item: _numbered_key(item[0])):
        if not isinstance(analysis, dict):
            continue

        quantities = []
        for quantity in analysis.get("result_quantity", []):
            if not isinstance(quantity, dict):
                continue
            quantities.append(
                {
                    "name": str(quantity.get("name", "")),
                    "equation": str(quantity.get("equation", "")),
                    "symbol": str(quantity.get("symbol", "")),
                    "value": str(quantity.get("value", "")),
                    "unit": str(quantity.get("unit", "")),
                }
            )

        normalized.append(
            {
                "step": str(step),
                "physical_theorem": str(analysis.get("physical_theorem", "")),
                "result_quantities": quantities,
            }
        )
    return normalized


def _normalize_theorems(problem: dict, step_analysis: list[dict]) -> list[str]:
    theorems = problem.get("Theorem", [])
    if isinstance(theorems, str):
        theorems = [theorems]
    if not isinstance(theorems, list):
        theorems = []

    normalized = [str(theorem) for theorem in theorems if str(theorem).strip()]
    if not normalized:
        normalized = list(dict.fromkeys(analysis["physical_theorem"] for analysis in step_analysis if analysis["physical_theorem"]))
    return normalized


def _normalize_image_captions(problem: dict) -> str:
    # Four mini-set problems use the singular source key.
    captions = problem.get("image_captions", problem.get("image_caption", ""))
    if isinstance(captions, list):
        captions = " ".join(str(caption) for caption in captions)
    return str(captions or "")


def _build_physreason_config(config_name: str) -> datasets.DatasetDict:
    archive_name, expected_root = PHYSREASON_ARCHIVES[config_name]
    archive_path = hf_hub_download(PHYSREASON_SOURCE, archive_name, repo_type="dataset", revision=SOURCE_REVISIONS[PHYSREASON_SOURCE])

    with tempfile.TemporaryDirectory(prefix=f"physreason-{config_name}-") as temp_dir:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temp_dir)

        root = Path(temp_dir) / expected_root
        if not root.is_dir():
            root = Path(temp_dir)

        rows = []
        for problem_path in sorted(root.iterdir()):
            json_path = problem_path / "problem.json"
            if not json_path.is_file():
                continue

            with open(json_path, encoding="utf-8") as file:
                problem = json.load(file)

            question_structure = problem.get("question_structure", {})
            sub_questions = []
            index = 1
            while f"sub_question_{index}" in question_structure:
                sub_questions.append(str(question_structure[f"sub_question_{index}"]))
                index += 1

            answers = problem.get("answer", [])
            if isinstance(answers, str):
                answers = [answers]

            image_file_names = [str(path) for path in problem.get("question_image_list", [])]
            images = []
            for relative_path in image_file_names:
                image_path = problem_path / relative_path
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing PhysReason image: {image_path}")
                images.append({"bytes": image_path.read_bytes(), "path": image_path.name})

            explanation_steps = _normalize_explanation_steps(problem)
            step_analysis = _normalize_step_analysis(problem)

            row = {
                "problem_id": problem_path.name,
                **{f"image_{index + 1}": images[index] if index < len(images) else None for index in range(PHYSREASON_MAX_IMAGES)},
                "context": str(question_structure.get("context", "")),
                "sub_questions": sub_questions,
                "answers": [str(answer) for answer in answers],
                "difficulty": str(problem.get("difficulty", "unknown")),
                "image_file_names": image_file_names,
                "num_images": len(images),
                "image_captions": _normalize_image_captions(problem),
                "theorems": _normalize_theorems(problem, step_analysis),
                "explanation_steps": explanation_steps,
                "step_analysis": step_analysis,
                "num_steps": len(explanation_steps),
            }
            rows.append(row)

    expected_size = 1200 if config_name == "full" else 200
    assert len(rows) == expected_size
    assert len({row["problem_id"] for row in rows}) == expected_size
    assert all(len(row["sub_questions"]) == len(row["answers"]) for row in rows)
    assert all(row["num_images"] == len(row["image_file_names"]) for row in rows)
    assert max(row["num_images"] for row in rows) <= PHYSREASON_MAX_IMAGES
    assert all(row["num_steps"] == len(row["explanation_steps"]) for row in rows)

    features = datasets.Features(
        {
            "problem_id": datasets.Value("string"),
            **{f"image_{index + 1}": datasets.Image() for index in range(PHYSREASON_MAX_IMAGES)},
            "context": datasets.Value("string"),
            "sub_questions": datasets.Sequence(datasets.Value("string")),
            "answers": datasets.Sequence(datasets.Value("string")),
            "difficulty": datasets.Value("string"),
            "image_file_names": datasets.Sequence(datasets.Value("string")),
            "num_images": datasets.Value("int64"),
            "image_captions": datasets.Value("string"),
            "theorems": datasets.Sequence(datasets.Value("string")),
            "explanation_steps": [
                {
                    "sub_question": datasets.Value("string"),
                    "step": datasets.Value("string"),
                    "explanation": datasets.Value("string"),
                }
            ],
            "step_analysis": [
                {
                    "step": datasets.Value("string"),
                    "physical_theorem": datasets.Value("string"),
                    "result_quantities": [
                        {
                            "name": datasets.Value("string"),
                            "equation": datasets.Value("string"),
                            "symbol": datasets.Value("string"),
                            "value": datasets.Value("string"),
                            "unit": datasets.Value("string"),
                        }
                    ],
                }
            ],
            "num_steps": datasets.Value("int64"),
        }
    )
    return datasets.DatasetDict({"test": datasets.Dataset.from_list(rows, features=features)})


def build_physreason() -> dict[str, datasets.DatasetDict]:
    return {config_name: _build_physreason_config(config_name) for config_name in PHYSREASON_ARCHIVES}


BUILDERS = {
    "physbench": build_physbench,
    "physgame": build_physgame,
    "physics_rw": build_physics_rw,
    "physreason": build_physreason,
}


def _upload_physics_rw_media(api: HfApi, target_repo: str, dataset: datasets.DatasetDict) -> None:
    source_to_target = {}
    for row in dataset["test"]:
        target_path = Path(row["video_path"])
        if target_path.is_absolute() or ".." in target_path.parts or not target_path.parts or target_path.parts[0] != "media":
            raise ValueError(f"Invalid Physics-RW target path: {target_path}")
        source_path = Path("Physics-RW", *target_path.parts[1:])
        source_to_target[source_path] = target_path

    hf_home = Path(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")))
    # Keep source staging separate from the task cache symlink created by
    # lmms-eval. Mixing the two can mutate a Hub snapshot and make later runs
    # silently reuse incomplete files.
    cache_root = hf_home / "physics_rw_sources" / PHYSICS_RW_REVISION

    def download_media(item: tuple[Path, Path]) -> None:
        source_path, target_path = item
        _download_url(_modelscope_url(source_path.as_posix()), cache_root / target_path)

    items = sorted(source_to_target.items(), key=lambda item: item[0].as_posix())
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(download_media, items))

    api.upload_folder(
        repo_id=target_repo,
        repo_type="dataset",
        folder_path=cache_root,
        allow_patterns="media/**",
        commit_message="data: publish official classification videos",
    )


def _copy_media(api: HfApi, dataset_name: str, target_repo: str, dataset) -> None:
    if dataset_name == "physbench":
        for filename in ("image.zip", "video.zip"):
            api.copy_files(
                f"hf://datasets/{PHYSBENCH_SOURCE}@{SOURCE_REVISIONS[PHYSBENCH_SOURCE]}/{filename}",
                f"hf://datasets/{target_repo}/{filename}",
            )
    elif dataset_name == "physgame":
        api.copy_files(
            f"hf://datasets/{PHYSGAME_SOURCE}@{SOURCE_REVISIONS[PHYSGAME_SOURCE]}/PhysGame-Benchmark.zip",
            f"hf://datasets/{target_repo}/PhysGame-Benchmark.zip",
        )
    elif dataset_name == "physics_rw":
        _upload_physics_rw_media(api, target_repo, dataset)


def _update_card(api: HfApi, dataset_name: str, target_repo: str) -> None:
    info = DATASET_METADATA[dataset_name]
    metadata_update(
        target_repo,
        {
            "license": info["license"],
            "language": ["en"],
            "pretty_name": info["pretty_name"],
            "task_categories": ["question-answering"],
        },
        repo_type="dataset",
        overwrite=True,
        commit_message="docs: add normalized dataset metadata",
    )

    readme_path = hf_hub_download(target_repo, "README.md", repo_type="dataset", force_download=True)
    readme = Path(readme_path).read_text(encoding="utf-8")
    marker = "<!-- lmms-eval-normalized -->"
    if marker in readme:
        readme = readme.split(marker, 1)[0].rstrip()
    readme += f"\n\n{marker}\n# {info['pretty_name']} for lmms-eval\n\n{info['description']}\n\nSource: [{info['source']}]({info['source']}). The source dataset license is preserved.\n"
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=target_repo,
        repo_type="dataset",
        commit_message="docs: document normalized lmms-eval dataset",
    )


def publish_dataset(dataset_name: str, dataset, *, copy_media: bool) -> None:
    api = HfApi()
    target_repo = f"{TARGET_ORG}/{DATASET_METADATA[dataset_name]['repo']}"
    api.create_repo(target_repo, repo_type="dataset", exist_ok=True)

    if dataset_name == "physreason":
        for config_name, config_dataset in dataset.items():
            config_dataset.push_to_hub(
                target_repo,
                config_name=config_name,
                commit_message=f"data: publish normalized {config_name} split",
            )
    else:
        dataset.push_to_hub(target_repo, commit_message="data: publish normalized annotations")

    if copy_media:
        _copy_media(api, dataset_name, target_repo, dataset)
    _update_card(api, dataset_name, target_repo)


def _print_summary(dataset_name: str, dataset) -> None:
    if dataset_name == "physreason":
        for config_name, config_dataset in dataset.items():
            print(f"{dataset_name}/{config_name}: {config_dataset}")
    else:
        print(f"{dataset_name}: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *BUILDERS], default="all")
    parser.add_argument("--push", action="store_true", help=f"Publish to the {TARGET_ORG} organization")
    parser.add_argument("--skip-media", action="store_true", help="Skip server-side media copies when publishing")
    args = parser.parse_args()

    dataset_names = list(BUILDERS) if args.dataset == "all" else [args.dataset]
    for dataset_name in dataset_names:
        dataset = BUILDERS[dataset_name]()
        _print_summary(dataset_name, dataset)
        if args.push:
            publish_dataset(dataset_name, dataset, copy_media=not args.skip_media)


if __name__ == "__main__":
    main()
