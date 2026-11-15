"""CPU-only public-output parity check, not fresh inference or leaderboard parity.

Downloads a pinned published pipeline, renders a deterministic sample, runs the
actual task and compares all official slots/dimensions against the official CLI.
Produces a from_log input for a separate public lmms_eval CLI replay. Nothing
is downloaded into the source tree unless --output explicitly points there.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

from lmms_eval.tasks.parsebench2 import utils

RESULTS_REVISION = "80df59e43b88c4474a1c9b90d0315edc70b2b1f8"
PIPELINE = "azure_di_layout"


def main() -> None:
    """Compare public saved predictions through the task and official file runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-subset", type=int, default=10)
    parser.add_argument("--pipeline", default=PIPELINE)
    args = parser.parse_args()
    pipeline = args.pipeline
    if not pipeline or "/" in pipeline or "\\" in pipeline or pipeline in {".", ".."}:
        parser.error("--pipeline must be a published pipeline name")
    if args.per_subset < 1:
        parser.error("--per-subset must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    utils.require_scorer()
    inputs = utils.load_inputs()
    counts = {}
    selected = []
    for doc in inputs:
        count = counts.get(doc["subset"], 0)
        if count < args.per_subset:
            selected.append(doc)
        counts[doc["subset"]] = count + 1
    # load_inputs interleaves deterministically; the prefix gives equal coverage
    # until a subset is exhausted. Map by stable ID, never by output file order.
    selected_ids = {doc["example_id"] for doc in selected}

    def fetch_records(subset: str) -> list[dict]:
        path = hf_hub_download("llamaindex/ParseBench-2-results", f"results/{pipeline}/{subset}.jsonl", repo_type="dataset", revision=RESULTS_REVISION)
        with open(path, encoding="utf-8") as handle:
            return [record for line in handle if (record := json.loads(line))["request"]["example_id"] in selected_ids]

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = {record["request"]["example_id"]: record for batch in pool.map(fetch_records, utils.SUBSETS) for record in batch}
    if set(records) != selected_ids:
        raise RuntimeError(f"Published output coverage differs; missing={selected_ids - set(records)}")
    items = []
    logs = []
    rendered_pages = 0
    for doc in selected:
        record = records[doc["example_id"]]
        images = utils.doc_to_visual(doc)
        rendered_pages += len(images)
        for image in images:
            image.close()
        item = utils.process_results(doc, [json.dumps(record)])["parsebench2_overall"]
        items.append(item)
        subset, stem = doc["example_id"].split("/")
        cases = root / "cases" / subset
        results = root / "direct" / subset / pipeline
        cases.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
        for filename in [doc["document"], f"{doc['example_id']}.test.json"]:
            shutil.copyfile(utils.fetch(filename), cases / Path(filename).name)
        (results / f"{stem}.result.json").write_text(json.dumps(record), encoding="utf-8")
        (results / "_metadata.json").write_text(json.dumps({"pipeline": {"pipeline_name": pipeline, "provider_name": pipeline, "product_type": "parse", "config": {}}}))
        logs.append({"doc_id": inputs.index(doc), "resps": [[json.dumps(record)]]})
    report_dir = root / "official-report"
    report_dir.mkdir()
    mappings = ",".join(f"{s}={root / 'direct' / s / pipeline}::{root / 'cases' / s}" for s in utils.SUBSETS)
    subprocess.run([sys.executable, "-m", "parse_bench.cli", "evaluation", "run", str(report_dir), f"--next={mappings}", "--next_only"], check=True)
    direct = json.loads((report_dir / "_next_score.json").read_text())
    actual = utils.combined_report(items)

    def compare(expected: object, observed: object) -> None:
        # Official CLI workers finish in varying orders; floating point sums
        # can differ in their last bit. All structure/counts remain exact.
        if isinstance(expected, float):
            if not isinstance(observed, (float, int)) or not math.isclose(expected, observed, rel_tol=0, abs_tol=1e-12):
                raise AssertionError(f"Score mismatch: {expected}, {observed}")
        elif isinstance(expected, dict):
            if not isinstance(observed, dict) or expected.keys() != observed.keys():
                raise AssertionError("Report keys differ")
            for key in expected:
                compare(expected[key], observed[key])
        elif isinstance(expected, list):
            if not isinstance(observed, list) or len(expected) != len(observed):
                raise AssertionError("Report slots differ")
            for left, right in zip(expected, observed):
                compare(left, right)
        elif expected != observed:
            raise AssertionError(f"Report value mismatch: {expected}, {observed}")

    # The CLI adds filesystem provenance, which is not part of combine_next_score.
    provenance = direct.pop("subsets")
    expected_provenance = {s: {"output_dir": str(root / "direct" / s / pipeline), "test_cases_dir": str(root / "cases" / s)} for s in utils.SUBSETS}
    if provenance != expected_provenance:
        raise AssertionError("Official CLI subset source paths differ")
    compare(direct, actual)
    for subset in utils.SUBSETS:
        summary = json.loads((root / "direct" / subset / pipeline / "_evaluation_report.json").read_text())
        expected = sum(doc["subset"] == subset for doc in selected)
        if (summary["total_examples"], summary["successful"], summary["failed"], summary["skipped"]) != (expected, expected, 0, 0):
            raise AssertionError(f"Official scorer did not evaluate every {subset} document")
    replay = root / "from-log"
    replay.mkdir()
    (replay / "published.json").write_text(json.dumps({"args": {"model": pipeline, "model_args": "published", "limit": None}, "model_configs": {"task": "parsebench2"}, "logs": logs}))
    receipt = {
        "dataset_revision": utils.DATA_REVISION,
        "results_revision": RESULTS_REVISION,
        "pipeline": pipeline,
        "scorer": f"parse-bench=={utils.SCORER_VERSION}",
        "documents": len(selected),
        "rendered_pages": rendered_pages,
        "manifest_documents": len(inputs),
        "official_cli_report_parity": True,
        "absolute_score_tolerance": 1e-12,
        "scope": "Deterministic first documents per subset; published-output scoring and rendering only. No fresh inference or full benchmark parity.",
        "report": actual,
    }
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    print(f"Replay via: python -m lmms_eval --model from_log --model_args logs={replay} --tasks parsebench2 --limit {len(selected)} --batch_size 1 --device cpu --log_samples --output_path {root / 'harness'}")


if __name__ == "__main__":
    main()
