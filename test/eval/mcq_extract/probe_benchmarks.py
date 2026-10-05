"""Replay benchmark attack fixtures through actual local parsing paths.

This diagnostic records observations, not desired behavior for native scorers.
It never calls a judge API, loads a dataset, or mocks a parsing function.
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer
from lmms_eval.tasks._task_utils.mmmu_mcq_utils import (
    parse_mmmu_multi_choice_response,
    parse_mmmu_pro_multi_choice_response,
    parse_videommmu_multi_choice_response,
)
from lmms_eval.tasks.ai2d.utils import MultiChoiceRegexFilter
from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator
from lmms_eval.tasks.seedbench.utils import seed_process_result


def native_prediction(case: dict[str, Any]) -> str | bool:
    """Call the fixture's named benchmark parser with synthetic option context."""
    context = case["benchmark_context"]
    response, choices, options = case["response"], case["choices"], context["options"]
    parser = context["parser"]
    if parser == "mmmu":
        return parse_mmmu_multi_choice_response(response, choices, options)
    if parser == "mmmu_pro":
        return parse_mmmu_pro_multi_choice_response(response, choices, options)
    if parser == "videommmu":
        return parse_videommmu_multi_choice_response(response, choices, options)
    if parser == "mmbench":
        # The real prefetch path runs before both static and API judging.
        return MMBench_Evaluator().can_infer(response, dict(options))
    if parser == "seedbench":
        doc = {"answer": choices[0], "data_type": "image", "question_id": case["id"]}
        return seed_process_result(doc, [response])["seed_all"]["pred"]
    if parser == "ai2d":
        # Use the settings from ai2d.yaml; the override consumes no doc fields.
        extractor = MultiChoiceRegexFilter(regex_pattern=r"([A-Z])\.", group_select=0, ignore_case=True, ignore_punctuation=True)
        return extractor.apply([[response]], [{"options": list(options.values())}])[0]
    raise ValueError(f"Unknown benchmark parser: {parser}")


def main() -> int:
    """Write raw native predictions and agreement with the reviewed contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write all observations as JSON")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    corpus_path = Path(__file__).with_name("cases.json")
    cases = json.loads(corpus_path.read_text(encoding="utf-8"))["cases"]
    cases = [case for case in cases if case["category"].startswith("hack_")]
    rows = []
    rng_state = random.getstate()
    try:
        for case in cases:
            seeds = (0, 1, 7, 42) if case["benchmark_context"]["parser"] in {"mmmu", "mmmu_pro"} else (None,)
            observations = []
            for seed in seeds:
                if seed is not None:
                    random.seed(seed)
                raw = native_prediction(case)
                # Retain False or unmatched strings in the artifact. Only a
                # returned offered letter denotes a native MCQ selection.
                selected = raw.upper() if isinstance(raw, str) and raw.upper() in case["choices"] else ""
                observations.append({"seed": seed, "raw": raw, "selected": selected, "matches_contract": selected == case["expected"]})
            shared = extract_mcq_answer(case["response"], choices=case["choices"])
            rows.append({"id": case["id"], "category": case["category"], "expected": case["expected"], "shared": shared, "shared_pass": shared == case["expected"], "native": observations})
    finally:
        random.setstate(rng_state)
    report = {
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "python": sys.version,
        "versions": {package: version(package) for package in ("numpy", "pandas")},
        "scope": "Synthetic outputs and option text; parser/prefetch observations, not dataset accuracy or judge API results.",
        "total": len(rows),
        "shared_passed": sum(row["shared_pass"] for row in rows),
        "native_disagreement_cases": sum(any(not observation["matches_contract"] for observation in row["native"]) for row in rows),
        "cases": rows,
    }
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False))
    return int(report["shared_passed"] != report["total"])


if __name__ == "__main__":
    raise SystemExit(main())
