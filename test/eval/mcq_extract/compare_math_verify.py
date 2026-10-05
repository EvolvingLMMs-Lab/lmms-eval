"""Compare the installed math-verify string extractor with the reviewed corpus.

This is an optional comparison report, not a test requiring another parser to
implement this project's policies. It does not count as model E2E evidence.
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from math_verify import StringExtractionConfig, parse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write per-case comparison results as JSON")
    args = parser.parse_args()
    cases = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))["cases"]
    report = {"package": "math-verify", "version": version("math-verify"), "strategies": {}}
    for strategy in ("uppercase_tokens", "case_variants", "anchored_case_variants"):
        rows = []
        for case in cases:
            allowed = tuple(choice.upper() for choice in case["choices"] or "ABCDEFGH")
            strings = allowed if strategy == "uppercase_tokens" else (*allowed, *(choice.lower() for choice in allowed))
            config = StringExtractionConfig(strings=strings, try_extract_without_anchor=strategy != "anchored_case_variants")
            parsed = parse(case["response"] or "", extraction_config=[config], fallback_mode="no_fallback")
            actual = str(parsed[0]).upper() if parsed else ""
            rows.append({"id": case["id"], "expected": case["expected"], "actual": actual, "match": actual == case["expected"]})
        matched = sum(row["match"] for row in rows)
        report["strategies"][strategy] = {"total": len(rows), "matched": matched, "mismatched": len(rows) - matched, "cases": rows}
        print(f"{strategy}: {matched}/{len(rows)} match this project's contract")
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
