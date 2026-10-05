"""Report the reviewed MCQ corpus against the checkout or a Git revision."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="Git revision to load; defaults to the current checkout")
    parser.add_argument("--output", type=Path, help="Write the full JSON report here")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    module_path = "lmms_eval/tasks/_task_utils/mcq_extract.py"
    if args.revision:
        source = subprocess.check_output(["git", "show", f"{args.revision}:{module_path}"], cwd=repo, text=True)
        module = ModuleType("mcq_under_test")
        sys.modules[module.__name__] = module
        exec(compile(source, f"{args.revision}:{module_path}", "exec"), module.__dict__)
        extract = module.extract_mcq_answer
    else:
        spec = importlib.util.spec_from_file_location("mcq_under_test", repo / module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module.__name__] = module
        spec.loader.exec_module(module)
        extract = module.extract_mcq_answer
    corpus = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
    rows = []
    for case in corpus["cases"]:
        actual = extract(case["response"], choices=case["choices"])
        rows.append({**case, "actual": actual, "pass": actual == case["expected"]})
    passed = sum(row["pass"] for row in rows)
    report = {"revision": args.revision or "working tree", "python": sys.version, "total": len(rows), "passed": passed, "failed": len(rows) - passed, "cases": rows}
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False))
    return int(passed != len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
