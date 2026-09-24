"""Print pinned GRT profile commands; execute only after inspecting the plan."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from importlib.resources import files
from pathlib import Path


def load_profiles() -> dict:
    """Load the public release's pinned model arguments from package data."""
    return json.loads(files("lmms_eval.models.model_utils.grt").joinpath("profiles.json").read_text())


def command(profile: str, role: str, output: Path) -> list[str]:
    """Construct one exact-profile command without loading models or data."""
    entry = load_profiles()["profiles"][profile]
    selected = entry["roles"][role]
    return [
        sys.executable,
        "-m",
        "lmms_eval.models.model_utils.grt.worker",
        "--model",
        selected["model"],
        "--model_args",
        selected["model_args"],
        "--tasks",
        entry["task"],
        "--batch_size",
        "1",
        "--limit",
        str(entry["samples"]),
        "--seed",
        "0,1234,1234,1234",
        "--gen_kwargs",
        f"max_new_tokens={entry['max_new_tokens']},temperature=0,top_p=1.0,num_beams=1,do_sample=False",
        "--log_samples",
        "--output_path",
        str(output / selected["method"]),
    ]


def main() -> None:
    """Print a shell command; no inference, downloads or output writes occur."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(load_profiles()["profiles"]), required=True)
    parser.add_argument("--role", choices=("base", "all", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 " + shlex.join(command(args.profile, args.role, args.output)))


if __name__ == "__main__":
    main()
