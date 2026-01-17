#!/usr/bin/env python3
"""
Subprocess helper for judge evaluation with GPU memory isolation.

Called by utils.py to run the 72B judge model in a clean GPU state.
"""

import pickle
import sys


def main():
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <input_pickle> <output_pickle>\n")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    with open(input_path, "rb") as f:
        pending_items = pickle.load(f)

    # Import here to ensure fresh GPU state
    from lmms_eval.tasks.captionqa.utils import evaluate_single_question

    try:
        from tqdm import tqdm

        iterator = tqdm(pending_items, desc="Judge Evaluating")
    except ImportError:
        iterator = pending_items

    results_dict = {}
    for item in iterator:
        cache_key = (item["image_id"], item["question_idx"])
        eval_result = evaluate_single_question(
            caption=item["caption"],
            image_id=item["image_id"],
            q_idx=item["question_idx"],
            question_text=item["question"],
            choices=item["choices"],
            answer=item["answer"],
        )
        if eval_result:
            results_dict[cache_key] = eval_result

    with open(output_path, "wb") as f:
        pickle.dump(results_dict, f)

    print(f"Judge evaluation complete. Evaluated {len(results_dict)} questions.")


if __name__ == "__main__":
    main()
