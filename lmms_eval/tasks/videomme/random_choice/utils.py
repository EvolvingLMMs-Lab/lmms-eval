# VideoMME Random Choice Setting Utils
import random
from typing import Dict, List

import datasets

from lmms_eval.tasks.videomme.utils import (
    videomme_aggregate_results,
    videomme_doc_to_text,
    videomme_doc_to_visual,
    videomme_process_results,
)


def videomme_process_docs_random_choice(dataset: datasets.Dataset) -> datasets.Dataset:
    """Shuffle options and update answer accordingly."""

    def shuffle_options(example):
        options = example["options"]
        answer = example["answer"]

        # Parse options (format: "A. content", "B. content", etc.)
        letters = ["A", "B", "C", "D"]
        contents = []
        for opt in options:
            # Extract content after "X. "
            if ". " in opt:
                contents.append(opt.split(". ", 1)[1])
            else:
                contents.append(opt)

        # Find original answer index
        answer_idx = letters.index(answer.upper()) if answer.upper() in letters else 0
        answer_content = contents[answer_idx]

        # Create shuffled indices with fixed seed based on question_id for reproducibility
        seed = hash(example.get("question_id", str(example))) % (2**32)
        rng = random.Random(seed)
        indices = list(range(len(contents)))
        rng.shuffle(indices)

        # Reorder contents
        new_contents = [contents[i] for i in indices]

        # Find new answer position
        new_answer_idx = new_contents.index(answer_content)
        new_answer = letters[new_answer_idx]

        # Format new options
        new_options = [f"{letters[i]}. {new_contents[i]}" for i in range(len(new_contents))]

        return {"options": new_options, "answer": new_answer}

    return dataset.map(shuffle_options)
