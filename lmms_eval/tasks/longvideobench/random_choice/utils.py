# LongVideoBench Random Choice Setting Utils
import random

import datasets


def longvideobench_process_docs_random_choice(dataset: datasets.Dataset) -> datasets.Dataset:
    """Shuffle options and update answer accordingly."""

    def shuffle_options(example):
        # LongVideoBench uses option_0, option_1, option_2, option_3 fields
        options = [
            example.get("option_0", ""),
            example.get("option_1", ""),
            example.get("option_2", ""),
            example.get("option_3", ""),
        ]
        correct_choice = example.get("correct_choice", "")

        if not correct_choice:
            return example

        # Get correct answer index
        correct_idx = ord(correct_choice.upper()) - ord("A")
        if correct_idx < 0 or correct_idx >= len(options):
            return example

        correct_content = options[correct_idx]

        # Create shuffled indices with fixed seed
        seed = hash(example.get("video_id", str(example))) % (2**32)
        rng = random.Random(seed)
        indices = list(range(len(options)))
        rng.shuffle(indices)

        # Reorder options
        new_options = [options[i] for i in indices]

        # Find new correct answer position
        new_correct_idx = new_options.index(correct_content)
        new_correct_choice = chr(ord("A") + new_correct_idx)

        return {
            "option_0": new_options[0],
            "option_1": new_options[1],
            "option_2": new_options[2],
            "option_3": new_options[3],
            "correct_choice": new_correct_choice,
        }

    return dataset.map(shuffle_options)
