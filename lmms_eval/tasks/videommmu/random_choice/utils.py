# VideoMMMU Random Choice Setting Utils
import random

import datasets


def videommmu_process_docs_random_choice(dataset: datasets.Dataset) -> datasets.Dataset:
    """Shuffle options and update answer accordingly for MCQ questions."""

    def shuffle_options(example):
        # Skip non-MCQ questions
        if example.get("question_type") != "multiple-choice":
            return example

        options = example.get("options", [])
        answer = example.get("answer", "")

        if not options or not answer:
            return example

        # Get answer letter
        answer_letter = answer.strip().upper()
        if answer_letter not in "ABCDEFGH":
            return example

        letters = [chr(ord("A") + i) for i in range(len(options))]
        answer_idx = ord(answer_letter) - ord("A")

        if answer_idx < 0 or answer_idx >= len(options):
            return example

        answer_content = options[answer_idx]

        # Create shuffled indices with fixed seed
        seed = hash(example.get("id", str(example))) % (2**32)
        rng = random.Random(seed)
        indices = list(range(len(options)))
        rng.shuffle(indices)

        # Reorder options
        new_options = [options[i] for i in indices]

        # Find new answer position
        new_answer_idx = new_options.index(answer_content)
        new_answer = letters[new_answer_idx]

        return {"options": new_options, "answer": new_answer}

    return dataset.map(shuffle_options)
