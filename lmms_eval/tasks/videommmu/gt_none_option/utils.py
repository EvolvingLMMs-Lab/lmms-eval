# VideoMMMU GT None Option Setting Utils
import datasets


def videommmu_process_docs_gt_none(dataset: datasets.Dataset) -> datasets.Dataset:
    """Replace correct answer content with 'None' for MCQ questions."""

    def replace_gt_with_none(example):
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

        answer_idx = ord(answer_letter) - ord("A")

        if answer_idx < 0 or answer_idx >= len(options):
            return example

        # Replace the correct answer content with "None"
        new_options = list(options)
        new_options[answer_idx] = "None"

        return {"options": new_options, "answer": answer}

    return dataset.map(replace_gt_with_none)
