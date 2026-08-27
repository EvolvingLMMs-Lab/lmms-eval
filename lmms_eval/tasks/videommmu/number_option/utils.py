# VideoMMMU Number Option Setting Utils
import datasets


def videommmu_process_docs_number_option(dataset: datasets.Dataset) -> datasets.Dataset:
    """Change option labels from A.B.C.D to 1.2.3.4 for MCQ questions."""

    def convert_to_number_options(example):
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

        # Convert answer from letter to number
        new_answer = str(answer_idx + 1)

        return {"answer": new_answer}

    return dataset.map(convert_to_number_options)
