# VideoMME GT None Option Setting Utils
# Replace correct answer content with "None" to test position bias

import datasets


def videomme_process_docs_gt_none(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Replace the content of the correct answer option with "None".
    Example: Original A. Apple, B. Banana, C. Orange, D. Grape, answer is B
    After processing: A. Apple, B. None, C. Orange, D. Grape, answer is still B
    """
    letters = ["A", "B", "C", "D"]

    def replace_gt_with_none(example):
        options = example["options"]
        answer = example["answer"]

        # Get the index of the correct answer
        answer_idx = ord(answer.upper()) - ord("A")

        # Extract option contents (remove "X. " prefix)
        contents = []
        for opt in options:
            if ". " in opt:
                contents.append(opt.split(". ", 1)[1])
            else:
                contents.append(opt)

        # Replace the correct answer content with "None"
        if 0 <= answer_idx < len(contents):
            contents[answer_idx] = "None"

        # Format new options
        new_options = [f"{letters[i]}. {contents[i]}" for i in range(len(contents))]

        return {"options": new_options, "answer": answer}

    return dataset.map(replace_gt_with_none)
