import re


def parse_reasoning_model_answer(model_answer):
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", model_answer, re.DOTALL)
    boxed_answer_match = re.search(r"\\boxed\{\s*(.*?)\s*\}", model_answer, re.DOTALL)
    if answer_match:
        return answer_match.group(1)
    elif boxed_answer_match:
        return boxed_answer_match.group(1)
    else:
        return model_answer
