import re


def parse_reasoning_model_answer(model_answer: str) -> str:
    # First strip <think> blocks if present
    cleaned = re.sub(r"<think>.*?</think>", "", model_answer, flags=re.DOTALL).strip()

    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", cleaned, re.DOTALL)
    boxed_answer_match = re.search(r"\\boxed\{\s*(.*?)\s*\}", cleaned, re.DOTALL)
    if answer_match:
        return answer_match.group(1)
    elif boxed_answer_match:
        return boxed_answer_match.group(1)
    else:
        return cleaned
