import re
import ast

MULTI_CHOICE_PROMPT = "Answer with the option letter from the given choices directly."
OPEN_ENDED_PROMPT = "Answer the question using a single word or phrase."


def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def parse_options(options):
    parsed_result = "\n"
    for index, option in enumerate(options):
        # Assign a letter (starting from 'A') to each option
        letter = chr(65 + index)  # 65 is the ASCII value for 'A'
        parsed_result += f"{letter}. {option}\n"
    return parsed_result


def doc_to_text(doc):
    question = doc["question"]
    if doc["question_type"] == "multiple-choice":
        # Weirdly, data["options"] is a string in MMMU Huggingface dataset
        parsed_options = parse_options(ast.literal_eval(doc["options"]))
        # parsed_options already prepends a newline so no need to add space here
        question = f"{question}{parsed_options}{MULTI_CHOICE_PROMPT}"
        question = replace_images_tokens(question)
    else:
        question = f"{question}{OPEN_ENDED_PROMPT}"
        question = replace_images_tokens(question)
    return question


def doc_to_visual(doc):
    prompt = doc_to_text(doc)
    image_tokens = re.findall(r"<image \d+>", prompt)
    # Remove <> and  swap space as _
    image_tokens = [image_token.strip("<>").replace(" ", "_") for image_token in image_tokens]
    visual = [doc[image_token].convert("RGB") for image_token in image_tokens]
    return visual
