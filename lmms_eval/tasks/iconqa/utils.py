import json
import os


def options_to_str(options_prompt):
    option_prompt_str = ""
    for i, option in enumerate(options_prompt):
        option_choice = chr(ord("A") + i)
        option_prompt_str += f"{option_choice}. {option}\n"

    option_prompt_str = option_prompt_str.rstrip("\n")
    return option_prompt_str


def doc_to_visual(doc):
    image_list = []
    if "query_image" in doc:
        image_list.append(doc["query_image"].convert("RGB"))
    for i in range(5):
        id = f"choice_image_{i}"
        if id in doc and doc[id] is not None:
            image_list.append(doc[id].convert("RGB"))
    assert len(image_list) < 6, "Maximum 5 images allowed for ICON-QA"
    return image_list


def doc_to_text(doc, model_specific_prompt_kwargs):
    question = doc["question"]
    ques_type = doc["ques_type"]
    options_prompt = []

    if ques_type == "choose_img":
        options_prompt.append("The first image.")
        options_prompt.append("The second image.")

        options_str = options_to_str(options_prompt)
        full_prompt = f"{model_specific_prompt_kwargs['pre_prompt']}{model_specific_prompt_kwargs['statement']}{model_specific_prompt_kwargs['options_statement'].format(question=question, options=options_str)}"

    elif ques_type == "choose_txt":
        choices = doc["choices"].split(",")
        for i, choice in enumerate(choices):
            options_prompt.append(f"{choice}")

        options_str = options_to_str(options_prompt)
        full_prompt = f"{model_specific_prompt_kwargs['pre_prompt']}{model_specific_prompt_kwargs['statement']}{model_specific_prompt_kwargs['options_statement'].format(question=question, options=options_str)}"

    elif ques_type == "fill_in_blank":
        full_prompt = f"{model_specific_prompt_kwargs['pre_prompt']}{model_specific_prompt_kwargs['statement']}{model_specific_prompt_kwargs['freeform_statement'].format(question=question)}"

    return full_prompt


def test_process_results(doc, results):
    pred = results[0]
    questionId = doc["question_id"]
    answer = doc["answer"]
    return {"anls": {"questionId": int(questionId), "answer": answer, "pred_answer": pred}}
