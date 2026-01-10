import logging
import re
from typing import List

eval_logger = logging.getLogger("lmms-eval")


def mantis_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question_type, question, option = (
        doc["question_type"],
        doc["question"],
        doc["options"],
    )
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if question_type == "short-answer":
        option = ""
        final_question = f'Given the images, answer the following short answer vqa question:\nQ: {question}\nYou can first give your analysis, and then give your final answer as "Final Answer:"'
    if question_type == "multi-choice":
        final_question = f"{question}\nAnswer with the option's letter from the given choices directly."

    return f"{pre_prompt}{final_question}{option}{post_prompt}"


def mantis_doc_to_visual(doc):
    image_list = [image.convert("RGB") for image in doc["images"]]
    return image_list


def mantis_doc_to_target(doc):
    return doc["answer"]


def parse_multi_choice_response(response):
    option_letter_regex = re.compile(r"^\s*([A-Z])\.")
    match = option_letter_regex.match(response)
    if match:
        filtered_resps = match.group(1)
    else:
        filtered_resps = response
    return filtered_resps


def parse_answer(raw_answer):
    if "final answer:" in raw_answer.lower():
        answer = raw_answer[
            raw_answer.lower().index("final answer:") + len("final answer:") :
        ].strip()
    elif "the answer is" in raw_answer.lower():
        answer = raw_answer[
            raw_answer.lower().index("the answer is") + len("the answer is") :
        ].strip()
    elif "answer:" in raw_answer.lower():
        answer = raw_answer[
            raw_answer.lower().index("answer:") + len("answer:") :
        ].strip()
    else:
        answer = raw_answer
    return answer


def get_option(final_answer):
    if re.match(r"Answer: [A-Z]", final_answer):
        return final_answer[8]
    for s in final_answer:
        if s.isalpha():
            return s.upper()
    return None


def get_prediction(
    question_type: str, raw_answer: str, ref_answer: str, options: List[str]
):
    answer = parse_answer(raw_answer)
    ref_answer = ref_answer.strip("()\n ")  # important for some datasets
    if question_type == "multi-choice":
        if not len(ref_answer) == 1:
            for c in ref_answer:
                if c.isalpha():
                    ref_answer = c
                    break
        assert len(ref_answer) == 1, (
            f"Ref answer is not a single character: {ref_answer}"
        )

        selected_option = get_option(answer)
        if selected_option and (ord(selected_option) - ord("A") < len(options)):
            correct = selected_option == ref_answer.upper()
            parsed_answer = selected_option
        else:
            ref_option_idx = ord(ref_answer.upper()) - ord("A")
            if ref_option_idx >= len(options):
                correct = False
                parsed_answer = raw_answer
            else:
                ref_raw_answer = options[ref_option_idx]
                if ref_raw_answer.startswith(ref_answer + "."):
                    correct = (
                        raw_answer.strip()
                        == ref_raw_answer[len(ref_answer + ".") :].strip()
                    )
                elif ref_raw_answer.startswith(ref_answer + ":"):
                    correct = (
                        raw_answer.strip()
                        == ref_raw_answer[len(ref_answer + ":") :].strip()
                    )
                elif ref_raw_answer.startswith("(" + ref_answer + ")"):
                    correct = (
                        raw_answer.strip()
                        == ref_raw_answer[len(ref_answer) + 2 :].strip()
                    )
                else:
                    correct = raw_answer.strip() == ref_raw_answer.strip()
            parsed_answer = raw_answer
    elif question_type == "short-answer":
        correct = ref_answer.lower() == answer.lower()
        parsed_answer = answer

    return {
        "raw_answer": raw_answer,
        "parsed_answer": parsed_answer,
        "correct": correct,
    }


def mantis_process_results(doc, results):
    pred = results[0]
    question_type, answer, options = doc["question_type"], doc["answer"], doc["options"]

    parsed_pred = get_prediction(question_type, pred, answer, options)
    data_dict = {
        "question_id": doc["id"],
        "pred_answer": parsed_pred["parsed_answer"],
        "answer": doc["answer"],
        "correct": parsed_pred["correct"],
    }

    return {f"mantis_score": data_dict}


def eval_multi_choice(gold_i, pred_i):
    correct = False
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:
        if gold_i == pred_i:
            correct = True
    return correct


def mantis_aggregation(results):
    score = 0
    for result in results:
        if result["correct"]:
            score += 1
    avg_score = score / len(results)

    return avg_score
