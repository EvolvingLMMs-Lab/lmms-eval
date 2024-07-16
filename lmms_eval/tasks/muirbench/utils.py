
from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter
import re
import pandas as pd


def muir_doc_to_text(doc, model_specific_prompt_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    post_prompt = model_specific_prompt_kwargs["post_prompt"]
    pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"


def muir_doc_to_visual(doc):
    image_list = [image.convert("RGB") for image in doc["image_list"]]
    return image_list 


def muir_doc_to_target(doc):
    return doc["answer"]


def muir_process_results(doc, result):
    pred = result[0]
    task = doc["task"]
    idx = doc["idx"]
    image_relation = doc["image_relation"]
    answer = doc["answer"]
    image_type = doc["image_type"]

    data_dict = {
            "pred" : pred,
            "task" : task,
            "idx" : idx,
            "image_relation" : image_relation,
            "answer" : answer,
            "image_type" : image_type,
        }

    return {"muirbench_score_overall" : data_dict}


def muir_aggregation(results):
    task_num = {}
    score = 0
    task_score = {}
    for result in results:
        if result["task"] not in task_score:
            task_score[result["task"]] = 0

        if result["task"] not in task_num:
            task_num[result["task"]] = 0

        if result["pred"].lower().strip() == result["answer"].lower().strip():
            task_score[result["task"]] += 1
            score += 1
        task_num[result["task"]] += 1

    score = score / len(results)

    task_score = {k : v / task_num[k] for k,v in task_score.items()}

    print("=" * 50)
    for k, v in task_score.items():
        print(f"{k} : {v:.2f}")
    print("=" * 50)
    
    return score




class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            # Regex to directly extract the option letter from the model response
            option_letter_regex = re.compile(r"^\s*([A-Z])\.")

            # Process each response
            filtered = []
            for resp in r:
                # Try to match the option letter at the start of the response
                match = option_letter_regex.match(resp)
                if match:
                    # If a match is found, append the matched letter
                    filtered.append(match.group(1))
                else:
                    # If no match, return the original response
                    filtered.append(resp)

            # Assuming we need the first response that matches or the original response
            filtered_resps.append(filtered[0])

        return filtered_resps