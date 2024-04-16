import os
import re
import sys
import datetime
import lmms_eval.tasks._task_utils.file_utils as file_utils
from lmms_eval.filters.extraction import ExtendedRegexFilter
import json
import logging
import yaml
from pathlib import Path

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

# A bit ugly here
# But the idea is that we will unzip all the zip files
# To HF HOME cache dir
# And load it here
HF_HOME = os.environ["HF_HOME"]
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, cache_dir)
cache_dir = os.path.join(cache_dir)


eval_logger = logging.getLogger("lmms-eval")


# Pass in video path here
# Can only work correctly with video llm
def mix_evals_video2text_doc_to_visual(doc):
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# This is the place where you format your question
def mix_evals_video2text_doc_to_text(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]

    user_prompt = doc["prompt"]

    if "options" in doc:
        option_prompt = "Here are the options:\n"
        for idx, option in enumerate(doc["options"]):
            char_idx = chr(ord("A") + idx)
            option = option.strip()
            option_prompt += f"{char_idx}. {option}\n"

        option_prompt = option_prompt.rstrip("\n")
        user_prompt = f"{user_prompt}\n{option_prompt}"

    if pre_prompt:
        user_prompt = f"{pre_prompt}\n{user_prompt}"

    if post_prompt:
        user_prompt = f"{user_prompt}\n{post_prompt}"
    return user_prompt


def mix_evals_video2text_doc_to_text_open_convs(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]

    formatted_prompt = ""
    first_turn_user_prompt = doc["first_turn_user_prompt"]

    if pre_prompt:
        formatted_prompt = f"{pre_prompt}\n{first_turn_user_prompt}"
    else:
        formatted_prompt = f"{first_turn_user_prompt}"

    if "round2" in doc and doc["round2"]:
        second_turn_user_prompt = doc["second_turn_user_prompt"]
        formatted_prompt += f"{formatted_prompt}\n{second_turn_user_prompt}"
        if post_prompt:
            formatted_prompt += f"{formatted_prompt}\n{post_prompt}"
        return formatted_prompt
    else:
        if post_prompt:
            formatted_prompt += f"{formatted_prompt}\n{post_prompt}"
        return formatted_prompt


def mix_evals_video2text_process_results_open_convs(doc, result):
    pred = result[0]
    return {"submission": {"pred": pred, "question_idx": doc["question_index"], "first_turn_video_caption": doc["first_turn_video_caption"], "target": ""}}


def mix_evals_video2text_process_results_freeform(doc, result):
    pred = result[0]
    return {"submission": {"pred": pred, "question_idx": doc["question_index"], "first_turn_video_caption": doc["first_turn_video_caption"], "target": ""}}


def mix_evals_video2text_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"mix_evals_video2text_{task}-{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Submission file saved to {path}")


# Factory into different aggregate
def mix_evals_video2text_aggregate_gen(results, args):
    mix_evals_video2text_aggregate_submissions(results, args, "OpenConvs")


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
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            options_prompt = ""
            for idx, option in enumerate(doc["options"]):
                char_idx = chr(ord("A") + idx)
                option = option.strip()
                options_prompt += f"{char_idx}. {option}\n"
            options_prompt = options_prompt.rstrip("\n")
            # Regex to extract multiple choice options from the question
            multiple_choices_regex = re.compile(r"\b([A-Z])\.\s+([^\n]*)")
            matches = multiple_choices_regex.findall(options_prompt)

            # Build regex patterns and mappings for each choice
            for m in matches:
                choice_text = m[1].strip()
                fallback_regexes.append(f"{re.escape(choice_text)}")
                choice_to_alpha[choice_text] = next_alpha

                next_alpha = chr(ord(next_alpha) + 1)

            # Compile regex to match any of the extracted choices
            fallback_regex = re.compile("|".join(fallback_regexes))

            # Process each response
            filtered = []
            for resp in r:
                # Remove any punctuation and extra spaces
                cleaned_resp = re.sub(r"[^\w\s]", "", resp).strip()
                # Try to match cleaned response with the choice text
                match = fallback_regex.search(cleaned_resp)
                if match and match.group() in choice_to_alpha:
                    # Map the matched choice text back to its corresponding letter
                    filtered.append(choice_to_alpha[match.group()])
                else:
                    # If no match, return the cleaned response
                    filtered.append(cleaned_resp)

            filtered_resps.append(filtered[0])

        return filtered_resps
