import ast
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import yaml

from lmms_eval.loggers.evaluation_tracker import GeneralConfigTracker
from lmms_eval.utils import sanitize_model_name

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def construct_prompt(doc, post_prompt="与えられた選択肢の中から最も適切な回答のアルファベットを直接記入してください。"):
    question = doc["question"]
    # Weirdly, data["options"] is a string in MMMU Huggingface dataset
    parsed_options = parse_options(ast.literal_eval(doc["options"]))
    # parsed_options already prepends a newline so no need to add space here
    question = f"{question}\n{parsed_options}\n\n{post_prompt}"
    return question


def jmmmu_pro_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    if "question" in doc and "options" in doc:  # original operation
        question = construct_prompt(doc, post_prompt)
        if config["metadata"]["interleaved_format"]:
            question = replace_images_tokens(question)
    return question


def jmmmu_pro_doc_to_visual(doc):
    if "question" in doc and "options" in doc:  # original operation
        prompt = construct_prompt(doc)
        image_tokens = re.findall(r"<image \d+>", prompt)
        # Remove <> and  swap space as _
        image_tokens = sorted(list(set([image_token.strip("<>").replace(" ", "_") for image_token in image_tokens])))
        visual = [doc[image_token].convert("RGB") for image_token in image_tokens]
    else:  # vision-only operation
        visual = [doc["image"].convert("RGB")]
    return visual


def jmmmu_pro_process_results(doc, results):
    pred = results[0]
    index2ans, all_choices = get_multi_choice_info(ast.literal_eval(doc["options"]))
    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    id = doc["id"]
    # Calculate correct flag by comparing answer and parsed_pred
    correct = eval_multi_choice(doc["answer"], parsed_pred)
    jmmmu_pro_acc = {"id": id, "subdomain": extract_subset_name(doc["id"]), "question_type": doc["question_type"], "answer": doc["answer"], "parsed_pred": parsed_pred}
    return {
        "jmmmu_pro_acc": jmmmu_pro_acc,
        "submission": {
            id: pred,
        },
        "correct": correct,
    }


def extract_subset_name(input_string):
    # Define a regex pattern to match "validation_" at the beginning and "_<number>" at the end
    split = input_string.split("_")[0]
    pattern = re.compile(rf"^{split}_(.+?)_\d+$")
    match = pattern.search(input_string)
    if match:
        return match.group(1)
    else:
        raise ValueError(f'No match found in "{input_string}"')


DOMAIN_CAT2SUB_CAT = {
    "Art and Psychology": ["Design", "Music", "Psychology"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": [
        "Biology",
        "Chemistry",
        "Math",
        "Physics",
    ],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
    "Culture-specific": [
        "Japanese_Art",
        "Japanese_Heritage",
        "Japanese_History",
        "World_History",
    ],
}

SUBDOMAINS = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Japanese_Art",
    "Japanese_Heritage",
    "Japanese_History",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "World_History",
]

CULTURE_SPECIFIC_SUBDOMAINS = ["Japanese_Art", "Japanese_Heritage", "Japanese_History", "World_History"]

CULTURE_AGNOSTIC_SUBDOMAINS = [s for s in SUBDOMAINS if s not in CULTURE_SPECIFIC_SUBDOMAINS]


def jmmmu_pro_aggregate_results(results, args):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["subdomain"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_jmmmu_pro(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict

    # Calculate domain-level results
    printable_results = {}
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
            else:
                pass
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum([cat_results["num_example"] for cat_results in in_domain_cat_results.values()])
        printable_results["Overall-" + domain] = {
            "num": int(in_domain_data_num),
            "acc": round(in_domain_ins_acc, 5),
        }
        # add sub category
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {
                "num": int(cat_results["num_example"]),
                "acc": round(cat_results["acc"], 5),
            }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 5),
    }

    # Calculate subdomain-specific results
    subdomain_results = {}
    for subdomain in SUBDOMAINS:
        if subdomain in evaluation_result:
            subdomain_results[subdomain] = {
                "num": int(evaluation_result[subdomain]["num_example"]),
                "acc": round(evaluation_result[subdomain]["acc"], 5),
            }
        else:
            subdomain_results[subdomain] = {
                "num": 0,
                "acc": 0.0,
            }

    # Calculate culture-specific and culture-agnostic results
    culture_specific_results = {}
    culture_agnostic_results = {}
    for subdomain in CULTURE_SPECIFIC_SUBDOMAINS:
        if subdomain in evaluation_result:
            culture_specific_results[subdomain] = evaluation_result[subdomain]
    for subdomain in CULTURE_AGNOSTIC_SUBDOMAINS:
        if subdomain in evaluation_result:
            culture_agnostic_results[subdomain] = evaluation_result[subdomain]

    culture_specific_acc = calculate_ins_level_acc(culture_specific_results)
    culture_agnostic_acc = calculate_ins_level_acc(culture_agnostic_results)

    culture_specific_num = sum([r["num_example"] for r in culture_specific_results.values()])
    culture_agnostic_num = sum([r["num_example"] for r in culture_agnostic_results.values()])

    # Add aggregated results to subdomain_results
    subdomain_results["culture-specific"] = {
        "num": int(culture_specific_num),
        "acc": round(culture_specific_acc, 5),
    }
    subdomain_results["culture-agnostic"] = {
        "num": int(culture_agnostic_num),
        "acc": round(culture_agnostic_acc, 5),
    }

    # Save subdomain results to JSON file
    # Only save if output_path is provided
    if args.output_path is not None:
        # Extract model name from model_args (e.g., pretrained=model_name)
        model_args = getattr(args, "model_args", "")
        model_name = GeneralConfigTracker._get_model_name(model_args)
        if model_name:
            model_name_sanitized = sanitize_model_name(model_name)
        else:
            model_name_sanitized = ""
        if model_name_sanitized:
            save_dir = os.path.join(args.output_path, model_name_sanitized)
        else:
            save_dir = args.output_path
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "jmmmu_pro_subdomain_results.json")

        with open(save_path, "w") as f:
            json.dump(subdomain_results, f, indent=2, ensure_ascii=False)

    return printable_results["Overall"]["acc"]


def jmmmu_pro_subdomain_results(results, args):
    """Return subdomain-specific results for all subdomains."""
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["subdomain"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_jmmmu_pro(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict

    subdomain_results = {}
    # Include all subdomains, even if they have no samples
    for subdomain in SUBDOMAINS:
        if subdomain in evaluation_result:
            subdomain_results[subdomain] = {
                "num": int(evaluation_result[subdomain]["num_example"]),
                "acc": round(evaluation_result[subdomain]["acc"], 5),
            }
        else:
            # Include subdomain with zero samples if not present
            subdomain_results[subdomain] = {
                "num": 0,
                "acc": 0.0,
            }

    # Calculate culture-specific and culture-agnostic results
    culture_specific_results = {}
    culture_agnostic_results = {}
    for subdomain in CULTURE_SPECIFIC_SUBDOMAINS:
        if subdomain in evaluation_result:
            culture_specific_results[subdomain] = evaluation_result[subdomain]
    for subdomain in CULTURE_AGNOSTIC_SUBDOMAINS:
        if subdomain in evaluation_result:
            culture_agnostic_results[subdomain] = evaluation_result[subdomain]

    culture_specific_acc = calculate_ins_level_acc(culture_specific_results)
    culture_agnostic_acc = calculate_ins_level_acc(culture_agnostic_results)

    culture_specific_num = sum([r["num_example"] for r in culture_specific_results.values()])
    culture_agnostic_num = sum([r["num_example"] for r in culture_agnostic_results.values()])

    # Add aggregated results to subdomain_results
    subdomain_results["culture-specific"] = {
        "num": int(culture_specific_num),
        "acc": round(culture_specific_acc, 5),
    }
    subdomain_results["culture-agnostic"] = {
        "num": int(culture_agnostic_num),
        "acc": round(culture_agnostic_acc, 5),
    }

    # Save to the same directory as results.json: output_path/{model_name_sanitized}/
    # Only save if output_path is provided
    if args.output_path is not None:
        model_name_sanitized = sanitize_model_name(getattr(args, "model", ""))
        if model_name_sanitized:
            save_dir = os.path.join(args.output_path, model_name_sanitized)
        else:
            save_dir = args.output_path
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "jmmmu_pro_subdomain_results.json")

        with open(save_path, "w") as f:
            json.dump(subdomain_results, f, indent=2, ensure_ascii=False)

    return None


##################
# Helper functions written by official MMMU repo.
##################


def calculate_ins_level_acc(results):
    """Calculate the instruction level accuracy for given Subject results
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
    """
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L175
    """
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def evaluate_jmmmu_pro(samples):
    """
    Batch evaluation for multiple choice questions.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L219
    """
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        correct = eval_multi_choice(gold_i, pred_i)
        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response using answer_parser logic.
    Return the predicted index e.g., A, B, C, D, or "X" if not parsed.
    """
    from typing import Optional

    # Precompile commonly used regex patterns
    FULLWIDTH_MAP = {chr(ord("Ａ") + i): chr(ord("A") + i) for i in range(26)}
    FULLWIDTH_TRANS = str.maketrans(FULLWIDTH_MAP)

    OPTION_LINE_RE = re.compile(
        r"""^\s*
            (?:[-*・>\u2022]\s*)?      # bullets
            [A-ZＡ-Ｚ]                 # option label
            [\.\)\u3001\u3002：:]      # separators: . ) 、 。 : ：
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    EXPLICIT_RE = re.compile(
        r"""(?ix)
        (?:answer|final|correct|solution|ans
          |正解(?:は)?
          |答え(?:は)?
          |解答(?:は)?
        )                                   # cue words
        \s*[:：]?\s*
        [【\[\(\u3010\u3011\*_-]*            # optional brackets/emphasis start
        ([A-Z])                             # capture the letter
        [】\]\)\*_-]*                       # optional brackets/emphasis end
        \b
        """
    )

    MARKDOWN_LETTER_RE = re.compile(
        r"""
        [【\[\(\*]*([A-Z])[】\]\)\*]*   # plain or emphasized letter
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def _normalize(text: str) -> str:
        # Normalize full-width letters and strip obvious bold markers.
        text = text.translate(FULLWIDTH_TRANS)
        return text

    def _is_option_line(line: str) -> bool:
        return bool(OPTION_LINE_RE.match(line))

    def _explicit_in_line(line: str) -> Optional[str]:
        m = EXPLICIT_RE.search(line)
        if m:
            return m.group(1).upper()
        return None

    def _last_standalone_letter(lines: list[str]) -> Optional[str]:
        """Fallback: pick the last standalone letter (A–Z) not on an option line."""
        candidates: list[str] = []
        for line in lines:
            if _is_option_line(line):
                continue
            for m in MARKDOWN_LETTER_RE.finditer(line):
                letter = m.group(1).upper()
                candidates.append(letter)
        return candidates[-1] if candidates else None

    def parse_answer(text: str) -> Optional[str]:
        """
        Parse a model free-form response and return an answer letter (A–Z) or None.
        Priority:
          1) Explicit marker ("Answer/正解/Final/...") on the first non-empty line.
          2) Explicit marker anywhere in the text.
          3) Last standalone letter (A–Z) that is not on an option line.
        """
        if not text:
            return None

        norm = _normalize(text)
        lines = [ln.strip() for ln in norm.splitlines() if ln.strip()]
        if not lines:
            return None

        # 1) First non-empty line explicit marker
        first_line = lines[0]
        hit = _explicit_in_line(first_line)
        if hit:
            return hit

        # 2) Explicit marker anywhere
        hit_any = _explicit_in_line(norm)
        if hit_any:
            return hit_any

        # 3) Last standalone letter outside option lines
        return _last_standalone_letter(lines)

    # Use answer_parser logic
    parsed_letter = parse_answer(response)

    # Validate that the parsed letter is in all_choices
    if parsed_letter and parsed_letter in all_choices:
        return parsed_letter

    # If not found or not valid, return "X"
    return "X"


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices
