import re
from math import isclose
from typing import Dict, List, Tuple


def parse_math_answer(raw_string):
    return remove_boxed(last_boxed_only_string(raw_string))


def remove_boxed(s):
    left = "oxed{"  # change
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        answer = s[len(left) : -1]
        if "=" in answer:
            answer = answer.split("=")[-1].lstrip(" ")
        return answer
    except:
        return None


def last_boxed_only_string(string):
    idx = string.rfind("oxed")  # change
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def extract_boxed_answers(text):
    # Find all boxed contents
    matches = re.findall(r"boxed{([^}]*)}", text)
    for m in matches:
        # Strip spaces
        candidate = m.strip()
        # Keep only the numeric ones (int or decimal, with optional sign)
        if re.fullmatch(r"[-+]?\d*\.?\d+", candidate):
            return candidate
    return None


def cal_not(inputs):
    # print("Input cal not: ", inputs, type(inputs), len(inputs))
    try:
        # print("Hi")
        x, ab = inputs
        # print("Hi also", x, ab)
        match_number = re.compile("10\^[{]?\ *-?[0-9]+\ *[}]?")
        ab = re.findall(match_number, ab)[0]
        ab = ab[ab.find("^") + 1 :]
        if "{" in ab:
            ab = ab[ab.find("{") + 1 :]
        if "}" in ab:
            ab = ab[: ab.find("}")]
        x = x.strip()
        out = float(x) * 10 ** float(ab)
        # print(float(x)*10**float(ab))
        return str(out)
    except:
        print("error")
    return inputs


def remove_not(x):
    match_number = re.compile("[\$]?\ *10\^[{]?\ *-?[0-9]+\ *[}]?\ *[\$]?")
    result = re.findall(match_number, x)
    if len(result) != 0:
        return re.split(match_number, x)[-1]
    return None


def parse_not(inputs):
    try:
        if not inputs:
            return "", ""
        if "\\times" in inputs:
            x, ab = inputs.split("\\times")
        elif "\times" in inputs:
            x, ab = inputs.split("\times")
        elif "*" in inputs:
            x, ab = inputs.split("*")
        else:
            return inputs
        return x, ab
    except:
        return "", ""


def equiv_with_unit(model_output, answer, unit):
    """Fixed to handle None inputs gracefully"""

    model_output = clean_number_string(model_output)
    try:
        ans = float(clean_number_string(answer))
        first = isclose(float(model_output.strip()), ans, rel_tol=0.05)
    except:
        first = False

    try:
        model = model_output.strip().split()[0]
        second = isclose(float(model.strip()), ans, rel_tol=0.05)
    except:
        second = False

    return first or second


def clean_number_string(s):
    """Fixed to handle None input"""
    if s is None:
        return ""
    return s.replace(",", "").replace("−", "-").strip()


def scibench_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict) -> str:
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    question = doc["problem_text"]
    unit_prob = doc["unit"].strip()
    if remove_not(doc["unit"].strip()):
        unit_prob = remove_not(doc["unit"]).strip()
    if unit_prob:
        question = doc["problem_text"].strip() + " The unit of the answer is " + unit_prob + "."
    else:
        question = doc["problem_text"].strip()
    return f"{pre_prompt}\nQuestion: {question}{post_prompt}"


def scibench_process_results(doc: Dict, result: List[str]) -> Dict[str, float]:
    """Fixed version with proper null handling"""
    pred = result[0]

    pred = parse_math_answer(pred)
    ans = doc["answer_number"].strip()
    unit = doc["unit"].strip()
    unit_prob = doc["unit"].strip()

    if remove_not(doc["unit"].strip()):
        unit_prob = remove_not(doc["unit"]).strip()
    if unit_prob != unit:
        pred = cal_not(parse_not(pred))
        ans = cal_not((ans, unit))
        if len(ans) > 1:
            ans = ans[0]
    try:
        res_equiv = equiv_with_unit(pred, ans, unit)
    except Exception as e:
        print(f"Error in equiv_with_unit: {e}")
        res_equiv = False

    score = 1 if res_equiv else 0
    return {"accuracy": score}
