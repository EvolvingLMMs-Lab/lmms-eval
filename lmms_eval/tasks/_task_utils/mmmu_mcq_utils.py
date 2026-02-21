import random
import re

import numpy as np


def get_multi_choice_info(options, start_chr="A"):
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        choice = chr(ord(start_chr) + i)
        index2ans[choice] = option
        all_choices.append(choice)
    return index2ans, all_choices


def parse_mmmu_multi_choice_response(response, all_choices, index2ans):
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []

    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index


def parse_jmmmu_multi_choice_response(response, all_choices, index2ans):
    for char in [",", ".", "!", "?", ";", ":", "'", "、", "。", "！", "？", "；", "："]:
        response = response.strip(char)
    response = " " + response + " "

    japanese_char_pattern = r"[\u3040-\u30FF\u4E00-\u9FFF]"
    index_ans = True
    ans_with_brack = False
    candidates = []

    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            pattern = rf"{japanese_char_pattern}{choice}{japanese_char_pattern}"
            if re.search(pattern, response):
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index


def parse_videommmu_multi_choice_response(response, all_choices, index2ans):
    if response == "API Error" or response == "":
        return "API Error"

    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    ans_with_period = False
    ans_with_colon = False
    candidates = []

    for choice in all_choices:
        if f"{choice}." in response:
            candidates.append(choice)
            ans_with_period = True

    for choice in all_choices:
        if f"{choice}:" in response:
            candidates.append(choice)
            ans_with_colon = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"({choice})" in response:
                candidates.append(choice)
                ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = "No Answer Found."
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_period:
                for can in candidates:
                    start_indexes.append(response.rfind(f"{can}."))
            elif ans_with_colon:
                for can in candidates:
                    start_indexes.append(response.rfind(f"{can}:"))
            elif ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index


def parse_jmmmu_pro_multi_choice_response(response, all_choices):
    fullwidth_map = {chr(ord("Ａ") + i): chr(ord("A") + i) for i in range(26)}
    fullwidth_trans = str.maketrans(fullwidth_map)

    option_line_re = re.compile(
        r"""^\s*
            (?:[-*・>\u2022]\s*)?
            [A-ZＡ-Ｚ]
            [\.\)\u3001\u3002：:]
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    explicit_re = re.compile(
        r"""(?ix)
        (?:answer|final|correct|solution|ans
          |正解(?:は)?
          |答え(?:は)?
          |解答(?:は)?
        )
        \s*[:：]?\s*
        [【\[\(\u3010\u3011\*_-]*
        ([A-Z])
        [】\]\)\*_-]*
        \b
        """
    )

    markdown_letter_re = re.compile(
        r"""
        [【\[\(\*]*([A-Z])[】\]\)\*]*
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def _normalize(text):
        return text.translate(fullwidth_trans)

    def _is_option_line(line):
        return bool(option_line_re.match(line))

    def _explicit_in_line(line):
        match = explicit_re.search(line)
        if match:
            return match.group(1).upper()
        return None

    def _last_standalone_letter(lines):
        candidates = []
        for line in lines:
            if _is_option_line(line):
                continue
            for match in markdown_letter_re.finditer(line):
                candidates.append(match.group(1).upper())
        return candidates[-1] if candidates else None

    def parse_answer(text):
        if not text:
            return None

        normalized = _normalize(text)
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        if not lines:
            return None

        first_line_hit = _explicit_in_line(lines[0])
        if first_line_hit:
            return first_line_hit

        any_line_hit = _explicit_in_line(normalized)
        if any_line_hit:
            return any_line_hit

        return _last_standalone_letter(lines)

    parsed_letter = parse_answer(response)
    if parsed_letter and parsed_letter in all_choices:
        return parsed_letter

    return "X"
