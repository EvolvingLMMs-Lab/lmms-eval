"""Common utilities shared across CC-OCR evaluators.

Directly ported from the official CC-OCR evaluation code:
https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/Benchmarks/CC-OCR/evaluation/evaluator
"""

import json
import re
from typing import Any, Dict, List, Union


def convert_to_halfwidth(text: str) -> str:
    """Convert full-width ASCII characters to half-width."""
    halfwidth_chars = str.maketrans(
        "！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠" "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀" "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～",
        "!\"#$%&'()*+,-./0123456789:;<=>?@" "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`" "abcdefghijklmnopqrstuvwxyz{|}~",
    )
    return text.translate(halfwidth_chars)


def fullwidth_to_halfwidth(text: str) -> str:
    """CJK-aware full-width to half-width conversion (matches KIE evaluator)."""
    result = ""
    for char in text:
        code_point = ord(char)
        if code_point == 0x3000:
            code_point = 0x0020
        elif 0xFF01 <= code_point <= 0xFF5E:
            code_point -= 0xFEE0
        result += chr(code_point)
    result = result.replace("、", ",")
    return result


def remove_unnecessary_spaces(text: str) -> str:
    """Normalize spacing around CJK / ASCII / punctuation boundaries."""
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[a-zA-Z0-9])", "", text)
    text = re.sub(r"(?<=[a-zA-Z0-9])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<![0-9])\s*([,.!?:;])\s*", r"\1 ", text)
    text = re.sub(r"(?<=[0-9])(?=[a-zA-Z])", " ", text)
    text = re.sub(r"(?<=[a-zA-Z])(?=[0-9])", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def post_process_to_json(qwen_info_str: str, file_name: str = None) -> Union[Dict, List, None]:
    """Extract a JSON object from a model response that may wrap it in ```json ... ```."""
    if not isinstance(qwen_info_str, str):
        return None
    try:
        if "```json" in qwen_info_str:
            if "```" not in qwen_info_str.split("```json", 1)[1]:
                qwen_info_str = qwen_info_str + "```"
            m = re.search(r"```json(.*?)```", qwen_info_str, re.DOTALL)
            if m is None:
                return None
            json_str = m.group(1).strip().replace("\n", "")
        else:
            json_str = qwen_info_str.strip().replace("\n", "")
        return json.loads(json_str)
    except Exception:
        return None


def normalize_values_of_nested_dict(d: Any, normalize_func) -> Any:
    """Recursively normalize string leaves of a nested dict/list."""
    if isinstance(d, dict):
        return {k: normalize_values_of_nested_dict(v, normalize_func) for k, v in d.items()}
    if isinstance(d, list):
        return [normalize_values_of_nested_dict(x, normalize_func) if isinstance(x, (dict, list)) else normalize_func(x) if isinstance(x, str) else x for x in d]
    if isinstance(d, str):
        return normalize_func(d)
    return d


def kie_normalize_text(text: str) -> str:
    """Full pipeline used for KIE string normalization."""
    return remove_unnecessary_spaces(fullwidth_to_halfwidth(str(text)))
