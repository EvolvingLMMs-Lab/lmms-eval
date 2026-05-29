### reference: https://github.com/flageval-baai/FlagEvalMM/blob/main/flagevalmm/evaluator/pre_process.py

import re


def strip_answer(answer):
    answer = re.sub("The", "", answer)
    answer = re.sub("If", "", answer)
    answer = re.sub("[INST]", "", answer)
    answer = re.sub("[/INST]", "", answer)
    answer = re.sub("<Img>", "", answer)
    answer = re.sub("</Img>", "", answer)
    answer = answer.strip()
    return answer


def remove_special_characters(text):
    pattern = r"[-`\\【】\*\$、,，。.；;:：？\?！!\s\n\u4e00-\u9fff0-9①②③④⑤⑥⑦\[\]\<>a-z=\'\"\(\)\{\}]+"
    cleaned_text = re.sub(pattern, "", text)

    return cleaned_text


def process_multiple_choice(answer):
    answer = strip_answer(answer)
    pattern = r"^([A-Z])\."
    matches = re.match(pattern, answer)
    if matches:
        return matches.group(1)
    key_words = [
        "boxed",
        "Answer:",
        "Answer is",
        "answer is",
        "option is",
        "Correct option",
        "correct option",
        "Answer",
        "answer",
        "故选",
        "选择",
        "正确选项为",
        "答案选",
        "答案为",
        "答案是",
        "因此",
        "答案",
    ]

    for key_word in key_words:
        if key_word in answer:
            answer = answer.split(key_word)[-1]
            break
    answer = remove_special_characters(answer)
    # keep the last line
    answer = answer.split("\n")[-1]
    pattern = r"[A-Z]"
    matches = re.findall(pattern, answer)
    return "".join(matches)


def remove_unit(value):
    units = [
        "cm",
        "m",
        "km",
        "mm",
        "s",
        "h",
        "kg",
        "g",
        "l",
        "ml",
        "mol",
        "厘米",
        "米",
        "千米",
        "°",
        "毫米",
        "月",
        "秒",
        "小时",
        "克",
        "千克",
        "升",
        "毫升",
        "摩尔",
    ]
    unit_pattern = r"^(\d+)(?:" + "|".join(units) + ")$"
    match = re.match(unit_pattern, value)
    if match:
        return match.group(1)
    else:
        return value


def convert_circled_numbers(text):
    circled_numbers = {
        "①": "1",
        "②": "2",
        "③": "3",
        "④": "4",
        "⑤": "5",
        "⑥": "6",
        "⑦": "7",
        "⑧": "8",
        "⑨": "9",
        "⑩": "10",
    }
    for circled, number in circled_numbers.items():
        text = text.replace(circled, number)
    return text


def normalize_string(raw_answer):
    if "$" not in raw_answer:
        wrong_answer_words = ["\\times", "不对", "不正确", "×"]
        for word in wrong_answer_words:
            raw_answer = raw_answer.replace(word, "错误")
    raw_answer = re.sub(r"\\text\s*\{(.*?)\}", r"\1", raw_answer)
    replace_dict = {
        "√": "正确",
        "：": ":",
        "$": "",
        "（": "(",
        "）": ")",
        "，": ",",
        "。": ".",
        "变小": "减小",
        "变大": "增大",
        "路程": "距离",
        "\\pi": "π",
        "＞": ">",
        "＜": "<",
        "；": ";",
    }
    # write to convert characters like ①②③④ to 1234

    for k, v in replace_dict.items():
        raw_answer = raw_answer.replace(k, v)

    # Convert circled numbers to regular numbers
    raw_answer = convert_circled_numbers(raw_answer)

    strict_replace_dict = {
        "错": "错误",
        "对": "正确",
        "(F)": "F",
        "(T)": "T",
        "(正确)": "正确",
        "(错误)": "错误",
        "“T”": "T",
        "“F”": "F",
    }
    if raw_answer in strict_replace_dict:
        raw_answer = strict_replace_dict[raw_answer]

    key_words = [
        "Answer:",
        "Answer is",
        "answer is",
        "Answer",
        "answer",
        "答案为",
        "答案是",
        "解是",
        "解为",
        "答案",
        "结果",
        "为",
        "因此",
        " = ",
    ]
    # get text after key_word
    for key_word in key_words:
        if key_word in raw_answer:
            raw_answer = raw_answer.split(key_word)[-1]
            break
    raw_answer = raw_answer.strip()
    # remove leading :
    if raw_answer.startswith(":"):
        raw_answer = raw_answer[1:]
    if len(raw_answer) > 0 and raw_answer[-1] in [".", ",", ":", ";"]:
        raw_answer = raw_answer[:-1]
    raw_answer = remove_unit(raw_answer)
    return raw_answer.strip()
