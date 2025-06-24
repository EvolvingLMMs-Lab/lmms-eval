import os
import random
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import datasets
import openai
from openai import OpenAI

QUERY_TEMPLATE = "{Question}\n\nA) {choice1}\nB) {choice2}\nC) {choice3}\nD) {choice4}"
QUERY_TEMPLATE_API = "{Question}\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"

if os.getenv("PROMPTLONG") is not None:
    QUERY_TEMPLATE += "\n\nAnswer after a long amount of thinking. If you feel like you are finished early, spend the extra time trying to double-check your work until you are absolutely sure that you have the correct answer."
elif os.getenv("PROMPTSHORT") is not None:
    QUERY_TEMPLATE += "\n\nAnswer after a short amount of thinking. Do not spend excessive time double-checking your work."
elif os.getenv("PROMPTTOKEN") is not None:
    QUERY_TEMPLATE += f"\n\nThink for up to " + os.getenv("PROMPTTOKEN") + " tokens."
elif os.getenv("PROMPTSTEP") is not None:
    QUERY_TEMPLATE += f"\n\nThink for up to " + os.getenv("PROMPTSTEP") + " steps."

# print("QUERY_TEMPLATE: ", QUERY_TEMPLATE)

# Adapted from https://github.com/openai/simple-evals/blob/c0dba4c7bfbc17f786aec7bd7c3585a36ad81f23/common.py#L23
# (?i): Enables case-insensitive matching. This means "Answer", "answer", "ANSWER", etc., will all be matched.
# Answer: Matches the literal string "Answer" (case-insensitive due to (?i)).
# \s*: Matches zero or more whitespace characters (spaces, tabs, etc.) after "Answer". This accounts for cases where there might or might not be space between "Answer" and the colon (:).
# :: Matches the literal colon character :.
# \s*: Matches zero or more whitespace characters after the colon. This handles cases where there might be spaces between the colon and the actual answer.
# (.*): The .* matches zero or more of any character (including none), except for newlines unless re.DOTALL is used (which allows newlines to be matched too).
# Note: This does not match e.g. "**Final Answer:** A" as it only matches "Answer: A" or "Answer: A) 7" etc.
ANSWER_PATTERN = r"(?i)Answer\s*:\s*(.*)"

EXTRACTION_TEMPLATE = r"""
Look at the following question and an attempt by a student and extract which choice among A, B, C, D the student picked. If the student did not pick any choice, respond with "-1".

Examples:

    Question: ...
    Attempt: Answer: **A**

A

    Question: A) Dinosaur B) Elephant C) Cat D) Dog
    Attempt: ...The answer is therefore Elephant...

B

    Question: ...
    Attempt: Answer: None of the above

-1

    Question: ...
    Attempt: ...Answer: D), because...

D

    Question: ...
(A) 7 
(B) 8 
(C) 4 
(D) 10
    Attempt: 4

C

    Question: ...
    Attempt: ...\\boxed{C}...

C

---

YOUR TASK


Respond only with the capitalized alphabetic letter (without quotes) or -1. Do not include a rationale.

    Question: %(expression1)s
    Attempt: %(expression2)s
""".strip()


def extract_answer(sampler, question: str, attempt: str):
    prompt = EXTRACTION_TEMPLATE % {"expression1": question, "expression2": attempt}
    response = sampler([dict(content=prompt, role="user")])
    return response


class ChatCompletionSampler:
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    metrics = {"exact_match": None, "extracted_answers": []}
    # Multiple results -> we are measuring cov/maj etc
    if isinstance(results[0], list):
        results = results[0]
        n_res = len(results)  # e.g. 64
        n_res_list = [2**i for i in range(1, int(n_res.bit_length()))]  # e.g. [2, 4, 8, 16, 32, 64]
        metrics = {
            **metrics,
            "exact_matches": [],
            **{f"cov@{n}": -1 for n in n_res_list},
            **{f"maj@{n}": -1 for n in n_res_list},
        }

    if os.getenv("PROCESSOR", "") == "gpt-4o-mini":
        sampler = ChatCompletionSampler(model="gpt-4o-mini")
        question = QUERY_TEMPLATE_API.format(Question=doc["Question"], choice1=doc["choice1"], choice2=doc["choice2"], choice3=doc["choice3"], choice4=doc["choice4"])
    else:
        print(f"Unknown processor: {os.getenv('PROCESSOR')}; set 'PROCESSOR=gpt-4o-mini' and 'OPENAI_API_KEY=YOUR_KEY' for best results.")
        sampler = None

    split_tokens = ["<|im_start|>answer\n", "<|im_start|>"]
    for i, a in enumerate(results, start=1):
        if split_tokens[0] in a:
            a = a.split(split_tokens[0])[-1]
        elif split_tokens[1] in a:
            a = a.split(split_tokens[1])[-1]
            if "\n" in a:
                a = "\n".join(a.split("\n")[1:])

        if (box := last_boxed_only_string(a)) is not None:
            a = remove_boxed(box)
        # re.DOTALL is key such that newlines are included e.g. if it does `Answer: Here is the solution:\n\n10`
        elif (matches := re.findall(ANSWER_PATTERN, a, re.DOTALL)) != []:
            a = matches[-1]  # Get the last match

        if a in ["a", "b", "c", "d"]:
            a = a.upper()

        if a not in ["A", "B", "C", "D"]:
            if sampler is not None:
                a = extract_answer(sampler, question, a)
            else:
                pass  # TODO: Maybe add back legacy processing

        if a not in ["A", "B", "C", "D"]:
            print(f"Warning: Default to A as given {results[i-1]} extracted {a}")
            a = "A"

        metrics["extracted_answers"].append(a)
        a = int(a == doc["answer"])
        if not (a):  # Optional logging
            print("Marked incorrect\na " + metrics["extracted_answers"][-1] + "\ndoc['answer'] " + doc["answer"])
        if i == 1:
            metrics["exact_match"] = a
            if "exact_matches" in metrics:
                metrics["exact_matches"].append(a)
        elif i > 1:
            metrics["exact_matches"].append(a)
            if i in n_res_list:
                metrics[f"cov@{i}"] = int(1 in metrics["exact_matches"])
                metrics[f"maj@{i}"] = int(doc["answer"] == Counter(metrics["extracted_answers"]).most_common(1)[0][0])

    return metrics


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            doc["Incorrect Answer 1"],
            doc["Incorrect Answer 2"],
            doc["Incorrect Answer 3"],
            doc["Correct Answer"],
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(doc["Correct Answer"])

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"{chr(65 + correct_answer_index)}",
        }
        return out_doc

    return dataset.map(_process_doc)


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
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

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def doc_to_text_gpqa(doc: dict) -> str:
    return QUERY_TEMPLATE.format(Question=doc["Question"], choice1=doc["choice1"], choice2=doc["choice2"], choice3=doc["choice3"], choice4=doc["choice4"])
