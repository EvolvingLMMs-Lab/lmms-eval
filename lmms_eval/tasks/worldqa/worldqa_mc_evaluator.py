import math
import os.path as osp
import random as rd
import string
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from loguru import logger as eval_logger
from tqdm import tqdm


class WorldQA_MC_Evaluator:
    def __init__(self, sys_prompt="There are several options:", API_KEY="", API_URL="", model_version="gpt-3.5-turbo-0613"):
        self.sys_prompt = sys_prompt
        self.model_version = model_version
        self.API_KEY = API_KEY
        self.API_URL = API_URL

    def build_prompt(self, question, options, prediction):
        tmpl = (
            "You are an AI assistant who will help me to match an answer "
            "with several options of a single-choice question. "
            "You are provided with a question, several options, and an answer, "
            "and you need to find which option is most similar to the answer. "
            "If the meaning of all options are significantly different "
            "from the answer, output E. "
            "Your should output a single uppercase character in A, B, C, D "
            "(if they are valid options), and E. \n"
            "Example 1: \n"
            "Question: What is the main object in image?\nOptions: A. teddy bear "
            "B. rabbit C. cat D. dog\nAnswer: a cute teddy bear\nYour output: A\n"
            "Example 2: \n"
            "Question: What is the main object in image?\nOptions: A. teddy bear "
            "B. rabbit C. cat D. dog\nAnswer: Spider\nYour output: E\n"
            "Example 3: \n"
            "Question: {}?\nOptions: {}\nAnswer: {}\nYour output: "
        )
        return tmpl.format(question, options, prediction)

    # Prefetch Answers
    def can_infer_option(self, answer, num_choice=5):
        choices = string.ascii_uppercase[:num_choice]
        if "Failed to obtain answer via API" in answer:
            return False

        def count(splits, choices="ABCD", prefix="", suffix=""):
            cnt = 0
            for c in choices:
                if prefix + c + suffix in splits:
                    cnt += 1
            return cnt

        splits = [x.strip() for x in answer.split()]
        if count(splits, choices) == 1:
            for ch in choices:
                if "A" in splits and len(splits) > 3:
                    eval_logger.info(f"A might be a quantifier in the string: {answer}.")
                    break
                if ch in splits:
                    return ch
        tups = [("", "."), ("", ","), ("", ":"), ("", ")"), ("", ")."), ("(", ")"), ("(", ")."), (":", ""), (":", ","), (":", "."), (":", ")"), (":", ").")]
        for tup in tups:
            if count(splits, choices, prefix=tup[0], suffix=tup[1]) == 1:
                for ch in choices:
                    if tup[0] + ch + tup[1] in splits:
                        return ch
        return False

    def _post_request(self, payload):
        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_chat_response(self, prompt, temperature=0, max_tokens=256, n=1, patience=5, sleep_time=3):
        messages = [
            {"role": "user", "content": prompt},
        ]
        payload = {"model": self.model_version, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "n": n}

        while patience > 0:
            patience -= 1
            try:
                response = self._post_request(payload)
                if n == 1:
                    prediction = response["choices"][0]["message"]["content"].strip()
                    if prediction and prediction != "":
                        return prediction
                else:
                    prediction = [choice["message"]["content"].strip() for choice in response["choices"]]
                    if prediction and prediction[0] != "":
                        return prediction

            except Exception as e:
                eval_logger.info(f"Attempt {patience + 1} failed with error: {e}")
                if sleep_time > 0:
                    time.sleep(sleep_time)

        return "Failed to obtain answer via API"

    def evaluate(self, results):
        answer = results["answer"].split(".")[0]
        if self.can_infer_option(results["pred"], num_choice=4):
            choice = self.can_infer_option(results["pred"], num_choice=4)
            return int(choice.lower().strip() == answer.lower().strip())
        else:
            prompt = self.build_prompt(question=results["question"], options="\n".join(results["option"]), prediction=results["pred"])
            prediction = self.get_chat_response(prompt)
            return int(prediction.lower().strip() == answer.lower().strip())
