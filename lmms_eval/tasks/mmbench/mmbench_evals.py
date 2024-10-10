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


class MMBench_Evaluator:
    def __init__(self, sys_prompt="There are several options:", API_KEY="", API_URL="", model_version="gpt-3.5-turbo-0613"):
        self.sys_prompt = sys_prompt
        self.model_version = model_version
        self.API_KEY = API_KEY
        self.API_URL = API_URL

    def create_options_prompt(self, row_data, option_candidate):
        available_keys = set(row_data.keys()) & set(option_candidate)
        options = {cand: row_data[cand] for cand in available_keys if row_data[cand]}
        sorted_options = dict(sorted(options.items()))
        options_prompt = f"{self.sys_prompt}\n"
        for key, item in sorted_options.items():
            if pd.notna(item) and item != "nan":
                options_prompt += f"{key}. {item}\n"
        return options_prompt.rstrip("\n"), sorted_options

    # Prompt Building
    def build_option_str(self, option_list):
        chars = string.ascii_uppercase
        s = "There are several options: \n"
        for c, opt in zip(chars, option_list):
            if not pd.isna(opt):
                s += f"{c}. {opt}\n"
            else:
                return s
        return s

    def extract_options(self, item):
        options = []
        for c in "ABCD":
            if c in item and not pd.isna(item[c]):
                options.append(item[c])
            else:
                return options
        return options

    def build_choices(self, item):
        ret = {}
        for ch in "ABCD":
            if not pd.isna(item[ch]):
                ret[ch] = item[ch]
        return ret

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

    def can_infer_text(self, answer, choices):
        answer = answer.lower()
        assert isinstance(choices, dict)
        for k in choices:
            assert k in "ABCD"
            choices[k] = str(choices[k]).lower()
        cands = []
        for k in choices:
            if choices[k] in answer:
                cands.append(k)
        if len(cands) == 1:
            return cands[0]
        return False

    def can_infer(self, answer, choices):
        copt = self.can_infer_option(answer)
        return copt if copt else self.can_infer_text(answer, choices)

    def prefetch_answer(self, item):
        choices = self.build_choices(item)
        return self.can_infer(item["prediction"], choices)

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

    def extract_answer_from_item(self, item):
        options = self.extract_options(item)
        option_str = self.build_option_str(options)

        prompt = self.build_prompt(item["question"], option_str, item["prediction"])
        retry = 3
        choices = self.build_choices(item)

        ret = self.can_infer(item["prediction"], choices)
        if ret:
            return ret, item["prediction"]

        while retry:
            ans = self.get_chat_response(prompt)
            if "Failed to obtain answer via API" in ans:
                msg = "GPT API failed to answer. "
                eval_logger.info(msg)
                retry -= 1
            else:
                ret = self.can_infer(ans, choices)
                if ret:
                    return ret, ans
                else:
                    eval_logger.info(f'GPT output includes 0 / >1 letter in "ABCD": {ans}')
                    retry -= 1

            if retry == 0:
                num_options = sum([ch in item for ch in "ABCD"])
                if num_options >= 2:
                    chars = string.ascii_uppercase[:num_options]
                    chars = chars + "E"
                    num_options += 1
                    tmp = rd.randint(0, num_options - 1)
                    return chars[tmp], "Failed to predict, thus randomly generate one. "

    # Extract answer from multiple rolling records
    def eval_sub_data(self, sub_data, answer_map):
        lt = len(sub_data)
        GT, PRED = [], []
        for i in range(lt):
            item = sub_data.iloc[i]
            idx = item["index"]
            GT.append(answer_map[idx])
            PRED.append(self.prefetch_answer(item))
            if PRED[-1] and (GT[-1] != PRED[-1]):
                return 0

        for i in range(lt):
            if PRED[i]:
                continue
            else:
                ret, _ = self.extract_answer_from_item(sub_data.iloc[i])
                PRED[i] = ret
                if PRED[i] != GT[i]:
                    return 0
        return 1

    def calculate_hit_rates(self, data):
        overall_hit_rate = data["hit"].mean()

        category_hit_rate = {}
        if "category" in data.columns:
            # Category-based hit rate
            category_hit_rate = data.groupby("category")["hit"].mean().to_dict()

        # l2-category based hit rate
        l2_category_hit_rate = {}
        if "l2-category" in data.columns:
            l2_category_hit_rate = data.groupby("l2-category")["hit"].mean().to_dict()

        return overall_hit_rate, category_hit_rate, l2_category_hit_rate

    # Evaluate Results
    def eval_result(self, results, eval_method):
        rd.seed(2680)
        assert eval_method == "openai"
        # Set a large retry number to avoid failure
        # model = OpenAI('gpt-3.5-turbo-0613', retry=99)

        # double_log(f'Evaluating {eval_file}', fout)

        # result_file = eval_file.replace('.xlsx', f'_{eval_method}_result.pkl')
        result = {}
        # if osp.exists(result_file):
        #     result = load(result_file)

        # data = load(eval_file)
        data = pd.DataFrame(results)
        data = data.sort_values(by="index")
        data["prediction"] = [str(x) for x in data["prediction"]]
        for k in data.keys():
            data[k.lower() if k not in "ABCD" else k] = data.pop(k)

        # meta = load(meta_file)

        data_main = data[data["index"] < int(1e6)]

        data_main["hit"] = 0
        cate_map = {i: c for i, c in zip(data["index"], data["category"])}
        answer_map = {i: c for i, c in zip(data["index"], data["answer"])}
        if "l2-category" in data.columns:
            l2_cate_map = {i: c for i, c in zip(data["index"], data["l2-category"])}

        lt = len(data_main)
        hit, tot = 0, 0

        for i in range(lt):
            # Dealing with the normal part
            item_main = data_main.iloc[i]
            idx = item_main["index"]

            if idx in result:
                correct = result[idx]
                assert correct in [0, 1]
                hit += correct
                tot += 1
                continue

            sub_data = data[data["index"] % int(1e6) == idx]
            ret = self.eval_sub_data(sub_data, answer_map)
            result[idx] = ret
            hit += ret
            tot += 1

            data_main.loc[data_main["index"] == idx, "hit"] = ret
            # if (i + 1) % 100 == 0:
            #     eval_logger.info(f"Evaluating: {i + 1}/{lt}, Acc: {hit / tot * 100: .2f}%. ")

        indices = data_main["index"]
        data_main = data_main.set_index("index")
        data_main["category"] = [cate_map[i] if not math.isnan(i) else "uncategorized" for i in indices]
        if "l2-category" in data_main.columns:
            data_main["l2-category"] = [l2_cate_map[i] if not math.isnan(i) else "uncategorized" for i in indices]

        overall_hit_rate, category_hit_rate, l2_category_hit_rate = self.calculate_hit_rates(data_main)

        if "category" in data_main.columns:
            print(f"Category Acc. (dev):")
            for category_key in category_hit_rate:
                if category_key == "split":
                    continue

                category_percentage = category_hit_rate[category_key] * 100
                print(f"\t{category_key}: {category_percentage:.3f}")

        if "l2-category" in data_main.columns:
            print(f"L2-category Acc. (dev):")
            for l2_category_key in l2_category_hit_rate:
                if l2_category_key == "split":
                    continue

                l2_category_percentage = l2_category_hit_rate[l2_category_key] * 100
                print(f"\t{l2_category_key}: {l2_category_percentage:.3f}")

        return overall_hit_rate, category_hit_rate, l2_category_hit_rate
