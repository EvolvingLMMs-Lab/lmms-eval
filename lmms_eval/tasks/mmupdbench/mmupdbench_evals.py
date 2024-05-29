import time
import random as rd
import string
from collections import defaultdict
import requests
import math
import numpy as np
import pandas as pd
import pickle
import logging
import json

eval_logger = logging.getLogger("lmms-eval")


def dump(data, f):

    def dump_pkl(data, pth):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth):
        json.dump(data, open(pth, 'w'))

    def dump_jsonl(data, f):
        lines = [json.dumps(x, ensure_ascii=False) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f):
        data.to_excel(f, index=False)

    def dump_csv(data, f):
        data.to_csv(f, index=False)

    def dump_tsv(data, f):
        data.to_csv(f, sep='\t', index=False)

    handlers = dict(pkl=dump_pkl,
                    json=dump_json,
                    jsonl=dump_jsonl,
                    xlsx=dump_xlsx,
                    csv=dump_csv,
                    tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f)


def load(f):
    """
        Loads data from various file formats.

        Parameters:
        - file_path: Path to the file to be loaded.

        Returns:
        - Loaded data.
    """
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        df = pd.read_excel(f)
        df = df.dropna(subset=['prediction'])

        return df

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl,
                    json=load_json,
                    jsonl=load_jsonl,
                    xlsx=load_xlsx,
                    csv=load_csv,
                    tsv=load_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](f)


class MMUPDBench_Evaluator:
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

    def is_none(self, value):
        if value is None:
            return True
        if pd.isna(value):
            return True
        if type(value) is float and math.isnan(value):
            return True
        if type(value) is str and value.lower() == 'nan':
            return True
        if type(value) is str and value.lower() == 'none':
            return True
        return False

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
        for c in 'ABCDE':
            try:
                if self.is_none(item[c]) is False:
                    options.append(item[c])
            except KeyError:
                continue
        return options

    def build_choices(self, item):
        ret = {}
        for ch in 'ABCDE':
            try:
                if self.is_none(item[ch]) is False:
                    ret[ch] = item[ch]
            except KeyError:
                continue
        return ret

    def build_option_str_w_gt(self, option_list, gt_text, eval_type, question_type, upd_type):
        chars = string.ascii_uppercase
        s = 'There are several options: \n'
        valid_option = []
        answer_option = []

        for c, opt in zip(chars, option_list):
            if self.is_none(opt) is False:
                s += f'{c}. {opt}\n'
                valid_option.append(c)
        if upd_type == "aad":
            if eval_type == "aad":
                gt_option = ""
                for option in ['C', 'D', 'E', 'F']:
                    if option not in valid_option:
                        gt_option = option
                        break
                none_option_mapping = {'C': 'D', 'D': 'E', 'E': 'F', 'F': 'G'}
                s += f'{gt_option}. {gt_text}\n'
                none_option = none_option_mapping[gt_option]
                s += f'{none_option}. The correct answer is No answer, None of the above, \
                    all provided options are incorrect, or I cannot answer.\n'
                valid_option.append(gt_option)
                valid_option.append(none_option)
                answer_option = [gt_option, none_option]
                if question_type == 'inst':
                    s += f'F. None of the above\n'
                    valid_option.append("F")
                    answer_option = [gt_option, none_option, "F"]

            if eval_type == "standard":
                none_option = ""
                for option in ['C', 'D', 'E', 'F']:
                    if option not in valid_option:
                        none_option = option
                        break
                s += f'{none_option}. The correct answer is No answer, None of the above, \
                    all provided options are incorrect, or I cannot answer.\n'
                valid_option.append(none_option)
                if question_type == 'inst':
                    s += 'F. None of the above\n'
                    valid_option.append("F")
        elif upd_type == "iasd":
            if eval_type == "iasd":
                gt_option = ""
                for option in ['C', 'D', 'E', 'F']:
                    if option not in valid_option:
                        gt_option = option
                        break

                s += f'{gt_option}. {gt_text}\n'
                valid_option.append(gt_option)

                if question_type == 'inst':
                    if gt_option == 'E':
                        s += f'F. None of the above\n'
                        valid_option.append('F')
                        s += 'G. The correct answer is No answer, None of the above, all provided options are irrelevant or incorrect, or I cannot answer.\n'
                        valid_option.append('G')
                        answer_option = [gt_option, 'F', 'G']
                    else:
                        none_option_mapping = {'C': 'D', 'D': 'E'}
                        none_option = none_option_mapping[gt_option]
                        s += f'{none_option}. The correct answer is No answer, None of the above, all provided options are irrelevant or incorrect, or I cannot answer.\n'
                        valid_option.append(none_option)
                        s += f'F. None of the above\n'
                        valid_option.append('F')
                        answer_option = [gt_option, none_option, 'F']
                else:
                    none_option_mapping = {'C': 'D', 'D': 'E', 'E': 'F', 'F': 'G'}
                    none_option = none_option_mapping[gt_option]
                    s += f'{none_option}. The correct answer is No answer, None of the above, all provided options are irrelevant or incorrect, or I cannot answer.\n'
                    valid_option.append(none_option)
                    answer_option = [gt_option, none_option]

            if eval_type == "standard":
                none_option = ""
                for option in ['C', 'D', 'E', 'F']:
                    if option not in valid_option:
                        none_option = option
                        break
                s += f'{none_option}. The correct answer is No answer, None of the above, \
                    all provided options are irrelevant or incorrect, or I cannot answer.\n'
                valid_option.append(none_option)
                if question_type == 'inst':
                    s += f'F. None of the above\n'
                    valid_option.append("F")
        elif upd_type == "ivqd":
            if eval_type == "ivqd":
                none_option = ""
                for option in ['C', 'D', 'E', 'F']:
                    if option not in valid_option:
                        none_option = option
                        break
                s += f'{none_option}. The correct answer is that The image is incompatible with the question, or I cannot answer.\n'
                valid_option.append(none_option)
                answer_option = [none_option]
                if question_type == 'inst':
                    s += f'F. The image and question are irrelevant.\n'
                    valid_option.append("F")
                    answer_option = [none_option, "F"]

            if eval_type == "standard":
                none_option = ""
                for option in ['C', 'D', 'E', 'F']:
                    if option not in valid_option:
                        none_option = option
                        break
                s += f'{none_option}. The correct answer is that The image is incompatible with the question, or I cannot answer.\n'
                valid_option.append(none_option)
                if question_type == 'inst':
                    s += f'F. The image and question are irrelevant.\n'
                    valid_option.append("F")
        return s, valid_option, answer_option

    def build_prompt(self, question, options, prediction, upd_type):
        if upd_type == "aad":
            tmpl = (
                'You are an AI assistant who will help me to match an answer '
                'with several options of a single-choice question. '
                'You are provided with a question, several options, and an answer, '
                'and you need to find which option is most similar to the answer. '
                'If the meaning of all options are significantly different '
                'from the answer, output H. '
                'Your should output a single uppercase character in A, B, C, D, E, F, G '
                '(if they are valid options), and H. \n'
                'Example 1: \n'
                'Question: What is the main object in image?\nOptions: A. teddy bear '
                'B. rabbit C. cat D. dog E. no answer \nAnswer: a cute teddy bear\nYour output: A\n'
                'Example 2: \n'
                'Question: What is the main object in image?\nOptions: A. teddy bear '
                'B. rabbit C. cat D. dog E. None of the above  \nAnswer: no answer \nYour output: E\n'
                'Example 3: \n'
                'Question: What is the main object in image?\nOptions: A. teddy bear '
                'B. rabbit C. cat D. dog E. None of the above \nAnswer: fish \nYour output: H\n'
                'Example 4: \n'
                'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: ')
        elif upd_type == "iasd":
            tmpl = (
                'You are an AI assistant who will help me to match an answer '
                'with several options of a single-choice question. '
                'You are provided with a question, several options, and an answer, '
                'and you need to find which option is most similar to the answer. '
                'If the meaning of all options are significantly different '
                'from the answer, output H. '
                'Your should output a single uppercase character in A, B, C, D, E, F, G '
                '(if they are valid options), and H. \n'
                'Example 1: \n'
                'Question: What is the main object in image?\nOptions: A. teddy bear '
                'B. rabbit C. cat D. dog E. no answer \nAnswer: a cute teddy bear\nYour output: A\n'
                'Example 2: \n'
                'Question: What is the main object in image?\nOptions: A. teddy bear '
                'B. rabbit C. cat D. dog E. None of the above  \nAnswer: no answer \nYour output: E\n'
                'Example 3: \n'
                'Question: What is the main object in image?\nOptions: A. teddy bear '
                'B. rabbit C. cat D. dog E. None of the above \nAnswer: fish \nYour output: H\n'
                'Example 4: \n'
                'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: ')
        elif upd_type == "ivqd":
            tmpl = (
                'You are an AI assistant who will help me to match an answer '
                'with several options of a single-choice question. '
                'You are provided with a question, several options, and an answer, '
                'and you need to find which option is most similar to the answer. '
                'If the meaning of all options are significantly different '
                'from the answer, output H. '
                'Your should output a single uppercase character in A, B, C, D, E, F, G '
                '(if they are valid options), and H. \n'
                'Example 1: \n'
                'Question: What is the main object in image?\nOptions: A. teddy bear '
                'B. rabbit C. cat D. dog E. The image and question are irrelevant \nAnswer: a cute teddy bear\nYour output: A\n'
                'Example 2: \n'
                'Question: What is the main object in image?\nOptions: A. teddy bear '
                'B. rabbit C. cat D. dog E. The image and question are irrelevant \nAnswer: The updloaded image and question are incompatible. \nYour output: E\n'
                'Example 3: \n'
                'Question: What is the main object in image?\nOptions: A. teddy bear '
                'B. rabbit C. cat D. dog E. The image and question are irrelevant \nAnswer: fish \nYour output: H\n'
                'Example 4: \n'
                'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: ')
        return tmpl.format(question, options, prediction)

    # Prefetch Answers
    def can_infer_option(self, answer, option_dict, question_type=None, valid_option=None):
        if valid_option is None:
            valid_option = list(option_dict.keys())
            if question_type == 'inst':
                valid_option.append("F")

        if 'Failed to obtain answer via API' in answer:
            return False

        answer = answer.strip()

        ch_cand_list = []

        punctuations = [".", ")", ","]
        if "A" in valid_option:
            characters = ["B", "C", "D", "E", "F", "G"]
            combinations = [char + punct for char in characters for punct in punctuations]
            start_patterns = ["A)", "A.", "A,", "(A)"]
            if answer == "A" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
                ch_cand_list.append("A")
        if "B" in valid_option:
            characters = ["A", "C", "D", "E", "F", "G"]
            combinations = [char + punct for char in characters for punct in punctuations]
            start_patterns = ["B)", "B.", "B,", "(B)"]
            if answer == "B" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
                ch_cand_list.append("B")
        if "C" in valid_option:
            characters = ["A", "B", "D", "E", "F", "G"]
            combinations = [char + punct for char in characters for punct in punctuations]
            start_patterns = ["C)", "C.", "C,", "(C)"]
            if answer == "C" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
                ch_cand_list.append("C")
        if "D" in valid_option:
            characters = ["A", "B", "C", "E", "F", "G"]
            combinations = [char + punct for char in characters for punct in punctuations]
            start_patterns = ["D)", "D.", "D,", "(D)"]
            if answer == "D" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
                ch_cand_list.append("D")
        if "E" in valid_option:
            characters = ["A", "B", "C", "D", "F", "G"]
            combinations = [char + punct for char in characters for punct in punctuations]
            start_patterns = ["E)", "E.", "E,", "(E)"]
            if answer == "E" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
                ch_cand_list.append("E")
        if "F" in valid_option:
            characters = ["A", "B", "C", "D", "E", "G"]
            combinations = [char + punct for char in characters for punct in punctuations]
            start_patterns = ["F)", "F.", "F,", "(F)"]
            if answer == "F" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
                ch_cand_list.append("F")
        if "G" in valid_option:
            characters = ["A", "B", "C", "D", "E", "F"]
            combinations = [char + punct for char in characters for punct in punctuations]

            start_patterns = ["G)", "G.", "G,", "(G)"]
            if answer == "G" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
                ch_cand_list.append("G")

        if len(ch_cand_list) == 1:
            return ch_cand_list[0]

        return False

    def can_infer(self, answer, choices, question_type=None, valid_option=None):
        copt = self.can_infer_option(answer, choices, question_type, valid_option=valid_option)
        return copt if copt else False

    def prefetch_answer(self, item, question_type):
        choices = self.build_choices(item)
        return self.can_infer(item["prediction"], choices, question_type=question_type)

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

    def extract_answer_from_item(self, item, gt_text, eval_type, question_type, upd_type):
        options = self.extract_options(item)
        option_str, valid_option, answer_option = self.build_option_str_w_gt(options, gt_text, eval_type, question_type=question_type, upd_type=upd_type)

        prompt = self.build_prompt(item['question'], option_str, item['prediction'], upd_type=upd_type)
        retry = 3
        choices = self.build_choices(item)

        ret = self.can_infer(item['prediction'], choices, valid_option=valid_option)
        if ret:
            return ret, item['prediction'], answer_option

        while retry:
            ans = self.get_chat_response(prompt, temperature=0.9)
            if 'Failed to obtain answer via API' in ans:
                msg = 'GPT API failed to answer. '
                eval_logger.info(msg)
                retry -= 1
            else:
                ret = self.can_infer(ans,  choices, valid_option=valid_option)
                if ret:
                    return ret, ans, answer_option
                else:
                    eval_logger.info(f'GPT output includes 0 / >1 letter in "{valid_option}": {ans}')
                    retry -= 1

            if retry == 0:
                return 'H', 'Failed to predict. ', answer_option

    def eval_sub_data(self, sub_data, answer_map, gt_text_map, question_type, eval_type, upd_type):
        lt = len(sub_data)
        GT, PRED = [], []

        for i in range(lt):
            item = sub_data.iloc[i]
            idx = item['index']
            GT.append(answer_map[idx])
            PRED.append(self.prefetch_answer(item, question_type))
            if PRED[-1] and (GT[-1] != PRED[-1]):
                return 0

        for i in range(lt):
            if PRED[i]:
                continue
            else:
                item = sub_data.iloc[i]
                idx = item['index']
                gt_text = gt_text_map[idx] if gt_text_map is not None else None
                ret, _, answer_option = self.extract_answer_from_item(sub_data.iloc[i], gt_text, eval_type, question_type=question_type, upd_type=upd_type)
                PRED[i] = ret
                if eval_type == "standard":
                    if PRED[i] != GT[i]:
                        return 0
                else:
                    if GT[i] == "F":
                        if PRED[i] not in answer_option:
                            return 0
                    else:
                        if PRED[i] != GT[i] and PRED[i] not in answer_option:
                            return 0
        return 1

    def calculate_hit_rates(self, data):
        overall_hit_rate = data["hit"].mean()

        category_hit_rate = {}
        if "category" in data.columns:
            # Category-based hit rate
            category_hit_rate = data.groupby("category")["hit"].mean().to_dict()

        return overall_hit_rate, category_hit_rate

    # Evaluate Results
    def eval_result(self, results, eval_method, upd_type, question_type, eval_type):
        """
        Parameters:
        - args: Arguments.
        - results: Results to evaluate.
        - eval_method: The evaluation method. either "openai".
        - upd_type: The type of UPD. either "aad", "iasd", or "ivqd".
        - question_type: The type of question. either "base", "option", or "inst".
        - eval_type: The type of evaluation. either "standard", "aad", "iasd", "ivqd".
        """

        rd.seed(2680)
        assert eval_method == "openai"

        result = {}
        data = pd.DataFrame(results)

        if eval_type == "standard":
            data = data[data["type"] == "standard"]
        else:
            data = data[data["type"] == "upd"]

        data = data.sort_values(by="index")

        data["prediction"] = [str(x) for x in data["prediction"]]
        for k in data.keys():
            data[k.lower() if k not in "ABCDE" else k] = data.pop(k)

        # meta = load(meta_file)

        data_main = data[data["index"] < int(1e6)]

        data_main["hit"] = 0
        cate_map = {i: c for i, c in zip(data["index"], data["category"])}
        answer_map = {i: c for i, c in zip(data["index"], data["answer"])}

        gt_text_map = {i: c for i, c in zip(data['index'], data['masked_answer'])}

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

            ret = self.eval_sub_data(sub_data, answer_map, gt_text_map,\
                                     question_type=question_type, eval_type=eval_type, upd_type=upd_type)
            result[idx] = ret
            hit += ret
            tot += 1

            data_main.loc[data_main["index"] == idx, "hit"] = ret

        indices = data_main["index"]

        data_main["category"] = [cate_map[i] if not math.isnan(i) else "uncategorized" for i in indices]

        overall_hit_rate, category_hit_rate = self.calculate_hit_rates(data_main)

        return overall_hit_rate, category_hit_rate, data_main

    def report_acc(self, df, groupd='category'):
        assert 'split' in df
        assert groupd in [None, 'category', 'l2-category']

        res = defaultdict(list)
        res['split'] = ['test']
        if groupd is None:
            res['overall'] = [
                np.mean(df['hit']),
            ]
            return pd.DataFrame(res)

        elif groupd in df:
            abilities = list(set(df[groupd]))
            abilities.sort()
            for ab in abilities:
                sub_df = df[df[groupd] == ab]
                res[ab] = [
                    np.mean(sub_df['hit']),
                ]
            return pd.DataFrame(res)

    def calculate_dual_acc(self, standard_df, upd_df):
        # dual results
        dual_df = pd.merge(standard_df, upd_df, on='index',
                           suffixes=('_standard', '_upd'))
        dual_df['hit'] = dual_df.apply(lambda row: 1 if row['hit_standard'] == 1 and row['hit_upd'] == 1 else 0, axis=1)
        dual_df['split'] = dual_df['split_standard']
        dual_df['category'] = dual_df['category_standard']

        # Evaluate dual results
        overall_hit_rate, category_hit_rate = self.calculate_hit_rates(dual_df)
        print("Overall Dual Acc.", overall_hit_rate)

        if "category" in dual_df.columns:
            print("Category Dual Acc.:")
            for category_key in category_hit_rate:
                if category_key == "split":
                    continue

                category_percentage = category_hit_rate[category_key] * 100
                print(f"\t{category_key}: {category_percentage:.3f}")

        return overall_hit_rate, category_hit_rate, dual_df
