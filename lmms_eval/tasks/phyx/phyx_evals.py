import re
import json
import os
import copy
import argparse
from tqdm import tqdm
from collections import defaultdict
import ast
from pathlib import Path
import yaml
from openai import OpenAI
from loguru import logger as eval_logger
import time
import pandas as pd

FAIL_MSG = 'Failed to obtain answer via API.'

with open(Path(__file__).parent / "phyx.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


class PhyXEvaluator:
    def __init__(self):
        if not config["metadata"]["quick_extract"]:
            self.juder_model = config["metadata"]["eval_model_name"]
            API_URL = "https://api.deepseek.com/v1/chat/completions"
            API_KEY = os.getenv("Deepseek_API", "YOUR_API_KEY")
            self.headers = {
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    }
            self.client = OpenAI(api_key=API_KEY, base_url=API_URL.rstrip("chat/completions"))
        else:
            self.juder_model = None
            self.headers = None
            self.client = None


    def judger_generate(self, prompt, temperature=0, max_tokens=128, n=1, patience=5, sleep_time=0):
        assert config["metadata"]["quick_extract"]==False, "TO employ LLM to extract answer, please set `quick_extract=False`"
        messages = [
            {"role": "user", "content": prompt},
        ]
        payload = {"model": self.juder_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

        while patience > 0:
            patience -= 1
            try:
                response = self.client.chat.completions.create(**payload)
                if n == 1:
                    prediction = response.choices[0].message.content.strip()
                    if prediction and prediction != "":
                        return prediction
                else:
                    prediction = [choice.message.content.strip() for choice in response.choices]
                    if prediction and prediction[0] != "":
                        return prediction

            except Exception as e:
                if "Rate limit" not in str(e):
                    eval_logger.error(e)

                if "Please reduce the length of the messages" in str(e):
                    eval_logger.error("!!Reduce prompt size")
                    # reduce input prompt and keep the tail
                    new_size = int(len(prompt) * 0.9)
                    new_start = len(prompt) - new_size
                    prompt = prompt[new_start:]
                    payload["messages"] = [
                        {"role": "user", "content": prompt},
                    ]

                if sleep_time > 0:
                    time.sleep(sleep_time)
        return FAIL_MSG

    def get_ICE(self):
        example_1 = """
    Ground truth answer: 502 \n
    Predicted answer: The mass of block (B) is:
    [
    \\boxed{ 50 \\sqrt{101} }
    ] \n
    Judegement: 1
    """

        example_2 = """
    Ground truth answer: 46.3 kN \n
    Predicted answer: The tension ( T_B ) in the cable is approximately:
    [
    \\boxed{46300 }
    ] \n
    Judegement: 1
    """

        example_3 = """
    Ground truth answer: 12 m/s \n
    Predicted answer: The speed of the box after 2.00 seconds is:
    [
    \\boxed{11.3, \\text{m/s}}
    ] \n
    Judegement: 0
    """

        example_4 = """
    Ground truth answer: 36.00 kg \n
    Predicted answer: The mass of the hanging block ( m_2 ) must be approximately:
    [
    \\boxed{36.1, \\text\\{kg\\}}
    ] \n
    Judegement: 1
    """

        example_5 = """
    Ground truth answer: 3.2 m \n
    Predicted answer: The stuntman and villain slide approximately \\frac\{10\}{3.1415} meters**.
    Judegement: 1
    """

        return [example_1, example_2, example_3, example_4, example_5]


    def get_ICE_MC(self):
        example_1 = """
    Ground truth answer: A \n
    Predicted answer: A \n
    Judegement: 1
    """

        example_2 = """
    Ground truth answer: B \n
    Predicted answer: A \n
    Judegement: 0
    """

        example_3 = """
    Ground truth answer: C \n
    Predicted answer: ### Step 1: Calculate ( l_1 )
    The lightbulb is ( 2.50, \\text\\{m\\}) above the floor, and the bottom of the mirror is (0.50, \\text\\{m\\}) \
    above the floor. The vertical distance from the lightbulb to the bottom of the mirror is:
    [
    \\Delta y_1 = 2.50, \\text\\{m\\} - 0.50, \\text\\{m\\} = 2.00, \\text\\{m\\}.
    ] \n
    Judegement: 0
    """

        example_4 = """
    Ground truth answer: D \n
    Predicted answer: The correct option is D. \n
    Judegement: 1
    """

        return [example_1, example_2, example_3, example_4]


    def build_phyx_gpt4_prompt(self, line, pred):
        task_description = """
    Please read the following example. Given predicted answer and ground truth answer,
    compare the these two answers, then ONLY output judegement 1/0 for matched/unmatched at the end of the prompt.
    If the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
    If the given predicted mentions "approximately", then allow the Approximation Error, \
    such as 0.49 and approximately 0.5, 0.81 and approximately 0.8. \n
    """
        gt_answer = line['answer']
        prompt = task_description
        examples = self.get_ICE()
        for example in examples:
            prompt += example + '\n'
        prompt += 'Ground truth answer: {} \n'.format(gt_answer)
        prompt += 'Predicted answer: {} \n'.format(pred)
        prompt += 'Judegement:'
        return prompt


    def build_phyx_gpt4_prompt_MC(self, line, pred):
        task_description = """
    Please read the following example. Given predicted answer and ground truth answer for Multi-Choice question.
    The ground truth answer would be A/B/C/D. The predicted answer would be some words containing A/B/C/D.
    Please compare the these two answers, then ONLY output judegement 1/0 for matched/unmatched at the end of the prompt. \n
    """
        gt_answer = line['answer']
        prompt = task_description
        examples = self.get_ICE_MC()
        for example in examples:
            prompt += example + '\n'
        prompt += 'Ground truth answer: {} \n'.format(gt_answer)
        prompt += 'Predicted answer: {} \n'.format(pred)
        prompt += 'Judegement:'
        return prompt


    def mapping_str(self, input):
        d = {"\dfrac": "\\frac", "\pi": "3.14"}
        output = input
        for k,v in d.items():
            try:
                output = output.replace(k, v)
            except:
                pass
        return output

    def safe_literal_eval(self, s):
        s = s.strip()
        try:
            return ast.literal_eval(s)
        except:
            pass  
        if not s.startswith("{"):
            s = "{" + s
        if not s.endswith("}"):
            s = s + "}"
        s = re.sub(r'([{,]\s*)([^"\{\}\:\,\s]+)\s*:', r'\1"\2":', s)
        try:
            return ast.literal_eval(s)
        except Exception as e:
            return None


    def extract_boxed_content(self, s):
        start = s.find(r'\boxed{')
        if start == -1:
            return None 
        content_start = start + len(r'\boxed{')
        rest = s[content_start:]
        depth = 0
        for i, ch in enumerate(rest):
            if ch == '{':
                depth += 1
            elif ch == '}':
                if depth == 0:
                    return rest[:i]
                else:
                    depth -= 1
        return None


    def PhyX_auxeval(self, line):
        log = ''
        retry = 5

        gt_answer = str(line['answer'])
        prediction = line['prediction']

        # try extract final answer using re rules
        tmp = self.PhyX_process_line(line)

        if tmp["extracted"] == "Fail to Call API":
            log += "Fail to Call API"
            prediction = "Fail to Call API"
            return dict(log=log, res=0, extracted=prediction)

        if tmp["extracted"] != "SAME as predict":
            prediction = tmp["extracted"]

        # judge via LLM
        if gt_answer.strip().lower() == prediction.strip().lower():
            return dict(log="Matched at string level", res=1, extracted=prediction)

        prompt = self.build_phyx_gpt4_prompt(line, prediction)
        for i in range(retry):
            res = self.judger_generate(prompt, temperature=i * 0.5)
            if FAIL_MSG in res:
                log += f'Try {i}: answer and prediction are {gt_answer} and {prediction}, failed to compare.\n'
            else:
                log += 'Compared at semantic level. '
                # print(res)
                if "1" in res or 1 == res:
                    log += "Semantic equal via LLM."
                    return dict(log=log, res=1, extracted=prediction)
                elif "0" in res or 0 == res:
                    log += "LLM judgement {}".format(res)
                    return dict(log=log, res=0, extracted=prediction)
        log += 'All 5 retries failed.\n'
        return dict(log=log, res=0, extracted=prediction)


    def PhyX_auxeval_MC(self, line):
        log = ''
        retry = 5

        gt_answer = str(line['answer'])
        prediction = line['prediction']

        tmp = self.PhyX_process_line_MC(line)

        if tmp["extracted"] == "Fail to Call API":
            log += "Fail to Call API"
            prediction = "Fail to Call API"
            return dict(log=log, res=0, extracted=prediction)
            
        if tmp["extracted"] != "SAME as predict":
            prediction = tmp["extracted"]

        # match at string level
        if gt_answer.strip().lower() == prediction.strip().lower():
            return dict(log="Matched at string level", res=1, extracted=prediction)
        else:
            # prediction is A/B/C/D, then labeled as unmatch
            if prediction.strip() in ["A", "B", "C", "D"]:
                return dict(log="Unmatched at string level", res=0, extracted=prediction)

        prompt = self.build_phyx_gpt4_prompt_MC(line, prediction)
        for i in range(retry):
            res = self.judger_generate(prompt, temperature=i * 0.5)
            if FAIL_MSG in res:
                log += f'Try {i}: answer and prediction are {gt_answer} and {prediction}, failed to compare.\n'
            else:
                log += 'Compared at semantic level. '
                if "1" in res or 1 == res:
                    log += "Semantic equal via LLM."
                    return dict(log=log, res=1, extracted=prediction)
                elif "0" in res or 0 == res:
                    log += "LLM judgement {}".format(res)
                    return dict(log=log, res=0, extracted=prediction)
        log += 'All 5 retries failed.\n'
        return dict(log=log, res=0, extracted=prediction)


    def PhyX_process_line(self, line):
        ret = {}

        answers = str(line['answer'])

        ret["index"] = line["index"]
        ret['gt'] = answers
        
        # with reasoning, extract content part
        prediction_str = line['prediction']
        with_reasoning = False
        try:
            pred_dict = self.safe_literal_eval(prediction_str)
            if isinstance(pred_dict, dict) and 'content' in pred_dict and pred_dict['content']!="":
                ret['pred'] = pred_dict['content'].strip()
                with_reasoning = True
        except:
            pass

        if not with_reasoning:
            ret['pred'] = prediction_str.strip()
        
        if ret['pred'] == FAIL_MSG:
            ret['match'] = 0
            ret["extracted"] = "Fail to Call API"
            return ret
        
        boxed_answer = self.extract_boxed_content(ret['pred'])
        if boxed_answer != None:
            boxed_answer = self.mapping_str(boxed_answer)
            ret["extracted"] = boxed_answer
        else:
            pattern = r'\b(?:final\s+answer|correct\s+answer)\b[^:：]*[:：]\s*(.*?)(?=\n\n\n|\Z)'
            flags = re.IGNORECASE | re.DOTALL
            match = re.search(pattern, ret['pred'], flags=flags)
            if match:
                extracted_answer = match.group(1)
                extracted_answer = self.mapping_str(extracted_answer)
                ret["extracted"] = extracted_answer
            else:
                ret["extracted"] = "SAME as predict"


        if ret['gt'].strip().lower() == ret["extracted"].strip().lower() or ret['gt'].strip().lower() == ret["pred"].strip().lower() or ret['gt'] in ret['pred']:
            ret['match'] = 1
            return ret

        # unmatch at string level
        ret['match'] = 0
        return ret


    def PhyX_process_line_MC(self, line):
        ret = {}

        answers = str(line['answer'])

        ret["index"] = line["index"]
        ret['gt'] = answers
        ret['pred'] = line['prediction'].strip()

        pattern = r'\b(?:correct|answer|option|Answer|Option|Correct)\b[\s\S]*?([A-D])'
        match = re.search(pattern, ret['pred'])

        if ret['pred'] == FAIL_MSG:
            ret['match'] = 0
            ret["extracted"] = "Fail to Call API"
            return ret

        if match:
            extracted_answer = match.group(1)
            # compare string
            ret["extracted"] = extracted_answer
            if ret['gt'].strip().lower() == extracted_answer.strip().lower():
                ret['match'] = 1
                return ret
        else:
            # try another match strategy
            matches = re.findall(r'([ABCD]):', ret['pred'])
            if matches:
                extracted_answer = matches[-1]
                ret["extracted"] = extracted_answer
                if ret['gt'].strip().lower() == extracted_answer.strip().lower():
                    ret['match'] = 1
                    return ret
            else:
                ret["extracted"] = "SAME as predict"

        if ret['gt'] + ":" in ret['pred'] or ret['gt'] + "**" in ret['pred'] or "**" + ret['gt'] in ret['pred']:
            ret['match'] = 1
        else:
            ret['match'] = 0

        return ret
