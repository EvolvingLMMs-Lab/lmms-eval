import time
import requests
import re
from Levenshtein import distance
import logging

eval_logger = logging.getLogger("lmms-eval")
DEMO_PROMPT = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""


class MathVistaEvaluator:
    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key, gpt_model="gpt-3.5-turbo", quick_extract=False):
        self.api_key = api_key
        self.gpt_model = gpt_model
        self.quick_extract = quick_extract

    def _post_request(self, payload):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_chat_response(self, prompt, temperature=0, max_tokens=256, n=1, patience=10000000, sleep_time=0):
        messages = [
            {"role": "user", "content": prompt},
        ]
        payload = {"model": self.gpt_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "n": n}

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
        return ""

    def verify_extraction(self, extraction):
        extraction = extraction.strip()
        if not extraction:
            return False
        return True

    def create_test_prompt(self, demo_prompt, query, response):
        demo_prompt = demo_prompt.strip()
        test_prompt = f"{query}\n\n{response}"
        full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
        return full_prompt

    def extract_answer(self, response, problem, quick_extract=False):
        question_type = problem["question_type"]
        answer_type = problem["answer_type"]
        choices = problem.get("choices", [])
        query = problem["query"]

        if not response:
            return ""

        if question_type == "multi_choice" and response in choices:
            return response

        if answer_type == "integer":
            try:
                extraction = int(response)
                return str(extraction)
            except ValueError:
                pass

        if answer_type == "float":
            try:
                extraction = str(float(response))
                return extraction
            except ValueError:
                pass

        # quick extraction
        if quick_extract:
            eval_logger.info("Quickly extracting answer...")
            # The answer is "text". -> "text"
            try:
                result = re.search(r'The answer is "(.*)"\.', response)
                if result:
                    extraction = result.group(1)
                    return extraction
            except re.error:
                pass

        # general extraction
        try:
            full_prompt = self.create_test_prompt(DEMO_PROMPT, query, response)
            extraction = self.get_chat_response(full_prompt, temperature=0, max_tokens=256, n=1)
            return extraction
        except Exception as e:
            eval_logger.error(e)
            eval_logger.error(f"Error in extracting answer for problem")

        return ""

    def get_most_similar(self, prediction, choices):
        """
        Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
        """
        distances = [distance(prediction, choice) for choice in choices]
        ind = distances.index(min(distances))
        return choices[ind]

    def normalize_extracted_answer(self, extraction, choices, question_type, answer_type, precision):
        """
        Normalize the extracted answer to match the answer type
        """
        if question_type == "multi_choice":
            # make sure the extraction is a string
            if isinstance(extraction, str):
                extraction = extraction.strip()
            else:
                try:
                    extraction = str(extraction)
                except:
                    extraction = ""

            # extract "A" from "(A) text"
            letter = re.findall(r"\(([a-zA-Z])\)", extraction)
            if len(letter) > 0:
                extraction = letter[0].upper()

            options = [chr(ord("A") + i) for i in range(len(choices))]

            if extraction in options:
                # convert option letter to text, e.g. "A" -> "text"
                ind = options.index(extraction)
                extraction = choices[ind]
            else:
                # select the most similar option
                extraction = self.get_most_similar(extraction, choices)
            assert extraction in choices

        elif answer_type == "integer":
            try:
                extraction = str(int(float(extraction)))
            except:
                extraction = None

        elif answer_type == "float":
            try:
                extraction = str(round(float(extraction), precision))
            except:
                extraction = None

        elif answer_type == "list":
            try:
                extraction = str(extraction)
            except:
                extraction = None

        return extraction

    def safe_equal(self, prediction, answer):
        """
        Check if the prediction is equal to the answer, even if they are of different types
        """
        try:
            if str(prediction).strip() == str(answer).strip():
                return True
            return False
        except Exception as e:
            eval_logger.info(e)
            return False

    def get_acc_with_contion(self, res_pd, key, value):
        """
        Calculate the accuracy of predictions with a specific condition
        """
        if key == "skills":
            total_pd = res_pd[res_pd[key].apply(lambda x: value in x)]
        else:
            total_pd = res_pd[res_pd[key] == value]

        correct_pd = total_pd[total_pd["true_false"] == True]
        acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100) if len(total_pd) > 0 else "0.00"
        return len(correct_pd), len(total_pd), acc

    def create_one_query(self, problem, shot_type, examples=None, shot_num=0, use_caption=False, use_ocr=False):
        ### [1] Demo prompt
        if shot_num == 0:
            demo_prompt = ""
        else:
            demos = []
            shot_num = min(shot_num, len(examples))
            for example in examples[:shot_num]:
                prompt = ""

                # question
                prompt += f"Question: {example['question']}"

                # choices
                if "choices" in example:
                    texts = ["Choices:"]
                    for i, choice in enumerate(example["choices"]):
                        texts.append(f"({chr(ord('A')+i)}) {choice}")
                    prompt += "\n" + "\n".join(texts)

                # caption
                if use_caption:
                    caption = example["caption"] if "caption" in example else ""
                    if caption != "":
                        prompt += "\n" + f"Image description: {caption}"

                # ocr
                if use_ocr:
                    ocr = example["ocr"] if "ocr" in example else ""
                    if ocr != "":
                        prompt += "\n" + f"Image detected text: {ocr}"

                # solution
                if shot_type == "solution":
                    solution = example["solution"].strip()
                    prompt += "\n" + f"Solution: {solution}"

                # step-by-step
                if shot_type == "step-by-step":
                    solution = example["solution"].strip()
                    prompt += "\n" + f"{solution}"

                # think-step-by-step
                if shot_type == "think-step-by-step":
                    solution = example["solution"].strip()
                    prompt += "\n" + f"{solution}"

                # direct
                if shot_type == "direct":
                    solution = example["solution"].strip()
                    prompt += "\n" + f"{solution}"

                # code
                if shot_type == "code":
                    code = example["code"].strip()
                    prompt += "\n" + f"Python code: {code}"

                demos.append(prompt)

            demo_prompt = "\n\n".join(demos)

        ### [2] Test query
        # problem info
        question = problem["question"]
        unit = problem["unit"]
        choices = problem["choices"]
        caption = problem["caption"]
        ocr = problem["ocr"]
        precision = problem["precision"]
        question_type = problem["question_type"]
        answer_type = problem["answer_type"]

        # hint
        if shot_type == "solution":
            if question_type == "multi_choice":
                assert answer_type == "text"
                hint_text = f"Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
            else:
                assert answer_type in ["integer", "float", "list"]
                if answer_type == "integer":
                    hint_text = f"Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end."

                elif answer_type == "float" and precision == 1:
                    hint_text = f"Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end."

                elif answer_type == "float" and precision == 2:
                    hint_text = f"Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end."

                elif answer_type == "list":
                    hint_text = f"Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end."
        # step-by-step
        elif shot_type == "format-prompt":
            if question_type == "multi_choice":
                assert answer_type == "text"
                hint_text = f"Answer with the option's letter from the given choices directly."
            else:
                if answer_type == "integer":
                    hint_text = f"Answer the question using a single integer number."

                elif answer_type == "float" and precision == 1:
                    hint_text = f"Answer the question using a single floating-point number with one decimal place."

                elif answer_type == "float" and precision == 2:
                    hint_text = f"Answer the question using a single floating-point number with two decimal places."

                elif answer_type == "list":
                    hint_text = f"Answer the question using a Python list."
        # step-by-step
        elif shot_type == "step-by-step":
            if question_type == "multi_choice":
                assert answer_type == "text"
                hint_text = f"Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
            else:
                assert answer_type in ["integer", "float", "list"]
                if answer_type == "integer":
                    hint_text = f"Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end."

                elif answer_type == "float" and precision == 1:
                    hint_text = f"Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end."

                elif answer_type == "float" and precision == 2:
                    hint_text = f"Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end."

                elif answer_type == "list":
                    hint_text = f"Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end."
        # step-by-step
        elif shot_type == "reason-first":
            if question_type == "multi_choice":
                assert answer_type == "text"
                hint_text = f"First perform reasoning, then finally select the question from the choices in the following format: Answer: xxx."
            else:
                assert answer_type in ["integer", "float", "list"]
                if answer_type == "integer":
                    hint_text = f"First perform reasoning, then finally answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end in the following format: Answer: xxx."

                elif answer_type == "float" and precision == 1:
                    hint_text = (
                        f"First perform reasoning, then finally answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end in the following format: Answer: xxx."
                    )

                elif answer_type == "float" and precision == 2:
                    hint_text = f"First perform reasoning, then finally answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end in the following format: Answer: xxx."

                elif answer_type == "list":
                    hint_text = f"First perform reasoning, then finally answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end in the following format: Answer: xxx."
        elif shot_type == "direct":
            hint_text = ""
        else:
            assert shot_type == "code"
            hint_text = "Hint: Please generate a python code to solve the problem"

        # question
        if shot_type == "format-prompt":
            question_text = f"{question}"
        else:
            question_text = f"Question: {question}"
        if unit:
            question_text += f" (Unit: {unit})"

        # choices
        if choices:
            if shot_type == "format-prompt":
                texts = []
                for i, choice in enumerate(choices):
                    texts.append(f"{chr(ord('A')+i)}. {choice}")
                choices_text = "\n".join(texts)
            else:
                # choices: (A) 1.2 (B) 1.3 (C) 1.4 (D) 1.5
                texts = ["Choices:"]
                for i, choice in enumerate(choices):
                    texts.append(f"({chr(ord('A')+i)}) {choice}")
                choices_text = "\n".join(texts)
        else:
            choices_text = ""

        # caption
        caption_text = ""
        if use_caption and caption != "":
            caption_text = f"Image description: {caption}"

        # ocr
        ocr_text = ""
        if use_ocr and ocr != "":
            ocr_text = f"Image detected text: {ocr}"

        # prompt
        if shot_type == "solution":
            prompt = "Solution: "
        elif shot_type == "format-prompt":
            prompt = ""
        elif shot_type == "step-by-step":
            prompt = ""
        elif shot_type == "reason-first":
            prompt = ""
        elif shot_type == "direct":
            prompt = ""
        else:
            assert shot_type == "code"
            prompt = "Python code: "

        if shot_type == "reason-first":
            elements = [hint_text, question_text, choices_text, caption_text, ocr_text, prompt]
            test_query = "\n".join([e for e in elements if e != ""])
        else:
            elements = [question_text, choices_text, caption_text, ocr_text, hint_text, prompt]
            test_query = "\n".join([e for e in elements if e != ""])

        ### [3] Final query
        query = demo_prompt + "\n\n" + test_query
        query = query.strip()
        return query
