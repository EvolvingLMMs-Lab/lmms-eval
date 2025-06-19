import os
import time

import pandas as pd
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.llm_judge import ServerConfig, get_server

DEMO_PROMPT_EXTRACT = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.

1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)

2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D

3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)

4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null

5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3

6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1

7.
"""

DEMO_PROMPT_SCORE = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.

[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0

[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0

[Question]: Given the graph of the ellipse that intersects with x-axis at 9 and -9 and with y-axis at 3 and -3, determine its equation.A. \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1 B. Can not determine.\n
[Standard Answer]: A
[Model_answer] : \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1
Judgement: 1

[Question]: {question}
[Standard Answer]: {gt}
[Model_answer] : {extraction}
Judgement: """


class MathVerseEvaluator:
    def __init__(self, api_key, gpt_model="gpt-3.5-turbo", quick_extract=False):
        self.api_key = api_key
        self.gpt_model = gpt_model
        self.quick_extract = quick_extract
        
        # Initialize the judge server
        API_TYPE = os.getenv("API_TYPE", "openai")
        server_config = ServerConfig(
            model_name=gpt_model,
        )
        self.server = get_server(server_name=API_TYPE, config=server_config)


    def get_chat_response(self, prompt, temperature=0, max_tokens=256, n=1, patience=10000000, sleep_time=0):
        while patience > 0:
            patience -= 1
            try:
                # Use the judge server for general text generation
                result = self.server.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n
                )
                
                if result["success"]:
                    if n == 1:
                        prediction = result["result"].strip()
                        if prediction and prediction != "":
                            return prediction
                    else:
                        # For multiple completions, we need to handle differently
                        # Since the judge server doesn't support n > 1 directly,
                        # we'll just return the single result
                        prediction = result["result"].strip()
                        if prediction and prediction != "":
                            return prediction
                else:
                    eval_logger.error(f"Generation failed: {result.get('raw_response', 'Unknown error')}")
                    
            except Exception as e:
                # some model may output repetitive answer, which ChatGPT will throw an error.
                if "repetitive patterns" in str(e):
                    print(str(e))
                    print("Continue with empty answer")
                    return ""
                # some answer may contain some sensitive words, like 'test'
                if "sensitive" in str(e) or "400" in str(e):
                    print(str(e))
                    print("Continue with empty answer")
                    return "0"

                if "Rate limit" not in str(e):
                    eval_logger.error(e)

                if "Please reduce the length of the messages" in str(e):
                    eval_logger.error("!!Reduce prompt size")
                    # reduce input prompt and keep the tail
                    new_size = int(len(prompt) * 0.9)
                    new_start = len(prompt) - new_size
                    prompt = prompt[new_start:]

                if sleep_time > 0:
                    time.sleep(sleep_time)
        return ""

    def verify_extraction(self, extraction):
        extraction = extraction.strip()
        if not extraction:
            return False
        return True

    def create_extract_prompt(self, demo_prompt, response):
        demo_prompt = demo_prompt.strip()
        test_prompt = f"Model response: '{response}'\nExtracted Answer: "
        full_prompt = f"{demo_prompt}\n\n{test_prompt}"
        return full_prompt

    def create_match_prompt(self, demo_prompt, question, answer, extraction):
        demo_prompt = demo_prompt.strip()
        full_prompt = demo_prompt.format(question=question, gt=answer, extraction=extraction)
        return full_prompt

    def extract_answer(self, response):
        if not response:
            return ""

        # general extraction
        try:
            full_prompt = self.create_extract_prompt(DEMO_PROMPT_EXTRACT, response)
            extraction = self.get_chat_response(full_prompt, temperature=0, max_tokens=256, n=1)
            return extraction
        except Exception as e:
            eval_logger.error(e)
            eval_logger.error(f"Error in extracting answer for problem")

        return ""

    def score_answer(self, question, answer, extraction, quick_match=False):
        if quick_match:
            return extraction == answer

        try:
            # Use the judge server for binary evaluation
            custom_prompt = """Below are two answers to a math question. Determine whether these two answers are consistent.
Please note that only when the Model Answer completely matches the Standard Answer means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.

Return only "Yes" if they are consistent or "No" if they are different.
Only return "Yes" or "No" with no additional text or formatting."""
            
            result = self.server.evaluate_binary(
                question=question,
                answer=str(answer),
                prediction=extraction,
                output_format="yes/no",
                custom_prompt=custom_prompt
            )
            
            if result["success"]:
                judge_response = result["result"]
                return judge_response and judge_response.lower() == "yes"
            else:
                eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(e)
            print(f"Error in matching answer")

        return False

    def get_acc_with_contion(self, res_pd, key, value):
        """
        Calculate the accuracy of predictions with a specific condition
        """
        total_pd = res_pd[res_pd[key] == value]

        correct_pd = total_pd[total_pd["true_false"] == True]
        acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100) if len(total_pd) > 0 else "0.00"
        return len(correct_pd), len(total_pd), acc

    def create_one_query(self, problem, shot_type, hint, query_type, examples=None, shot_num=0, use_caption=False, use_ocr=False):
        ### [1] Demo prompt
        if shot_num == 0:
            demo_prompt = ""
        else:
            demos = []
            shot_num = min(shot_num, len(examples))
            for example in examples[:shot_num]:
                prompt = ""

                # question
                prompt += f"Question: {example[query_type]}"

                # solution
                if shot_type == "solution":
                    solution = example["solution"].strip()
                    prompt += "\n" + f"Solution: {solution}"

                # step-by-step
                if shot_type == "step-by-step":
                    solution = example["solution"].strip()
                    prompt += "\n" + f"{solution}"

                # direct
                if shot_type == "direct":
                    solution = example["solution"].strip()
                    prompt += "\n" + f"{solution}"

                demos.append(prompt)

            demo_prompt = "\n\n".join(demos)

        ### [2] Test query
        # problem info
        question = problem["question"]
        question_type = problem["question_type"]

        # hint
        # format-prompt
        if shot_type == "format-prompt":
            hint_text = ""
        # custom-prompt
        elif shot_type == "custom-prompt":
            if question_type == "multi-choice":
                hint_text = hint["multi-choice"]
            else:  # free-form
                hint_text = hint["free-form"]

        # question
        if shot_type == "format-prompt":
            question_text = f"{problem[query_type]}"
        elif shot_type == "custom-prompt":
            question_text = f"Question: {question}"

        elements = [hint_text, question_text]
        test_query = "\n".join([e for e in elements if e != ""])

        ### [3] Final query
        query = demo_prompt + "\n\n" + test_query
        query = query.strip()
        return query

    def eval_results(self, results, config):
        # extract and score for each question
        for inst in tqdm(results):
            full_prediction = inst["prediction"].strip()
            problem = {
                "question_type": inst["question_type"],
                "answer": inst["answer"] if "answer" in inst else None,
                "question_for_eval": inst["question_for_eval"],
            }
            if config["metadata"].get("trunk_response", -1) > 0:
                prediction = " ".join(full_prediction.split(" ")[-config["metadata"]["trunk_response"] :])
            else:
                prediction = full_prediction
            extraction = self.extract_answer(prediction)
            # set test set answer to None
            true_false = self.score_answer(problem["question_for_eval"], problem["answer"], extraction, config["metadata"]["quick_match"]) if problem["answer"] is not None else False

            inst["extraction"] = extraction
            inst["prediction"] = prediction
            inst["true_false"] = true_false

        # calculate total scores
        sample_index = [result["sample_index"] for result in results]
        total = len(results)
        correct = sum(1 for idx, pid in enumerate(sample_index) if results[idx]["true_false"])
        accuracy = round(correct / total * 100, 2)
        scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

        for result in results:
            result.update(result.pop("metadata"))

        results_dict = {result["sample_index"]: result for result in results}
        df = pd.DataFrame(results_dict).T
        target_keys = ["problem_version", "subfield"]

        for key in target_keys:
            values = df[key].unique()
            scores[key] = {}
            for value in values:
                correct, total, acc = self.get_acc_with_contion(df, key, value)
                if total > 0:
                    scores[key][value] = {"accuracy": acc, "correct": correct, "total": total}
            scores[key] = dict(sorted(scores[key].items(), key=lambda item: float(item[1]["accuracy"]), reverse=True))

        return results_dict, scores
