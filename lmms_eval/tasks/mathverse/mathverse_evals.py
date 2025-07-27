import os

import pandas as pd
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.llm_judge import ServerConfig, get_server


class MathVerseEvaluator:
    def __init__(self, api_key, gpt_model="gpt-3.5-turbo", quick_extract=False):
        self.api_key = api_key
        self.gpt_model = gpt_model
        self.quick_extract = quick_extract

        # Initialize the judge server using environment variables
        API_TYPE = os.getenv("API_TYPE", "openai")
        server_config = ServerConfig(
            model_name=gpt_model,
        )
        self.server = get_server(server_name=API_TYPE, config=server_config)

    def create_match_prompt(self, demo_prompt, question, answer, extraction):
        demo_prompt = demo_prompt.strip()
        full_prompt = demo_prompt.format(question=question, gt=answer, extraction=extraction)
        return full_prompt

    def score_answer(self, question, answer, model_response, quick_match=False):
        if quick_match:
            return model_response == answer

        try:
            result = self.server.evaluate_binary(question=question, answer=str(answer), prediction=model_response, output_format="0/1")

            if result["success"]:
                judge_response = result["result"]
                return judge_response
            else:
                eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
                return 0

        except Exception as e:
            print(e)
            print(f"Error in matching answer")

        return 0

    def get_acc_with_contion(self, res_pd, key, value):
        """
        Calculate the accuracy of predictions with a specific condition
        """
        total_pd = res_pd[res_pd[key] == value]

        correct_pd = total_pd[total_pd["true_false"] == True]
        acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100) if len(total_pd) > 0 else "0.00"
        return len(correct_pd), len(total_pd), acc

    def create_one_query(self, problem, shot_type, hint, query_type, examples=None, shot_num=0):
        ### [1] Demo prompt
        if shot_num == 0:
            demo_prompt = ""
        else:
            demos = []
            shot_num = min(shot_num, len(examples)) if examples else 0
            for example in examples[:shot_num] if examples else []:
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
        # score each question directly without extraction
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

            # Skip extraction and pass model's full response directly to scoring
            # set test set answer to None
            if "true_false" in inst:
                true_false = inst["true_false"]
            else:
                true_false = self.score_answer(problem["question_for_eval"], problem["answer"], prediction, config["metadata"]["quick_match"]) if problem["answer"] is not None else False

            inst["extraction"] = prediction  # Store the full prediction as extraction
            inst["prediction"] = prediction
            inst["true_false"] = true_false

        # calculate total scores
        sample_index = [result["sample_index"] for result in results]
        total = len(results)
        correct = sum(1 for idx in range(len(sample_index)) if results[idx]["true_false"])
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
