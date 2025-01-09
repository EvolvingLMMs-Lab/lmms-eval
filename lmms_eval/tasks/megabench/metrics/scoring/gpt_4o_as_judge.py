import os
import requests
import logging
import re

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OpenAIGPT4Judger(OpenAI):
    """
        The OpenAI model class for calling GPT4o or textonly gpt as the juedge
    for open-ended generation tasks
    """

    def __init__(self, metric_config, verbose=True):
        self.judge_model_type = metric_config["judge_model_type"]
        self.eval_prompt = metric_config["eval_criteria_prompt"]
        self.reference_type = metric_config["reference_type"]
        self.template_mapping = metric_config["template_mapping"]
        model = "gpt-4o-2024-08-06"
        super().__init__(
            os.getenv("OPENAI_API_KEY"), model, None, print_response=verbose
        )
        if os.getenv("MEGABENCH_OPEN_API_KEY") is not None:
            self.api_key = os.getenv("MEGABENCH_OPEN_API_KEY")
            self.url = os.getenv("MEGABENCH_OPEN_API_URL")
            if os.getenv("MEGABENCH_OPEN_API_MODEL") is not None:
                self.model = os.getenv("MEGABENCH_OPEN_API_MODEL")
            assert self.url, "You must set up the API URL for evaluating the Open tasks using your own API"

    def prepare_eval_prompt(
        self, reference, response, images, question, eval_context=None
    ):
        content = []
        if self.judge_model_type == "with image":
            for image_path in images:
                content.append(self.create_image_content(image_path))

        prompt_mapping = {}
        for key, val in self.template_mapping.items():
            if val == "model_output":
                prompt_mapping[key] = response
            elif val == "example_question":
                prompt_mapping[key] = question
            elif val.split(".")[0] == "answers":
                key_name = val.split(".")[1]
                prompt_mapping[key] = reference[key_name]
            elif val.split(".")[0] == "eval_context":
                key_name = val.split(".")[1]
                prompt_mapping[key] = eval_context[key_name]

        full_eval_prompt = self.eval_prompt.format(**prompt_mapping)

        content.append({"type": "text", "text": full_eval_prompt})
        return content

    def query(self, reference_info, response, images, question, eval_context=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        context = self.prepare_eval_prompt(
            reference_info, response, images, question, eval_context
        )

        query_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": context}],
            "temperature": 0,
        }

        response_data = None
        while response_data is None:
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=query_payload,
                )
                response_ = response.json()
            except (requests.exceptions.JSONDecodeError, requests.exceptions.ConnectionError) as e:
                logging.info(f'Error in requests: {e}')
                logging.info('Retry...')
                continue

            if "error" in response_:
                error_info = response_["error"]
                logging.info(
                    f"Got error with type: {error_info['type']}. Message: {error_info['message']}"
                )
                if (
                    error_info["message"]
                    == "Sorry! We've encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt."
                ):
                    logging.info(query_payload)
                    # If the model's response has too many repetitive tokens, then we give it a score of 0.
                    logging.error(f"gpt-4o judge query failed...")
                    return f"**Score explanation**: {error_info['message']}\n\n**Score**: 0"
                logging.info(f"Retry...")
            else:
                response_data = response_
                break

        total_tokens = response_data.get("usage", {}).get("total_tokens", "N/A")

        # Extracting the 'content' field from the response
        if response_data and "choices" in response_data:
            choices = response_data["choices"]
            if choices and "message" in choices[0]:
                message_content = choices[0]["message"]["content"]
                if self.print_response:
                    logging.info(
                        f"gpt-4o judge results: {message_content}; tokens:{total_tokens}"
                    )
        else:
            logging.error(f"gpt-4o judge query failed...")
            message_content = ""

        return message_content


class GPT4OJudgeScore:
    """Using GPT-4o as a adjuge to evaluate open-ended generation tasks"""

    def __init__(self, metric_config):
        self.model = OpenAIGPT4Judger(metric_config)

    def parse_results(self, eval_results):
        """
        This parsing function is based on the output prompt setting in the
        file "gpt4o_judge_prompt.json"
        """
        score_pattern = r"\*\*Score\*\*\s*:\s*(\d+)"
        explanation_pattern = r"\*\*Score explanation\*\*\s*:\s*(.*)"

        # Extract the score
        score_match = re.search(score_pattern, eval_results)
        score = int(score_match.group(1)) if score_match else None

        # Extract the score explanation
        explanation_match = re.search(explanation_pattern, eval_results, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        info_str = f"Score: {score}; Explanation: {explanation}"
        if score is None:
            return 0, f"Score is NULL: {eval_results};"

        return score / 10.0, info_str

    def match(
        self, response, reference_dict, images, question, eval_context=None
    ) -> int:
        eval_results = self.model.query(
            reference_dict, response, images, question, eval_context
        )
        score = self.parse_results(eval_results)
        return score
