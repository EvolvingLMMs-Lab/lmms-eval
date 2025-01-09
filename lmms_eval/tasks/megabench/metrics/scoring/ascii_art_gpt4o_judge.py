"""Return if two ASCII art images depict the same thing."""

import logging
from numbers import Number
import os
import requests
from metrics.scoring.common.conversions import ascii_text_to_image
from models.OpenAI import OpenAI


class AsciiArtGPT4Judger(OpenAI):
    """A GPT-4o judge for assessing if two ASCII art images depict the same thing."""

    def __init__(self, verbose=True):
        self.eval_prompt = 'Determine if the following two ASCII art images depict the same object. Your answer should be either "yes" or "no", but without the quotation marks.'
        model = "gpt-4o-2024-08-06"
        super().__init__(
            os.getenv("OPENAI_API_KEY"),
            model,
            None,
            resize=False,
            print_response=verbose,
        )
        if os.getenv("MEGABENCH_OPEN_API_KEY") is not None:
            self.api_key = os.getenv("MEGABENCH_OPEN_API_KEY")
            self.url = os.getenv("MEGABENCH_OPEN_API_URL")
            if os.getenv("MEGABENCH_OPEN_API_MODEL") is not None:
                self.model = os.getenv("MEGABENCH_OPEN_API_MODEL")
            assert self.url, "You must set up the API URL for evaluating the Open tasks using your own API"

    def encode_image(self, image):
        """Encode an image into base64 and return its mime type."""
        mime_type = "image/jpeg"
        image_format = "JPEG"

        if image.mode == "RGBA":
            image = self._rgba_to_rgb(image)

        if self.resize and max(image.size) > self.max_side:
            image = self._resize_image(image)
            encoded_image = self._encode_image(image, image_format)
        else:
            encoded_image = self._encode_image(image, image_format)

        return encoded_image, mime_type

    def prepare_eval_prompt(self, images):
        """Prepare the evaluation prompt."""
        content = []
        for image_path in images:
            content.append(self.create_image_content(image_path))

        content.append({"type": "text", "text": self.eval_prompt})
        return content

    def query(self, images):
        """Query GPT4o to determine if the ASCII images show the same thing."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        context = self.prepare_eval_prompt(images)

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
            except (requests.exceptions.JSONDecodeError, requests.exceptions.ConnectionError) as e:
                logging.info(f'Error in requests: {e}')
                logging.info('Retry...')
                continue

            response_ = response.json()
            if "error" in response_:
                error_info = response_["error"]
                logging.info(
                    f"Got error with type: {error_info['type']}. Message: {error_info['message']}"
                )
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


judge = AsciiArtGPT4Judger()


class AsciiArtGPT4OJudge:
    """Compute the cosine similarity between two pieces of ASCII art."""

    @classmethod
    def match(cls, response, correct_answer) -> Number:
        """Compute the cosine similarity between two pieces of ASCII art."""
        if not isinstance(response, str) or not isinstance(correct_answer, str):
            return 0
        if not response:
            return 0
        response_image = ascii_text_to_image(response, 224, 224)
        correct_answer_image = ascii_text_to_image(correct_answer, 224, 224)

        eval_results = judge.query([response_image, correct_answer_image])
        return 1 if "yes" in eval_results.lower() else 0
