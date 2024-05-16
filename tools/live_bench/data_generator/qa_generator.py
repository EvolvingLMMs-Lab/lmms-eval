import io
import re
import os
import openai
import base64
import json
import random
import logging
import pathlib
import textwrap
import google.generativeai as genai
from time import sleep
from PIL import Image
from typing import List
from abc import ABC, abstractmethod
from lmms_eval.live_bench.data_generator.response import Response
from lmms_eval.live_bench.screen_shoter import ScreenImage
from lmms_eval.live_bench.data_generator.utils.gpt4v import format_gpt4v_images, gpt4v_generate_response
from lmms_eval.live_bench.data_generator.utils.gemini import gemini_generate_response

logger = logging.getLogger("lmms-eval")


class QAData(object):
    def __init__(self, question: str = None, answer: str = None, subtask: str = None):
        self.question = question
        self.answer = answer
        self.subtask = subtask

    def to_dict(self):
        return {"question": self.question, "answer": self.answer}


class QAGenerator(ABC):
    def __init__(self, prompt_file: str = os.path.join(os.path.dirname(__file__), "prompt.md")):
        self.prompt_file = prompt_file
        self.prompt = self._load_prompt()

    def _load_prompt(self):
        with open(self.prompt_file, "r") as f:
            return f.read()

    def __call__(self, images: ScreenImage, *args, **kwargs):
        return self.generate(images, *args, **kwargs)

    def generate(self, images: ScreenImage, *, test=False, **kwargs) -> Response:
        if test:
            return Response(success=True, content="This is a test response.", full_log={})
        return self._generate(images, **kwargs)

    def check(self, images: ScreenImage, question, answer, subtask, *, test=False, **kwargs) -> Response:
        if test:
            return Response(success=True, content="This is a test response.", full_log={})
        return self._check(images, question, answer, subtask, **kwargs)

    @abstractmethod
    def _generate(self, images: ScreenImage, **kwargs) -> Response:
        raise NotImplementedError("_generate not implemented")

    @abstractmethod
    def _check(self, images: ScreenImage, question, answer, subtask, **kwargs) -> Response:
        raise NotImplementedError("_check not implemented")

    def format_response(self, response: Response) -> QAData:
        if response.success:
            qa_data = self._format_response(response)
            if qa_data is None:
                return []
            else:
                return qa_data
        else:
            return []

    @abstractmethod
    def _format_response(self, response: Response) -> str:
        raise NotImplementedError("format_response not implemented")

    @abstractmethod
    def format_checked_response(self, response: Response) -> QAData:
        raise NotImplementedError("format_checked_response not implemented")

    def get_name(self) -> str:
        raise NotImplementedError("get_name not implemented")


class GeneratorRegistry:
    def __init__(self):
        self.generators = {}

    def register_generator(self, name):
        def decorator(cls):
            self.generators[name] = cls
            cls.get_name = lambda self: name
            return cls

        return decorator

    def get_generator(self, name) -> QAGenerator:
        return self.generators[name]

    def get_random_generator(self) -> QAGenerator:
        return random.choice(list(self.generators.values()))


generator_registry = GeneratorRegistry()


def register_generator(name):
    return generator_registry.register_generator(name)


def get_generator(name, *args, **kwargs) -> QAGenerator:
    return generator_registry.get_generator(name)(*args, **kwargs)


def get_random_generator(*args, **kwargs) -> QAGenerator:
    return generator_registry.get_random_generator()(*args, **kwargs)


@register_generator("gpt4v")
class GPT4Generator(QAGenerator):
    def __init__(
        self,
        prompt_file: str = os.path.join(os.path.dirname(__file__), "prompt.md"),
        model="gpt-4-turbo",
        example_path=os.path.join(os.path.dirname(__file__), "example"),
        check_prompt=os.path.join(os.path.dirname(__file__), "check_prompt.md"),
    ):
        super().__init__(prompt_file)
        API_KEY = os.getenv("OPENAI_API_KEY")
        if not API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.api_key = API_KEY
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        if os.path.exists(example_path):
            self.example_path = example_path
        else:
            self.example_path = None
        if os.path.exists(check_prompt):
            with open(check_prompt, "r") as f:
                self.check_prompt = f.read()
        else:
            self.check_prompt = check_prompt

    def format_messages(self, images: List[Image.Image], example_image: Image.Image, example_output: str):
        example = [
            format_gpt4v_images(example_image),
            {
                "type": "text",
                "text": example_output,
            },
        ]
        content = example + [format_gpt4v_images(image) for image in images]
        content.append(
            {
                "type": "text",
                "text": "Please generate high-quality questions focusing on the information displayed within this webpage. Your response should be in the format of the examples provided above and in JSON format.",
            },
        )
        messages = [
            {
                "role": "system",
                "content": self.prompt,
            },
            {
                "role": "user",
                "content": content,
            },
        ]
        return messages

    def _generate(self, images: ScreenImage, *, max_tokens=4096, max_try_times=5, **kwargs):
        if self.example_path:
            example_image_path = os.path.join(self.example_path, "example_website.png")
            example_output_path = os.path.join(self.example_path, "example_output.json")
            example_image = Image.open(example_image_path)
            with open(example_output_path, "r") as f:
                example_output = f.read()

        messages = self.format_messages(images.images, example_image, example_output)

        return gpt4v_generate_response(self.client, self.model, messages, max_tokens, max_try_times, **kwargs)

    def get_check_prompt(self, question: str, answer: str, subtask, images: List[Image.Image]):
        messages = [
            {
                "role": "system",
                "content": self.check_prompt,
            }
        ]
        content = []
        for img in images:
            content.append(format_gpt4v_images(img))
        content.append(
            {
                "type": "text",
                "text": f"Question: {question}\nQuestioner's Answer: {answer}\nSubtask: {subtask}",
            },
        )
        content.append(
            {
                "type": "text",
                "text": "Please rephrase or rewrite the high-quality question focusing on the information displayed within this webpage. Your response should be in the format of the examples provided above and in JSON format.",
            },
        )
        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )
        return messages

    def _check(self, images: ScreenImage, question, answer, subtask, *, max_tokens=4096, max_try_times=5, **kwargs):
        messages = self.get_check_prompt(question, answer, subtask, images.images)
        return gpt4v_generate_response(self.client, self.model, messages, max_tokens, max_try_times, **kwargs)

    def format_checked_response(self, response: Response):
        data = json.loads(response.content)
        question = data.get("question", None)
        answer = data.get("answer", None)
        subtask = data.get("subtask", None)
        return QAData(question=question, answer=answer, subtask=subtask)

    def _format_response(self, response: Response) -> List[QAData]:
        try:
            qa_data = []
            content = json.loads(response.content)
            for subtask, message in content.items():
                subtask = subtask.lower()
                message_lower = {k.lower(): v for k, v in message.items()}
                try:
                    question = message_lower["question"]
                    answer = message_lower["answer"]
                    qa_data.append(QAData(question=question, answer=answer, subtask=subtask))
                except KeyError as e:
                    logger.error(f"Failed to parse response: {message}")
                    logger.error(f"Error: {e}")
            return qa_data
        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return []


@register_generator("gemini")
class GeminiGenerator(QAGenerator):
    def __init__(
        self,
        prompt_file: str = os.path.join(os.path.dirname(__file__), "prompt.md"),
        model="gemini-pro-vision",
        example_path=os.path.join(os.path.dirname(__file__), "example"),
        check_prompt=os.path.join(os.path.dirname(__file__), "check_prompt.md"),
    ):
        super().__init__(prompt_file)
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=GOOGLE_API_KEY)

        self.api_key = GOOGLE_API_KEY
        self.model = model
        self.client = genai.GenerativeModel(model)
        if os.path.exists(example_path):
            self.example_path = example_path
        else:
            self.example_path = None
        if os.path.exists(check_prompt):
            with open(check_prompt, "r") as f:
                self.check_prompt = f.read()
        else:
            self.check_prompt = check_prompt

    def format_messages(self, images: List[Image.Image], example_image: Image.Image, example_output: str):
        content = [self.prompt, "\n", "Example Image:", example_image, "\n", "Example Output:", example_output]
        content.extend(images)
        content.append("Please generate high-quality questions focusing on the information displayed within this webpage. Your response should be in the format of the examples provided above and in JSON format.")
        return content

    def _generate(self, images: ScreenImage, *, max_tokens=4096, max_try_times=5, **kwargs):
        if self.example_path:
            example_image_path = os.path.join(self.example_path, "example_website.png")
            example_output_path = os.path.join(self.example_path, "example_output.json")
            example_image = Image.open(example_image_path)
            with open(example_output_path, "r") as f:
                example_output = f.read()

        messages = self.format_messages(images.images, example_image, example_output)

        return gemini_generate_response(self.client, messages, max_tokens, max_try_times, **kwargs)

    def get_check_prompt(self, question: str, answer: str, subtask, images: List[Image.Image]):
        content = [self.check_prompt] + images
        content.append(f"Question: {question}\nQuestioner's Answer: {answer}\nSubtask: {subtask}")
        content.append("Your response should be strictly in the below format:\n\nQuestion: <question>\nAnswer: <answer>\nSubtask: <subtask>")
        return content

    def _check(self, images: ScreenImage, question, answer, subtask, *, max_tokens=4096, max_try_times=5, **kwargs):
        messages = self.get_check_prompt(question, answer, subtask, images.images)
        return gemini_generate_response(self.client, messages, max_tokens, max_try_times, **kwargs)

    def format_checked_response(self, response: Response):
        # Extract the question, answer, and subtask from the normalized content
        question_match = re.search(r"question:\s*(.*?)\nAnswer:", response.content, re.IGNORECASE | re.DOTALL)
        answer_match = re.search(r"answer:\s*(.*?)\n(Subtask:|$)", response.content, re.IGNORECASE | re.DOTALL)
        subtask_match = re.search(r"subtask:\s*(.*)", response.content, re.IGNORECASE)

        question = answer = subtask = None

        if question_match:
            # Extract the matched groups
            question = question_match.group(1).strip()
        if answer_match:
            answer = answer_match.group(1).strip()
        if subtask_match:
            subtask = subtask_match.group(1).strip()

        return QAData(question=question, answer=answer, subtask=subtask)

    def _format_response(self, response: Response) -> List[QAData]:
        try:
            qa_data = []
            content = json.loads(response.content)
            for subtask, message in content.items():
                subtask = subtask.lower()
                message_lower = {k.lower(): v for k, v in message.items()}
                try:
                    question = message_lower["question"]
                    answer = message_lower["answer"]
                    qa_data.append(QAData(question=question, answer=answer, subtask=subtask))
                except KeyError as e:
                    logger.error(f"Failed to parse response: {message}")
                    logger.error(f"Error: {e}")
            return qa_data
        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return []
