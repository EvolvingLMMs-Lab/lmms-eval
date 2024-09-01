import json
import os
import random
from abc import ABC, abstractmethod
from typing import List

import anthropic
import openai
from live_bench.data_generator.qa_generator import Response
from live_bench.data_generator.utils.claude import (
    claude_generate_response,
    format_claude_images,
)
from live_bench.data_generator.utils.gpt4v import (
    format_gpt4v_images,
    gpt4v_generate_response,
)
from live_bench.screen_shoter import ScreenImage
from PIL import Image


class Score(object):
    def __init__(self, score: int, reason: str):
        self.score = score
        self.reason = reason


class ScoreGetter(ABC):
    def get_name(self):
        return self.name

    @abstractmethod
    def get_score(self, question: str, answer: str, images: ScreenImage):
        raise NotImplementedError("get_score not implemented")

    def __call__(self, question: str, answer: str, images: ScreenImage, **kwargs):
        return self.get_score(question, answer, images, **kwargs)


class ScoreGetterRegistry:
    def __init__(self):
        self.score_getters = {}

    def register_score_getter(self, name):
        def decorator(cls):
            self.score_getters[name] = cls
            cls.name = name
            return cls

        return decorator

    def get_score_getter(self, name) -> ScoreGetter:
        return self.score_getters[name]

    def get_random_score_getter(self) -> ScoreGetter:
        return random.choice(list(self.score_getters.values()))


generator_registry = ScoreGetterRegistry()


def register_score_getter(name):
    return generator_registry.register_score_getter(name)


def get_score_getter(name, *args, **kwargs) -> ScoreGetter:
    return generator_registry.get_score_getter(name)(*args, **kwargs)


def get_random_score_getter(*args, **kwargs) -> ScoreGetter:
    return generator_registry.get_random_score_getter()(*args, **kwargs)


@register_score_getter("gpt4v")
class GPT4VScoreGetter(ScoreGetter):
    def __init__(self, prompt: str = os.path.join(os.path.dirname(__file__), "score_prompt.md"), model="gpt-4o", example_path=os.path.join(os.path.dirname(__file__), "example")):
        super().__init__()
        if os.path.exists(prompt):
            with open(prompt, "r") as f:
                self.prompt = f.read()
        else:
            self.prompt = prompt
        API_KEY = os.getenv("OPENAI_API_KEY")
        if not API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.api_key = API_KEY
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        if os.path.exists(example_path) and os.path.isfile(os.path.join(example_path, "example_score_input.md")):
            with open(example_path, "r") as f:
                self.example = f.read()
        else:
            self.example = None

    def _format_prompt(self, question: str, answer: str, images: List[Image.Image]):
        prompt = [{"role": "system", "content": self.prompt}]
        messages = []
        for image in images:
            messages.append(format_gpt4v_images(image))
        messages.append({"type": "text", "text": f"Question: {question}\nQuestioner's Answer: {answer}"})
        messages.append({"type": "text", "text": 'You should format you answer into json format like this: {"reason": "some reason", "score": 10}'})
        prompt.append({"role": "user", "content": messages})
        return prompt

    def get_score(self, question: str, answer: str, images: ScreenImage, *, max_tokens=4096, max_try_times=5, **kwargs) -> Score:
        prompt = self._format_prompt(question, answer, images)
        try:
            response = gpt4v_generate_response(client=self.client, model=self.model, messages=prompt, max_tokens=max_tokens, max_try_times=max_try_times, json_format=True, **kwargs)
            if response.success:
                content = json.loads(response.content)
                score = content.get("score", None)
                reason = content.get("reason", None)
                return Score(score=score, reason=reason)
            else:
                return Score(score=None, reason=response.content)
        except Exception as e:
            return Score(score=None, reason=str(e))


@register_score_getter("claude")
class ClaudeScoreGetter(ScoreGetter):
    def __init__(self, prompt: str = os.path.join(os.path.dirname(__file__), "score_prompt.md"), model="claude-3-5-sonnet-20240620", example_path=os.path.join(os.path.dirname(__file__), "example")):
        super().__init__()
        if os.path.exists(prompt):
            with open(prompt, "r") as f:
                self.prompt = f.read()
        else:
            self.prompt = prompt
        API_KEY = os.getenv("ANTHROPIC_API_KEY")
        if not API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        self.api_key = API_KEY
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        if os.path.exists(example_path) and os.path.isfile(os.path.join(example_path, "example_score_input.md")):
            with open(example_path, "r") as f:
                self.example = f.read()
        else:
            self.example = None

    def _format_prompt(self, question: str, answer: str, images: List[Image.Image]):
        # prompt = [{"role": "system", "content": self.prompt}]
        prompt = []
        messages = []
        for image in images:
            messages.append(format_claude_images(image))
        messages.append({"type": "text", "text": f"Question: {question}\nQuestioner's Answer: {answer}"})
        messages.append({"type": "text", "text": 'You should format you answer into JSON format like this: { "reason": "some reason", "score": 10 }'})
        prompt.append({"role": "user", "content": messages})
        return prompt

    def get_score(self, question: str, answer: str, images: ScreenImage, *, max_tokens=4096, max_try_times=5, **kwargs) -> Score:
        prompt = self._format_prompt(question, answer, images)
        try:
            response = claude_generate_response(client=self.client, model=self.model, messages=prompt, system=self.prompt, max_tokens=max_tokens, max_try_times=max_try_times, **kwargs)
            if response.success:
                content = json.loads(response.content)
                score = content.get("score", None)
                reason = content.get("reason", None)
                return Score(score=score, reason=reason)
            else:
                return Score(score=None, reason=response.content)
        except Exception as e:
            return Score(score=None, reason=str(e))
