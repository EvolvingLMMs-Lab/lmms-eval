import json
import os
from abc import ABC, abstractmethod
from typing import List

import anthropic
import google.generativeai as genai
import openai
from live_bench.data_generator.qa_generator import QAData
from live_bench.data_generator.utils.claude import (
    claude_generate_response,
    format_claude_images,
)
from live_bench.data_generator.utils.gemini import gemini_generate_response
from live_bench.data_generator.utils.gpt4v import (
    format_gpt4v_images,
    get_openai_client,
    gpt4v_generate_response,
)
from PIL import Image

REJECT_TO_ANSWER = "Reject to answer"


class AnswerGetter(ABC):
    @abstractmethod
    def get_answer(self, question: str, images: List[Image.Image]):
        raise NotImplementedError("get_answer not implemented")


class GPT4VAnswerGetter(AnswerGetter):
    def __init__(self, model: str = "gpt-4o", api_key=None):
        self.model = model
        self.client = get_openai_client()

    def get_answer(self, question: str, images: List[Image.Image]):
        messages = [{"role": "user", "content": format_gpt4v_images(images) + [{"type": "text", "text": question}]}]
        response = gpt4v_generate_response(messages=messages, client=self.client, model=self.model)
        if response.success:
            return response.content
        else:
            return REJECT_TO_ANSWER


class ClaudeAnswerGetter(AnswerGetter):
    def __init__(self, model: str = "claude-3-5-sonnet-20240620", api_key=None):
        self.model = model
        if api_key is None:
            self.api_key = os.getenv("ANTHROPIC_API_KEY", None)
        else:
            self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def get_answer(self, question: str, images: List[Image.Image]):
        messages = [{"role": "user", "content": format_claude_images(images) + [{"type": "text", "text": question}]}]
        response = claude_generate_response(self.client, self.model, messages)
        if response.success:
            return response.content
        else:
            return REJECT_TO_ANSWER


class GeminiAnswerGetter(AnswerGetter):
    def __init__(self, model: str = "gemini-1.5-pro", api_key=None):
        self.model = model
        self.client = genai.GenerativeModel(model)

    def get_answer(self, question: str, images: List[Image.Image]):
        response = gemini_generate_response(self.client, images + [question], max_tokens=2048)
        if response.success:
            return response.content
        else:
            return REJECT_TO_ANSWER


QUESTION_FINALIZER_PROMPT = """\
You are a question setter, and your task is to modify the following question, answer, and scoring criteria to ensure:

1. The question is clear and unambiguous.
2. The answer is correct and reasonable (although the original ground truth answer is mostly correct, it may not be perfect, and sometimes the answer maybe incorrect).
3. The scoring criteria are rational and facilitate the accurate assessment of responses.
4. The full score for the scoring criteria must be 10 points, and it must directly relate to the specific answer.
5. Except for some multiple-choice questions or other questions with only one possible answer, the scoring criteria should not be an all-or-nothing system. Partially correct answers should receive proportional points.
6. Ensure that the scoring system is flexible enough to accommodate slight variations in correct answers while still maintaining a standard for what is considered an acceptable answer.
7. Clearly define what constitutes a full score, partial score, and zero score.
8. The criteria should be as detailed as possible, even if a LLM without image understanding capabilities could still score the answer based on the criteria and ground truth answer correctly.

Your task is to finalize these standards, thus ensuring the correctness of the answer and the rationality of the scoring criteria.

Some tips:

1. For some extremely hard open-ended questions where answers may vary, hitting all points perfectly may not be realistic. In such cases, you can relax the criteria slightly. For example, if there are five possible points in an answer, but answering three adequately could merit full points. An other option is to change the question to a multiple-choice / multi-select question. But remember, it only applies to extremely hard open-ended questions which are impossible to answer perfectly.
2. For some questions, changing the format might be beneficial. You can consider transforming them into different types of questions such as essay, fill-in-the-blank, multiple-choice (or multiple-select), true/false, ranking (e.g., based on time, importance, etc.), or matching questions to enhance the difficulty and rationality of the scoring criteria.
"""

FINALIZER_OUTPUT_FORMAT_PROMPT = """\
Please provide the final question, answer, and scoring criteria in the following json format:
{
    "question": "The final question",
    "answer": "The final answer",
    "criteria": "The final scoring criteria"
}

One thing as a reminder is that if you want to add a new line in the json string, you should use the escape character "\\n" instead to represent the new line.
"""


class QuestionFinalizer(object):
    def __init__(self, gpt4v_model: str = "gpt-4o", claude_model: str = "claude-3-5-sonnet-20240620", gemini_model: str = "gemini-1.5-pro"):
        self.models = {"GPT4V": GPT4VAnswerGetter(gpt4v_model), "Claude": ClaudeAnswerGetter(claude_model), "Gemini": GeminiAnswerGetter(gemini_model)}
        self.client = get_openai_client()
        # self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", None))
        # self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", None))

    def finalize_question(self, question, answer, criteria, images: List[Image.Image]):
        information = [f"[Original Question]\n{question}", f"[Original Answer]\n{answer}", f"[Original Criteria]\n{criteria}"]
        information.append(
            "Below are answers from three candidates for reference. These answers may not be correct but are reasonably credible (but mostly correct). If any candidate rejects to answer, consider whether there is an issue with the question (such as containing violent or graphic content, or having a clear political bias). If so, please make necessary modifications. For open-ended questions, also consider the reasonableness of these answers. If they are reasonable, you may need to adjust the scoring criteria or the answer itself."
        )
        for model_name, model in self.models.items():
            information.append(f"[{model_name} Answer]\n{model.get_answer(question, images)}")
        information.append(FINALIZER_OUTPUT_FORMAT_PROMPT)
        prompt = "\n\n".join(information)
        messages = [{"role": "user", "content": format_gpt4v_images(images) + [{"type": "text", "text": prompt}]}]
        try:
            response = gpt4v_generate_response(client=self.client, model="gpt-4o", messages=messages, system=QUESTION_FINALIZER_PROMPT)
            if response.success:
                data = json.loads(response.content)
                return {
                    "question": data["question"],
                    "answer": data["answer"],
                    "criteria": data["criteria"],
                }
        except Exception as e:
            print(f"Failed to generate response: {e}")
            return None
