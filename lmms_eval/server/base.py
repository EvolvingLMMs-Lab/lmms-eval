import abc
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .protocol import Request, Response, ServerConfig
from .utils import JudgePromptBuilder, ResponseParser


class ServerInterface(abc.ABC):
    """Abstract base class for judge implementations"""

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig(model_name="gpt-4")

    @abc.abstractmethod
    def evaluate(self, request: Request) -> Response:
        """
        Evaluate the given request and return a response

        Args:
            request: JudgeRequest containing the evaluation context

        Returns:
            JudgeResponse with the evaluation result
        """
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the judge service is available"""
        pass

    def prepare_messages(self, request: Request) -> List[Dict[str, Any]]:
        """Prepare messages in the format expected by the API"""
        messages = request.messages.copy()

        # Add system prompt if configured
        if self.config.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        return messages

    def evaluate_binary(self, question: str, answer: str, prediction: str, output_format: str = "0/1", custom_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Evaluate binary correctness"""
        # Build prompt
        prompt = JudgePromptBuilder.build_binary_prompt(question=question, answer=answer, prediction=prediction, output_format=output_format, custom_prompt=custom_prompt, **kwargs)

        # Create request
        request = Request(messages=[{"role": "user", "content": prompt}], question=question, answer=answer, prediction=prediction, config=self.config)

        # Evaluate
        response = self.evaluate(request)

        # Parse result
        parsed_result = ResponseParser.parse_binary_response(response.content, output_format)

        return {"result": parsed_result, "raw_response": response.content, "model": response.model_used, "prompt": prompt, "success": response.success}

    def evaluate_comparative(
        self, question: str, response1: str, response2: str, context: Optional[str] = None, score_range: Tuple[int, int] = (1, 10), custom_prompt: Optional[str] = None, images: Optional[List[Union[str, bytes]]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Evaluate comparative responses"""
        # Build prompt
        prompt = JudgePromptBuilder.build_comparative_prompt(question=question, response1=response1, response2=response2, context=context, score_range=score_range, custom_prompt=custom_prompt, **kwargs)

        # Create request
        request = Request(messages=[{"role": "user", "content": prompt}], question=question, response1=response1, response2=response2, context=context, images=images, config=self.config)

        # Evaluate
        response = self.evaluate(request)

        # Parse result
        scores = ResponseParser.parse_comparative_response(response.content)

        return {"scores": scores, "raw_response": response.content, "model": response.model_used, "prompt": prompt, "success": response.success}

    def evaluate_with_rubric(self, question: str, prediction: str, rubric: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Evaluate with a custom rubric"""
        # Build rubric prompt
        rubric_text = "\n".join([f"- {k}: {v}" for k, v in rubric.items()])

        prompt = f"""Evaluate the following response according to the given rubric.

Question: {question}

Response: {prediction}

Rubric:
{rubric_text}

Provide a JSON response with scores for each rubric item."""

        request = Request(messages=[{"role": "user", "content": prompt}], config=self.config)

        # Evaluate
        response = self.evaluate(request)

        # Parse JSON result
        parsed_result = ResponseParser.parse_json_response(response.content)

        return {"scores": parsed_result, "raw_response": response.content, "model": response.model_used, "prompt": prompt, "success": response.success}


class AsyncServerInterface(ServerInterface):
    """Abstract base class for async judge implementations"""

    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(config)
        self.semaphore = asyncio.Semaphore(config.max_concurrent)

    @abc.abstractmethod
    async def evaluate_async(self, request: Request) -> Response:
        """
        Asynchronously evaluate the given request and return a response

        Args:
            request: JudgeRequest containing the evaluation context

        Returns:
            JudgeResponse with the evaluation result
        """
        pass

    async def evaluate_batch(self, requests: List[Request]) -> List[Response]:
        """
        Evaluate multiple requests concurrently

        Args:
            requests: List of JudgeRequests to evaluate

        Returns:
            List of JudgeResponses in the same order as requests
        """
        tasks = [self.evaluate_async(request) for request in requests]
        return await asyncio.gather(*tasks)

    def evaluate(self, request: Request) -> Response:
        """Synchronous wrapper for async evaluation"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.evaluate_async(request))
