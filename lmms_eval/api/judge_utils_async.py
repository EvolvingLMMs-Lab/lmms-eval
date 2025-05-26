"""
Async judge utilities for high-throughput evaluation scenarios
"""

import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger as eval_logger

from .judge import JudgeConfig
from .judge_async import (
    AsyncJudgeFactory,
    JudgeRequest,
    evaluate_batch_async,
    evaluate_with_fallback,
)
from .judge_utils import JudgePromptBuilder, ResponseParser


class AsyncSimplifiedJudge:
    """Async simplified interface for common judge operations"""

    def __init__(self, model_name: Optional[str] = None, api_type: Optional[str] = None, api_key: Optional[str] = None, azure_endpoint: Optional[str] = None, api_version: Optional[str] = None, max_concurrent: int = 10, **config_kwargs):
        """Initialize async judge with optional configuration

        Args:
            model_name: Model name to use (defaults to MODEL_VERSION env var)
            api_type: API type ('openai', 'azure', 'anthropic', 'cohere', 'together')
            api_key: API key (defaults to appropriate env var based on api_type)
            azure_endpoint: Azure endpoint (only for Azure, defaults to AZURE_ENDPOINT env var)
            api_version: API version (only for Azure, defaults to API_VERSION env var)
            max_concurrent: Maximum number of concurrent requests
            **config_kwargs: Additional configuration parameters
        """
        # Get defaults from environment
        if api_type is None:
            api_type = os.getenv("API_TYPE", "openai")

        # Set environment variables if provided (for backward compatibility)
        if api_type == "azure":
            if azure_endpoint:
                os.environ["AZURE_ENDPOINT"] = azure_endpoint
            if api_key:
                os.environ["AZURE_API_KEY"] = api_key
            if api_version:
                os.environ["API_VERSION"] = api_version
        elif api_type == "openai":
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        elif api_type == "anthropic":
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
        elif api_type == "cohere":
            if api_key:
                os.environ["COHERE_API_KEY"] = api_key
        elif api_type == "together":
            if api_key:
                os.environ["TOGETHER_API_KEY"] = api_key

        self.config = JudgeConfig(model_name=model_name or os.getenv("MODEL_VERSION", "gpt-4o-2024-08-06"), **config_kwargs)
        self.api_type = api_type
        self.max_concurrent = max_concurrent
        self.judge = AsyncJudgeFactory.create_judge(api_type=api_type, config=self.config)
        self.judge.semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_binary_async(self, question: str, answer: str, prediction: str, output_format: str = "0/1", custom_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Asynchronously evaluate binary correctness"""
        # Build prompt
        prompt = JudgePromptBuilder.build_binary_prompt(question=question, answer=answer, prediction=prediction, output_format=output_format, custom_prompt=custom_prompt, **kwargs)

        # Create request
        request = JudgeRequest(messages=[{"role": "user", "content": prompt}], question=question, answer=answer, prediction=prediction, config=self.config)

        # Evaluate
        response = await self.judge.evaluate_async(request)

        # Parse result
        parsed_result = ResponseParser.parse_binary_response(response.content, output_format)

        return {"result": parsed_result, "raw_response": response.content, "model": response.model_used, "prompt": prompt, "success": response.success}

    async def evaluate_binary_batch_async(self, questions: List[str], answers: List[str], predictions: List[str], output_format: str = "0/1", custom_prompt: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Asynchronously evaluate multiple binary correctness tasks"""
        if not (len(questions) == len(answers) == len(predictions)):
            raise ValueError("All input lists must have the same length")

        tasks = []
        for q, a, p in zip(questions, answers, predictions):
            task = self.evaluate_binary_async(q, a, p, output_format, custom_prompt, **kwargs)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def evaluate_comparative_async(
        self, question: str, response1: str, response2: str, context: Optional[str] = None, score_range: Tuple[int, int] = (1, 10), custom_prompt: Optional[str] = None, images: Optional[List[Union[str, bytes]]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Asynchronously evaluate comparative responses"""
        # Build prompt
        prompt = JudgePromptBuilder.build_comparative_prompt(question=question, response1=response1, response2=response2, context=context, score_range=score_range, custom_prompt=custom_prompt, **kwargs)

        # Create request
        request = JudgeRequest(messages=[{"role": "user", "content": prompt}], question=question, response1=response1, response2=response2, context=context, images=images, config=self.config)

        # Evaluate
        response = await self.judge.evaluate_async(request)

        # Parse result
        scores = ResponseParser.parse_comparative_response(response.content)

        return {"scores": scores, "raw_response": response.content, "model": response.model_used, "prompt": prompt, "success": response.success}

    async def evaluate_comparative_batch_async(
        self,
        questions: List[str],
        responses1: List[str],
        responses2: List[str],
        contexts: Optional[List[Optional[str]]] = None,
        score_range: Tuple[int, int] = (1, 10),
        custom_prompt: Optional[str] = None,
        images_list: Optional[List[Optional[List[Union[str, bytes]]]]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Asynchronously evaluate multiple comparative response tasks"""
        if not (len(questions) == len(responses1) == len(responses2)):
            raise ValueError("Questions and responses lists must have the same length")

        if contexts is None:
            contexts = [None] * len(questions)
        if images_list is None:
            images_list = [None] * len(questions)

        tasks = []
        for q, r1, r2, ctx, imgs in zip(questions, responses1, responses2, contexts, images_list):
            task = self.evaluate_comparative_async(q, r1, r2, ctx, score_range, custom_prompt, imgs, **kwargs)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def evaluate_with_rubric_async(self, question: str, prediction: str, rubric: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Asynchronously evaluate with a custom rubric"""
        # Build rubric prompt
        rubric_text = "\n".join([f"- {k}: {v}" for k, v in rubric.items()])

        prompt = f"""Evaluate the following response according to the given rubric.

Question: {question}

Response: {prediction}

Rubric:
{rubric_text}

Provide a JSON response with scores for each rubric item."""

        # Create request with JSON response format
        config = JudgeConfig(model_name=self.config.model_name, response_format="json", temperature=self.config.temperature, max_tokens=self.config.max_tokens)

        request = JudgeRequest(messages=[{"role": "user", "content": prompt}], question=question, prediction=prediction, config=config)

        # Evaluate
        response = await self.judge.evaluate_async(request)

        # Parse JSON result
        parsed_result = ResponseParser.parse_json_response(response.content)

        return {"scores": parsed_result, "raw_response": response.content, "model": response.model_used, "prompt": prompt, "success": response.success}


# Convenience async functions
async def get_binary_judge_response_async(question: str, answer: str, prediction: str, model_name: Optional[str] = None, output_format: str = "0/1", api_type: Optional[str] = None, **kwargs) -> Union[int, bool]:
    """Quick async function to get binary judge response"""
    judge = AsyncSimplifiedJudge(model_name=model_name, api_type=api_type)
    result = await judge.evaluate_binary_async(question, answer, prediction, output_format, **kwargs)
    return result["result"]


async def get_comparative_scores_async(question: str, response1: str, response2: str, model_name: Optional[str] = None, api_type: Optional[str] = None, **kwargs) -> Tuple[float, float]:
    """Quick async function to get comparative scores"""
    judge = AsyncSimplifiedJudge(model_name=model_name, api_type=api_type)
    result = await judge.evaluate_comparative_async(question, response1, response2, **kwargs)
    return result["scores"]


async def evaluate_batch_with_progress(
    requests: List[Dict[str, Any]], eval_type: str = "binary", model_name: Optional[str] = None, api_type: Optional[str] = None, max_concurrent: int = 10, progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate a batch of requests with progress tracking

    Args:
        requests: List of request dictionaries
        eval_type: Type of evaluation ('binary' or 'comparative')
        model_name: Model to use
        api_type: API type to use
        max_concurrent: Maximum concurrent requests
        progress_callback: Optional callback function(completed, total)

    Returns:
        List of evaluation results
    """
    judge = AsyncSimplifiedJudge(model_name=model_name, api_type=api_type, max_concurrent=max_concurrent)

    results = []
    completed = 0
    total = len(requests)

    # Process in batches
    for i in range(0, total, max_concurrent):
        batch = requests[i : i + max_concurrent]
        batch_tasks = []

        for req in batch:
            if eval_type == "binary":
                task = judge.evaluate_binary_async(req["question"], req["answer"], req["prediction"], req.get("output_format", "0/1"), req.get("custom_prompt"))
            elif eval_type == "comparative":
                task = judge.evaluate_comparative_async(req["question"], req["response1"], req["response2"], req.get("context"), req.get("score_range", (1, 10)), req.get("custom_prompt"), req.get("images"))
            else:
                raise ValueError(f"Unknown eval_type: {eval_type}")

            batch_tasks.append(task)

        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)

        completed += len(batch)
        if progress_callback:
            progress_callback(completed, total)

    return results


# Example usage function
async def run_async_evaluation_example():
    """Example of how to use the async judge utilities"""
    # Single evaluation
    result = await get_binary_judge_response_async(question="What is 2+2?", answer="4", prediction="The answer is 4", api_type="openai")
    print(f"Single result: {result}")

    # Batch evaluation
    judge = AsyncSimplifiedJudge(api_type="openai", max_concurrent=5)

    questions = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]
    answers = ["4", "6", "8"]
    predictions = ["4", "6", "8"]

    results = await judge.evaluate_binary_batch_async(questions, answers, predictions)

    for i, result in enumerate(results):
        print(f"Question {i+1}: {result['result']}")

    # Evaluation with fallback
    request = JudgeRequest(messages=[{"role": "user", "content": "Evaluate if 2+2=4 is correct"}])

    response = await evaluate_with_fallback(request, primary_api="openai", fallback_apis=["anthropic", "together"])
    print(f"Fallback result: {response.content}")
