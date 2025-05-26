"""
Example integration of async judges with existing evaluation tasks
This shows how to use the async judge interface for high-throughput evaluation
"""

import asyncio
import time
from typing import Dict, List, Optional

from loguru import logger as eval_logger

from .judge import JudgeConfig
from .judge_async import AsyncJudgeFactory, evaluate_batch_async
from .judge_utils_async import AsyncSimplifiedJudge, evaluate_batch_with_progress


class HighThroughputEvaluator:
    """Example class showing how to integrate async judges for high-throughput evaluation"""

    def __init__(self, api_type: str = "openai", model_name: str = "gpt-4o-2024-08-06", max_concurrent: int = 20, use_fallback: bool = True):
        self.api_type = api_type
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.use_fallback = use_fallback

        # Initialize primary judge
        self.judge = AsyncSimplifiedJudge(api_type=api_type, model_name=model_name, max_concurrent=max_concurrent)

        # Initialize fallback judges if enabled
        if use_fallback:
            self.fallback_judges = {
                "anthropic": AsyncSimplifiedJudge(api_type="anthropic", model_name="claude-3-opus-20240229", max_concurrent=max_concurrent),
                "together": AsyncSimplifiedJudge(api_type="together", model_name="meta-llama/Llama-2-70b-chat-hf", max_concurrent=max_concurrent),
            }

    async def evaluate_dataset_async(self, dataset: List[Dict], eval_type: str = "binary", batch_size: Optional[int] = None) -> List[Dict]:
        """
        Evaluate an entire dataset asynchronously

        Args:
            dataset: List of evaluation items
            eval_type: Type of evaluation ('binary' or 'comparative')
            batch_size: Optional batch size for processing

        Returns:
            List of evaluation results
        """
        if batch_size is None:
            batch_size = self.max_concurrent

        total_items = len(dataset)
        results = []
        start_time = time.time()

        eval_logger.info(f"Starting async evaluation of {total_items} items")

        # Process in batches
        for i in range(0, total_items, batch_size):
            batch = dataset[i : i + batch_size]
            batch_start = time.time()

            # Create evaluation tasks
            if eval_type == "binary":
                batch_results = await self.judge.evaluate_binary_batch_async(questions=[item["question"] for item in batch], answers=[item["answer"] for item in batch], predictions=[item["prediction"] for item in batch])
            elif eval_type == "comparative":
                batch_results = await self.judge.evaluate_comparative_batch_async(
                    questions=[item["question"] for item in batch], responses1=[item["response1"] for item in batch], responses2=[item["response2"] for item in batch], contexts=[item.get("context") for item in batch]
                )
            else:
                raise ValueError(f"Unknown eval_type: {eval_type}")

            results.extend(batch_results)

            batch_time = time.time() - batch_start
            completed = min(i + batch_size, total_items)
            eval_logger.info(f"Processed batch {i//batch_size + 1}: " f"{completed}/{total_items} items " f"({batch_time:.2f}s, {len(batch)/batch_time:.1f} items/s)")

        total_time = time.time() - start_time
        eval_logger.info(f"Completed evaluation: {total_items} items in {total_time:.2f}s " f"({total_items/total_time:.1f} items/s)")

        return results

    async def evaluate_with_retry_and_fallback(self, item: Dict, eval_type: str = "binary", max_retries: int = 3) -> Dict:
        """
        Evaluate a single item with retry and fallback logic

        Args:
            item: Evaluation item
            eval_type: Type of evaluation
            max_retries: Maximum retries per API

        Returns:
            Evaluation result
        """
        apis_to_try = [self.api_type]
        if self.use_fallback:
            apis_to_try.extend(self.fallback_judges.keys())

        last_error = None

        for api in apis_to_try:
            judge = self.judge if api == self.api_type else self.fallback_judges.get(api)
            if not judge:
                continue

            for attempt in range(max_retries):
                try:
                    if eval_type == "binary":
                        result = await judge.evaluate_binary_async(question=item["question"], answer=item["answer"], prediction=item["prediction"])
                    else:
                        result = await judge.evaluate_comparative_async(question=item["question"], response1=item["response1"], response2=item["response2"], context=item.get("context"))

                    result["api_used"] = api
                    result["attempts"] = attempt + 1
                    return result

                except Exception as e:
                    last_error = e
                    eval_logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {api}: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)  # Exponential backoff

            eval_logger.warning(f"All retries exhausted for {api}")

        # All APIs failed
        return {"success": False, "error": str(last_error), "api_used": "none", "attempts": max_retries * len(apis_to_try)}


# Example integration with existing evaluation pipeline
class AsyncJudgeIntegration:
    """Shows how to integrate async judges into existing evaluation pipelines"""

    @staticmethod
    async def convert_existing_evaluation(eval_function, dataset: List[Dict], model_name: Optional[str] = None, api_type: Optional[str] = None, max_concurrent: int = 10) -> List[Dict]:
        """
        Convert an existing synchronous evaluation to use async judges

        Args:
            eval_function: Existing evaluation function
            dataset: Dataset to evaluate
            model_name: Model to use
            api_type: API type
            max_concurrent: Max concurrent requests

        Returns:
            Evaluation results
        """
        judge = AsyncSimplifiedJudge(model_name=model_name, api_type=api_type, max_concurrent=max_concurrent)

        # Progress tracking
        completed = 0
        total = len(dataset)

        def progress_callback(done, total):
            nonlocal completed
            completed = done
            if done % 100 == 0 or done == total:
                eval_logger.info(f"Progress: {done}/{total} ({100*done/total:.1f}%)")

        # Convert dataset to request format
        requests = []
        for item in dataset:
            # Adapt based on your evaluation function's expected format
            request = {"question": item.get("question", ""), "answer": item.get("answer", ""), "prediction": item.get("prediction", ""), "output_format": "0/1"}
            requests.append(request)

        # Run async evaluation
        results = await evaluate_batch_with_progress(requests, eval_type="binary", model_name=model_name, api_type=api_type, max_concurrent=max_concurrent, progress_callback=progress_callback)

        return results


# Example usage
async def example_high_throughput_evaluation():
    """Example of high-throughput evaluation using async judges"""

    # Create sample dataset
    dataset = [{"question": f"What is {i} + {i}?", "answer": str(2 * i), "prediction": str(2 * i)} for i in range(100)]

    # Initialize evaluator
    evaluator = HighThroughputEvaluator(api_type="openai", max_concurrent=20, use_fallback=True)

    # Run evaluation
    results = await evaluator.evaluate_dataset_async(dataset, eval_type="binary", batch_size=50)

    # Analyze results
    correct = sum(1 for r in results if r.get("result") == 1)
    eval_logger.info(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")

    # Example with retry and fallback
    single_result = await evaluator.evaluate_with_retry_and_fallback(dataset[0], eval_type="binary")
    eval_logger.info(f"Single evaluation result: {single_result}")


# Integration helper for existing code
def run_async_evaluation(coro):
    """Helper to run async evaluation in synchronous code"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# Example of adapting existing synchronous code
def adapt_sync_to_async(sync_eval_function):
    """Decorator to convert synchronous evaluation to async"""

    async def async_wrapper(*args, **kwargs):
        # Extract evaluation parameters
        dataset = kwargs.get("dataset", [])
        model_name = kwargs.get("model_name")
        api_type = kwargs.get("api_type", "openai")

        # Use async evaluation
        judge = AsyncSimplifiedJudge(model_name=model_name, api_type=api_type, max_concurrent=20)

        # Run evaluations concurrently
        tasks = []
        for item in dataset:
            task = judge.evaluate_binary_async(question=item["question"], answer=item["answer"], prediction=item["prediction"])
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    return async_wrapper


if __name__ == "__main__":
    # Run example
    asyncio.run(example_high_throughput_evaluation())
