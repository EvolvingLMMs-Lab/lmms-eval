import json
import os
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import requests
import yaml
from loguru import logger as eval_logger
from lmms_eval.tasks.genai_rqa.prompt import (
    EVALUATION_SYSTEM_PROMPT,
    EVALUATION_USER_PROMPT,
    EVALUATION_RESPONSE_SCHEMA
)

NUM_SECONDS_TO_SLEEP = 5
RAG_METRICS = [
    "gpt_eval_rag_correctness",
    "gpt_eval_rag_richness",
    "gpt_eval_rag_completeness"
]

# Correctness label to score mapping
LABEL_TO_SCORE = {
    "Perfect": 1.0,
    "Acceptable": 0.5,
    "Missing": 0.0,
    "Incorrect": -1.0
}

# 載入配置文件
with open(Path(__file__).parent / "genai_rqa.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))
GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]
API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }
    
def get_eval(content: str, max_tokens: int, retries: int = 5):
    """Call API for evaluation using structured prompts"""
    global headers
    messages = [
        {
            "role": "system",
            "content": EVALUATION_SYSTEM_PROMPT
        },
        {"role": "user", "content": content},
    ]
    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": min(max_tokens, 16384),  # Output limit max_tokens to leave room for input (input + output )
        "response_format": EVALUATION_RESPONSE_SCHEMA
    }
    if API_TYPE == "azure":
        payload.pop("model")
        payload.pop("response_format")  # Azure OpenAI might not support this yet
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            content = json.loads(content)
            if content != "":
                return content, response_data["model"]
            break
        except Exception as e:
            if isinstance(e, requests.exceptions.HTTPError):
                print(f"Response error details: {response.text}")  # Print full error response
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries:
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:
                eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
                return "", ""
    return "", ""
    
def parse_evaluation(review):
    """Parse the evaluation response"""
    try:
        return {
            'correctness': LABEL_TO_SCORE[review['correctness_label']],
            'richness': float(review['richness_score']),
            'completeness': float(review['completeness_score']),
            'reasoning': review['reasoning_steps'],
            'comments': review['output_comments'],
            'richness_comments': review['richness_comments'],
            'completeness_comments': review['completeness_comments']
        }
    except Exception as e:
        eval_logger.debug(f"Error parsing evaluation: {e}. Returning default scores")
        return {
            'correctness': 0.0,
            'richness': 1.0,
            'completeness': 1.0,
            'reasoning': f"Error parsing evaluation: {str(e)}",
            'comments': "Failed to parse evaluation response",
            'richness_comments': "Error occurred while parsing richness evaluation",
            'completeness_comments': "Error occurred while parsing completeness evaluation"
        }
    
def genai_rqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """將文檔轉換為文本格式"""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"

def genai_rqa_process_results(doc, results):
    """Process evaluation results using LLM-as-judge approach"""
    try:
        model_response = results[0].strip()
        
        question = doc.get("question", "")
        context = doc.get("context", "")
        expected_answer = doc.get("answer", "")

        # Format content using prompt from prompt.py
        content = EVALUATION_USER_PROMPT.format(
            chunk_text=context,
            question=question,
            # expected_answer=expected_answer,
            generated_answer=model_response
        )
        
        # Get evaluation from LLM
        review, model_name = get_eval(content=content, max_tokens=16384)
        eval_results = parse_evaluation(review)
        
        # Create result dictionary
        result_dict = {
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": model_response,
            "review": review,
            "eval_model": model_name,
        }
        
        # Create metric-specific dictionaries
        data_dict = {}
        metrics_map = {
            "gpt_eval_rag_correctness": ("correctness", eval_results['correctness']),
            "gpt_eval_rag_richness": ("richness", eval_results['richness']),
            "gpt_eval_rag_completeness": ("completeness", eval_results['completeness'])
        }
        
        for metric, (score_key, score) in metrics_map.items():
            metric_dict = deepcopy(result_dict)
            metric_dict["score_type"] = score_key
            metric_dict["score"] = score
            data_dict[metric] = metric_dict
            
        return data_dict
        
    except Exception as e:
        eval_logger.error(f"Error processing results: {e}")
        error_dict = {
            "score": -1,
            "error": str(e)
        }
        return {metric: error_dict for metric in RAG_METRICS}

def genai_rqa_metric_aggregation(results, metric_key):
    """Aggregate results for a specific metric"""
    try:
        scores = [result["score"] for result in results if "score" in result]
        if not scores:
            return None
        return round(float(np.mean(scores)), 3)
    except Exception as e:
        eval_logger.info(f"Error in {metric_key} aggregation: {e}")
        return None

def genai_rqa_correctness_aggregation(results):
    """Aggregate correctness scores"""
    return genai_rqa_metric_aggregation(results, "correctness")

def genai_rqa_richness_aggregation(results):
    """Aggregate richness scores"""
    return genai_rqa_metric_aggregation(results, "richness")

def genai_rqa_completeness_aggregation(results):
    """Aggregate completeness scores"""
    return genai_rqa_metric_aggregation(results, "completeness")
