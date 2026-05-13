import base64
import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path

import yaml
from loguru import logger as eval_logger

_default_template_path = Path(__file__).parent / "_default_template_yaml"
_browsecomp_config_path = Path(__file__).parent / "browsecomp.yaml"


def _load_yaml_stripped(path: Path) -> dict:
    with open(path, "r") as f:
        raw_data = f.readlines()
    safe_data = [line for line in raw_data if "!function" not in line]
    return yaml.safe_load("".join(safe_data)) or {}


_config = _load_yaml_stripped(_default_template_path)
_config.update(_load_yaml_stripped(_browsecomp_config_path))

_judge_server = None
_judge_server_config = None
if _config.get("metadata", {}).get("use_lmms_judge"):
    try:
        from lmms_eval.llm_judge import get_server
        from lmms_eval.llm_judge.protocol import ServerConfig

        API_TYPE = os.getenv("API_TYPE", "openai").lower()
        DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME") or os.getenv("OPENAI_API_MODEL", "gpt-4o")

        _judge_server_config = ServerConfig(model_name=DEPLOYMENT_NAME)
        _judge_server = get_server(server_name=API_TYPE, config=_judge_server_config)
        eval_logger.info("Using LMMS judge server for BrowseComp task.")
    except Exception as err:
        eval_logger.warning("Failed to initialize LMMS judge for BrowseComp: {}", err)
        _judge_server = None
        _judge_server_config = None


BROWSECOMP_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct based on the [correct_answer].

[question]: {question}
[response]: {response}
[correct_answer]: {correct_answer}

Determine if the response's final answer matches the correct answer. Answer only "Correct" or "Incorrect".
"""

EXACT_ANSWER_PATTERN = re.compile(r"^\s*Exact\s*Answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
FALLBACK_ANSWER_PATTERN = re.compile(r"^\s*(?:Final\s+)?Answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
CONFIDENCE_PATTERN = re.compile(r"^\s*Confidence\s*:\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*$", re.IGNORECASE | re.MULTILINE)


def _derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def _get_password(doc: dict) -> str:
    canary = doc.get("canary", "")
    if canary is None:
        return ""

    password = str(canary)
    try:
        decoded = base64.b64decode(password, validate=True).decode()
        if decoded and decoded.isprintable():
            return decoded
    except Exception:
        pass

    return password


def _safe_decrypt(ciphertext_b64: str, password: str) -> str:
    try:
        return _decrypt(ciphertext_b64, password)
    except Exception as err:
        eval_logger.warning("Failed to decrypt BrowseComp field: {}", err)
        return ""


def _extract_exact_answer(response: str) -> str:
    if not response:
        return ""

    match = EXACT_ANSWER_PATTERN.search(response)
    if match:
        return match.group(1).strip()

    fallback_match = FALLBACK_ANSWER_PATTERN.search(response)
    if fallback_match:
        return fallback_match.group(1).strip()

    return response.strip()


def _extract_confidence(response: str):
    if not response:
        return None

    match = CONFIDENCE_PATTERN.search(response)
    if not match:
        return None

    try:
        score = float(match.group(1))
    except ValueError:
        return None

    return max(0.0, min(100.0, score))


def _normalize_for_exact_match(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _judge_is_correct(question: str, response: str, correct_answer: str):
    if _judge_server is None or _judge_server_config is None:
        return None

    try:
        from lmms_eval.llm_judge.protocol import Request

        submit_prompt = BROWSECOMP_JUDGE_PROMPT.format(
            question=question,
            response=response,
            correct_answer=correct_answer,
        )
        request = Request(messages=[{"role": "user", "content": submit_prompt}], config=_judge_server_config)
        judge_response_obj = _judge_server.evaluate(request)
        judge_result = str(judge_response_obj.content).strip().lower()

        if "incorrect" in judge_result:
            return False
        if "correct" in judge_result:
            return True
    except Exception as err:
        eval_logger.debug("BrowseComp LLM judge failed, fallback to exact match: {}", err)

    return None


def browsecomp_doc_to_visual(doc) -> list:
    return []


def browsecomp_doc_to_text(doc, lmms_eval_specific_kwargs=None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    password = _get_password(doc)
    question = _safe_decrypt(str(doc.get("problem", "")), password)

    prompt = (
        f"{question}\n\n"
        "Your response should be in the following format:\n"
        "Explanation: {your explanation for your final answer}\n"
        "Exact Answer: {your succinct, final answer}\n"
        "Confidence: {your confidence score between 0% and 100% for your answer}"
    )

    return f"{pre_prompt}{prompt}{post_prompt}"


def browsecomp_doc_to_target(doc) -> str:
    password = _get_password(doc)
    return _safe_decrypt(str(doc.get("answer", "")), password)


def browsecomp_process_results(doc, results) -> dict:
    response = results[0] if results else ""
    if not isinstance(response, str):
        response = str(response)

    password = _get_password(doc)
    question = _safe_decrypt(str(doc.get("problem", "")), password)
    ground_truth = _safe_decrypt(str(doc.get("answer", "")), password)

    extracted_answer = _extract_exact_answer(response)
    confidence_score = _extract_confidence(response)

    response_for_judge = extracted_answer if extracted_answer else response.strip()
    judge_decision = _judge_is_correct(
        question=question,
        response=response_for_judge,
        correct_answer=ground_truth,
    )

    if judge_decision is None:
        is_correct = _normalize_for_exact_match(extracted_answer) == _normalize_for_exact_match(ground_truth)
    else:
        is_correct = bool(judge_decision)

    return {
        "browsecomp_acc": {
            "question_topic": doc.get("problem_topic", "unknown"),
            "prediction": extracted_answer,
            "target": ground_truth,
            "confidence": confidence_score,
            "correct": float(is_correct),
        }
    }


def browsecomp_aggregate_results(results) -> float:
    if not results:
        eval_logger.warning("No BrowseComp samples were provided for aggregation.")
        return 0.0

    topic_correct = defaultdict(float)
    topic_total = defaultdict(int)
    total_correct = 0.0

    for sample in results:
        topic = str(sample.get("question_topic", "unknown"))
        correct = float(sample.get("correct", 0.0))
        topic_total[topic] += 1
        topic_correct[topic] += correct
        total_correct += correct

    total_count = len(results)
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

    eval_logger.info("BrowseComp overall accuracy: {:.4f} ({}/{})", overall_accuracy, int(total_correct), total_count)
    for topic in sorted(topic_total):
        accuracy = topic_correct[topic] / topic_total[topic]
        eval_logger.info("BrowseComp topic [{}] accuracy: {:.4f} ({}/{})", topic, accuracy, int(topic_correct[topic]), topic_total[topic])

    return float(overall_accuracy)
