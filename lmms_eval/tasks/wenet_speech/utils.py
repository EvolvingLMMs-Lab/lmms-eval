import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger as eval_logger

import lmms_eval.tasks._task_utils.file_utils as file_utils
from lmms_eval.llm_judge import ServerConfig, get_server

API_TYPE = os.getenv("API_TYPE", "openai")
# Use JUDGE_MODEL_VERSION instead of MODEL_VERSION
JUDGE_MODEL_VERSION = os.getenv("JUDGE_MODEL_VERSION", "gpt-4o-mini")

server_config = ServerConfig(
    model_name=JUDGE_MODEL_VERSION,
)
server = get_server(server_name=API_TYPE, config=server_config)


def get_column_value(doc, candidates):
    for candidate in candidates:
        if candidate in doc and doc[candidate] is not None:
            return doc[candidate]
    return ""


def tokenize(text):
    tokens = []
    i = 0
    while i < len(text):
        char = text[i]
        if "\u4e00" <= char <= "\u9fff":
            tokens.append(char)
            i += 1
        else:
            match = re.match(r"[a-zA-Z']+\w*", text[i:])
            if match:
                tokens.append(match.group(0))
                i += match.end()
            else:
                i += 1
    return tokens


def levenshtein_distance(ref_tokens, hyp_tokens):
    m, n = len(ref_tokens), len(hyp_tokens)
    if m == 0 and n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compute_mer(ref, hyp):
    ref_tokens = tokenize(ref)
    hyp_tokens = tokenize(hyp)
    distance = levenshtein_distance(ref_tokens, hyp_tokens)
    max_len = max(len(ref_tokens), len(hyp_tokens))
    return distance / max_len if max_len > 0 else 0.0


def process_opus_audio(audio_data, target_sample_rate=16000):
    try:
        if isinstance(audio_data, dict) and "array" in audio_data and "sampling_rate" in audio_data:
            current_sr = audio_data["sampling_rate"]
            audio_array = audio_data["array"]

            if current_sr != target_sample_rate:
                try:
                    import librosa

                    audio_array = librosa.resample(audio_array, orig_sr=current_sr, target_sr=target_sample_rate)
                except ImportError:
                    eval_logger.warning("librosa not available for resampling, using original sample rate")
                except Exception as e:
                    eval_logger.warning(f"Resampling failed: {e}, using original audio")

            return {"array": audio_array, "sampling_rate": target_sample_rate}

        return audio_data
    except Exception as e:
        eval_logger.error(f"Error processing opus audio: {e}")
        return audio_data


def wenet_speech_doc_to_audio(doc):
    audio_file = get_column_value(doc, ["audio"])

    if audio_file:
        try:
            decoded_audio = audio_file.get_all_samples()
            audio_array = decoded_audio.data

            audio_array = audio_array.cpu().numpy()
            sampling_rate = 16000
            sampling_rate = decoded_audio.sample_rate

            audio_dict = {"array": audio_array, "sampling_rate": sampling_rate}
            audio_dict = process_opus_audio(audio_dict)
            return [audio_dict]
        except Exception as e:
            print(f"Error converting AudioDecoder object: {e}")
            print(f"AudioDecoder type: {type(audio_file)}")
            print(f"AudioDecoder attributes: {dir(audio_file)}")
            return []
    else:
        print(f"Warning: No audio file found in document. Available keys: {list(doc.keys())}")
        return []


def wenet_speech_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    default_prompt = "Please listen to the audio and transcribe what you hear. Please only provide the transcription without any additional commentary. Do not include any punctuation."

    return f"{pre_prompt}{default_prompt}{post_prompt}"


def wenet_speech_process_results(doc, results):
    if not results:
        return {"mer": 100.0, "accuracy": 0.0}

    reference_text = get_column_value(doc, ["text"])

    if not reference_text:
        eval_logger.warning("No reference transcription found for ASR evaluation")
        return {"mer": 100.0, "accuracy": 0.0}

    hypothesis_list = []
    reference_list = []

    if isinstance(results, str):
        results = [results]
    if isinstance(reference_text, str):
        reference_text = [reference_text]

    for result in results:
        hypothesis = str(result).strip() if result is not None else ""
        hypothesis_list.append(hypothesis)

    for ref in reference_text:
        reference_list.append(str(ref).strip())

    min_len = min(len(hypothesis_list), len(reference_list))
    hypothesis_list = hypothesis_list[:min_len]
    reference_list = reference_list[:min_len]

    if not hypothesis_list or not reference_list:
        return {"mer": 100.0, "accuracy": 0.0}

    mer_scores = []
    for ref, hyp in zip(reference_list, hypothesis_list):
        mer = compute_mer(ref, hyp)
        mer_scores.append(mer)

    avg_mer = sum(mer_scores) / len(mer_scores) if mer_scores else 1.0
    avg_mer_percent = avg_mer * 100

    accuracy = max(0, (1 - avg_mer) * 100)

    eval_logger.info(f"ASR Evaluation - MER: {avg_mer_percent:.2f}%, Accuracy: {accuracy:.2f}%")

    return {"mer": avg_mer_percent, "accuracy": accuracy, "mer_raw": avg_mer}
