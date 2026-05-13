import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger as eval_logger
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_column_value(doc, candidates):
    """Helper function to get value from document with multiple possible column names"""
    for candidate in candidates:
        if candidate in doc and doc[candidate] is not None:
            return doc[candidate]
    return ""


def song_describer_doc_to_audio(doc):
    """Extract audio from song-describer dataset document

    Returns audio array and sampling rate (16kHz for song-describer).
    """
    audio_file = doc.get("audio_path") or doc.get("audio")

    if not audio_file:
        eval_logger.warning(f"No audio found in document. Available keys: {list(doc.keys())}")
        return []

    try:
        # Handle different audio formats
        if hasattr(audio_file, "get_all_samples"):
            decoded_audio = audio_file.get_all_samples()
        else:
            decoded_audio = audio_file

        # Extract array
        if hasattr(decoded_audio, "data"):
            audio_array = decoded_audio.data
        elif hasattr(decoded_audio, "array"):
            audio_array = decoded_audio.array
        elif hasattr(decoded_audio, "samples"):
            temp = decoded_audio.samples
            if hasattr(temp, "data"):
                audio_array = temp.data
            else:
                audio_array = temp
        else:
            audio_array = decoded_audio

        # Convert torch tensor to numpy if needed
        if hasattr(audio_array, "cpu") and hasattr(audio_array, "numpy"):
            audio_array = audio_array.cpu().numpy()
        elif hasattr(audio_array, "numpy"):
            audio_array = audio_array.numpy()

        # Ensure it's a numpy array
        if not isinstance(audio_array, np.ndarray):
            try:
                audio_array = np.array(audio_array)
            except (ValueError, TypeError) as e:
                eval_logger.error(f"Cannot convert audio to numpy array: {e}")
                if hasattr(audio_array, "tolist"):
                    audio_array = np.array(audio_array.tolist())
                else:
                    raise

        # Ensure it's 1D array (flatten if multi-channel)
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        # Ensure float32 dtype
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Get sampling rate (song-describer is 16kHz)
        sampling_rate = getattr(audio_file, "_desired_sample_rate", 16000)

        eval_logger.debug(f"Audio array shape: {audio_array.shape}, dtype: {audio_array.dtype}, sampling_rate: {sampling_rate}")

        return [{"array": audio_array, "sampling_rate": sampling_rate}]

    except Exception as e:
        eval_logger.error(f"Error extracting audio: {e}")
        eval_logger.error(f"Audio type: {type(audio_file)}, attributes: {dir(audio_file)}")
        return []


def song_describer_doc_to_text(doc, lmms_eval_specific_kwargs):
    """Generate prompt for music captioning"""
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    # Default prompt for music captioning
    default_prompt = "Listen to this music and describe what you hear. Include details about the instruments, genre, mood, and any distinctive musical elements."

    return f"{pre_prompt}{default_prompt}{post_prompt}"


# Load config for evaluation
with open(Path(__file__).parent / "song_describer_validation.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))


# Local Qwen model configuration
EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
eval_logger.info(f"Using local evaluation model: {EVAL_MODEL_NAME}")

# Global model and tokenizer (lazy loading)
_eval_model = None
_eval_tokenizer = None


def get_eval_model():
    """Lazy load evaluation model"""
    global _eval_model, _eval_tokenizer

    if _eval_model is None:
        eval_logger.info(f"Loading evaluation model: {EVAL_MODEL_NAME}")
        _eval_tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL_NAME, trust_remote_code=True)
        _eval_model = AutoModelForCausalLM.from_pretrained(EVAL_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
        eval_logger.info("Evaluation model loaded successfully")

    return _eval_model, _eval_tokenizer


eval_prompt = """
            [Music Description Task]
            Reference Description: {ground_truth}

            Model's Description: {model_response}

            [Evaluation Task]
            Rate the model's music description based on its alignment with the reference description, focusing on accuracy of musical elements, genre, mood, instruments, and overall coherence.
            
            Criteria: 
            - Assess if the model's description accurately captures the key musical elements mentioned in the reference
            - Consider the description of instruments, genre, mood, tempo, and style
            - Evaluate the coherence and naturalness of the description
            - Be critical on factual accuracy while allowing for reasonable paraphrasing
            
            Score Guidelines:
            Score 0: The description is completely inaccurate or irrelevant to the reference
            Score 1: The description shows minimal alignment, missing most key musical elements
            Score 2: The description recognizes some elements but misses or misrepresents significant aspects
            Score 3: The description aligns generally but lacks detail or has minor inaccuracies
            Score 4: The description is mostly accurate with good detail, closely following the reference
            Score 5: The description is highly accurate, detailed, and captures all key elements from the reference

            Your response should be formatted as follows:
            Explanation: (Provide a concise explanation of your rating, comparing the reference with the model's response. "The reference describes [XXX], while the model's description mentions [YYY]. I think ...")
            Rating: (int)
            
            There should be no more content after the rating. Only provide the explanation and the rating in your response."""


def get_eval(max_tokens: int, content: str):
    """Call local Qwen model for evaluation"""
    model, tokenizer = get_eval_model()

    messages = [
        {"role": "system", "content": "You are a professional music critic and evaluator. Provide objective and detailed assessments."},
        {"role": "user", "content": content},
    ]

    try:
        # Format messages using chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize and generate
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
            )

        # Decode only the generated part (excluding input)
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if response:
            return response, EVAL_MODEL_NAME

        eval_logger.warning("Empty response from evaluation model")
        return "", ""

    except Exception as e:
        eval_logger.error(f"Error during evaluation: {e}")
        return "", ""


def song_describer_process_results(doc, results):
    """Process results for song-describer captioning task"""
    pred = results[0]
    ground_truth_str = get_column_value(doc, ["text", "caption", "description"])

    if not ground_truth_str:
        eval_logger.warning("No ground truth text found in document")
        return {"eval_score": {"eval_answer": "", "model_name": ""}}

    content = eval_prompt.format(model_response=pred, ground_truth=ground_truth_str)

    eval_answer, model_name = get_eval(max_tokens=1024, content=content)

    return {
        "eval_score": {"eval_answer": eval_answer, "model_name": model_name},
    }


def song_describer_aggregate_results(results):
    """Aggregate evaluation scores from local model"""
    score = 0
    valid_count = 0

    for result in results:
        eval_answer = result["eval_answer"]

        if not eval_answer:
            continue

        try:
            match = re.search(r"Rating:\s*([0-5])\s*$", eval_answer)
            eval_score = match.group(1) if match else 0
            eval_score = float(eval_score)
            score += eval_score
            valid_count += 1
        except Exception as e:
            eval_logger.error(f"Error parsing eval_score: {e}")
            continue

    if valid_count == 0:
        eval_logger.error("No valid evaluation scores found")
        return 0.0

    return score / valid_count
