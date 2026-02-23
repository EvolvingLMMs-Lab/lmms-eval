import os
import re

import numpy as np
from loguru import logger as eval_logger


def get_column_value(doc, candidates):
    """Helper function to get value from document with multiple possible column names"""
    for candidate in candidates:
        if candidate in doc and doc[candidate] is not None:
            return doc[candidate]
    return ""


def cn_college_mcq_doc_to_audio(doc):
    """Extract audio from CN College MCQ dataset

    Dataset uses 'context' field for audio.
    """
    audio_file = doc.get("context")

    if not audio_file:
        eval_logger.warning(f"No audio found in document. Available keys: {list(doc.keys())}")
        return []

    try:
        # Handle AudioDecoder type (like AMI dataset)
        if str(type(audio_file).__name__) == "AudioDecoder":
            decoded_audio = audio_file.get_all_samples()

            eval_logger.debug(f"decoded_audio type: {type(decoded_audio)}, type name: {type(decoded_audio).__name__}")

            # Extract array from AudioSamples or similar objects
            if hasattr(decoded_audio, "data"):
                audio_array = decoded_audio.data
                eval_logger.debug("Extracted audio from .data attribute")
            elif hasattr(decoded_audio, "samples"):
                audio_array = decoded_audio.samples
                eval_logger.debug("Extracted audio from .samples attribute")
            elif hasattr(decoded_audio, "array"):
                audio_array = decoded_audio.array
                eval_logger.debug("Extracted audio from .array attribute")
            else:
                audio_array = decoded_audio
                eval_logger.debug("Using decoded_audio directly")

            eval_logger.debug(f"audio_array type before conversion: {type(audio_array)}")

            # Convert torch tensor to numpy if needed
            if hasattr(audio_array, "cpu") and hasattr(audio_array, "numpy"):
                audio_array = audio_array.cpu().numpy()
                eval_logger.debug("Converted from torch tensor (cpu)")
            elif hasattr(audio_array, "numpy"):
                audio_array = audio_array.numpy()
                eval_logger.debug("Converted from tensor (numpy)")
            elif hasattr(audio_array, "detach"):
                audio_array = audio_array.detach().cpu().numpy()
                eval_logger.debug("Converted from torch tensor (detach)")

            # Ensure it's numpy array
            if not isinstance(audio_array, np.ndarray):
                try:
                    audio_array = np.array(audio_array, dtype=np.float32)
                    eval_logger.debug("Converted to numpy array")
                except Exception as e:
                    eval_logger.error(f"Failed to convert to numpy array: {e}")
                    eval_logger.error(f"audio_array type: {type(audio_array)}, value: {audio_array}")
                    return []

            # Flatten if multi-dimensional
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()
                eval_logger.debug(f"Flattened to shape: {audio_array.shape}")

            # Ensure float32 dtype
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
                eval_logger.debug(f"Converted to float32, dtype: {audio_array.dtype}")

            # Get sampling rate
            sampling_rate = 16000  # default
            if hasattr(decoded_audio, "sample_rate"):
                sampling_rate = decoded_audio.sample_rate
            elif hasattr(decoded_audio, "sampling_rate"):
                sampling_rate = decoded_audio.sampling_rate
            elif hasattr(audio_file, "_desired_sample_rate"):
                sampling_rate = audio_file._desired_sample_rate

            eval_logger.debug(f"Final audio shape: {audio_array.shape}, sampling_rate: {sampling_rate}")

            return [{"array": audio_array, "sampling_rate": sampling_rate}]

        # Handle dict-like audio (standard HF format)
        elif isinstance(audio_file, dict):
            if "array" in audio_file and "sampling_rate" in audio_file:
                return [audio_file]

        # Handle direct array
        elif isinstance(audio_file, (list, np.ndarray)):
            return [{"array": np.array(audio_file, dtype=np.float32), "sampling_rate": 16000}]

        eval_logger.warning(f"Unknown audio type: {type(audio_file)}")
        return []

    except Exception as e:
        eval_logger.error(f"Error extracting audio: {e}")
        eval_logger.error(f"Audio type: {type(audio_file)}, attributes: {dir(audio_file)}")
        return []


def cn_college_mcq_doc_to_text(doc, lmms_eval_specific_kwargs):
    """Generate prompt for the audio model"""
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    # Get question and choices
    instruction = doc.get("instruction", "")
    choices = doc.get("choices", "")

    # Build prompt
    prompt = f"{instruction}\n\n{choices}"

    return f"{pre_prompt}{prompt}{post_prompt}"


def cn_college_mcq_process_results(doc, results):
    """
    Process results for Chinese College Listening MCQ task.
    Extract the predicted answer and compare with ground truth.
    """

    def normalize(text):
        """Normalize text for comparison"""
        if not isinstance(text, str):
            text = str(text)
        return text.lower().strip()

    def extract_answer(response):
        """Extract answer from model response"""
        if not response:
            return None

        response = normalize(response)

        patterns = [
            r"answer\s+is\s+([ABCD])",
            r"answer:\s*([ABCD])",
            r"the\s+answer\s+is\s+([ABCD])",
            r"correct\s+answer\s+is\s+([ABCD])",
            r"correct\s+answer:\s*([ABCD])",
            r"choose\s+([ABCD])",
            r"option\s+([ABCD])",
            r"\(([ABCD])\)",
            r"^([ABCD])[\.,。]",
            r"^([ABCD])$",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Check if response starts with A/B/C/D
        for choice in ["A", "B", "C", "D"]:
            if response.startswith(choice.lower()):
                return choice

        return None

    # Get ground truth answer and extract the option letter
    ground_truth_raw = doc.get("answer", "").strip()
    # Extract option letter from ground truth (e.g., "(A) Find a place." -> "A")
    ground_truth = extract_answer(ground_truth_raw)

    # If extraction failed, try direct match
    if not ground_truth:
        ground_truth = ground_truth_raw.upper()

    # Extract predicted answer from first result
    pred = results[0] if results else ""
    predicted_answer = extract_answer(pred)

    # Calculate accuracy
    correct = 1 if predicted_answer and predicted_answer == ground_truth else 0

    # Calculate failure rate (unable to extract valid answer)
    failure = 1 if predicted_answer is None else 0

    eval_logger.debug(f"Ground truth raw: {ground_truth_raw}, extracted: {ground_truth}, Predicted: {predicted_answer}, Correct: {correct}")

    return {"accuracy": correct, "failure_rate": failure}


def cn_college_mcq_aggregate_results(results):
    """Aggregate results across all samples"""
    if not results:
        return 0.0

    total_count = len(results)
    correct_count = sum(results)
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    eval_logger.info(f"CN College MCQ evaluation: {correct_count}/{total_count} correct, accuracy: {accuracy:.4f}")

    return accuracy
