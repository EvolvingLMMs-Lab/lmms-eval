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


def europal_asr_doc_to_audio(doc):
    """Extract audio from europal-asr dataset document

    Returns audio array and sampling rate (16kHz for europal-asr).
    """
    audio_file = doc.get("audio")

    if not audio_file:
        eval_logger.warning(f"No audio found in document. Available keys: {list(doc.keys())}")
        return []

    try:
        # Handle AudioDecoder type with get_all_samples() method
        if hasattr(audio_file, "get_all_samples"):
            decoded_audio = audio_file.get_all_samples()
        else:
            decoded_audio = audio_file

        # Extract array - check for data attribute first (AudioSamples object)
        if hasattr(decoded_audio, "data"):
            # AudioSamples object from torchcodec
            audio_array = decoded_audio.data
        elif hasattr(decoded_audio, "samples"):
            temp = decoded_audio.samples
            # Check if samples itself has a data attribute
            if hasattr(temp, "data"):
                audio_array = temp.data
            else:
                audio_array = temp
        elif hasattr(decoded_audio, "array"):
            audio_array = decoded_audio.array
        else:
            audio_array = decoded_audio

        # Convert torch tensor to numpy if needed
        if hasattr(audio_array, "cpu") and hasattr(audio_array, "numpy"):
            audio_array = audio_array.cpu().numpy()
        elif hasattr(audio_array, "numpy"):
            audio_array = audio_array.numpy()

        # Ensure it's a numpy array and flatten if needed
        if not isinstance(audio_array, np.ndarray):
            try:
                audio_array = np.array(audio_array)
            except (ValueError, TypeError) as e:
                eval_logger.error(f"Cannot convert audio to numpy array: {e}, type: {type(audio_array)}")
                # Try tolist() first if it exists
                if hasattr(audio_array, "tolist"):
                    audio_array = np.array(audio_array.tolist())
                else:
                    raise

        # Ensure it's 1D array (flatten if multi-channel)
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        # Ensure float32 dtype for librosa compatibility
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Get sampling rate (europal-asr is 16kHz)
        sampling_rate = getattr(audio_file, "_desired_sample_rate", 16000)

        eval_logger.debug(f"Audio array shape: {audio_array.shape}, dtype: {audio_array.dtype}, sampling_rate: {sampling_rate}")

        return [{"array": audio_array, "sampling_rate": sampling_rate}]

    except Exception as e:
        eval_logger.error(f"Error extracting audio: {e}")
        eval_logger.error(f"Audio type: {type(audio_file)}, attributes: {dir(audio_file)}")
        return []


def europal_asr_doc_to_text(doc, lmms_eval_specific_kwargs):
    """Generate prompt for the audio model"""
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    # Default prompt for speech recognition
    default_prompt = "Please transcribe the following audio. Only provide the transcription without any additional explanation or formatting."

    return f"{pre_prompt}{default_prompt}{post_prompt}"


def europal_asr_process_results_asr(doc, results):
    """
    Process results for Automatic Speech Recognition (ASR) task.
    Calculates Word Error Rate (WER) using a simple implementation.
    """
    scores = []

    # Get ground truth
    ground_truth = get_column_value(doc, ["text_verbatim", "transcript", "transcription"])
    if not ground_truth:
        eval_logger.warning("No ground truth text found in document")
        return {"wer": 1.0}

    ground_truth = ground_truth.strip().upper()

    for pred in results:
        prediction = pred.strip() if isinstance(pred, str) else str(pred)
        prediction = prediction.strip()

        # Extract transcription from various formats
        prediction = extract_transcription(prediction)
        prediction = prediction.upper()

        # Calculate Word Error Rate
        wer = calculate_wer(ground_truth, prediction)
        scores.append(wer)

    avg_wer = sum(scores) / len(scores) if scores else 1.0
    return {"wer": avg_wer}


def extract_transcription(text):
    """
    Extract transcription from various text formats.
    Handles XML tags, quotes, prefixes, etc.
    """
    if not isinstance(text, str):
        return str(text)

    text = text.strip()

    # Pattern 1: XML-style tags
    for tag in ["<answer>", "<response>", "<result>", "<transcription>", "<text>"]:
        closing_tag = tag.replace("<", "</")
        pattern = f"{re.escape(tag)}\\s*([\\s\\S]*?)\\s*{re.escape(closing_tag)}"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Pattern 2: "The original content of this audio is:" followed by text in quotes
    patterns = [
        r"(?:the\s+)?original\s+content\s+(?:of\s+)?(?:this\s+)?audio\s+is\s*:\s*['\"](.+?)['\"]\s*$",
        r"(?:the\s+)?(?:audio|speech)\s+(?:content|transcription|says)\s*:\s*['\"](.+?)['\"]\s*$",
        r"transcription\s*:\s*['\"](.+?)['\"]\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    # Pattern 3: Text enclosed in quotes (single or double)
    quote_patterns = [r"^['\"](.+?)['\"]$", r"['\"]([^'\"]{20,})['\"]"]  # Entire text in quotes  # Long text in quotes (at least 20 chars)

    for pattern in quote_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Pattern 4: Remove common prefixes
    prefixes_to_remove = [
        r"^(?:here\s+is\s+)?(?:the\s+)?transcription\s*:\s*",
        r"^(?:the\s+)?(?:audio|speech)\s+(?:says|contains)\s*:\s*",
        r"^(?:answer|response|result)\s*:\s*",
    ]

    for prefix in prefixes_to_remove:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)

    return text.strip()


def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.
    WER = (S + D + I) / N
    where:
    - S = number of substitutions
    - D = number of deletions
    - I = number of insertions
    - N = number of words in reference
    """
    # Split into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Build edit distance matrix
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)

    # Calculate WER
    if n == 0:
        return 0.0 if m == 0 else 1.0

    wer = dp[n][m] / n
    return wer
