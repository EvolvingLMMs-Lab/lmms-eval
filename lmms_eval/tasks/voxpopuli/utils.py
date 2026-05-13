import os
import re
from collections import defaultdict

import numpy as np
from loguru import logger as eval_logger

# Language code mapping (voxpopuli uses integer codes)
LANGUAGE_MAP = {0: "en", 1: "de", 2: "fr", 3: "es", 4: "pl", 5: "it", 6: "ro", 7: "hu", 8: "cs", 9: "nl", 10: "fi", 11: "hr", 12: "sk", 13: "sl", 14: "et", 15: "lt"}


def _fallback_silent_audio(sampling_rate: int = 16000):
    return [{"array": np.zeros(sampling_rate, dtype=np.float32), "sampling_rate": sampling_rate}]


def get_column_value(doc, candidates):
    """Helper function to get value from document with multiple possible column names"""
    for candidate in candidates:
        if candidate in doc and doc[candidate] is not None:
            return doc[candidate]
    return ""


def voxpopuli_doc_to_audio(doc):
    """Extract audio from VoxPopuli dataset document

    VoxPopuli dataset structure:
    {
      'audio': {
        'path': '/path/to/audio.wav',
        'array': array([...], dtype=float32),
        'sampling_rate': 16000
      },
      ...
    }
    """
    audio_data = doc.get("audio")

    if not audio_data:
        eval_logger.warning(f"No audio found in document. Available keys: {list(doc.keys())}")
        return _fallback_silent_audio()

    try:
        # VoxPopuli audio is already a dict with 'array' and 'sampling_rate'
        if isinstance(audio_data, dict):
            audio_array = audio_data.get("array")
            sampling_rate = audio_data.get("sampling_rate", 16000)
        else:
            # If it's an AudioDecoder object
            decoded_audio = audio_data.get_all_samples() if hasattr(audio_data, "get_all_samples") else audio_data

            # Extract array - check for data attribute first (AudioSamples object from torchcodec)
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

            sampling_rate = getattr(audio_data, "_desired_sample_rate", 16000)

        # Convert torch tensor to numpy if needed
        if hasattr(audio_array, "cpu") and hasattr(audio_array, "numpy"):
            audio_array = audio_array.cpu().numpy()
        elif hasattr(audio_array, "numpy"):
            audio_array = audio_array.numpy()

        # Ensure it's a numpy array
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)

        # Ensure it's 1D array (flatten if multi-channel)
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        if audio_array.size == 0:
            eval_logger.warning("Decoded audio is empty, using 1s silent fallback audio")
            return _fallback_silent_audio(int(sampling_rate) if sampling_rate else 16000)

        # Ensure float32 dtype
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        eval_logger.debug(f"Audio array shape: {audio_array.shape}, dtype: {audio_array.dtype}, sampling_rate: {sampling_rate}")

        return [{"array": audio_array, "sampling_rate": sampling_rate}]

    except Exception as e:
        eval_logger.error(f"Error extracting audio: {e}")
        eval_logger.error(f"Audio type: {type(audio_data)}, attributes: {dir(audio_data)}")
        return _fallback_silent_audio()


def voxpopuli_doc_to_text(doc, lmms_eval_specific_kwargs):
    """Generate prompt for the audio model"""
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    # Get language information
    language_code = doc.get("language", 0)
    language = LANGUAGE_MAP.get(language_code, "unknown")

    # Default prompt for speech recognition
    default_prompt = f"Please transcribe the following audio in {language}."

    return f"{pre_prompt}{default_prompt}{post_prompt}"


def voxpopuli_process_results_asr(doc, results):
    """
    Process results for Automatic Speech Recognition (ASR) task.
    Calculates Word Error Rate (WER) and stores language information for aggregation.
    """
    scores = []

    # Get ground truth
    ground_truth = get_column_value(doc, ["normalized_text", "text", "transcript"])
    if not ground_truth:
        eval_logger.warning("No ground truth text found in document")
        return {"wer": 1.0, "language": doc.get("language", 0)}

    ground_truth = ground_truth.strip().lower()  # VoxPopuli uses lowercase normalized text

    # Get language for later aggregation
    language_code = doc.get("language", 0)

    for pred in results:
        prediction = pred.strip() if isinstance(pred, str) else str(pred)
        prediction = prediction.strip()

        # Extract transcription from various formats
        prediction = extract_transcription(prediction)
        prediction = prediction.lower()

        # Calculate Word Error Rate
        wer = calculate_wer(ground_truth, prediction)
        scores.append(wer)

    avg_wer = sum(scores) / len(scores) if scores else 1.0

    # Return both WER and language for aggregation
    return {"wer": avg_wer, "language": language_code, "wer_by_language": {language_code: avg_wer}}


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

    # Dynamic programming
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
    wer = dp[n][m] / max(n, 1)  # Avoid division by zero
    return wer


def voxpopuli_aggregate_results_overall(results):
    """Aggregate overall WER results for VoxPopuli evaluation"""
    if not results:
        return 0.0

    total_count = len(results)

    # Calculate overall average WER
    if all(isinstance(r, (int, float)) for r in results):
        avg_wer = sum(results) / total_count
        eval_logger.info(f"VoxPopuli Overall WER: {avg_wer:.4f}")
        return avg_wer

    return 0.0


def voxpopuli_aggregate_results_by_language(results):
    """
    Aggregate WER results by language for VoxPopuli evaluation.
    Returns a dictionary with per-language WER scores.
    """
    if not results:
        return {}

    # Group results by language
    language_results = defaultdict(list)

    for result in results:
        if isinstance(result, dict) and "wer_by_language" in result:
            for lang_code, wer in result["wer_by_language"].items():
                language_results[lang_code].append(wer)
        elif isinstance(result, dict) and "language" in result and "wer" in result:
            language_results[result["language"]].append(result["wer"])

    # Calculate average WER per language
    language_wer = {}
    for lang_code, wer_scores in language_results.items():
        avg_wer = sum(wer_scores) / len(wer_scores)
        lang_name = LANGUAGE_MAP.get(lang_code, f"lang_{lang_code}")
        language_wer[lang_name] = avg_wer
        eval_logger.info(f"VoxPopuli WER for {lang_name}: {avg_wer:.4f} ({len(wer_scores)} samples)")

    # Also log overall average
    all_wers = [wer for wers in language_results.values() for wer in wers]
    if all_wers:
        overall_avg = sum(all_wers) / len(all_wers)
        eval_logger.info(f"VoxPopuli Overall Average WER: {overall_avg:.4f}")
        language_wer["overall"] = overall_avg

    return language_wer
