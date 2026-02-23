import os
import re
import string

import numpy as np
from loguru import logger as eval_logger

from lmms_eval.llm_judge import ServerConfig, get_server

API_TYPE = os.getenv("API_TYPE", "openai")
JUDGE_MODEL_VERSION = os.getenv("JUDGE_MODEL_VERSION", "gpt-4o-mini")

server_config = ServerConfig(
    model_name=JUDGE_MODEL_VERSION,
)
server = get_server(server_name=API_TYPE, config=server_config)


def get_column_value(doc, candidates):
    """Helper function to get value from document with multiple possible column names"""
    for candidate in candidates:
        if candidate in doc and doc[candidate] is not None:
            return doc[candidate]
    return ""


def remove_punctuation_except_apostrophe(text):
    """
    Remove all punctuation except apostrophe (') from text.
    Used for AMI ASR evaluation to normalize text before WER calculation.
    """
    # Create a translation table that removes all punctuation except apostrophe
    # string.punctuation contains: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    punctuation_to_remove = string.punctuation.replace("'", "")
    translator = str.maketrans("", "", punctuation_to_remove)
    return text.translate(translator)


def ami_doc_to_audio(doc):
    """Extract audio from AMI dataset document

    AMI dataset uses AudioDecoder type with get_all_samples() method.
    Returns audio array and sampling rate (16kHz for AMI).
    """
    audio_file = doc.get("audio")

    if not audio_file:
        eval_logger.warning(f"No audio found in document. Available keys: {list(doc.keys())}")
        return []

    try:
        # AMI uses AudioDecoder type with get_all_samples() method
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

        # Get sampling rate (AMI is 16kHz)
        sampling_rate = getattr(audio_file, "_desired_sample_rate", 16000)

        eval_logger.debug(f"Audio array shape: {audio_array.shape}, dtype: {audio_array.dtype}, sampling_rate: {sampling_rate}")

        return [{"array": audio_array, "sampling_rate": sampling_rate}]

    except Exception as e:
        eval_logger.error(f"Error extracting audio: {e}")
        eval_logger.error(f"Audio type: {type(audio_file)}, attributes: {dir(audio_file)}")
        # Re-raise to help debug
        import traceback

        eval_logger.error(f"Traceback: {traceback.format_exc()}")
        return []


def ami_doc_to_text(doc, lmms_eval_specific_kwargs):
    """Generate prompt for the audio model"""
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    # Get meeting context if needed
    meeting_id = get_column_value(doc, ["meeting_id"])
    speaker_id = get_column_value(doc, ["speaker_id"])

    # Default prompt for speech recognition
    default_prompt = "Please transcribe the following audio. Only provide the transcription without any additional explanation or formatting."

    return f"{pre_prompt}{default_prompt}{post_prompt}"


def ami_process_results_asr(doc, results):
    """
    Process results for Automatic Speech Recognition (ASR) task.
    Calculates Word Error Rate (WER) - case insensitive.
    """
    scores = []

    # Get ground truth
    ground_truth = get_column_value(doc, ["text", "transcript", "transcription"])
    if not ground_truth:
        eval_logger.warning("No ground truth text found in document")
        return {"wer": 1.0}

    # Normalize: strip and lowercase for case-insensitive comparison
    ground_truth = ground_truth.strip().lower()

    # Remove all punctuation except apostrophe
    ground_truth = remove_punctuation_except_apostrophe(ground_truth)

    for pred in results:
        prediction = pred.strip() if isinstance(pred, str) else str(pred)

        # Extract transcription from various formats
        prediction = extract_transcription(prediction)

        # Normalize: strip and lowercase for case-insensitive comparison
        prediction = prediction.strip().lower()

        # Remove all punctuation except apostrophe
        prediction = remove_punctuation_except_apostrophe(prediction)

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

    # Pattern 2: "The transcription of the audio is:" followed by text in quotes
    patterns = [
        r"(?:the\s+)?transcription\s+(?:of\s+)?(?:the\s+)?audio\s+is\s*:\s*['\"](.+?)['\"]\s*\.?\s*$",
        r"(?:the\s+)?original\s+content\s+(?:of\s+)?(?:this\s+)?audio\s+is\s*:\s*['\"](.+?)['\"]\s*\.?\s*$",
        r"(?:the\s+)?(?:audio|speech)\s+(?:content|transcription|says)\s*:\s*['\"](.+?)['\"]\s*\.?\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    # Pattern 3: Text enclosed in quotes (single or double)
    quote_patterns = [r"^['\"](.+?)['\"]\s*\.?\s*$", r"['\"]([^'\"]{20,})['\"]"]  # Entire text in quotes  # Long text in quotes (at least 20 chars)

    for pattern in quote_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Pattern 4: Remove common prefixes
    prefixes_to_remove = [
        r"^(?:here\s+is\s+)?(?:the\s+)?transcription\s*(?:of\s+(?:the\s+)?audio)?\s*:\s*",
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


def ami_process_results_llm_judge(doc, results):
    """
    Process results using LLM as judge for open-ended evaluation.
    Similar to voicebench's open-ended evaluation.
    """
    parsed_preds = []
    scores = []

    # Evaluation prompt template
    meta_prompt = """I need your help to evaluate the quality of an automatic speech recognition (ASR) transcription.

You will be provided with:
1. [Reference]: The ground truth transcription
2. [Hypothesis]: The model's transcription output

Please evaluate the hypothesis on a scale of 1 to 5 based on accuracy, completeness, and overall quality:
1 point: The transcription is largely incorrect with many missing or wrong words.
2 points: The transcription captures some words correctly but has significant errors.
3 points: The transcription is mostly accurate but has some errors or omissions.
4 points: The transcription is accurate with only minor errors.
5 points: The transcription is perfect or near-perfect with negligible errors.

Below are the reference and hypothesis transcriptions:
### [Reference]: {reference}
### [Hypothesis]: {hypothesis}

After evaluating, please output the score only without anything else.
You don't need to provide any explanations."""

    for pred in results:
        prediction = pred.strip() if isinstance(pred, str) else str(pred)

        # Extract answer from tags if present
        if isinstance(prediction, str):
            for tag in ["<answer>", "<response>", "<result>", "<transcription>"]:
                closing_tag = tag.replace("<", "</")
                pattern = f"{re.escape(tag)}\\s*([\\s\\S]*?)\\s*{re.escape(closing_tag)}"
                match = re.search(pattern, prediction)
                if match:
                    prediction = match.group(1).strip()
                    break

        # Get reference text
        reference_text = get_column_value(doc, ["text", "transcript", "transcription"])

        formatted_prompt = meta_prompt.format(reference=reference_text, hypothesis=prediction)

        try:
            from lmms_eval.llm_judge.protocol import Request, ServerConfig

            custom_config = ServerConfig(model_name=JUDGE_MODEL_VERSION, temperature=0.5, max_tokens=10)

            request = Request(messages=[{"role": "system", "content": "You are a helpful assistant who evaluates speech recognition quality."}, {"role": "user", "content": formatted_prompt}], config=custom_config)

            response = server.evaluate(request)

            if response.success:
                judge_response = response.content.strip()
                try:
                    judge_score = int(judge_response)
                except ValueError:
                    eval_logger.error(f"Failed to parse score: {judge_response}")
                    judge_score = 0.0
            else:
                eval_logger.error(f"Judge evaluation failed: {response.content}")
                judge_score = 0.0

        except Exception as e:
            eval_logger.error(f"Error getting judge response: {e}")
            judge_score = 0.0

        scores.append(judge_score)
        parsed_preds.append(prediction)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {"llm_as_judge_eval": avg_score}


def ami_aggregate_results(results):
    """Aggregate results for AMI evaluation"""
    if not results:
        return 0.0

    total_count = len(results)

    # If results are WER scores, return average
    if all(isinstance(r, (int, float)) for r in results):
        avg_score = sum(results) / total_count
        eval_logger.info(f"AMI evaluation: Average score: {avg_score:.4f}")
        return avg_score

    # If results are boolean (correct/incorrect), calculate accuracy
    correct_count = sum(results)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    eval_logger.info(f"AMI evaluation: {correct_count}/{total_count} correct, accuracy: {accuracy:.4f}")

    return accuracy
