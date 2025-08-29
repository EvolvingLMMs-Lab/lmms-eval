import base64
import json
import os
import time
from copy import deepcopy
from io import BytesIO
from typing import Any, List, Tuple

import librosa
import numpy as np
import soundfile as sf
from accelerate import Accelerator, DistributedType
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from scipy import signal

    scipy_available = True
except ImportError:
    scipy_available = False

from loguru import logger as eval_logger

# File: lmms_eval/models/simple/gpt4o_audio.py

API_TYPE = os.getenv("API_TYPE", "openai")
NUM_SECONDS_TO_SLEEP = 10
if API_TYPE == "openai":
    API_URL = os.getenv(
        "OPENAI_API_URL",
        "https://api.openai.com/v1/chat/completions",
    )
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
elif API_TYPE == "azure":
    API_URL = os.getenv(
        "AZURE_ENDPOINT",
        "https://your-resource-name.openai.azure.com",
    )
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    API_VERSION = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")
else:
    raise ValueError(f"Unsupported API_TYPE '{API_TYPE}'. Expected 'openai' or 'azure'.")


@register_model("gpt4o_audio")
class GPT4OAudio(lmms):
    def __init__(
        self,
        model_version: str = "gpt-4o-audio-preview",
        modality: str = "audio",
        timeout: int = 120,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        audio_voice: str = "alloy",
        audio_format: str = "wav",
        **kwargs,
    ) -> None:
        super().__init__()

        if librosa is None or sf is None:
            raise ImportError("librosa and soundfile are required for GPT-4o audio. Please install with: pip install librosa soundfile")

        self.model_version = model_version
        self.modality = modality
        self.audio_token = "<audio>"  # Audio token placeholder
        self.timeout = timeout
        self.continual_mode = continual_mode
        self.audio_voice = audio_voice
        self.audio_format = audio_format

        if self.continual_mode:
            if response_persistent_folder is None:
                raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")

            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        if API_TYPE == "openai":
            self.client = OpenAI(api_key=API_KEY)
        elif API_TYPE == "azure":
            self.client = AzureOpenAI(api_key=API_KEY, azure_endpoint=API_URL, api_version=API_VERSION)

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

    def encode_audio(self, audio_input: Any, max_size_mb: float = 24.0) -> str:
        """
        Encode audio input into a base64-encoded WAV string.

        Accepts: file path, dict{array,sampling_rate}, numpy array, objects with
        array/sampling_rate attributes, path, bytes, or a callable returning such a dict.

        Args:
            audio_input: Audio data in various formats
            max_size_mb: Maximum size in MB for the encoded audio

        Returns:
            Base64-encoded WAV audio string
        """
        if isinstance(audio_input, dict):
            audio_array = audio_input["array"]
            sample_rate = audio_input.get("sampling_rate", 16000)
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}. Only HuggingFace dataset format (dict with 'array' and 'sampling_rate' keys) is supported.")

        if hasattr(audio_array, "dtype") and audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        elif not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)

        # Handle multi-channel audio by taking first channel
        if len(audio_array.shape) > 1:
            audio_array = audio_array[0] if audio_array.shape[0] < audio_array.shape[1] else audio_array[:, 0]

        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))

        # Compress audio if it's too long (reduce duration or sample rate)
        max_bytes = int(max_size_mb * 1024 * 1024 * 0.75)

        # Compression strategies
        compression_attempts = [
            {"sample_rate": sample_rate, "duration": None},  # Original
            {"sample_rate": 16000, "duration": None},  # Downsample to 16kHz
            {"sample_rate": 8000, "duration": None},  # Downsample to 8kHz
            {"sample_rate": 16000, "duration": 60},  # 16kHz + max 60 seconds
            {"sample_rate": 8000, "duration": 60},  # 8kHz + max 60 seconds
            {"sample_rate": 16000, "duration": 30},  # 16kHz + max 30 seconds
        ]

        for attempt in compression_attempts:
            try:
                if attempt["sample_rate"] != sample_rate and scipy_available:
                    target_length = int(len(audio_array) * attempt["sample_rate"] / sample_rate)
                    compressed_audio = signal.resample(audio_array, target_length)
                    compressed_sr = attempt["sample_rate"]
                elif attempt["sample_rate"] != sample_rate:
                    compressed_audio = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=attempt["sample_rate"])
                    compressed_sr = attempt["sample_rate"]
                else:
                    compressed_audio = audio_array.copy()
                    compressed_sr = sample_rate

                if attempt["duration"] is not None:
                    max_samples = int(compressed_sr * attempt["duration"])
                    if len(compressed_audio) > max_samples:
                        compressed_audio = compressed_audio[:max_samples]

                buffer = BytesIO()
                sf.write(buffer, compressed_audio, compressed_sr, format="WAV")
                audio_bytes = buffer.getvalue()

                if len(audio_bytes) <= max_bytes:
                    eval_logger.info(f"Audio compressed: {len(audio_array)/sample_rate:.1f}s@{sample_rate}Hz -> {len(compressed_audio)/compressed_sr:.1f}s@{compressed_sr}Hz ({len(audio_bytes)/(1024*1024):.2f}MB)")
                    break

            except Exception as e:
                eval_logger.warning(f"Compression attempt failed: {e}")
                if attempt["duration"] is not None:
                    max_samples = int(sample_rate * attempt["duration"])
                    if len(audio_array) > max_samples:
                        truncated_audio = audio_array[:max_samples]
                    else:
                        truncated_audio = audio_array
                else:
                    truncated_audio = audio_array

                buffer = BytesIO()
                sf.write(buffer, truncated_audio, sample_rate, format="WAV")
                audio_bytes = buffer.getvalue()

                if len(audio_bytes) <= max_bytes:
                    eval_logger.info(f"Audio truncated to {len(truncated_audio)/sample_rate:.1f}s ({len(audio_bytes)/(1024*1024):.2f}MB)")
                    compressed_audio = truncated_audio
                    compressed_sr = sample_rate
                    break

        else:
            eval_logger.warning(f"Could not compress audio below {max_size_mb}MB limit. Using truncated version.")

        buffer = BytesIO()
        sf.write(buffer, compressed_audio if "compressed_audio" in locals() else audio_array, compressed_sr if "compressed_sr" in locals() else sample_rate, format="WAV")
        audio_bytes = buffer.getvalue()

        if len(audio_bytes) == 0:
            raise ValueError("Generated audio bytes are empty")

        base64_str = base64.b64encode(audio_bytes).decode("utf-8")

        if not base64_str:
            raise ValueError("Base64 encoding resulted in empty string")

        eval_logger.debug(f"Encoded audio: {len(audio_bytes)} bytes -> {len(base64_str)} base64 chars")
        return base64_str

    def flatten(self, input_list):
        """Flatten nested lists."""
        new_list = []
        for i in input_list:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode is True and self.cache_mode == "resume":
                doc_uuid = f"{task}___{split}___{doc_id}"
                if doc_uuid in self.response_cache:
                    response_text = self.response_cache[doc_uuid]
                    # Ensure cached response is not None
                    if response_text is not None and response_text:
                        res.append(response_text)
                        pbar.update(1)
                        continue

            audios = [doc_to_visual(self.task_dict[task][split][doc_id])]
            if None in audios:
                audios = []
                encoded_audios = []
            else:
                audios = self.flatten(audios)
                encoded_audios = []

                for audio in audios:
                    try:
                        encoded_audio = self.encode_audio(audio, max_size_mb=20.0)
                        encoded_audios.append(encoded_audio)
                    except Exception as e:
                        eval_logger.warning(f"Failed to encode audio: {e}")
                        continue

            payload = {"messages": []}
            payload["model"] = self.model_version

            payload["modalities"] = ["text"]
            # GPT-4o Audio supports audio output also:
            # payload["audio"] = {"voice": self.audio_voice, "format": self.audio_format}

            user_content = []

            for encoded_audio in encoded_audios:
                if encoded_audio and len(encoded_audio) > 0:
                    user_content.append({"type": "input_audio", "input_audio": {"data": encoded_audio, "format": "wav"}})

            if contexts and contexts.strip():
                user_content.append({"type": "text", "text": contexts})

            if not user_content:
                eval_logger.warning("No audio or text content to send to API")
                res.append("")
                pbar.update(1)
                continue

            payload["messages"].append({"role": "user", "content": user_content})

            # Generation parameters
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if gen_kwargs["max_new_tokens"] > 4096:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = 0.95
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            payload["max_tokens"] = gen_kwargs["max_new_tokens"]
            payload["temperature"] = gen_kwargs["temperature"]
            if gen_kwargs.get("top_p") is not None:
                payload["top_p"] = gen_kwargs["top_p"]

            MAX_RETRIES = 5
            response_text = ""

            debug_payload = deepcopy(payload)
            if "messages" in debug_payload:
                for msg in debug_payload["messages"]:
                    if "content" in msg:
                        for content in msg["content"]:
                            if content.get("type") == "input_audio":
                                audio_data_size = len(content["input_audio"]["data"])
                                content["input_audio"]["data"] = f"[AUDIO_DATA_TRUNCATED_{audio_data_size}_BYTES]"
            # For debugging purposes
            # eval_logger.info(f"API payload structure: {debug_payload}")

            total_audio_size = 0
            for msg in payload["messages"]:
                if "content" in msg:
                    for content in msg["content"]:
                        if content.get("type") == "input_audio":
                            audio_size = len(content["input_audio"]["data"])
                            total_audio_size += audio_size

            # For debugging purposes
            # eval_logger.info(f"Total audio data size: {total_audio_size} bytes ({total_audio_size / (1024*1024):.2f} MB)")

            # Check if audio size is reasonable (OpenAI has limits)
            if total_audio_size > 20 * 1024 * 1024:  # 20MB limit (conservative)
                eval_logger.warning(f"Audio data size ({total_audio_size / (1024*1024):.2f} MB) may exceed API limits")

            for attempt in range(MAX_RETRIES):
                try:
                    # For debugging purposes
                    # eval_logger.info(f"Making API call attempt {attempt + 1}/{MAX_RETRIES}")
                    # eval_logger.info(f"Using model: {payload['model']}")
                    # eval_logger.info(f"API type: {API_TYPE}")

                    if "audio" not in payload["model"].lower():
                        eval_logger.warning(f"Model name '{payload['model']}' may not support audio. Consider using 'gpt-4o-audio-preview'")

                    response = self.client.chat.completions.create(**payload)

                    # For debugging purposes
                    # eval_logger.info(f"API response structure: {response}")
                    # eval_logger.info(f"Response choices: {response.choices}")
                    # eval_logger.info(f"First choice: {response.choices[0]}")
                    # eval_logger.info(f"Message: {response.choices[0].message}")
                    # eval_logger.info(f"Message content: {response.choices[0].message.content}")
                    # eval_logger.info(f"Message role: {response.choices[0].message.role}")

                    if hasattr(response.choices[0].message, "audio") and response.choices[0].message.audio:
                        eval_logger.info(f"Audio response detected: {response.choices[0].message.audio}")

                    response_text = response.choices[0].message.content
                    # For debugging purposes
                    # eval_logger.info(f"Response text: {response_text}")
                    # if response_text is None:
                    #     message = response.choices[0].message
                    #     if hasattr(message, 'audio') and message.audio:
                    #         eval_logger.info("Response contains audio, but we need text. Audio responses not supported yet.")
                    #         response_text = "[AUDIO_RESPONSE_NOT_SUPPORTED]"
                    #     elif hasattr(message, 'content') and isinstance(message.content, list):
                    #         text_parts = [part.get('text', '') for part in message.content if part.get('type') == 'text']
                    #         response_text = ' '.join(text_parts) if text_parts else ""
                    #     else:
                    #         response_text = ""
                    #     eval_logger.warning("API returned None response, using empty string")
                    #     eval_logger.warning(f"Full message object: {response.choices[0].message}")
                    # eval_logger.info(f"API call successful on attempt {attempt + 1}")
                    break

                except Exception as e:
                    error_msg = str(e)
                    eval_logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES} failed with error: {error_msg}")

                    if hasattr(e, "response") and hasattr(e.response, "text"):
                        eval_logger.info(f"Response body: {e.response.text}")
                    if hasattr(e, "response") and hasattr(e.response, "status_code"):
                        eval_logger.info(f"Status code: {e.response.status_code}")
                    if hasattr(e, "response") and hasattr(e.response, "headers"):
                        eval_logger.info(f"Response headers: {dict(e.response.headers)}")

                    if attempt == MAX_RETRIES - 1:
                        eval_logger.error(f"All {MAX_RETRIES} attempts failed. Last error: {error_msg}")
                        response_text = ""
                    else:
                        time.sleep(NUM_SECONDS_TO_SLEEP)

            res.append(response_text)
            pbar.update(1)

            if self.continual_mode is True and self.accelerator.is_local_main_process:
                doc_uuid = f"{task}___{split}___{doc_id}"
                cache_value = response_text if response_text is not None else ""
                self.response_cache[doc_uuid] = cache_value
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f, indent=4)
        pbar.close()
        return res

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("GPT4O-Audio not support")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4O-Audio not support"