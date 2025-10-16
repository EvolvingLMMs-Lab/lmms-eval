import base64
import os
from io import BytesIO
from typing import List, Tuple

import numpy as np
import requests
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from scipy.io import wavfile
from tqdm import tqdm
from transformers import AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.audio_processing import downsample_audio

# Model sampling rate
SAMPLING_RATE = 16_000


@register_model("whisper_tt")
class WhisperTT(lmms):
    """
    Whisper Audio Model - HTTP API Client
    
    This implementation uses HTTP calls to the tt-media-server instead of
    direct ttnn/tt-metal execution, allowing evals to run outside docker.
    """

    def __init__(
        self,
        pretrained: str = "openai/whisper-large-v3",
        device: str = "cuda",
        device_map: str = "cuda",
        batch_size: int = 1,
        use_cache: bool = True,
        language: str = "en",
        task: str = "transcribe",
        base_url: str = None,
        timeout: int = 300,
        max_retries: int = 3,
        num_concurrent: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        # Log warning for unexpected kwargs but don't fail
        if kwargs:
            eval_logger.warning(f"Ignoring unexpected kwargs: {kwargs}")
        
        # Get base URL from env var or argument
        self.base_url = base_url or os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8000")
        self.timeout = timeout
        self.max_retries = max_retries
        self.pretrained = pretrained
        
        # Get API key from environment
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        
        eval_logger.info(f"Initializing WhisperTT HTTP client with base_url: {self.base_url}")
        
        # Setup processor for tokenization
        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.processor.tokenizer.set_prefix_tokens(language=language, task=task)
        self._tokenizer = self.processor.tokenizer
        
        # Setup accelerator for distributed evaluation
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = f"cuda:{accelerator.local_process_index}"
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._device = device
            self._rank = 0
            self._world_size = 1
        
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def encode_audio_to_base64_wav(self, audio_array: np.ndarray, sampling_rate: int) -> str:
        """
        Convert audio numpy array to base64-encoded WAV format.
        
        Args:
            audio_array: Audio data as numpy array
            sampling_rate: Sampling rate of the audio
            
        Returns:
            Base64-encoded WAV file string
        """
        # Ensure float32 to create 32-bit WAV files (not 64-bit)
        # This prevents "Unsupported bit depth: 64" errors on the server
        audio_array = audio_array.astype(np.float32)
        
        # Create WAV file in memory
        wav_buffer = BytesIO()
        wavfile.write(wav_buffer, sampling_rate, audio_array)
        wav_bytes = wav_buffer.getvalue()
        
        # Encode to base64
        base64_str = base64.b64encode(wav_bytes).decode('utf-8')
        return base64_str

    def transcribe_audio(self, audio_array: np.ndarray, sampling_rate: int) -> str:
        """
        Transcribe audio using the tt-media-server HTTP API.
        
        Args:
            audio_array: Audio data as numpy array
            sampling_rate: Sampling rate of the audio
            
        Returns:
            Transcription text
        """
        # Encode audio to base64 WAV
        base64_audio = self.encode_audio_to_base64_wav(audio_array, sampling_rate)
        
        # Prepare request
        url = f"{self.base_url}/audio/transcriptions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "file": base64_audio,
            "stream": False
        }
        
        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                
                # Extract transcription text from response
                # The response format should contain the transcription
                if isinstance(result, dict):
                    # Try common keys for transcription text
                    transcription = result.get('text') or result.get('transcription') or result.get('result')
                    if transcription:
                        return transcription
                    # If no known key, return the entire dict as string
                    eval_logger.warning(f"Unexpected response format: {result}")
                    return str(result)
                else:
                    return str(result)
                    
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    eval_logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    continue
                else:
                    eval_logger.error(f"All retry attempts failed: {e}")
                    raise
        
        return ""

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Whisper")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        
        # Group requests by their generation_kwargs
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_audios = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            flattened_audios = self.flatten(batched_audios)

            # Process until tokens from gen_kwargs
            gen_kwargs = all_gen_kwargs[0]
            until = [self.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            # Process inputs
            sampling_rate = self.processor.feature_extractor.sampling_rate
            assert sampling_rate == SAMPLING_RATE, f"Expected sampling rate {SAMPLING_RATE}, but got {sampling_rate}"
            audios = [downsample_audio(audio["array"], audio["sampling_rate"], sampling_rate) for audio in flattened_audios]

            try:
                # Transcribe using HTTP API
                answers = []
                for audio in audios:
                    answer = self.transcribe_audio(audio, sampling_rate)
                    
                    # Apply until tokens
                    for term in until:
                        if len(term) > 0:
                            answer = answer.split(term)[0]
                    
                    answers.append(answer)

            except Exception as e:
                eval_logger.error(f"Error while generating: {e}")
                answers = [""] * len(contexts)

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        # Reorder results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
