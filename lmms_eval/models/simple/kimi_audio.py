import os
import tempfile
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("kimi_audio")
class KimiAudio(lmms):
    """
    Kimi-Audio Model
    https://github.com/MoonshotAI/Kimi-Audio
    """

    def __init__(
        self,
        pretrained: str = "moonshotai/Kimi-Audio-7B-Instruct",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        load_detokenizer: bool = False,
        text_temperature: float = 0.0,
        text_top_k: int = 5,
        text_repetition_penalty: float = 1.0,
        text_repetition_window_size: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            device_index = accelerator.local_process_index
        else:
            # Ensure we have a proper device index
            if device == "cuda" or device == "cuda:0":
                device_index = 0
            elif device.startswith("cuda:"):
                device_index = int(device.split(":")[1])
            else:
                device_index = 0
            self._device = torch.device(f"cuda:{device_index}")

        # Set the current CUDA device before loading the model
        if self._device.type == "cuda":
            torch.cuda.set_device(device_index)

        # Import KimiAudio from the kimia_infer package
        from kimia_infer.api.kimia import KimiAudio as KimiAudioModel

        # Load the model (it internally moves to CUDA)
        self._model = KimiAudioModel(model_path=pretrained, load_detokenizer=load_detokenizer)

        # Store generation parameters
        self.text_temperature = text_temperature
        self.text_top_k = text_top_k
        self.text_repetition_penalty = text_repetition_penalty
        self.text_repetition_window_size = text_repetition_window_size

        # Get tokenizer from prompt manager
        self._tokenizer = self._model.prompt_manager.text_tokenizer

        self._config = None
        self.batch_size_per_gpu = int(batch_size)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for KimiAudio")

    def _save_audio_to_temp(self, audio_array: np.ndarray, sampling_rate: int) -> str:
        """Save audio array to a temporary file and return the path."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio_array, sampling_rate)
        return temp_file.name

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0], bos=False, eos=False)
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]

            # Get audio data from task
            batched_audios = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]

            gen_kwargs = all_gen_kwargs[0]

            # Process generation kwargs
            max_new_tokens = gen_kwargs.pop("max_new_tokens", 256)
            temperature = gen_kwargs.pop("temperature", self.text_temperature)
            top_k = gen_kwargs.pop("top_k", self.text_top_k)

            until = [self.tokenizer.decode([self.eot_token_id])]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]

            # Process each item (KimiAudio doesn't support batching)
            temp_files = []
            for batch_idx, (context, audios) in enumerate(zip(contexts, batched_audios)):
                try:
                    # Build chat messages
                    messages = []

                    # Add text prompt first (as per official example)
                    if context and context.strip():
                        messages.append(
                            {
                                "role": "user",
                                "message_type": "text",
                                "content": context,
                            }
                        )

                    # Add audio content after text
                    for audio in audios:
                        audio_array = audio["array"]
                        sampling_rate = audio["sampling_rate"]

                        # Save audio to temp file (KimiAudio requires file path)
                        temp_path = self._save_audio_to_temp(audio_array, sampling_rate)
                        temp_files.append(temp_path)

                        messages.append(
                            {
                                "role": "user",
                                "message_type": "audio",
                                "content": temp_path,
                            }
                        )

                    # Generate response
                    _, generated_text = self.model.generate(
                        chats=messages,
                        output_type="text",
                        text_temperature=temperature,
                        text_top_k=top_k,
                        text_repetition_penalty=self.text_repetition_penalty,
                        text_repetition_window_size=self.text_repetition_window_size,
                        max_new_tokens=max_new_tokens,
                    )

                    # Apply until tokens
                    for term in until:
                        if len(term) > 0:
                            generated_text = generated_text.split(term)[0]

                    answer = generated_text

                except Exception as e:
                    eval_logger.debug(f"Error while generating: {e}. Context: {context[:100]}")
                    answer = ""

                res.append(answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                pbar.update(1)

            # Clean up temp files
            for temp_path in temp_files:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

        # Reorder results back to original order
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented")
