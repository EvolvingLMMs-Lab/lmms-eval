import os
import tempfile
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("audio_flamingo_3")
class AudioFlamingo3(lmms):
    """
    Audio-Flamingo-3 Model
    https://github.com/NVIDIA/audio-flamingo
    """

    def __init__(
        self,
        pretrained: str = "nvidia/audio-flamingo-3-hf",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self._model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype="auto",
            device_map=self.device_map,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.processor.tokenizer.padding_side = "left"
        self._tokenizer = self.processor.tokenizer

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
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
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
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
        raise NotImplementedError("Loglikelihood is not implemented for AudioFlamingo3")

    def _save_audio_to_temp(self, audio_array: np.ndarray, sampling_rate: int) -> str:
        """Save audio array to a temporary file and return the path."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio_array, sampling_rate)
        return temp_file.name

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
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

            until = [self.tokenizer.decode([self.eot_token_id])]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] " f"but got {type(until)}")

            # Build conversations for each item in the batch
            conversations = []
            temp_files = []

            for batch_idx, (context, audios) in enumerate(zip(contexts, batched_audios)):
                conv = [{"role": "user", "content": []}]

                # Add text prompt first (as per official example)
                if context and context.strip():
                    conv[0]["content"].append({"type": "text", "text": context})

                # Add audio content after text
                for audio in audios:
                    audio_array = audio["array"]
                    sampling_rate = audio["sampling_rate"]

                    # Save audio to temp file (processor can handle file paths)
                    temp_path = self._save_audio_to_temp(audio_array, sampling_rate)
                    temp_files.append(temp_path)

                    conv[0]["content"].append({"type": "audio", "path": temp_path})

                conversations.append(conv)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 256
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            try:
                # Process each conversation individually to avoid batching issues
                answers = []
                for conv in conversations:
                    inputs = self.processor.apply_chat_template(
                        conv,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                    ).to(self.device if self.device_map != "auto" else "cuda")

                    cont = self.model.generate(
                        **inputs,
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        min_new_tokens=1,
                        use_cache=self.use_cache,
                    )

                    # Trim input tokens from output
                    generated_ids_trimmed = cont[:, inputs.input_ids.shape[1] :]
                    answer = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    answers.append(answer)

                # Apply until tokens
                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans

            except Exception as e:
                eval_logger.debug(f"Error while generating: {e}. Contexts: {contexts}")
                answers = [""] * len(contexts)

            # Clean up temp files
            for temp_path in temp_files:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        # Reorder results back to original order
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented")
