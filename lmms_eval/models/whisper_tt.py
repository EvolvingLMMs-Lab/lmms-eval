from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoProcessor, WhisperForConditionalGeneration

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.audio_processing import downsample_audio

import os
from scipy.io import wavfile
import ttnn
from models.demos.whisper.demo.demo import (
    create_functional_whisper_for_conditional_generation_inference_pipeline,
)
from models.demos.whisper.tt import (
    ttnn_optimized_functional_whisper as ttnn_model,
)
from models.demos.whisper.tt.ttnn_optimized_functional_whisper import (
    WHISPER_L1_SMALL_SIZE,
)


# Model sampling rate
SAMPLING_RATE = 16_000


# Warmup the model on app startup
def warmup_model():
    # create device, these constants are specific to n150 & n300
    device_id = 0
    device_params = {"l1_small_size": WHISPER_L1_SMALL_SIZE}

    # use WORKER for n150, ETH for n300
    dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
    if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
        dispatch_core_type = ttnn.device.DispatchCoreType.ETH

    dispatch_core_axis = ttnn.DispatchCoreAxis.ROW
    dispatch_core_config = ttnn.DispatchCoreConfig(
        dispatch_core_type, dispatch_core_axis
    )
    device_params["dispatch_core_config"] = dispatch_core_config
    device = ttnn.CreateDevice(device_id=device_id, **device_params)
    device.enable_program_cache()

    # create model pipeline
    model_pipeline = (
        create_functional_whisper_for_conditional_generation_inference_pipeline(
            ttnn_model,
            device,
        )
    )

    # warmup model pipeline
    try:
        dir_path = Path(os.environ["TT_METAL_HOME"])
    except KeyError:
        raise RuntimeError("Must set TT_METAL_HOME environment variable")
    input_file_path = dir_path / "/tt-metal/models/demos/whisper/demo/dataset/conditional_generation/17646385371758249908.wav"
    sampling_rate, data = wavfile.read(input_file_path)
    _ttnn_output = model_pipeline(data, sampling_rate, stream=False)

    eval_logger.info("Loading Stable Diffusion model...")
    eval_logger.info("Model loaded and ready!")
    return model_pipeline


@register_model("whisper_tt")
class WhisperTT(lmms):
    """
    Whisper Audio Model
    """

    def __init__(
        self,
        pretrained: str = "openai/whisper-tiny",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        language: str = "en",
        task: str = "transcribe",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        self._model = warmup_model()

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

        # self._config = self.model.config
        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.processor.tokenizer.set_prefix_tokens(language=language, task=task)
        self._tokenizer = self.processor.tokenizer
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._rank = 0
        self._word_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

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
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_audios = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            flattened_audios = self.flatten(batched_audios)

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            # process inputs
            sampling_rate = self.processor.feature_extractor.sampling_rate
            assert sampling_rate == SAMPLING_RATE, f"Expected sampling rate {SAMPLING_RATE}, but got {sampling_rate}"
            audios = [downsample_audio(audio["array"], audio["sampling_rate"], sampling_rate) for audio in flattened_audios]

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 256
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            try:
                answer = self.model(audios[0], sampling_rate)
                answers = [answer]
                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans

            except Exception as e:
                eval_logger.debug(f"Error while generating: {e}")
                answers = [""] * len(contexts)

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
