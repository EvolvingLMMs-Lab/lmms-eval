import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoProcessor, WhisperForConditionalGeneration
from vllm import LLM, SamplingParams

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.audio_processing import downsample_audio


@register_model("whisper_vllm")
class WhisperVllm(lmms):
    """
    Whisper Audio Model VLLM
    """

    def __init__(
        self,
        pretrained: str = "openai/whisper-tiny",
        batch_size: Optional[Union[int, str]] = 1,
        **model_kwargs,
    ) -> None:
        super().__init__()
        self._batch_size = batch_size

        self._model = LLM(
            model=pretrained,
            **model_kwargs,
        )

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        raise NotImplementedError()

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
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

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
            
            # TODO
            #toks = self.tokenizer.encode(x[0])
            #return -len(toks), x[0]
            return 0, x[0]

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

            inputs = [
                {
                    "prompt": "<|startoftranscript|>",
                    "multi_modal_data": {
                        "audio": (audio["array"], audio["sampling_rate"]),
                    },
                }
                for audio in flattened_audios
            ]

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            sampling_params = SamplingParams(
                temperature=gen_kwargs.get("temperature", 0),
                top_p=gen_kwargs.get("top_p", 0),
                max_tokens=gen_kwargs.get("max_new_tokens", 256),
            )

            outputs = self.model.generate(inputs, sampling_params)
            answers = [output.outputs[0].text for output in outputs]

            try:
                pass

            except Exception as e:
                eval_logger.debug(f"Error while generating: {e}. It is possibly due to blank audio in {contexts}")
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
