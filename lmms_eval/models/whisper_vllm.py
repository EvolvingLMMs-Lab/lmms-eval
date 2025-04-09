from typing import List, Optional, Tuple, Union

from tqdm import tqdm
from vllm import LLM, SamplingParams

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("whisper_vllm")
class WhisperVllm(lmms):
    """
    Whisper Audio Model VLLM
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: Optional[Union[int, str]] = 1,
        **model_kwargs,
    ) -> None:
        super().__init__()
        self._batch_size = int(batch_size)

        self._model = LLM(
            model=pretrained,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **model_kwargs,
        )

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        raise NotImplementedError()

    @property
    def tokenizer(self):
        return self._model.get_tokenizer()

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
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batched_requests = [requests[i : i + self.batch_size] for i in range(0, len(requests), self.batch_size)]
        for batch_requests in batched_requests:
            batched_prompts = []
            for idx in range(len(batch_requests)):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = batch_requests[idx].arguments

                # generation parameters
                sampling_params = SamplingParams(
                    temperature=gen_kwargs.get("temperature", 0),
                    top_p=gen_kwargs.get("top_p", 0),
                    max_tokens=gen_kwargs.get("max_new_tokens", 256),
                )

                # prepare multimodal inputs
                audio = doc_to_visual(self.task_dict[task][split][doc_id])
                assert len(audio) == 1
                audio = audio[0]

                pre_prompt = gen_kwargs.get("pre_prompt", "")
                post_prompt = gen_kwargs.get("post_prompt", "")

                # prepare prompt for task "fleurs"
                task_name = str(task).strip()
                if task_name.startswith("fleurs"):
                    language = self.task_dict[task][split][doc_id]["language"]

                    if language in ["Mandarin Chinese", "Cantonese Chinese"]:
                        whisper_prompt = f"<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>"
                        prompt_text = f"{pre_prompt}{whisper_prompt}{post_prompt}"
                    elif language == "en":
                        whisper_prompt = f"<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
                        prompt_text = f"{pre_prompt}{whisper_prompt}{post_prompt}"
                    else:
                        prompt_text = f"{pre_prompt}Please recognize the speech and only output the recognized content:{post_prompt}"
                else:
                    prompt_text = "<|startoftranscript|>"

                # prepare input
                prompt = {
                    "prompt": prompt_text,
                    "multi_modal_data": {
                        "audio": (audio["array"], audio["sampling_rate"]),
                    },
                }
                batched_prompts.append(prompt)

            outputs = self.model.generate(batched_prompts, sampling_params, use_tqdm=False)
            transcriptions = [output.outputs[0].text for output in outputs]
            answers = [self.model.get_tokenizer().normalize(transcription) for transcription in transcriptions]  # whisper post processing

            assert len(answers) == len(batch_requests)
            res.extend(answers)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
