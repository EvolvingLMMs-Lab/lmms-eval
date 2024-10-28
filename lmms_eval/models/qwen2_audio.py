import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.audio_processing import downsample_audio


@register_model("qwen2_audio")
class Qwen2_Audio(lmms):
    """
    Qwen2_Audio Model
    "https://github.com/QwenLM/Qwen2-Audio"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-Audio-7B",  # Qwen/Qwen2-Audio-7B-Instruct
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        add_generation_prompt: bool = True,
        add_system_prompt: bool = True,
        simple_prompt: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self.add_generation_prompt = add_generation_prompt
        self.add_system_prompt = add_system_prompt
        # If using simple prompt, only add "<|audio_bos|><|AUDIO|><|audio_eos|>"
        # and then prompt to align with original Qwen2 Audio
        self.simple_prompt = simple_prompt
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype="auto",
            device_map=device_map,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.processor.tokenizer.padding_side = "left"
        self._tokenizer = self.processor.tokenizer

        if not self.add_system_prompt:
            # Overwrite chat template to exclude system prompt
            self.processor.chat_template = (
                "{% set audio_count = namespace(value=0) %}"
                "{% for message in messages %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{% if message['content'] is string %}"
                "{{ message['content'] }}<|im_end|>\n"
                "{% else %}"
                "{% for content in message['content'] %}"
                "{% if 'audio' in content or 'audio_url' in content %}"
                "{% set audio_count.value = audio_count.value + 1 %}"
                "Audio {{ audio_count.value }}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
                "{% elif 'text' in content %}"
                "{{ content['text'] }}"
                "{% endif %}"
                "{% endfor %}"
                "<|im_end|>\n"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "<|im_start|>assistant\n"
                "{% endif %}"
            )

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2_Audio")

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

            # contexts = "<|audio_bos|><|AUDIO|><|audio_eos|>" + contexts

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            if not self.simple_prompt:
                conversations = []
                for idx, context in enumerate(contexts):
                    conv = [{"role": "user", "content": []}]
                    for _ in batched_audios[idx]:
                        # This placeholder is just use to make chat template work
                        # We already have the sampled audio array
                        conv[0]["content"].append({"type": "audio", "audio_url": "placeholder.wav"})
                    conv[0]["content"].append({"type": "text", "text": context})
                    conversations.append(conv)

                text = [self.processor.apply_chat_template(conversation, add_generation_prompt=self.add_generation_prompt, tokenize=False) for conversation in conversations]
            else:
                text = ["<|audio_bos|><|AUDIO|><|audio_eos|>" + context for context in contexts]
            audios = [downsample_audio(audio["array"], audio["sampling_rate"], self.processor.feature_extractor.sampling_rate) for audio in flattened_audios]

            inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True, sampling_rate=self.processor.feature_extractor.sampling_rate)

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 256
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            try:
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

                # cont = self.model.generate(**inputs, max_new_tokens=256, min_new_tokens=1, do_sample=False)

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                # generated_ids_trimmed = cont[:, inputs.input_ids.size(1):]
                answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans

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
