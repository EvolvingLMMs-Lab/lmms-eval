import re
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_pil

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

DEFAULT_IMAGE_TOKEN = "<image>"


@register_model("aria")
class Aria(lmms):
    """
    Llava Model for Hugging Face Transformers: https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llava

    Adapted from the LLaVA-HF model in lmms_eval/models/llava_hf.py

    Example usage:

    accelerate launch --num_processes=8 --main_process_port 12345 -m lmms_eval \
        --model aria \
        --model_args pretrained=rhymes-ai/Aria \
        --tasks seedbench \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_samples
    """

    def __init__(
        self,
        pretrained: str = "rhymes-ai/Aria",
        revision: str = "main",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        attn_implementation: Optional[str] = None,
        device_map: str = "auto",
        chat_template: Optional[str] = None,
        use_cache: bool = True,
        specified_eot_token_id: Optional[int] = None,
        max_frames_num: Optional[int] = 64,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        self.max_frames_num = max_frames_num
        self._model = AutoModelForCausalLM.from_pretrained(pretrained, revision=revision, device_map=self.device_map, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation=attn_implementation)

        self.pretrained = pretrained
        self._image_processor = AutoProcessor.from_pretrained(pretrained, revision=revision, trust_remote_code=True)
        self._tokenizer = self._image_processor.tokenizer

        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.chat_template = chat_template
        self.specified_eot_token_id = specified_eot_token_id
        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with pipeline parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1
        self.accelerator = accelerator

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
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
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

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Not implemented for Aria.")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if isinstance(video_path, list):
            video_path = video_path[0]
        return read_video_pyav_pil(video_path, num_frm=max_frames_num)
        # if type(video_path) == str:
        #     vr = VideoReader(video_path, ctx=cpu(0))
        # else:
        #     vr = VideoReader(video_path[0], ctx=cpu(0))
        # total_frame_num = len(vr)
        # uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        # frame_idx = uniform_sampled_frames.tolist()
        # spare_frames = vr.get_batch(frame_idx).asnumpy()
        # spare_frames = [Image.fromarray(x) for x in spare_frames]
        # return spare_frames  # (frames, height, width, channels)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            if len(visuals) == 0:
                task_type = "text"
            elif isinstance(visuals[0], PIL.Image.Image):
                task_type = "image"
            elif isinstance(visuals[0], str):
                task_type = "video"
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            text_context = contexts[0]
            text_context = text_context.replace("\n\n", "\n")

            context = []

            if task_type == "video":
                try:
                    visuals = self.load_video(visuals, self.max_frames_num)
                except Exception as e:
                    res.append("")
                    eval_logger.info(f"Error {e} when loading video : {visuals}")
                    pbar.update(1)

            if DEFAULT_IMAGE_TOKEN not in context:
                context += [{"text": None, "type": "image"}] * len(visuals)
                context += [{"text": "\n" + text_context, "type": "text"}]
            else:
                assert text_context.count(DEFAULT_IMAGE_TOKEN) == len(visuals)
                for i, text_chunk in enumerate(text_context.split(DEFAULT_IMAGE_TOKEN)):
                    context += [{"text": text_chunk, "type": "text"}]
                    if i < len(visuals):
                        context += [{"text": None, "type": "image"}] * len(visuals)
                        context += [{"text": "\n", "type": "text"}]

            # Apply chat template
            messages = [{"role": "user", "content": context}]

            text = self._image_processor.apply_chat_template(messages, add_generation_prompt=True)

            # removing redundant placeholders
            text = re.sub(r"<image \d+>", "", text)
            text = re.sub(r"<image>", "", text)

            eval_logger.debug("DEBUGGING FOR ARIA:" + text)

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

            print(visuals)

            if task_type == "video":
                inputs = self._image_processor(images=visuals, text=text, return_tensors="pt", max_image_size=490)
            else:
                inputs = self._image_processor(images=visuals, text=text, return_tensors="pt", max_image_size=980)

            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            gen_kwargs["do_sample"] = False
            gen_kwargs["max_new_tokens"] = 1024

            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            eval_logger.debug(f"generate kwargs: {gen_kwargs}")

            try:
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    output = self.model.generate(
                        **inputs,
                        stop_strings=["<|im_end|>"],
                        tokenizer=self._image_processor.tokenizer,
                        **gen_kwargs,
                    )
                    output_ids = output[0][inputs["input_ids"].shape[1] :]
                    text_outputs = self._image_processor.decode(output_ids, skip_special_tokens=True).replace("<|im_end|>", "")

                    ### Basic Model-wise Parsing for CoT-alike Outputs
                    """
                    keywords = [
                        "Answer:",
                        "answer is:", "choice is:", "option is:", 
                        "Answer is:", "Choice is:", "Option is:",
                        "answer is", "choice is", "option is",
                        "Answer is", "Choice is", "Option is"
                    ]

                    for keyword in keywords:
                        if keyword in text_outputs:
                            eval_logger.debug(f"ARIA Original generated output simplified by keyword [{keyword}]: {text_outputs}")
                            text_outputs = text_outputs.split(keyword, 1)[-1]
                            break
                    """
                    eval_logger.debug(f"Generated output: {text_outputs}")
            except Exception as ex:
                eval_logger.debug(f"Generation failed: {ex}")
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{text_outputs}\n")

            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAHF")
