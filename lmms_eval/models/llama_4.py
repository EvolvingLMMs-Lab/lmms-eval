import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import (  # Modified import
    AutoConfig,
    AutoProcessor,
    Llama4ForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_pil

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

DEFAULT_IMAGE_TOKEN = "<|image|>"


@register_model("llama_4")
class Llama4(lmms):
    def __init__(
        self,
        # Modified default pretrained model ID for Llama4:
        pretrained: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        revision: str = "main",
        device: str = "cuda",
        # For Llama4, default dtype "auto" means torch.bfloat16
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        # Modified default: use "flex_attention" for Llama4
        attn_implementation: Optional[str] = "flex_attention",
        # Modified default device_map:
        device_map: str = "auto",
        max_frames_num: Optional[int] = 20,
        fps: Optional[int] = None,
        max_image_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        # ✅ Add this here to suppress TorchInductor backend errors
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map
        if isinstance(dtype, str) and dtype == "auto":
            # For Llama4, default dtype is bfloat16
            dtype = torch.bfloat16
        elif isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        self.fps = fps
        self.max_frames_num = max_frames_num
        self.max_image_size = max_image_size
        # Modified: load Llama4 model
        self._model = Llama4ForConditionalGeneration.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=dtype,
            device_map=self.device_map,
            # trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(pretrained)
        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
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
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented"

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            messages = [{"role": "user", "content": []}]
            images = []
            for visual in visuals:
                if isinstance(visual, str):
                    frames = self.load_video(visual, self.max_frames_num)
                    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
                    images.extend([to_pil_image(frame) for frame in frames])
                elif isinstance(visual, PIL.Image.Image):
                    images.append(visual)
            for _ in range(len(images)):
                messages[-1]["content"].append({"type": "image"})
            messages[-1]["content"].append({"type": "text", "text": contexts})
            # Modified: update apply_chat_template for Llama4 with additional parameters.
            # prompt = self.processor.apply_chat_template(
            #     messages,
            #     add_generation_prompt=True,
            #     tokenize=True,
            #     return_dict=True,
            #     return_tensors="pt"
            # ).to(self.model.device)
            # print(">>> prompt type:", type(prompt), "prompt:", prompt)
            # inputs = self.processor(images, prompt, return_tensors="pt").to(self.model.device)
            # print(">>> prompt type:", type(prompt), "prompt:", prompt)
            # prompt = self.processor.apply_chat_template(
            #     messages,
            #     add_generation_prompt=True,
            #     tokenize=True,
            #     return_dict=True,
            #     return_tensors="pt"
            # )
            # print(self.processor.tokenizer.special_tokens_map)
            # print(self.processor.tokenizer.all_special_tokens)

            prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            # print(self.processor.tokenizer.special_tokens_map)
            # print(self.processor.tokenizer.all_special_tokens)
            # print("Number of <|image|> tokens in prompt:", prompt.count("<|image|>"))
            # print("Number of images:", len(images))

            # print(">>> prompt type:", type(prompt), "prompt:", prompt)

            inputs = self.processor(images, prompt, add_special_tokens=True, return_tensors="pt").to(self.model.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    # max_new_tokens=gen_kwargs["max_new_tokens"],
                    max_new_tokens=512,
                    cache_implementation="hybrid",
                    temperature=gen_kwargs["temperature"],
                    do_sample=gen_kwargs["do_sample"],
                )
                output = output[:, inputs["input_ids"].shape[-1] :]
                res.append(self.processor.decode(output[0]))
            pbar.update(1)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAHF")
