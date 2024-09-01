import math
import os
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import requests
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

wd = Path(__file__).parent.parent.parent.resolve()
import sys

sys.path.append(os.path.join(str(wd), "InternVL", "internvl_chat"))
from loguru import logger as eval_logger

if not hasattr(eval_logger, "internvl_warning_logged"):
    eval_logger.internvl_warning_logged = False

try:
    from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
    from internvl.model.internvl_chat import InternVLChatModel
    from internvl.model.internvl_chat.configuration_internvl_chat import (
        InternVLChatConfig,
    )
    from internvl.model.internvl_chat.modeling_intern_vit import InternVisionModel
    from internvl.train.dataset import build_transform, dynamic_preprocess
except ImportError:
    eval_logger.debug("InternVL is not installed. Please install InternVL to use this model.")
    if not eval_logger.internvl_warning_logged:
        eval_logger.debug("InternVL is not installed. Please install InternVL to use this model.")
        eval_logger.internvl_warning_logged = True

import re
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.utils.checkpoint
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModel,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel


@register_model("internvl")
class InternVLChat(lmms):
    # config_class = InternVLChatConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["InternVisionEncoderLayer", "LlamaDecoderLayer"]

    """
    0. Install lmms-eval
    cd lmms-eval
    pip install -e .

    How to Install InternVL:
    1. Clone the InternVL repository:
    git clone https://github.com/OpenGVLab/InternVL.git

    2. Install the requirements:
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

    3. Install flash-attn==2.3.6:
    pip install flash-attn==2.3.6 --no-build-isolation
    """

    """
    How to download the pretrained model:
    1. Download the pretrained model from hugginface:
    cd pretrained/
    # pip install -U huggingface_hub
    huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL-Chat-V1-5 --local-dir InternVL-Chat-V1-5

    2. the pretrained model should be in the following directory:
    pretrained
    └── InternVL-Chat-V1-5
    """

    #
    # The above steps can be optional, I add snapshot download, so now can just use hf repo_id
    # model_args pretrained=OpenGVLab/InternVL-Chat-V1-5
    #

    """
    InternVL-Chat-V1-5 Model for OpenGVLab https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py
    Example usage:

    accelerate launch --num_processes=8 --main_process_port 12345 -m lmms_eval \
        --model internvl \
        --model_args pretrained=OpenGVLab/InternVL-Chat-V1-5 \
        --tasks llava_wilder_small \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_samples
    """

    def __init__(
        self,
        config=None,
        pretrained: str = "OpenGVLab/InternVL-Chat-V1-5",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = False,
        revision=None,
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config=None,  # ends in json
        dynamic=True,
        load_in_8bit=False,
        vision_model=None,
        language_model=None,
        max_num=12,
        **kwargs,
    ) -> None:
        super().__init__()

        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.dynamic = dynamic  # dynamic image_size
        self.max_num = max_num
        if accelerator.is_main_process:
            cache_dir = snapshot_download(repo_id=pretrained, cache_dir="cache_dir", local_dir="cache_dir", local_dir_use_symlinks=False)
        accelerator.wait_for_everyone()
        # So what I did is that I let main process to download the repo, and then
        # other process can just simply read from this repo
        cache_dir = snapshot_download(repo_id=pretrained, cache_dir="cache_dir", local_dir="cache_dir", local_dir_use_symlinks=False)
        config = InternVLChatConfig.from_pretrained(cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(cache_dir, trust_remote_code=True, use_fast=False)
        model = InternVLChatModel.from_pretrained(cache_dir, low_cpu_mem_usage=True, config=config, torch_dtype=torch.bfloat16, load_in_8bit=load_in_8bit).eval()
        if not load_in_8bit:
            model = model.cuda()
        # self.model=model
        # self.device=self._device
        self._tokenizer = tokenizer
        # self.tokenizer=tokenizer
        self._model = model
        self._config = self._model.config
        self.use_thumbnail = self.model.config.use_thumbnail
        self.model.eval()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        if accelerator.num_processes > 1:
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
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        # from internvl model

        self.image_size = config.force_image_size or config.vision_config.image_size

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r, target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"], lora_alpha=lora_alpha, lora_dropout=lora_dropout, task_type="CAUSAL_LM"
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        if self.ps_version == "v1":
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, " "which results in a transposed image.")
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def noised_embed(self, vit_embeds, noise_alpha=5):
        dims = torch.tensor(vit_embeds.size(1) * vit_embeds.size(2))
        mag_norm = noise_alpha / torch.sqrt(dims)
        noise = torch.zeros_like(vit_embeds).uniform_(-mag_norm, mag_norm)
        return vit_embeds + noise

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        if self.training and self.neftune_alpha is not None:
            vit_embeds = self.noised_embed(vit_embeds, self.neftune_alpha)

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)  # .to(pixel_values.device)
        return vit_embeds

    def multi_image_chat(self, tokenizer, pixel_values, image_counts, question, generation_config, history=None, return_history=False, IMG_START_TOKEN="<img>", IMG_END_TOKEN="</img>", IMG_CONTEXT_TOKEN="<IMG_CONTEXT>"):
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        if tokenizer.convert_tokens_to_ids("<|im_end|>") != 0:
            eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")  # 92542, InternLM2
        else:
            eos_token_id = tokenizer.eos_token_id

        from internvl.conversation import get_conv_template

        template = get_conv_template(self.template)

        if history is None:
            history = []
            image_tokens = ""
            image_bs = pixel_values.shape[0]
            # print(f"dynamic ViT batch size: {image_bs}, image_counts: {image_counts}")
            for idx, image_count in enumerate(image_counts):
                image_tokens += f"<image {idx+1}> (图{idx+1}):" + IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * image_count + IMG_END_TOKEN
            question = image_tokens + "\n" + question
        else:
            for old_question, old_answer in history:
                template.append_message(template.roles[0], old_question)
                template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].cuda()
        attention_mask = model_inputs["attention_mask"].cuda()
        generation_config["eos_token_id"] = eos_token_id

        generation_output = self.generate(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, **generation_config)
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split("<|im_end|>")[0].strip()  # for InternLM2
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(image_tokens, "<image>")
            # print(query_to_print, response)
            return response
        return response

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
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def post_processing(self, response):
        response = response.replace("\n", "").replace("不是", "No").replace("是", "Yes").replace("否", "No")
        response = response.lower().replace("true", "yes").replace("false", "no")
        pattern = re.compile(r"[\u4e00-\u9fa5]")
        response = re.sub(pattern, "", response)
        return response

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.img_context_token_id
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_image(self, flattened_visuals, input_size=224):
        assert flattened_visuals[0].mode == "RGB"
        image = flattened_visuals[0].convert("RGB")
        transform = build_transform(is_train=False, input_size=input_size)
        if self.dynamic:
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=self.use_thumbnail, max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

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
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N]
            flattened_visuals = self.flatten(batched_visuals)
            try:
                pixel_values = self.load_image(flattened_visuals, self.image_size).cuda().to(torch.bfloat16)
            except IndexError:
                pixel_values = None
            gen_kwargs = all_gen_kwargs[0]

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            generation_config = dict(
                do_sample=False,
                top_k=50,
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                eos_token_id=self.tokenizer.eos_token_id,
            )
            question = contexts[0]
            response = self.model.chat(tokenizer=self.tokenizer, pixel_values=pixel_values, question=question, generation_config=generation_config)
            # TODO(choiszt) try batch_chat for multiple inputs
            response = self.post_processing(response)
            res.append(response)
            self.cache_hook.add_partial("generate_until", (question, gen_kwargs), response)
            pbar.update(1)
        res = re_ords.get_original(res)
        return res
        # print(chunk)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        pass
