import os
import uuid
import base64
import re

import warnings
from typing import List, Optional, Tuple, Union
from PIL import Image
import decord
from io import BytesIO

import torch
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

from loguru import logger as eval_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration


@register_model("gemma3")
class Gemma3(lmms):
    """
    Gemma3 Model
    https://huggingface.co/google/gemma-3-27b-it
    """

    def __init__(
        self,
        pretrained: str = "google/gemma-3-27b-it",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache=True,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        interleave_visuals: Optional[bool] = False,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        self._model = Gemma3ForConditionalGeneration.from_pretrained(pretrained, device_map=self.device_map, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code,  device_map=self.device_map)
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)

        # self.tokenizer.padding_side = "left"
        # self.tokenizer.pad_token_id = self.tokenizer.eod_id
        
        self.prompt = "<img>{}</img>{}"
        self._config = self._model.config
        self.model.tie_weights()
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
            
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
            self._world_size = 1
        self.model.eval()

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
        # return self.tokenizer.eod_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    # should be deleted since max_new_tokens is decided by gen_kwargs not a model property
    # @property
    # def max_new_tokens(self) -> int:
    #     return 256

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
        raise NotImplementedError("Not implemented for Gemma3.")

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
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]
            
            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
            
            # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]
            
            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}]
                }]
                
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                for visual in visual_list[i]:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        vr = decord.VideoReader(visual)
                        first_frame = vr[0].asnumpy()
                        height, width = first_frame.shape[:2]
                        # max_pixels = height * width
                        processed_visuals.append({"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                    elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                        # processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                        # image_path = "/data/mllm/laolao77/ViRFT_COCO_base65/train-00000-of-00007-images/image_92.png"
                 
                        # processed_visuals.append({"type": "image", "image": image_path})
                        
                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                    # image_path = "/data/mllm/laolao77/ViRFT_COCO_base65/train-00000-of-00007-images/image_92.png"
                    # message.append(
                    #     {
                    #         "role": "user",
                    #         "content": [
                    #             {"type": "image", "image": image_path},
                    #             {"type": "text", "text": "Detect all objects"}
                    #         ]
                    #     }
                    # )
                    
                else:  # currently support find <image x> in the context
                    assert True, "Interleaving visuals is not implemented yet."
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)
            
            inputs = self.processor.apply_chat_template(
                batched_messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding="max_length", pad_to_multiple_of=8, max_length=4096

            ).to(self.model.device, dtype=torch.bfloat16)
            
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.device != self.model.device:
                        eval_logger.error(f"[Mismatch] Input {k} is on {v.device}, model is on {self.model.device}")
            # inputs = inputs.to("cuda")
            
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)
            from transformers.tokenization_utils_base import BatchEncoding

            # inputs = BatchEncoding({
            #     k: (v.contiguous() if isinstance(v, torch.Tensor) else v)
            #     for k, v in inputs.items()
            # })


            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}

            # pad_token_id = self.tokenizer.pad_token_id
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

            # cont = self.model.generate(
            #     **inputs,
            #     eos_token_id=self.tokenizer.eos_token_id,
            #     pad_token_id=pad_token_id,
            #     do_sample=True if current_gen_kwargs["temperature"] > 0 else False,
            #     temperature=current_gen_kwargs["temperature"],
            #     top_p=current_gen_kwargs["top_p"],
            #     num_beams=current_gen_kwargs["num_beams"],
            #     max_new_tokens=current_gen_kwargs["max_new_tokens"],
            #     use_cache=self.use_cache,
            # )
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    assert v.device == self.model.device, f"Input {k} on {v.device}, but model is on {self.model.device}"
            
            # try:
            #     with torch.inference_mode():
            #         cont = self.model.generate(
            #             **inputs,
            #             do_sample=False,
            #             max_new_tokens=100,
            #         )
            # except RuntimeError as e:
            #     eval_logger.error(f"RuntimeError during generation: {e}")
            #     for k, v in inputs.items():
            #         eval_logger.error(f"Input dtype {k}: {v.dtype}, shape: {v.shape}, device: {v.device}")
            #     exit(1)
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
        
            try:
                # cont = self.model.generate(
                #     **inputs,
                #     eos_token_id=self.tokenizer.eos_token_id,
                #     pad_token_id=pad_token_id,
                #     do_sample=True if gen_kwargs["temperature"] > 0 else False,
                #     temperature=gen_kwargs["temperature"],
                #     top_p=gen_kwargs["top_p"],
                #     num_beams=gen_kwargs["num_beams"],
                #     max_new_tokens=gen_kwargs["max_new_tokens"],
                # )
                
                cont = self.model.generate(
                    **inputs,
                    # eos_token_id=self.tokenizer.eos_token_id,
                    # pad_token_id=pad_token_id,
                    do_sample=False,
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )
            except RuntimeError as e:
                import traceback
                eval_logger.error(f"RuntimeError during generation: {e}")
                tb_str = traceback.format_exc()
                # print(tb_str)
                # eval_logger.error(tb_str)
                for k, v in inputs.items():
                    eval_logger.error(f"Input dtype {k}: {v.dtype}, shape: {v.shape}, device: {v.device}")
                # print(f"batched_messages: {batched_messages}")
                print(f"current_gen_kwargs: {current_gen_kwargs}")
                print(f"Dtype of model : {self.model.dtype}")

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                # print(f"Raw answer {i}: {ans}")
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

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
