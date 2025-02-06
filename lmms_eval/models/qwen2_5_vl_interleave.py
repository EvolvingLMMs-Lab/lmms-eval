import base64
import json
import os
from io import BytesIO
from re import I
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_5_vl_interleave")
class Qwen2_5_VL_Interleave(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] =  False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 256 * 28 * 28,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        # if self.max_image_size and not self.use_custom_video_loader:
        #     raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

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

        if use_flash_attention_2:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", device_map=self.device_map).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

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
            self._rank = 0
            self._word_size = 1

        # self.continual_mode = continual_mode
        # if self.continual_mode and response_persistent_folder is None:
        #     raise ValueError("Continual mode requires a persistent path for the response. We will cache the Gemini API response in this path and use it for future requests. Please provide a valid path.")
        # if self.continual_mode:
        #     self.response_persistent_folder = response_persistent_folder
        #     if not os.path.exists(self.response_persistent_folder):
        #         os.makedirs(self.response_persistent_folder)
        #     self.model_version = pretrained.split("/")[-1].replace("-", "_").lower()
        #     self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

        #     if os.path.exists(self.response_persistent_file):
        #         with open(self.response_persistent_file, "r") as f:
        #             self.response_cache = json.load(f)
        #         self.cache_mode = "resume"
        #     else:
        #         self.response_cache = {}
        #         self.cache_mode = "start"

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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

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
            # if self.continual_mode and self.cache_mode == "resume":
            #     doc_uuid = get_uuid(task, split, doc_id)
            #     # print(doc_uuid)
            #     if doc_uuid in self.response_cache:
            #         content = self.response_cache[doc_uuid]
            #         if content:
            #             res.extend(content)
            #             pbar.update(1)
            #             continue
            visuals = [doc_to_visual[i](self.task_dict[task][split][ids]) for i, ids in enumerate(doc_id)]

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

            # for i in range(len(contexts)):
            #     for j in range(32):
            #         if f"<image {j}>" in contexts[i]:
            #             contexts[i] = contexts[i].replace(f"<image {j}>", "<image>")
            #         if f"\\<image {j}\\>" in contexts[i]:
            #             contexts[i] = contexts[i].replace(f"\\<image {j}\\>", "<image>")
            # if "<image>" in contexts[i]:
            #     contexts[i] = contexts[i].replace("<image>", "")
            # print(contexts[i])

            messages = []
            processed_visuals = []
            for i, context in enumerate(contexts):
                # context += "\nPlease think step by step."

                # print("context", context)

                if "<image 1>" in context:
                    context = context.split("<image 1>")
                    # print(json.dumps(context, indent=4))
                else:
                    context = [context]

                message = [{"role": "system", "content": "You are a helpful assistant."}]

                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    # print("visuals", visual)
                    if isinstance(visual, Image.Image):
                        visual = [visual]
                    if isinstance(visual, (list, tuple)) and isinstance(visual[0], str):
                        assert len(visual) == 1, f"Expected a single video file but got {len(visual)} files"
                        visual = visual[0]
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        if self.use_custom_video_loader:
                            try:
                                visual = read_video_pyav_base64(
                                    visual,
                                    num_frm=self.max_num_frames,
                                    fps=self.fps,
                                    img_format="JPEG",
                                    max_image_size=self.max_image_size,
                                )
                            except Exception as e:
                                eval_logger.error(f"Failed to load video: {visual}. Error: {e}")
                                continue

                            image_contents = list(map(lambda x: f"data:image/jpeg;base64,{x}", visual))
                            if len(context) == 2:
                                if len(image_contents) == 1:
                                    message.append(
                                        {
                                            "role": "user",
                                            "content": [{"type": "video", "video": image_contents[:-1]}, {"type": "text", "text": context[0]}, {"type": "image", "image": image_contents[-1]}, {"type": "text", "text": context[1]}],
                                        }
                                    )
                                else:
                                    message.append({"role": "user", "content": [{"type": "text", "text": context[0]}, {"type": "image", "image": image_contents[-1]}, {"type": "text", "text": context[1]}]})
                            else:
                                message.append({"role": "user", "content": [{"type": "video", "video": image_contents}, {"type": "text", "text": context}]})
                            # with open("alb.json", "w") as f:
                            #     json.dump(message, f, indent=4)
                        else:
                            vr = decord.VideoReader(visual)
                            first_frame = vr[0].asnumpy()
                            height, width = first_frame.shape[:2]
                            # max_pixels = height * width
                            message.append({"role": "user", "content": [{"type": "video", "video": visual, "max_pixels": self.max_pixels}, {"type": "text", "text": context}]})
                    elif isinstance(visual, Image.Image):  # Single image
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}, {"type": "text", "text": context}]})
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        image_content = []
                        i = 0
                        for v in visual:
                            base64_image = v.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                            v.save(f"test_{i}.jpg")
                            i += 1
                        # message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                        assert len(image_content) + 1 == len(context), f"Number of images and context do not match, {len(image_content)} images and {len(context)} context\n{json.dumps(context)}"
                        content = []
                        for i in range(len(image_content)):
                            content.append({"type": "text", "text": context[i]})
                            content.append(image_content[i])
                        content.append({"type": "text", "text": context[-1]})
                        # print("content", content)
                    else:
                        message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})

                messages.append(message)
            # print(messages)
            
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                #fps=self.fps,
                return_tensors="pt"
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 128
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            #cont = self.model.generate(**inputs, max_new_tokens=1024)

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                answers[i] = ans

            content = []
            for ans, context in zip(answers, contexts):
                res.append(ans)
                content.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            # reorder this group of results back to original unsorted form

            # if self.continual_mode is True:  # Cache the response
            #     doc_uuid = get_uuid(task, split, doc_id)
            #     self.response_cache[doc_uuid] = content
            #     with open(self.response_persistent_file, "w") as f:
            #         json.dump(self.response_cache, f)

        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")