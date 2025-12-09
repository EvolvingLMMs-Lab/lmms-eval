import base64
import re
import os
import json
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.model_utils.load_video import read_video_decord_base64

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_vl")
class Qwen3_VL(lmms):
    """
    Qwen3_VL Model
    "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_name = pretrained
        self.continual_mode = continual_mode
        if self.continual_mode:
            if response_persistent_folder is None:
                response_persistent_folder = "./logs/qwen3_vl_persistent_folder"
            self.response_persistent_folder = response_persistent_folder
            if not os.path.exists(self.response_persistent_folder):
                os.makedirs(self.response_persistent_folder)
            
            safe_model_name = self.model_name.replace("/", "_")
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{safe_model_name}_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        
        # Validation: custom video loader specific parameters
        if self.use_custom_video_loader:
            eval_logger.info("=" * 80)
            eval_logger.info("🔧 DEBUG MODE ENABLED - Using custom video loader with decord")
            eval_logger.info(f"🔧 DEBUG: fps={self.fps}, max_num_frames={max_num_frames}, max_image_size={self.max_image_size}")
            eval_logger.info("=" * 80)
            
            # Write to INIT file
            init_file = "/mnt/sfs-common/krhu/lmms-eval/INIT_CALLED.txt"
            import datetime
            with open(init_file, "w") as f:
                f.write(f"=== __init__ CALLED at {datetime.datetime.now()} ===\n")
                f.write(f"use_custom_video_loader=True\n")
                f.write(f"fps={self.fps}, max_num_frames={max_num_frames}, max_image_size={self.max_image_size}\n")
                f.flush()
            
            if self.fps is None and max_num_frames is None:
                eval_logger.warning("Neither fps nor max_num_frames specified for custom video loader, defaulting to max_num_frames=32")
        else:
            eval_logger.info("Using official Qwen3-VL processor for video loading")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # check whether its an MoE model
        match = re.search(r"A\d+B", pretrained)
        model_fn = Qwen3VLMoeForConditionalGeneration if match else Qwen3VLForConditionalGeneration
        self._model = model_fn.from_pretrained(pretrained, **model_kwargs).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if self.continual_mode:
                eval_logger.warning("Continual mode is not supported with distributed inference. Disabling continual mode.")
                self.continual_mode = False
            
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
            self._world_size = 1

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
        # FORCE write to file - this WILL work
        import sys
        import datetime
        debug_file = "/mnt/sfs-common/krhu/lmms-eval/GENERATE_UNTIL_CALLED.txt"
        with open(debug_file, "w") as f:
            f.write(f"=== generate_until CALLED at {datetime.datetime.now()} ===\n")
            f.write(f"use_custom_video_loader = {self.use_custom_video_loader}\n")
            f.write(f"Number of requests: {len(requests)}\n")
            f.write(f"fps={self.fps}, max_num_frames={self.max_num_frames}, max_image_size={self.max_image_size}\n")
            f.flush()
        
        # Removed the test exception - let's see if we get here naturally
        
        # Force print to stderr
        print("\n" + "=" * 80, file=sys.stderr)
        print("🚀🚀🚀 generate_until CALLED 🚀🚀🚀", file=sys.stderr)
        print(f"🚀 use_custom_video_loader = {self.use_custom_video_loader}", file=sys.stderr)
        print(f"🚀 Number of requests: {len(requests)}", file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)
        sys.stderr.flush()
        
        # Also use ERROR level
        eval_logger.error("=" * 80)
        eval_logger.error("🚀🚀🚀 generate_until CALLED 🚀🚀🚀")
        eval_logger.error(f"🚀 use_custom_video_loader = {self.use_custom_video_loader}")
        eval_logger.error(f"🚀 Number of requests: {len(requests)}")
        eval_logger.error("=" * 80)
        
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
            
            orig_contexts = contexts
            orig_doc_id = doc_id
            orig_task = task
            orig_split = split
            cached_results_map = {}
            non_cached_indices = []

            if self.continual_mode:
                def get_uuid(t, s, d):
                    return f"{t}___{s}___{d}"

                for i in range(len(contexts)):
                    uuid = get_uuid(task[i], split[i], doc_id[i])
                    if uuid in self.response_cache:
                        cached_results_map[i] = self.response_cache[uuid]
                    else:
                        non_cached_indices.append(i)

                if len(non_cached_indices) == 0:
                    for i in range(len(contexts)):
                        res.append(cached_results_map[i])
                        pbar.update(1)
                    continue

                contexts = tuple([contexts[i] for i in non_cached_indices])
                doc_id = tuple([doc_id[i] for i in non_cached_indices])

            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            eval_logger.error(f"🔍 Processing chunk: {len(visual_list)} items, visual_list={visual_list}")
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

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                
                if visual_list[i] is not None:
                    for idx, visual in enumerate(visual_list[i]):
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            if self.use_custom_video_loader:
                                # Custom mode: convert to base64 images using decord
                                debug_file = "/mnt/sfs-common/krhu/lmms-eval/GENERATE_UNTIL_CALLED.txt"
                                import datetime
                                with open(debug_file, "a") as f:
                                    f.write(f"🎥 Found video file: {visual}\n")
                                    f.write(f"🎥 About to use custom decord loader\n")
                                    f.flush()
                                
                                eval_logger.error(f"🎥 Custom video loader: Processing video: {visual}")
                                eval_logger.error(f"🎥 Params: num_frm={self.max_num_frames}, fps={self.fps}, max_image_size={self.max_image_size}")
                                visual_frames = read_video_decord_base64(
                                    visual, 
                                    num_frm=self.max_num_frames, 
                                    fps=self.fps, 
                                    img_format="JPEG", 
                                    max_image_size=self.max_image_size
                                )
                                eval_logger.error(f"🎥 Extracted {len(visual_frames)} frames using DECORD")
                                # Add as images
                                for base64_frame in visual_frames:
                                    processed_visuals.append({
                                        "type": "image", 
                                        "image": f"data:image/jpeg;base64,{base64_frame}", 
                                        "max_pixels": self.max_pixels, 
                                        "min_pixels": self.min_pixels
                                    })
                            else:
                                # Official mode: add video path (processor will handle loading)
                                eval_logger.info(f"[DEBUG] Official mode: Adding video path: {visual}")
                                processed_visuals.append({
                                    "type": "video", 
                                    "video": visual,
                                })
                        elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                            base64_image = visual.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})

                eval_logger.error(f"📦 Sample {i}: Processed {len(processed_visuals)} items, types={[v['type'] for v in processed_visuals]}")
                
                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:  # currently support find <image x> in the context
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
            
            # ====== OFFICIAL MODE vs CUSTOM MODE ======
            debug_file = "/mnt/sfs-common/krhu/lmms-eval/GENERATE_UNTIL_CALLED.txt"
            import datetime
            with open(debug_file, "a") as f:
                f.write(f"\n🎬 Processing Mode: {'CUSTOM' if self.use_custom_video_loader else 'OFFICIAL'} at {datetime.datetime.now()}\n")
                f.write(f"use_custom_video_loader={self.use_custom_video_loader}\n")
                f.flush()
            
            eval_logger.error(f"🎬 Processing Mode: {'CUSTOM' if self.use_custom_video_loader else 'OFFICIAL'}")
            
            if self.use_custom_video_loader:
                with open(debug_file, "a") as f:
                    f.write("✅ ENTERING CUSTOM MODE\n")
                    f.flush()
                # Custom mode: videos already converted to base64 images
                # Check what's in batched_messages
                for idx, msg in enumerate(batched_messages):
                    for m in msg:
                        if m["role"] == "user":
                            content_types = [c.get("type") for c in m["content"] if isinstance(c, dict)]
                            eval_logger.error(f"🎬 Message {idx} content types: {content_types}")
                            # Check for any video entries
                            has_video = any(c.get("type") == "video" for c in m["content"] if isinstance(c, dict))
                            if has_video:
                                eval_logger.error(f"⚠️⚠️⚠️ WARNING: Found 'video' type in message {idx}!")
                
                texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
                
                # Extract only images from batched_messages (videos already converted to images)
                eval_logger.error(f"📞 About to call process_vision_info with {len(batched_messages)} messages")
                image_inputs, _ = process_vision_info(batched_messages)
                video_inputs = None  # No videos to process since they're already converted to images
                eval_logger.error(f"📞 process_vision_info returned {len(image_inputs) if image_inputs else 0} images")
                eval_logger.error(f"📞 video_inputs=None (custom mode)")
                
                padding_side = "left" if self.batch_size > 1 else "right"
                inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, padding_side=padding_side, return_tensors="pt")
            else:
                # Official mode: let processor handle video loading and sampling
                proc_kwargs = {}
                if self.fps is not None:
                    proc_kwargs["fps"] = self.fps
                if self.max_num_frames is not None:
                    proc_kwargs["num_frames"] = self.max_num_frames
                if self.max_image_size is not None:
                    proc_kwargs["max_image_size"] = self.max_image_size
                
                # Use processor with messages directly (official way)
                padding_side = "left" if self.batch_size > 1 else "right"
                texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True, **proc_kwargs)
                image_inputs, video_inputs = process_vision_info(batched_messages, **proc_kwargs)
                inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, padding_side=padding_side, return_tensors="pt")

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 32768,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            generated_results = []
            for ans, context in zip(answers, contexts):
                clean_ans = parse_reasoning_model_answer(ans)
                generated_results.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)

            if self.continual_mode:
                gen_ptr = 0
                for i in range(len(orig_contexts)):
                    if i in cached_results_map:
                        res.append(cached_results_map[i])
                        pbar.update(1)
                    else:
                        clean_ans = generated_results[gen_ptr]
                        res.append(clean_ans)

                        uuid = get_uuid(orig_task[i], orig_split[i], orig_doc_id[i])
                        self.response_cache[uuid] = clean_ans

                        pbar.update(1)
                        gen_ptr += 1

                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)
            else:
                for ans in generated_results:
                    res.append(ans)
                    pbar.update(1)

                # eval_logger.debug(f"Question: {context}")
                # eval_logger.debug(f"Model Raw Response: {ans}")
                # eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")