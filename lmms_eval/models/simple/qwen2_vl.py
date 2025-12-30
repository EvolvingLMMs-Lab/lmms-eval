import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

# TODO: Consider moving flatten to lmms_eval.utils
# from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import load_video_decord

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_vl")
class Qwen2_VL(lmms):
    """
    Qwen2_VL Model
    "https://github.com/QwenLM/Qwen2-VL"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        max_length: Optional[int] = 2048,  # Added max_length parameter
        max_pixels: int = 602112,
        min_pixels: int = 3136,
        max_num_frames: int = 32,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
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
        # Simplified logic for single process
        else:  # accelerator.num_processes == 1
            self._device = torch.device(device)
            # Respect device_map if provided and not empty, otherwise use the determined device string
            self.device_map = device_map if device_map else device

        if use_flash_attention_2:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype="auto",
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", device_map=self.device_map).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self._config = self.model.config
        # Initialize _max_length using the parameter or config (adjust attribute as needed)
        # self._max_length = max_length if max_length is not None else self._config.max_position_embeddings
        self._max_length = max_length  # Using the provided parameter for now

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
        # Ensure _max_length is initialized
        if not hasattr(self, "_max_length") or self._max_length is None:
            # Fallback or raise error if not initialized
            # Example: Attempt to get from config if not set
            try:
                self._max_length = self.model.config.max_position_embeddings
            except AttributeError:
                raise AttributeError("'_max_length' was not initialized and could not be inferred from model config.")
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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2_VL")

    # TODO: Consider moving flatten to lmms_eval.utils if it's general purpose
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

        # Import utils here if flatten is moved
        import lmms_eval.utils as utils

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

            # TODO: Clarify the behavior of doc_to_visual for documents without visual info.
            # The current logic might incorrectly discard all visuals if one doc lacks them.
            # Ensure flatten is appropriate here based on doc_to_visual's return type.
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            if None in visual_list:  # This check might need refinement
                # If a mix of visual/non-visual is possible, this needs careful handling
                # Currently sets all visuals to empty if any doc returns None
                visual_list = []
            else:
                visual_list = self.flatten(visual_list)  # Assumes doc_to_visual returns list of lists

            gen_kwargs = all_gen_kwargs[0] if all_gen_kwargs else {}

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until_from_kwargs = gen_kwargs.pop("until")
                if isinstance(until_from_kwargs, str):
                    until = [until_from_kwargs]
                elif isinstance(until_from_kwargs, list):
                    until = until_from_kwargs
                else:
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until_from_kwargs)}")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            # Remove image tags from context text itself, as they are handled separately
            contexts = [ctx.replace("<image>", "") for ctx in contexts]

            batched_messages = []
            # TODO: Consider refactoring message construction logic (especially visual processing)
            # into helper methods for clarity (e.g., _prepare_message, _process_visuals).
            for i, context in enumerate(contexts):
                message = [{"role": "system", "content": self.system_prompt}]
                current_context = context  # Use a temporary variable

                if self.reasoning_prompt:
                    current_context = current_context.strip() + self.reasoning_prompt
                    # Update the original contexts list as well if needed elsewhere, otherwise just use current_context
                    # contexts[i] = current_context # Uncomment if contexts needs to be updated

                processed_visuals = []
                # Use the potentially flattened visual_list relevant to this context 'i'
                # This assumes visual_list aligns correctly with contexts after potential flattening
                # Needs careful review based on doc_to_visual output structure
                # For simplicity, assuming visual_list contains all visuals for the batch for now
                # A more robust approach might map visuals back to their original context index.
                relevant_visuals = visual_list  # Placeholder: needs logic to get visuals for context 'i'

                for visual in relevant_visuals:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        try:
                            vr = decord.VideoReader(visual)
                            if len(vr) > 0:
                                first_frame = vr[0].asnumpy()
                                height, width = first_frame.shape[:2]
                                # max_pixels = height * width # This seems incorrect, should use instance config
                                processed_visuals.append({"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                            else:
                                eval_logger.warning(f"Skipping empty video: {visual}")
                        except Exception as e:
                            eval_logger.error(f"Failed to process video {visual}: {e}")
                    elif isinstance(visual, Image.Image):  # Handle PIL Image
                        try:
                            base64_image = visual.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                        except Exception as e:
                            eval_logger.error(f"Failed to process PIL image: {e}")
                    # Add handling for other potential visual types if necessary

                if not self.interleave_visuals:
                    # Add all visuals first, then the text
                    content_payload = processed_visuals + [{"type": "text", "text": current_context}]
                    message.append(
                        {
                            "role": "user",
                            "content": content_payload,
                        }
                    )
                else:  # Handle interleaving based on <image x> placeholders
                    image_placeholders = re.findall(r"<image \d+>", current_context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", current_context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for idx, placeholder in enumerate(image_placeholders):
                        try:
                            img_idx_match = re.search(r"<image (\d+)>", placeholder)
                            if img_idx_match:
                                img_idx = int(img_idx_match.group(1)) - 1  # 1-based index in text
                                # Map text index to available processed visuals
                                if 0 <= img_idx < len(processed_visuals):
                                    content_parts.append(processed_visuals[img_idx])
                                else:
                                    eval_logger.warning(f"Image index {img_idx + 1} out of range for available visuals ({len(processed_visuals)}) in context.")
                            else:
                                eval_logger.warning(f"Could not parse index from placeholder: {placeholder}")
                        except Exception as e:
                            eval_logger.error(f"Error processing placeholder {placeholder}: {e}")

                        # Add the text part following this placeholder
                        if idx + 1 < len(text_parts) and text_parts[idx + 1]:
                            content_parts.append({"type": "text", "text": text_parts[idx + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)

            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            # TODO: Consider moving video frame sampling logic into process_vision_info or a helper.
            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None and len(video_inputs) > 0 and video_inputs[0] is not None:
                # Assuming video_inputs is a list where the first element holds the tensor
                video_tensor = video_inputs[0]
                if isinstance(video_tensor, torch.Tensor) and video_tensor.ndim > 0 and video_tensor.shape[0] > 0:
                    total_frames = video_tensor.shape[0]
                    indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int, endpoint=True)  # Ensure endpoint=True
                    # Ensure unique indices if linspace produces duplicates for few frames
                    indices = np.unique(indices)
                    # Append the last frame index if not already included and needed
                    # if total_frames > 0 and total_frames - 1 not in indices:
                    #     indices = np.append(indices, total_frames - 1)
                    #     indices = np.unique(indices) # Ensure uniqueness again

                    # Limit to max_num_frames if appending last frame exceeded it
                    if len(indices) > self.max_num_frames:
                        # This might happen if linspace already picked close indices including the end
                        # Or if max_num_frames is very small. Prioritize evenly spaced.
                        indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int, endpoint=True)
                        indices = np.unique(indices)

                    video_inputs[0] = video_tensor[indices]
                else:
                    eval_logger.warning(f"Unexpected video_inputs format or empty tensor: {type(video_tensor)}")

            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            if self.device_map == "auto":
                inputs = inputs.to("cuda")  # Assuming 'cuda' is the target for 'auto' on single GPU
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs first, then override with user-provided ones
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Default to greedy
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}  # Provided gen_kwargs override defaults

            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if current_gen_kwargs["temperature"] > 0 else False,
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            # Decode generated sequences, excluding input tokens
            generated_ids_trimmed = []
            for in_ids, out_ids in zip(inputs.input_ids, cont):
                # Find the first position where output differs from input, or start after input length
                input_len = len(in_ids)
                # Handle potential padding in output; eos might appear before max length
                try:
                    # Find first eos token in the generated part
                    eos_pos = (out_ids[input_len:] == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        # Slice generated part up to (but not including) the first EOS token
                        generated_ids_trimmed.append(out_ids[input_len : input_len + eos_pos[0]])
                    else:
                        # No EOS found, take the whole generated part
                        generated_ids_trimmed.append(out_ids[input_len:])
                except IndexError:  # Handle cases where output is shorter than input (shouldn't happen with generate)
                    generated_ids_trimmed.append(torch.tensor([], dtype=torch.long, device=out_ids.device))

            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Process answers to remove text after stop tokens
            for i, ans in enumerate(answers):
                stop_pos = len(ans)  # Default to end of string
                for term in until:
                    if term and term in ans:  # Ensure term is not empty and exists
                        stop_pos = min(stop_pos, ans.index(term))
                answers[i] = ans[:stop_pos].strip()  # Trim whitespace from final answer

            for ans, context in zip(answers, contexts):
                res.append(ans)
                # Use original gen_kwargs for caching, not the merged one
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        import lmms_eval.utils as utils

        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_to_text, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]

            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]
            if None in batched_visuals:
                batched_visuals = [None] * len(batched_visuals)
            else:
                batched_visuals = [self.flatten([visuals]) if visuals is not None else [] for visuals in batched_visuals]

            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            until = [self.tokenizer.decode(self.eot_token_id)]

            round_idx = 0
            batched_round_res = []
            batched_previous_round_info = None
            while True:
                contexts = []
                visuals_list = []

                if round_idx != 0:
                    visuals_list, contexts, batched_terminal_signal, batched_round_res, batched_previous_round_info = list(
                        zip(
                            *[
                                batched_doc_to_text[0](
                                    self.task_dict[task][split][ids],
                                    previous_output=[round_res[ids_idx] for round_res in batched_round_res],
                                    round_idx=round_idx,
                                    previous_round_info=batched_previous_round_info[ids_idx] if batched_previous_round_info is not None else None,
                                )
                                for ids_idx, ids in enumerate(batched_doc_id)
                            ]
                        )
                    )
                    batched_round_res = list(zip(*batched_round_res))
                    if batched_terminal_signal[0]:
                        break
                else:
                    visuals_list = batched_visuals
                    contexts = list(batched_contexts)

                contexts = [ctx.replace("<image>", "") for ctx in contexts]

                batched_messages = []
                for i, context in enumerate(contexts):
                    message = [{"role": "system", "content": self.system_prompt}]
                    current_context = context

                    if self.reasoning_prompt:
                        current_context = current_context.strip() + self.reasoning_prompt

                    processed_visuals = []
                    relevant_visuals = visuals_list[i] if i < len(visuals_list) and visuals_list[i] is not None else []

                    for visual in relevant_visuals:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            try:
                                vr = decord.VideoReader(visual)
                                if len(vr) > 0:
                                    first_frame = vr[0].asnumpy()
                                    height, width = first_frame.shape[:2]
                                    processed_visuals.append({"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                                else:
                                    eval_logger.warning(f"Skipping empty video: {visual}")
                            except Exception as e:
                                eval_logger.error(f"Failed to process video {visual}: {e}")
                        elif isinstance(visual, Image.Image):
                            try:
                                base64_image = visual.convert("RGB")
                                buffer = BytesIO()
                                base64_image.save(buffer, format="JPEG")
                                base64_bytes = base64.b64encode(buffer.getvalue())
                                base64_string = base64_bytes.decode("utf-8")
                                processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                            except Exception as e:
                                eval_logger.error(f"Failed to process PIL image: {e}")

                    if not self.interleave_visuals:
                        if processed_visuals:
                            content_payload = processed_visuals + [{"type": "text", "text": current_context}]
                        else:
                            content_payload = [{"type": "text", "text": current_context}]
                        message.append(
                            {
                                "role": "user",
                                "content": content_payload,
                            }
                        )
                    else:
                        image_placeholders = re.findall(r"<image \d+>", current_context)
                        content_parts = []
                        text_parts = re.split(r"<image \d+>", current_context)
                        if text_parts[0]:
                            content_parts.append({"type": "text", "text": text_parts[0]})

                        for idx, placeholder in enumerate(image_placeholders):
                            try:
                                img_idx_match = re.search(r"<image (\d+)>", placeholder)
                                if img_idx_match:
                                    img_idx = int(img_idx_match.group(1)) - 1
                                    if 0 <= img_idx < len(processed_visuals):
                                        content_parts.append(processed_visuals[img_idx])
                                    else:
                                        eval_logger.warning(f"Image index {img_idx + 1} out of range for available visuals ({len(processed_visuals)}) in context.")
                                else:
                                    eval_logger.warning(f"Could not parse index from placeholder: {placeholder}")
                            except Exception as e:
                                eval_logger.error(f"Error processing placeholder {placeholder}: {e}")

                            if idx + 1 < len(text_parts) and text_parts[idx + 1]:
                                content_parts.append({"type": "text", "text": text_parts[idx + 1]})

                        message.append(
                            {
                                "role": "user",
                                "content": content_parts,
                            }
                        )

                    batched_messages.append(message)

                texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
                image_inputs, video_inputs = process_vision_info(batched_messages)

                if video_inputs is not None and len(video_inputs) > 0 and video_inputs[0] is not None:
                    video_tensor = video_inputs[0]
                    if isinstance(video_tensor, torch.Tensor) and video_tensor.ndim > 0 and video_tensor.shape[0] > 0:
                        total_frames = video_tensor.shape[0]
                        indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int, endpoint=True)
                        if len(indices) > self.max_num_frames:
                            indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int, endpoint=True)
                            indices = np.unique(indices)

                        video_inputs[0] = video_tensor[indices]
                    else:
                        eval_logger.warning(f"Unexpected video_inputs format or empty tensor: {type(video_tensor)}")

                inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                default_gen_kwargs = {
                    "max_new_tokens": 128,
                    "temperature": 0.0,
                    "top_p": None,
                    "num_beams": 1,
                }
                current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}

                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=True if current_gen_kwargs["temperature"] > 0 else False,
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )

                generated_ids_trimmed = []
                for in_ids, out_ids in zip(inputs.input_ids, cont):
                    input_len = len(in_ids)
                    try:
                        eos_pos = (out_ids[input_len:] == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                        if len(eos_pos) > 0:
                            generated_ids_trimmed.append(out_ids[input_len : input_len + eos_pos[0]])
                        else:
                            generated_ids_trimmed.append(out_ids[input_len:])
                    except IndexError:
                        generated_ids_trimmed.append(torch.tensor([], dtype=torch.long, device=out_ids.device))

                answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                clean_answers = []
                for ans in answers:
                    stop_pos = len(ans)
                    for term in until:
                        if term and term in ans:
                            stop_pos = min(stop_pos, ans.index(term))
                    clean_ans = ans[:stop_pos].strip()
                    clean_answers.append(clean_ans)

                batched_round_res.append(clean_answers)
                round_idx += 1

            transposed_res = list(zip(*batched_round_res))
            res.extend(transposed_res)

            self.cache_hook.add_partial("generate_until_multi_round", (batched_contexts[0], gen_kwargs), batched_round_res)
            pbar.update(1)

        res = re_ords.get_original(res)

        pbar.close()
        return res
