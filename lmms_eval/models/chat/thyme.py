import copy
import os
import re
import tempfile
import time

# pandayin: used for temporary image file creation
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Tuple

from loguru import logger as eval_logger
from tqdm import tqdm
from transformers.cache_utils import DynamicCache

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.model_utils.thyme.sandbox import execute_code_in_sandbox
from lmms_eval.models.model_utils.thyme.utils import (
    REASONING_SYS_PROMPT,
    SIMPLE_SYS_PROMPT,
    SPECIAL_STRING_LIST,
    generate_prompt_final_qa,
    generate_prompt_simple_qa,
)
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VLSimple
from lmms_eval.protocol import ChatMessages

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@contextmanager
def extract_user_input(message_list: List[Dict[str, Any]]) -> Generator[Tuple[str, str], None, None]:
    """
    Extracts the user's image, and save the image to a temporary file.
    Args:
        message_list (List[Dict[str, Any]]): A list of user input.

    Yields:
        Generator[Tuple[str, str], None, None]: output image_path.

    Raises:
        ValueError: Not found valid user image.
    """
    user_image = None

    for message in message_list:
        if message.get("role") == "user":
            content = message.get("content", [])
            for part in content:
                if part.get("type") == "image":
                    user_image = part.get("image", None)

    if user_image is None:
        raise ValueError("Not found valid image.")
    if not isinstance(user_image, str):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as temp_image_file:
            try:
                user_image.save(temp_image_file, format="JPEG")
                temp_image_file.flush()

                yield temp_image_file.name

            finally:
                pass
    else:
        yield user_image


@register_model("thyme")
class Thyme(Qwen2_5_VLSimple):
    is_simple = False

    def __init__(self, max_iterations=5, max_retry=5, verbose=True, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        self.max_retry = max_retry
        self.verbose = verbose

    # pandayin: Default behaviour. Thinking with images.
    def _generate_reasoning_mode(self, messages, user_image_path, temp_output_dir=None):
        formatted_message = self._prepare_content_reasoning(messages, user_image_path)

        # Main retry loop
        retry_generations = self.max_retry
        has_valid_answer = False

        while retry_generations > 0 and not has_valid_answer:
            conversation_history = copy.deepcopy(formatted_message)
            kv_cache = DynamicCache()
            previous_execution_context = {}
            total_tokens = 0
            if self.verbose:
                eval_logger.info(f"Generation {self.max_retry - retry_generations + 1}")

            # Inner iteration loop
            retry_iterations = self.max_iterations
            # pandayin: I'm tired. So just hard-code the generation kwargs here.
            generate_kwargs = {
                "max_new_tokens": 2048,
                "temperature": 0.01,
                "top_p": 0.001,
                "top_k": 1,
                "repetition_penalty": 1.0,
                "stop_strings": SPECIAL_STRING_LIST,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "tokenizer": self.tokenizer,
            }

            while retry_iterations > 0:
                retry_iterations -= 1
                generated_content = []

                if self.verbose:
                    eval_logger.info(f"Iteration {self.max_iterations - retry_iterations}")

                # Prepare inputs
                text = self.processor.apply_chat_template([conversation_history], tokenize=False, add_generation_prompt=(retry_iterations == self.max_iterations - 1))

                if retry_iterations != self.max_iterations - 1:
                    if text[0].endswith("<|im_end|>\n"):
                        text[0] = text[0][: -len("<|im_end|>\n")]

                images, videos = process_vision_info([conversation_history])
                inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                # Backup for rollback
                last_kv_cache = copy.deepcopy(kv_cache)
                last_execution_context = copy.deepcopy(self._remove_unpickable_values(previous_execution_context))

                # Generate
                generated_ids = self.model.generate(**inputs, **generate_kwargs, past_key_values=kv_cache, use_cache=self.use_cache)

                generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]

                total_tokens += len(generated_ids[0])

                out = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                generated_text_segment = out[0]

                # Check for direct answer
                if "</answer>" in generated_text_segment:
                    generated_content.append({"type": "text", "text": generated_text_segment})
                    has_valid_answer = True

                # Check for code block
                code_regex = re.compile(r"<code>\s*(?:```\s*)?(?:python\s*)?([\s\S]*?)\s*(?:```\s*)?</code>", re.IGNORECASE)
                code_match = code_regex.search(generated_text_segment)

                if code_match:
                    code_to_execute = code_match.group(1).strip()

                    if self.verbose:
                        eval_logger.info(f"Found code block: {code_to_execute}")

                    # Execute code
                    (
                        processed_img_paths,
                        captured_stdout,
                        error_msg,
                        current_execution_context,
                    ) = execute_code_in_sandbox(code_to_execute, user_image_path, temp_output_dir=temp_output_dir, previous_execution_context=previous_execution_context)

                    if not processed_img_paths:
                        # Rollback on failure
                        kv_cache = last_kv_cache
                        previous_execution_context = last_execution_context
                        if self.verbose:
                            eval_logger.warning(f"Code execution failed: {error_msg}")
                        continue

                    previous_execution_context = current_execution_context

                    # Add generated content
                    generated_content += [{"type": "text", "text": generated_text_segment}, {"type": "text", "text": "<sandbox_output>"}]

                    # Add images or text output
                    first_path = processed_img_paths[0]
                    if os.path.exists(first_path):
                        for img_path in processed_img_paths:
                            if os.path.exists(img_path):
                                generated_content.append({"type": "image", "image": img_path})
                    else:
                        generated_content.append({"type": "text", "text": first_path})

                    generated_content.append({"type": "text", "text": "</sandbox_output>"})
                else:
                    # No code and no answer - might be repetition
                    if "</answer>" not in generated_text_segment:
                        if self.verbose:
                            eval_logger.warning("No code or answer found, adjusting temperature")
                        generate_kwargs["temperature"] = 1.0
                        break

                # Update conversation history
                if conversation_history[-1]["role"] == "user":
                    conversation_history.append({"role": "assistant", "content": generated_content})
                elif conversation_history[-1]["role"] == "assistant":
                    conversation_history[-1]["content"] += generated_content

                # Check for final answer
                if "</answer>" in generated_text_segment:
                    has_valid_answer = True
                    if self.verbose:
                        eval_logger.info("Final answer found")
                    break

                # Check for EOS
                if generated_ids[0][-1] == self.tokenizer.eos_token_id:
                    if self.verbose:
                        eval_logger.info("Model generated EOS")
                    break

            if has_valid_answer:
                break

            retry_generations -= 1
            generate_kwargs["temperature"] = 1.0

        # Extract final response
        final_assistant_response = ""
        for msg in reversed(conversation_history):
            if msg["role"] != "assistant":
                continue
            current_content_str = ""
            for item in msg["content"]:
                if item["type"] == "text":
                    current_content_str += item["text"]
            final_assistant_response = current_content_str
            break

        return final_assistant_response, has_valid_answer, total_tokens

    # pandayin: Fall back to simple QA mode when reasoning fails. Directly answer the question.
    def _generate_simple_mode(self, messages):
        formatted_message = self._prepare_content_simple(messages)
        conversation_history = copy.deepcopy(formatted_message)
        total_tokens = 0

        text = self.processor.apply_chat_template([conversation_history], tokenize=False, add_generation_prompt=True)

        images, videos = process_vision_info([conversation_history])
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
        if self.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(self.device)

        generate_kwargs = {"max_new_tokens": 2048, "temperature": None, "do_sample": False, "eos_token_id": self.tokenizer.eos_token_id, "use_cache": True}

        generated_ids = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]

        total_tokens += len(generated_ids[0])

        out = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        generated_text = out[0]

        # Wrap in answer tags if not present
        answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
        if not answer_match:
            generated_text = f"<answer>{generated_text}</answer>"

        return generated_text, total_tokens

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                video_kwargs["nframes"] = self.max_num_frames
            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]
            # pandayin: For ease of implementation. We assume always batch_size = 1 for generation.
            # pandayin: Only support single image input for now.
            answers = []
            cache_contexts = []
            start_time = time.time()
            for current_message in batched_messages:
                with extract_user_input(current_message) as temp_image_path:
                    # pandayin: Try reasoning mode first.
                    # We automatically clean up all the intermediate files in the reasoning mode.
                    with tempfile.TemporaryDirectory() as temp_dir:
                        final_response, has_valid_answer, generated_total_tokens = self._generate_reasoning_mode(current_message, temp_image_path, temp_dir)
                    if not has_valid_answer:
                        # pandayin: Fall back to simple QA mode if reasoning fails.
                        final_response, generated_total_tokens = self._generate_simple_mode(current_message)

                total_tokens += generated_total_tokens
                answers.append(final_response)
                cache_context = self.processor.apply_chat_template(current_message, tokenize=False, add_generation_prompt=True)
                cache_contexts.append(cache_context)
            end_time = time.time()

            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time

            for answer, context in zip(answers, cache_contexts):
                clean_ans = parse_reasoning_model_answer(answer)
                res.append(clean_ans)
                # # ### SQLite-based caching of LMM responses
                # # pandayin: Pass in attr, req, res, and will store/cache the results for MLLM:
                # # Like this: dict[hash([attr, req])] = res
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {answer}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0

        e2e_latency = end_time - start_time
        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res

    def _prepare_content_reasoning(self, inputs: list[dict[str, str | List]], user_image_path: str) -> list[dict[str, str | List]]:
        new_inputs = []
        new_inputs.append({"role": "system", "content": REASONING_SYS_PROMPT})
        for conv_round in inputs:
            if conv_round["role"] != "user":
                continue
            content = []
            for s in conv_round["content"]:
                if s["type"] == "image":
                    item = {"type": "image", "image": s["image"]}
                    if self.min_pixels is not None:
                        item["min_pixels"] = self.min_pixels
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
                elif s["type"] == "text":
                    item = {
                        "type": "text",
                        "text": generate_prompt_final_qa(s["text"], user_image_path),
                    }
                else:
                    raise ValueError(f"Invalid message type: {s['type']}, {s}")
                content.append(item)
            new_inputs.append({"role": "user", "content": content})
        return new_inputs

    def _prepare_content_simple(self, inputs: list[dict[str, str | List]]) -> list[dict[str, str | List]]:
        new_inputs = []
        new_inputs.append({"role": "system", "content": SIMPLE_SYS_PROMPT})
        for conv_round in inputs:
            if conv_round["role"] != "user":
                continue
            content = []
            for s in conv_round["content"]:
                if s["type"] == "image":
                    item = {"type": "image", "image": s["image"]}
                    if self.min_pixels is not None:
                        item["min_pixels"] = self.min_pixels
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
                elif s["type"] == "text":
                    item = {
                        "type": "text",
                        "text": generate_prompt_simple_qa(s["text"]),
                    }
                else:
                    raise ValueError(f"Invalid message type: {s['type']}, {s}")
                content.append(item)
            new_inputs.append({"role": "user", "content": content})
        return new_inputs

    def _remove_unpickable_values(self, dictionary):
        import pickle

        def is_pickable(obj):
            try:
                pickle.dumps(obj)
                return True
            except (pickle.PicklingError, TypeError, AttributeError):
                return False

        keys_to_remove = []
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self._remove_unpickable_values(value)
            elif not is_pickable(value):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del dictionary[key]
        return dictionary
