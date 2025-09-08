import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

from tqdm import tqdm
from transformers import AutoModel

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.vllm import VLLM as VLLMSimple
from lmms_eval.protocol import ChatMessages

try:
    from vllm import LLM, SamplingParams
except ImportError:
    vllm = None

WORKERS = int(os.getenv("WORKERS", "32"))


@register_model("longvila")
class LongVila(VLLMSimple):
    is_simple = False

    def __init__(
        self,
        model="Efficient-Large-Model/LongVILA-R1-7B",
        tensor_parallel_size=1,
        data_parallel_size=1,
        gpu_memory_utilization=0.5,
        batch_size=1,
        max_frame_num=32,
        trust_remote_code=True,
        chat_template=None,
        max_pixels: int = 1605632,
        min_image_pixels=28,
        fps: Optional[int] = None,
        device_map: Optional[str] = "cuda",
        **kwargs,
    ):
        # vLLM requires the path to the autoregressive llm weights under the model root
        model_root = model
        llm_path = os.path.join(model_root, "llm")
        # Enable prompt embeddings so we can pass encoder-produced embeddings directly
        kwargs["enable_prompt_embeds"] = True
        self.fps = fps
        self.max_pixels = max_pixels

        # Set up imports from the model's remote_code directory
        # The LongVILA repo provides preprocessing utilities we must call directly
        try:
            from remote_code.media import extract_media as _extract_media
            from remote_code.mm_utils import process_images as _process_images
            from remote_code.tokenizer_utils import (
                tokenize_conversation as _tokenize_conversation,
            )
        except Exception as e:
            raise ImportError(f"Failed to import LongVILA remote_code utilities from '{model_root}'. Ensure the model path contains remote_code. Original error: {e}")

        self.extract_media = _extract_media
        self.process_images = _process_images
        self.tokenize_conversation = _tokenize_conversation

        # Load the encoder that produces prompt embeddings for the LLM
        # llm_only_need_embed reduces memory usage to only what's needed for embedding
        self.model_encoder = AutoModel.from_pretrained(
            model_root,
            trust_remote_code=True,
            device_map=device_map,
            llm_only_need_embed=True,
        )
        super().__init__(llm_path, tensor_parallel_size, data_parallel_size, gpu_memory_utilization, batch_size, max_frame_num, trust_remote_code, chat_template, min_image_pixels, **kwargs)

    def _to_remote_conversation(self, chat_messages: ChatMessages) -> list:
        """
        Convert ChatMessages to LongVILA remote_code conversation format.
        [{"from": "human"|"gpt", "value": [str | {"path": media_path}, ...]}, ...]
        """
        role_map = {"user": "human", "assistant": "gpt", "system": "human"}
        conversation = []
        for msg in chat_messages.messages:
            from_role = role_map.get(msg.role, "human")
            value_parts = []
            for content in msg.content:
                # ChatTextContent
                if getattr(content, "type", None) == "text":
                    value_parts.append(content.text)
                # Images, Videos, Audios -> use path dicts as required by tokenizer_utils
                elif getattr(content, "type", None) in ("image", "video", "audio"):
                    value_parts.append({"path": content.url})
            if value_parts:
                conversation.append({"from": from_role, "value": value_parts})
        return conversation

    def make_one_request(self, request: Instance) -> Tuple["object", dict]:
        """
        Build prompt embeddings and per-request sampling params from an Instance.
        Returns (inputs_embeds, params_dict). Does not mutate input.
        """
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages = ChatMessages(messages=raw_messages)

        # Copy to avoid side-effects across threads
        _gen = dict(gen_kwargs or {})
        _gen.setdefault("max_new_tokens", 4096)
        _gen.setdefault("temperature", 0)
        _gen.setdefault("top_p", 0.95)

        params = {
            "temperature": _gen["temperature"],
            "max_tokens": _gen["max_new_tokens"],
            "top_p": _gen["top_p"],
        }

        # Convert to LongVILA remote_code conversation format
        conversation = self._to_remote_conversation(chat_messages)

        # Extract and preprocess media
        media = self.extract_media(conversation, self.model_encoder.config)
        if "video" in media and media["video"] is not None:
            media["video"] = [self.process_images(images, self.model_encoder.vision_tower.image_processor, self.model_encoder.config).half() for images in media["video"]]

        # Tokenize conversation and move to CUDA for embedding
        input_ids = self.tokenize_conversation(conversation, self.model_encoder.tokenizer, add_generation_prompt=True).unsqueeze(0).cuda()

        # Create prompt embeddings using the model encoder
        inputs_embeds, _, _ = self.model_encoder._embed(input_ids, media, {"video": {}}, None, None)

        return inputs_embeds, params

    def generate_until(self, requests) -> List[str]:
        res = []
        self.load_cache()
        res, requests = self.get_response_from_cache(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        e2e_latency = 0
        for batch_requests in batched_requests:
            prompt_embeds_list = []
            params_list = []
            # Build embeddings sequentially to avoid GPU contention in the encoder
            for req in tqdm(batch_requests, disable=(self.rank != 0), desc="Building embeddings"):
                inputs_embeds, params = self.make_one_request(req)
                prompt_embeds_list.append({"prompt_embeds": inputs_embeds.squeeze(0)})
                params_list.append(params)

            # For now, assume homogeneous sampling params within a batch
            sampling_params = SamplingParams(**params_list[-1])

            start_time = time.time()
            response = self.client.generate(prompts=prompt_embeds_list, sampling_params=sampling_params)
            end_time = time.time()

            response_text = [o.outputs[0].text for o in response]
            for req, text in zip(batch_requests, response_text):
                self.add_request_response_to_cache(req, text)

            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time

            assert len(response_text) == len(batch_requests)
            res.extend(response_text)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
