import math
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

try:
    from cambrian.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from cambrian.conversation import conv_templates
    from cambrian.mm_utils import tokenizer_image_token
    from cambrian.model.cambrian_arch import unpad_image
except ImportError:
    eval_logger.error("Cambrian is not installed. pip install git+https://github.com/cambrian-mllm/cambrian-s.git")

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.cambrians.qwen2_monkey_patch import Qwen2SdpaAttention, cambrian_qwen2_forward
from lmms_eval.models.simple.cambrians import CambrianS, is_video_file


def process_video_with_decord_vsr(video_file: str, num_threads: int = -1):
    if num_threads < 1:
        vr = VideoReader(video_file, ctx=cpu(0))
    else:
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=num_threads)
    frame_idx = list(range(len(vr)))
    video = vr.get_batch(frame_idx).asnumpy()
    vr.seek(0)
    return video


def process_videos_vsr(videos, image_processor, num_threads: int = -1):
    new_videos_aux_list = []
    video_sizes = []

    for video in videos:
        video = process_video_with_decord_vsr(video, num_threads=num_threads)
        video_sizes.append((video.shape[2], video.shape[1], video.shape[0]))
        video = [Image.fromarray(video[idx], mode="RGB") for idx in range(video.shape[0])]

        video_aux_list = []
        for processor_aux in image_processor:
            video_aux_list.append(processor_aux.preprocess(video, return_tensors="pt")["pixel_values"])
        new_videos_aux_list.append(video_aux_list)

    new_videos_aux_list = [list(batch_video_aux) for batch_video_aux in zip(*new_videos_aux_list)]
    new_videos_aux_list = [torch.stack(video_aux) for video_aux in new_videos_aux_list]
    return new_videos_aux_list, video_sizes, None


def downsample_cache_states(cache_states: torch.Tensor, downsample_ratio: int, visual_features: torch.Tensor) -> torch.Tensor:
    cache_states_shape = cache_states.shape
    cache_states = cache_states.flatten(0, 1).unflatten(1, (visual_features.size(1), visual_features.size(2) + 1)).permute(0, 3, 1, 2)
    cache_states = torch.nn.functional.avg_pool2d(cache_states, kernel_size=downsample_ratio, stride=downsample_ratio)
    cache_states = cache_states.flatten(2, 3).unflatten(0, cache_states_shape[:2]).permute(0, 1, 3, 2)
    return cache_states


def _append_cache_entry(cache, key_states, value_states, modality, length, surprise_score):
    cache["key_states"].append(key_states)
    cache["value_states"].append(value_states)
    cache["modalities"].append(modality)
    cache["lengths"].append(length)
    cache["surprising_scores"].append(surprise_score)


def _maybe_consolidate_global_cache(cache, surprise_threshold: float, consolidation_method: str, consolidation_mem_budget: int):
    if sum(cache["lengths"]) <= consolidation_mem_budget:
        return

    if consolidation_method == "drop_merge":
        index = 1
        while index < len(cache["surprising_scores"]) - 1:
            if cache["surprising_scores"][index] >= surprise_threshold and cache["surprising_scores"][index + 1] >= surprise_threshold:
                cache["key_states"][index] = (cache["key_states"][index] + cache["key_states"][index + 1]) / 2.0
                cache["value_states"][index] = (cache["value_states"][index] + cache["value_states"][index + 1]) / 2.0
                cache["surprising_scores"][index] = (cache["surprising_scores"][index] + cache["surprising_scores"][index + 1]) / 2.0
                cache["key_states"].pop(index + 1)
                cache["value_states"].pop(index + 1)
                cache["modalities"].pop(index + 1)
                cache["lengths"].pop(index + 1)
                cache["surprising_scores"].pop(index + 1)
                torch.cuda.empty_cache()
                index -= 1
            index += 1

        while sum(cache["lengths"]) > consolidation_mem_budget:
            index = int(np.array(cache["surprising_scores"][1:]).argmin())
            cache["key_states"].pop(index + 1)
            cache["value_states"].pop(index + 1)
            cache["modalities"].pop(index + 1)
            cache["lengths"].pop(index + 1)
            cache["surprising_scores"].pop(index + 1)
            torch.cuda.empty_cache()

            if index > 0 and index < len(cache["surprising_scores"]) - 1 and cache["surprising_scores"][index] >= surprise_threshold and cache["surprising_scores"][index + 1] >= surprise_threshold:
                cache["key_states"][index] = (cache["key_states"][index] + cache["key_states"][index + 1]) / 2.0
                cache["value_states"][index] = (cache["value_states"][index] + cache["value_states"][index + 1]) / 2.0
                cache["surprising_scores"][index] = (cache["surprising_scores"][index] + cache["surprising_scores"][index + 1]) / 2.0
                cache["key_states"].pop(index + 1)
                cache["value_states"].pop(index + 1)
                cache["modalities"].pop(index + 1)
                cache["lengths"].pop(index + 1)
                cache["surprising_scores"].pop(index + 1)
                torch.cuda.empty_cache()
    elif consolidation_method == "drop":
        while sum(cache["lengths"]) > consolidation_mem_budget:
            index = int(np.array(cache["surprising_scores"][1:]).argmin())
            cache["key_states"].pop(index + 1)
            cache["value_states"].pop(index + 1)
            cache["modalities"].pop(index + 1)
            cache["lengths"].pop(index + 1)
            cache["surprising_scores"].pop(index + 1)
            torch.cuda.empty_cache()
    else:
        raise NotImplementedError(f"Unsupported consolidation_method: {consolidation_method}")


def _patch_cambrian_qwen2(model) -> None:
    for layer in model.model.layers:
        layer.self_attn.__class__ = Qwen2SdpaAttention
    from cambrian.model.language_model.cambrian_qwen2 import CambrianQwenModel

    CambrianQwenModel.forward = cambrian_qwen2_forward


@register_model("cambrians_vsr")
class CambriansVSR(CambrianS):
    def __init__(
        self,
        pretrained: str = "",
        torch_dtype: Optional[Union[str, torch.dtype]] = "float16",
        batch_size: Optional[Union[int, str]] = 1,
        device_map: str = "cuda:0",
        conv_template: str = "qwen_2",
        use_cache: bool = True,
        truncate_context: bool = False,
        video_max_frames: int = -1,
        video_fps: int = 1,
        video_force_sample: bool = False,
        add_time_instruction: bool = False,
        miv_token_len: int = 64,
        si_token_len: int = 729,
        image_aspect_ratio: str = "anyres",
        anyres_max_subimages: int = 9,
        enable_visual_feature_caching: bool = False,
        sensory_window_size: int = 32,
        surprise_threshold: float = 0.0,
        compression_downsample_ratio: int = 2,
        consolidation_method: str = "drop",
        consolidation_mem_budget: int = 8192,
        retrieval_topk: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            torch_dtype=torch_dtype,
            batch_size=batch_size,
            device_map=device_map,
            conv_template=conv_template,
            use_cache=use_cache,
            truncate_context=truncate_context,
            video_max_frames=video_max_frames,
            video_fps=video_fps,
            video_force_sample=video_force_sample,
            add_time_instruction=add_time_instruction,
            miv_token_len=miv_token_len,
            si_token_len=si_token_len,
            image_aspect_ratio=image_aspect_ratio,
            anyres_max_subimages=anyres_max_subimages,
            **kwargs,
        )

        self.enable_visual_feature_caching = enable_visual_feature_caching
        self.sensory_window_size = sensory_window_size
        self.surprise_threshold = surprise_threshold
        self.compression_downsample_ratio = compression_downsample_ratio
        self.consolidation_method = consolidation_method
        self.consolidation_mem_budget = consolidation_mem_budget
        self.retrieval_topk = retrieval_topk

        eval_logger.info(f"sensory_window_size: {sensory_window_size}")
        eval_logger.info(f"surprise_threshold: {surprise_threshold}")
        eval_logger.info(f"compression_downsample_ratio: {compression_downsample_ratio}")
        eval_logger.info(f"consolidation_method: {consolidation_method}")
        eval_logger.info(f"consolidation_mem_budget: {consolidation_mem_budget}")
        eval_logger.info(f"retrieval_topk: {retrieval_topk}")

        if self.enable_visual_feature_caching:
            self.cache_dir = os.path.join(".cache", self.pretrained, "visual_features")
            os.makedirs(self.cache_dir, exist_ok=True)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, requests, task_dict, tokenizer, image_processor, model_config, conv_template, enable_visual_feature_caching, cache_dir):
                self.requests = requests
                self.task_dict = task_dict
                self.tokenizer = tokenizer
                self.image_processor = image_processor
                self.model_config = model_config
                self.conv_template = conv_template
                self.enable_visual_feature_caching = enable_visual_feature_caching
                self.cache_dir = cache_dir

            def __len__(self):
                return len(self.requests)

            def __getitem__(self, idx):
                def feature_path_exists(paths):
                    if not self.enable_visual_feature_caching or not self.cache_dir:
                        return False
                    for path in paths:
                        if not os.path.exists(os.path.join(self.cache_dir, path.replace("/", "_") + ".pt")):
                            return False
                    return True

                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = self.requests[idx].args
                visuals = doc_to_visual(self.task_dict[task][split][doc_id])

                visual_tensors_type = "none"
                visual_tensor_path = None

                if visuals is not None:
                    qs = contexts
                    assert len(visuals) == 1
                    assert isinstance(visuals[0], str)
                    assert is_video_file(visuals[0])

                    visual_tensor_path = visuals[0].replace("/", "_") + ".pt"
                    if feature_path_exists(visuals):
                        visual_tensors = torch.load(os.path.join(self.cache_dir, visual_tensor_path))
                        vit_visual_tensors = torch.load(os.path.join(self.cache_dir, visual_tensor_path.replace(".pt", "_vit.pt")))
                        visual_sizes = torch.load(os.path.join(self.cache_dir, visual_tensor_path.replace(".pt", "_size.pt")))
                        visual_tensors = (visual_tensors, vit_visual_tensors)
                        visual_tensors_type = "feature"
                    else:
                        visual_tensors, visual_sizes, _ = process_videos_vsr(visuals, self.image_processor, num_threads=-1)
                        visual_tensors_type = "raw"

                    if isinstance(qs, str):
                        if self.model_config.mm_use_im_start_end:
                            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                        else:
                            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                    else:
                        raise NotImplementedError
                else:
                    visual_tensors = None
                    visual_sizes = None
                    qs = contexts

                conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
                return input_ids, visual_tensors_type, visual_tensors, visual_sizes, prompt, gen_kwargs, visual_tensor_path

        dataset = Dataset(
            requests,
            self.task_dict,
            self.tokenizer,
            self._image_processor,
            self._config,
            self.conv_template,
            self.enable_visual_feature_caching,
            getattr(self, "cache_dir", None),
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=0, pin_memory=True)

        for _, (input_ids, visual_tensors_type, visual_tensors, visual_sizes, cur_prompt, gen_kwargs, visual_tensor_path) in enumerate(dataloader):
            gen_kwargs.setdefault("max_new_tokens", 16)
            gen_kwargs.setdefault("temperature", 0)
            gen_kwargs.setdefault("top_p", None)
            gen_kwargs.setdefault("num_beams", 1)

            with torch.inference_mode():
                def add_newline_tokens(visual_features: torch.Tensor) -> torch.Tensor:
                    visual_features = torch.cat(
                        [visual_features, self.model.model.image_newline[None, None, None, :].expand(*visual_features.size()[:2], 1, -1)],
                        dim=2,
                    )
                    return visual_features.flatten(1, 2).flatten(0, 1)

                if visual_tensors_type == "raw":
                    visual_tensors = visual_tensors[0].flatten(0, 1)
                    visual_features = []
                    vit_visual_features = []
                    block_size = 128
                    miv_token_len = self.model.get_model().config.miv_token_len
                    miv_side_len = int(math.sqrt(miv_token_len))

                    for block_idx in range(math.ceil(visual_tensors.size(0) / block_size)):
                        chunked_visual_features = visual_tensors[block_idx * block_size : (block_idx + 1) * block_size].half().to(self._device)
                        chunked_visual_features = self.model.encode_images([chunked_visual_features])[0]
                        vit_chunked_visual_features = chunked_visual_features.clone()
                        chunked_visual_features = self.model.get_model().mm_projector(chunked_visual_features)

                        feature_side_len = int(math.sqrt(chunked_visual_features.size(1)))
                        chunked_visual_features = chunked_visual_features.unflatten(1, (feature_side_len, feature_side_len)).permute(0, 3, 1, 2)
                        vit_chunked_visual_features = vit_chunked_visual_features.unflatten(1, (feature_side_len, feature_side_len)).permute(0, 3, 1, 2)

                        if feature_side_len != miv_side_len:
                            chunked_visual_features = torch.nn.functional.interpolate(
                                chunked_visual_features,
                                size=(miv_side_len, miv_side_len),
                                mode="bilinear",
                                align_corners=False,
                            ).permute(0, 2, 3, 1)
                            vit_chunked_visual_features = torch.nn.functional.interpolate(
                                vit_chunked_visual_features,
                                size=(miv_side_len, miv_side_len),
                                mode="bilinear",
                                align_corners=False,
                            ).permute(0, 2, 3, 1)

                        visual_features.append(chunked_visual_features)
                        vit_visual_features.append(vit_chunked_visual_features)

                    visual_features = torch.cat(visual_features, dim=0)
                    vit_visual_features = torch.cat(vit_visual_features, dim=0)

                    if self.enable_visual_feature_caching and visual_tensor_path:
                        os.makedirs(self.cache_dir, exist_ok=True)
                        torch.save(visual_features.cpu(), os.path.join(self.cache_dir, visual_tensor_path))
                        torch.save(vit_visual_features.cpu(), os.path.join(self.cache_dir, visual_tensor_path.replace(".pt", "_vit.pt")))
                        torch.save(visual_sizes, os.path.join(self.cache_dir, visual_tensor_path.replace(".pt", "_size.pt")))
                elif visual_tensors_type == "feature":
                    visual_tensors, vit_visual_features = visual_tensors
                    visual_features = visual_tensors.to(self._device)
                else:
                    raise NotImplementedError("Cambrian VSR requires one video input.")

                visual_features = unpad_image(visual_features, visual_sizes[0][:2])
                vit_visual_features = unpad_image(vit_visual_features, visual_sizes[0][:2])

                _patch_cambrian_qwen2(self.model)

                input_ids = input_ids.to(self._device)
                image_token_pos = int(torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0][0].item())
                pre_img_tokens = input_ids[:, :image_token_pos]
                post_img_tokens = input_ids[:, image_token_pos + 1 :]
                pre_img_embeds = self.model.get_input_embeddings()(pre_img_tokens)

                global_kv_cache = [{"key_states": [], "value_states": [], "modalities": [], "lengths": [], "surprising_scores": []} for _ in range(self.model.config.num_hidden_layers)]
                runtime_kv_cache = [{"key_states": [], "value_states": [], "modalities": [], "lengths": [], "surprising_scores": []} for _ in range(self.model.config.num_hidden_layers)]

                out = self.model(
                    input_ids=None,
                    inputs_embeds=pre_img_embeds,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=None,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

                for layer_idx, (key_states, value_states) in enumerate(out.past_key_values):
                    _append_cache_entry(global_kv_cache[layer_idx], key_states, value_states, "T", key_states.size(2), 1.0)
                    _append_cache_entry(runtime_kv_cache[layer_idx], key_states, value_states, "T", key_states.size(2), 1.0)

                for frame_idx in range(visual_features.size(0)):
                    past_key_values = []
                    for layer_cache in runtime_kv_cache:
                        past_key_values.append((torch.cat(layer_cache["key_states"], dim=2), torch.cat(layer_cache["value_states"], dim=2)))

                    frame_feature = visual_features[frame_idx : frame_idx + 1]
                    if frame_idx == 0:
                        surprisingness_score = 1.0
                    else:
                        frame_feature_prediction = frame_feature_prediction.unflatten(1, (vit_visual_features.size(1), vit_visual_features.size(2) + 1))[:, :, :-1]
                        surprisingness_score = 1 - torch.cosine_similarity(
                            frame_feature_prediction.flatten(1, 2),
                            vit_visual_features[frame_idx : frame_idx + 1].flatten(1, 2).to(frame_feature_prediction.device),
                            dim=-1,
                        ).mean(1).item()

                    out = self.model(
                        input_ids=None,
                        inputs_embeds=add_newline_tokens(frame_feature).unsqueeze(0),
                        attention_mask=None,
                        position_ids=None,
                        use_cache=True,
                        return_dict=True,
                        past_key_values=past_key_values,
                        output_attentions=False,
                        output_hidden_states=True,
                    )

                    frame_feature_prediction = self.model.model.nfp_head(out.hidden_states)

                    for layer_idx, (layer_wise_past_key_values, layer_wise_output_cache) in enumerate(zip(past_key_values, out.past_key_values)):
                        input_seq_len = layer_wise_past_key_values[0].size(2)
                        key_states = layer_wise_output_cache[0][..., input_seq_len:, :].clone()
                        value_states = layer_wise_output_cache[1][..., input_seq_len:, :].clone()
                        _append_cache_entry(runtime_kv_cache[layer_idx], key_states, value_states, "I", key_states.size(2), surprisingness_score)

                        if self.sensory_window_size > 0 and len(runtime_kv_cache[layer_idx]["key_states"]) > self.sensory_window_size + 1:
                            key_states = runtime_kv_cache[layer_idx]["key_states"].pop(1)
                            value_states = runtime_kv_cache[layer_idx]["value_states"].pop(1)
                            modality = runtime_kv_cache[layer_idx]["modalities"].pop(1)
                            surprise_score = runtime_kv_cache[layer_idx]["surprising_scores"].pop(1)
                            length = runtime_kv_cache[layer_idx]["lengths"].pop(1)

                            if self.compression_downsample_ratio > 1 and surprise_score < self.surprise_threshold:
                                key_states = downsample_cache_states(key_states, self.compression_downsample_ratio, visual_features)
                                value_states = downsample_cache_states(value_states, self.compression_downsample_ratio, visual_features)
                                length = key_states.size(2)

                            _append_cache_entry(global_kv_cache[layer_idx], key_states, value_states, modality, length, surprise_score)
                            _maybe_consolidate_global_cache(
                                global_kv_cache[layer_idx],
                                self.surprise_threshold,
                                self.consolidation_method,
                                self.consolidation_mem_budget,
                            )

                for layer_idx in range(self.model.config.num_hidden_layers):
                    for cache_idx in range(1, len(runtime_kv_cache[layer_idx]["key_states"])):
                        key_states = runtime_kv_cache[layer_idx]["key_states"][cache_idx]
                        value_states = runtime_kv_cache[layer_idx]["value_states"][cache_idx]
                        surprise_score = runtime_kv_cache[layer_idx]["surprising_scores"][cache_idx]
                        length = runtime_kv_cache[layer_idx]["lengths"][cache_idx]

                        if self.compression_downsample_ratio > 1 and surprise_score < self.surprise_threshold:
                            key_states = downsample_cache_states(key_states, self.compression_downsample_ratio, visual_features)
                            value_states = downsample_cache_states(value_states, self.compression_downsample_ratio, visual_features)
                            length = key_states.size(2)

                        _append_cache_entry(
                            global_kv_cache[layer_idx],
                            key_states,
                            value_states,
                            runtime_kv_cache[layer_idx]["modalities"][cache_idx],
                            length,
                            surprise_score,
                        )
                        _maybe_consolidate_global_cache(
                            global_kv_cache[layer_idx],
                            self.surprise_threshold,
                            self.consolidation_method,
                            self.consolidation_mem_budget,
                        )

                past_key_values = []
                for layer_idx in range(self.model.config.num_hidden_layers):
                    global_kv_cache[layer_idx]["key_states"] = torch.cat(global_kv_cache[layer_idx]["key_states"], dim=2)
                    global_kv_cache[layer_idx]["value_states"] = torch.cat(global_kv_cache[layer_idx]["value_states"], dim=2)
                    past_key_values.append((global_kv_cache[layer_idx]["key_states"], global_kv_cache[layer_idx]["value_states"]))

                if self.retrieval_topk > 1:
                    for layer_idx, layer in enumerate(self.model.model.layers):
                        layer.self_attn.use_retrieval = True
                        layer.self_attn.retrieval_topk = self.retrieval_topk
                        layer.self_attn.cache_modalities = global_kv_cache[layer_idx]["modalities"]
                        layer.self_attn.cache_lengths = global_kv_cache[layer_idx]["lengths"]

                post_img_embeds = self.model.get_input_embeddings()(post_img_tokens)
                out = self.model(
                    input_ids=None,
                    inputs_embeds=post_img_embeds,
                    attention_mask=None,
                    position_ids=None,
                    use_cache=True,
                    return_dict=True,
                    past_key_values=past_key_values,
                    output_attentions=False,
                    output_hidden_states=True,
                )
                past_key_values = out.past_key_values

                logits = out.logits[:, -1, :]
                pred = logits.argmax(dim=-1)
                output_ids = torch.cat([torch.zeros_like(pred)[:, None].long().fill_(self._tokenizer.pad_token_id), pred[:, None]], dim=1)

                for _ in range(gen_kwargs["max_new_tokens"] - 1):
                    if pred == self._tokenizer.eos_token_id:
                        break
                    out = self.model(
                        input_ids=output_ids[:, -1:],
                        inputs_embeds=None,
                        attention_mask=None,
                        position_ids=None,
                        use_cache=True,
                        return_dict=True,
                        past_key_values=past_key_values,
                        output_attentions=False,
                        output_hidden_states=True,
                    )
                    past_key_values = out.past_key_values
                    logits = out.logits[:, -1, :]
                    score = torch.gather(logits, 1, output_ids)
                    score = torch.where(score < 0, score * 1.1, score / 1.1)
                    logits.scatter_(1, output_ids, score)
                    pred = logits.argmax(dim=-1)
                    output_ids = torch.cat([output_ids, pred[:, None]], dim=-1)

            outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            res.append(outputs)
            pbar.update(1)

        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError
