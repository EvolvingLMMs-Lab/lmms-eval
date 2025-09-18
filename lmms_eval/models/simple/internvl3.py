import logging
import math
import json
import os
import random
import atexit
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..model_utils.internvl_utils import adaptive_keyframe_sampling

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("eval_logger")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Set global seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    repetition_penalty=1.0,
)


def build_transform(input_size):
    """Build a transformation pipeline for image preprocessing.

    Args:
        input_size (int): The target size for the input images.

    Returns:
        T.Compose: The image transformation pipeline.
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from a set of target ratios.

    Args:
        aspect_ratio (float): The aspect ratio of the original image.
        target_ratios (List[Tuple[int, int]]): A list of target aspect ratios to consider.
        width (int): The width of the original image.
        height (int): The height of the original image.
        image_size (int): The size of the image after resizing.

    Returns:
        Tuple[int, int]: The closest aspect ratio as a tuple (width, height).
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamically preprocess the input image by resizing and splitting it into multiple patches.

    Args:
        image (PIL.Image): The input image to preprocess.
        min_num (int, optional): The minimum number of patches to create. Defaults to 1.
        max_num (int, optional): The maximum number of patches to create. Defaults to 12.
        image_size (int, optional): The size of the image after resizing. Defaults to 448.
        use_thumbnail (bool, optional): Whether to use a thumbnail of the image. Defaults to False.

    Returns:
        List[PIL.Image]: A list of preprocessed image patches.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12):
    """Load and preprocess the input image.

    Args:
        image (PIL.Image): The input image to preprocess.
        input_size (int, optional): The size of the image after resizing. Defaults to 448.
        max_num (int, optional): The maximum number of patches to create. Defaults to 12.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """Get the frame indices for each segment.

    Args:
        bound (tuple): The start and end time bounds for the segment.
        fps (float): The frames per second of the video.
        max_frame (int): The maximum number of frames in the video.
        first_idx (int, optional): The index of the first frame to include. Defaults to 0.
        num_segments (int, optional): The number of segments to divide the video into. Defaults to 32.

    Returns:
        np.ndarray: The frame indices for each segment.
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=6, num_segments=32, use_adaptive_sampling=True, query=None):
    """Load and preprocess the input video - matching internvl_utils.py implementation

    Args:
        video_path (str): The path to the video file.
        bound (tuple, optional): The start and end time bounds for the segment. Defaults to None.
        input_size (int, optional): The size of the video frames after resizing. Defaults to 448.
        max_num (int, optional): The maximum number of patches to create. Defaults to 6.
        num_segments (int, optional): The number of segments to divide the video into. Defaults to 32.
        use_adaptive_sampling (bool, optional): Whether to use adaptive keyframe sampling. Defaults to True.
        query (str, optional): The query for adaptive sampling. Defaults to None.

    Returns:
        tuple: (pixel_values, num_patches_list, frame_times, video_length)
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    video_length = len(vr) / fps

    pixel_values_list, num_patches_list, frame_times = [], [], []
    transform = build_transform(input_size=input_size)

    # Frame selection - match internvl_utils.py logic
    if use_adaptive_sampling and query:
        frame_indices = adaptive_keyframe_sampling(video_path, num_segments, query)
    else:
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    eval_logger.debug(f"DEBUG: load_video - video_path: {video_path}, total_frames: {len(vr)}, requested_segments: {num_segments}, selected_indices: {len(frame_indices)}, video_length: {video_length:.2f}s")

    # Process frames like workshop
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)  # Note: use_thumbnail=True like workshop
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
        frame_times.append(frame_index / fps)  # Match internvl_utils.py naming

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list, frame_times, video_length  # Return frame_times instead of image_times


def split_model(model_name, num_layers=None):
    """Split the model into smaller parts for distributed training.

    Args:
        model_name (str): The name of the model to split.
        num_layers (int, optional): The number of layers in the model. Defaults to None.

    Returns:
        dict: A mapping of layer names to device IDs.
    """
    device_map = {}
    world_size = torch.cuda.device_count()
    if num_layers is None:
        # InternVL3-8B uses Qwen2.5-7B which has 28 layers
        num_layers = {
            "InternVL3-1B": 16,  # Based on Qwen2.5-0.5B
            "InternVL3-2B": 24,  # Based on Qwen2.5-1.5B
            "InternVL3-8B": 28,  # Based on Qwen2.5-7B
            "InternVL3-14B": 40,  # Based on Qwen2.5-14B
            "InternVL3-38B": 64,  # Based on Qwen2.5-32B
            "InternVL3-78B": 80,  # Based on Qwen2.5-72B
        }.get(
            model_name.split("/")[-1], 28  # Default to 28 for InternVL3-8B
        )

    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


@register_model("internvl3")
class InternVL3(lmms):
    """InternVL3-8B model implementation for lmms-eval

    Args:
        lmms (lmms): The base lmms class.
    """

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3-8B",
        modality: str = "image",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        num_frame: int = 32,
        dynamic_image_size=False,
        use_temporal_context: bool = True,  # Enable enhanced temporal context by default
        use_adaptive_sampling: bool = False,  # Disable adaptive sampling by default
        num_layers=None,
        max_num: int = 6,  # Maximum number of image tiles
        **kwargs,
    ):
        super().__init__()

        self.path = pretrained
        self.num_frame = num_frame
        self.max_num = max_num
        self.use_temporal_context = use_temporal_context
        self.use_adaptive_sampling = use_adaptive_sampling

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for InternVL3, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        self.dynamic_image_size = dynamic_image_size

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            device_map = split_model(pretrained, num_layers=num_layers)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # Load the model with trust_remote_code=True for InternVL3
        self._model = AutoModel.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=self.device_map,
        ).eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.path,
            trust_remote_code=True,
            use_fast=False,  # InternVL3 recommendation
        )

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."

            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

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
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self.modality = modality

        # Initialize debug log file (disabled to avoid unwanted files)
        self.debug_log_file = None
        
        # Initialize cache file tracking for cleanup
        self._cache_file_path = "mammalps_enhanced_prompts.json"

        # Register cleanup function to run at exit
        atexit.register(self._cleanup_cache_file)

    def _cleanup_cache_file(self):
        """Helper method to clean up the cache file."""
        try:
            import os

            if hasattr(self, "_cache_file_path") and os.path.exists(self._cache_file_path):
                os.remove(self._cache_file_path)
                eval_logger.debug(f"Cleaned up cache file: {self._cache_file_path}")
        except Exception as e:
            eval_logger.debug(f"Could not clean up cache file: {e}")

    def __del__(self):
        """Cleanup cache file when model instance is destroyed."""
        self._cleanup_cache_file()

    @property
    def config(self):
        return self._model.config

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

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        """Generate responses until a certain condition is met.

        Args:
            requests (List[Request]): The list of requests to process.

        Raises:
            ValueError: If the requests list is empty or invalid.

        Returns:
            List[str]: The generated responses.
        """
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="InternVL3 Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            pop_keys = []
            for k, v in gen_kwargs.items():
                if k not in DEFAULT_GEN_KWARGS:
                    pop_keys.append(k)

            for k in pop_keys:
                gen_kwargs.pop(k)

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            if self.modality == "image":
                if visuals:
                    visuals = [load_image(visual, max_num=self.max_num).to(torch.bfloat16).cuda() for visual in visuals]
                    pixel_values = torch.cat(visuals, dim=0)
                    num_patches_list = [visual.size(0) for visual in visuals]
                    image_tokens = ["<image>"] * len(visuals)
                    image_tokens = " ".join(image_tokens)
                    contexts = image_tokens + "\n" + contexts
                else:
                    pixel_values = None
                    num_patches_list = None

                response, history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    contexts,
                    gen_kwargs,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )

            elif self.modality == "video":
                assert len(visuals) == 1, f"Only one video is supported, but got {len(visuals)} videos."
                video_path = visuals[0]
                pixel_values, num_patches_list, frame_times, video_length = load_video(
                    video_path,
                    bound=None,
                    input_size=448,
                    max_num=self.max_num,  # Use the configured max_num
                    num_segments=self.num_frame,
                    use_adaptive_sampling=self.use_adaptive_sampling,  # Use the configured adaptive sampling setting
                    query=contexts,  # Pass the query for potential adaptive sampling
                )
                pixel_values = pixel_values.to(torch.bfloat16).cuda()

                # DEBUG: Print actual frame count and temporal info
                eval_logger.debug(f"DEBUG: Video processing - requested frames: {self.num_frame}, actual frames: {len(num_patches_list)}, patches per frame: {num_patches_list}")
                eval_logger.debug(f"DEBUG: Video temporal info - video_length: {video_length:.2f}s, frame_timestamps: {[f'{t:.2f}s' for t in frame_times[:5]]}{'...' if len(frame_times) > 5 else ''}")

                # Create enhanced video prefix with temporal information
                if hasattr(self, "use_temporal_context") and self.use_temporal_context:
                    # Match exact format from animal_dataset.py
                    special_tokens = "\n".join(["Frame-{} at second {:.2f}: <image>".format(i + 1, frame_times[i]) for i in range(len(num_patches_list))])
                    special_tokens = "The video is {:.2f} second(s) long and you can see the frames below:\n".format(video_length) + special_tokens
                    # Replace <video>\n exactly like animal_dataset.py
                    question = contexts.replace("<video>\n", special_tokens + "\n")
                    # Also handle case without newline for robustness
                    if "<video>" in question and "<video>\n" not in contexts:
                        question = contexts.replace("<video>", special_tokens + "\n")
                else:
                    # Standard version (current implementation)
                    video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
                    question = video_prefix + contexts

                # DEBUG: Log the complete final prompt with timestamps that will be sent to the model
                eval_logger.debug(f"DEBUG: ===== FINAL PROMPT TO MODEL =====")
                eval_logger.debug(f"DEBUG: Video path: {video_path}")
                eval_logger.debug(f"DEBUG: Video length: {video_length:.2f}s, Frames: {len(num_patches_list)}")
                eval_logger.debug(f"DEBUG: Prompt length: {len(question)} chars")
                eval_logger.debug(f"DEBUG: Full prompt:")
                eval_logger.debug(f"DEBUG: ----START----")
                eval_logger.debug(question)
                eval_logger.debug(f"DEBUG: ----END----")
                eval_logger.debug(f"DEBUG: ================================")

                # Save enhanced prompt with timestamps for utils.py to access
                if hasattr(self, "use_temporal_context") and self.use_temporal_context:
                    try:
                        # Save the enhanced prompt to a cache file that utils.py can read
                        prompt_cache_file = "mammalps_enhanced_prompts.json"
                        prompt_cache = {}

                        # Load existing cache if it exists
                        if os.path.exists(prompt_cache_file):
                            try:
                                with open(prompt_cache_file, "r", encoding="utf-8") as f:
                                    prompt_cache = json.load(f)
                            except Exception:
                                prompt_cache = {}

                        # Create cache key from document info
                        doc = self.task_dict[task][split][doc_id]
                        cache_key = f"{doc.get('id', doc_id)}"

                        # Determine subtask (animal, action, activity)
                        subtask = None

                        # Method 1: Try to get subtask from task config
                        task_name = None
                        if hasattr(self, "task") and hasattr(self.task, "_config") and hasattr(self.task._config, "task"):
                            task_name = self.task._config.task
                        elif hasattr(self, "_task_name"):
                            task_name = self._task_name

                        if task_name:
                            eval_logger.debug(f"DEBUG: Task name from config: {task_name}")
                            if "action" in task_name:
                                subtask = "action"
                            elif "activity" in task_name:
                                subtask = "activity"
                            elif "animal" in task_name:
                                subtask = "animal"

                        # Method 2: Robust inference from prompt content
                        if not subtask:
                            eval_logger.debug("DEBUG: No subtask from config, using content analysis")
                            combined_text = contexts + " " + question

                            # Look for specific label spaces and keywords
                            if "activity label space" in combined_text or "activities label space" in combined_text:
                                subtask = "activity"
                            elif "action label space" in combined_text or "actions label space" in combined_text:
                                subtask = "action"
                            elif "animal label space" in combined_text or "animals label space" in combined_text:
                                subtask = "animal"
                            # Additional pattern matching
                            elif "list of activities" in combined_text.lower() or "animal activities" in combined_text.lower():
                                subtask = "activity"
                            elif "list of actions" in combined_text.lower() or "animal actions" in combined_text.lower():
                                subtask = "action"
                            elif "list of animals" in combined_text.lower() or "animal types" in combined_text.lower():
                                subtask = "animal"
                            else:
                                subtask = "unknown"

                        eval_logger.debug(f"DEBUG: Detected subtask: {subtask} for doc_id: {cache_key}")
                        eval_logger.debug(f"DEBUG: Content preview: {combined_text[:300]}...")

                        # Store the enhanced prompt in a subtask-specific way
                        if cache_key not in prompt_cache:
                            prompt_cache[cache_key] = {}
                        prompt_cache[cache_key][subtask] = {"original_prompt": contexts, "enhanced_prompt_with_timestamps": question, "video_length": video_length, "frame_timestamps": frame_times, "num_frames": len(num_patches_list)}

                        eval_logger.debug(f"DEBUG: Saving enhanced prompt to cache for doc_id: {cache_key}, subtask: {subtask}")
                        eval_logger.debug(f"DEBUG: Enhanced prompt preview: {question[:200]}...")

                        # Save back to cache file
                        with open(prompt_cache_file, "w", encoding="utf-8") as f:
                            json.dump(prompt_cache, f, ensure_ascii=False, indent=2)

                        eval_logger.debug(f"DEBUG: Successfully saved enhanced prompt to cache file: {prompt_cache_file}")

                    except Exception as e:
                        eval_logger.debug(f"DEBUG: Failed to save enhanced prompt to cache: {e}")

                response, history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    gen_kwargs,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )

            else:
                raise ValueError(f"Unsupported modality: {self.modality}")

            res.append(response)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented yet."

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for InternVL3")
