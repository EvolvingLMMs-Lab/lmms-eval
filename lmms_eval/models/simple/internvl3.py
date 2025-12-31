import logging
from datetime import timedelta
from typing import List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt
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

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("eval_logger")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


def build_transform(input_size: int) -> T.Compose:
    """Build image transformation pipeline for preprocessing.

    Args:
        input_size: Target size for the image (both width and height).

    Returns:
        A torchvision Compose transform pipeline.
    """
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return transform


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Set[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """Find the closest aspect ratio from a set of target ratios.

    Args:
        aspect_ratio: The aspect ratio of the input image.
        target_ratios: Set of candidate (width, height) ratio tuples.
        width: Original image width.
        height: Original image height.
        image_size: Base image size for area calculation.

    Returns:
        The best matching (width, height) ratio tuple.
    """
    best_ratio_diff = float("inf")
    best_ratio: Tuple[int, int] = (1, 1)
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


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    """Dynamically preprocess an image by splitting it into tiles.

    Args:
        image: Input PIL Image to process.
        min_num: Minimum number of tiles.
        max_num: Maximum number of tiles.
        image_size: Size of each tile.
        use_thumbnail: Whether to append a thumbnail of the original image.

    Returns:
        List of processed image tiles.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios: Set[Tuple[int, int]] = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num)
    sorted_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, set(sorted_ratios), orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image: Image.Image, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    """Load and preprocess an image into pixel values.

    Args:
        image: Input PIL Image.
        input_size: Target size for image tiles.
        max_num: Maximum number of tiles.

    Returns:
        Stacked tensor of preprocessed image tiles.
    """
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values_tensor = torch.stack(pixel_values)
    return pixel_values_tensor


def get_index(
    bound: Optional[Tuple[float, float]],
    fps: float,
    max_frame: int,
    first_idx: int = 0,
    num_segments: int = 32,
) -> npt.NDArray[np.int64]:
    """Get frame indices for video sampling.

    Args:
        bound: Optional tuple of (start, end) timestamps in seconds.
        fps: Frames per second of the video.
        max_frame: Maximum frame index in the video.
        first_idx: First frame index to consider.
        num_segments: Number of frames to sample.

    Returns:
        Array of frame indices to sample.
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000.0, 100000.0
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def load_video(
    video_path: str,
    bound: Optional[Tuple[float, float]] = None,
    input_size: int = 448,
    max_num: int = 1,
    num_segments: int = 32,
) -> Tuple[torch.Tensor, List[int]]:
    """Load and preprocess a video into pixel values.

    Args:
        video_path: Path to the video file.
        bound: Optional tuple of (start, end) timestamps in seconds.
        input_size: Target size for image tiles.
        max_num: Maximum number of tiles per frame.
        num_segments: Number of frames to sample.

    Returns:
        Tuple of (pixel_values tensor, list of patch counts per frame).
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list: List[torch.Tensor] = []
    num_patches_list: List[int] = []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values_tensor = torch.cat(pixel_values_list)
    return pixel_values_tensor, num_patches_list


@register_model("internvl3")
class InternVL3(lmms):
    """InternVL3 model wrapper for lmms-eval.

    This class provides support for evaluating InternVL3 models on various
    multimodal benchmarks. It supports both image and video modalities.

    Args:
        pretrained: HuggingFace model path or local path.
        modality: Input modality, either "image" or "video".
        device: Device to use for inference.
        device_map: Device mapping strategy. Use "auto" for multi-GPU.
        batch_size: Batch size (must be 1 for this model).
        num_frame: Number of frames to sample for video inputs.
        max_num: Maximum number of image tiles.
        use_flash_attn: Whether to use flash attention.
    """

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3-8B",
        modality: str = "image",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        num_frame: int = 32,
        max_num: int = 12,
        use_flash_attn: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.path = pretrained
        self.num_frame = num_frame
        self.max_num = max_num

        batch_size_int = int(batch_size)
        assert batch_size_int == 1, f"Batch size should be 1 for InternVL3, but got {batch_size_int}."
        self.batch_size_per_gpu = batch_size_int

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        self._use_auto_device_map = False

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device("cuda:0")
            self.device_map = "auto"
            self._use_auto_device_map = True
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self._model = AutoModel.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=True,
            device_map=self.device_map,
        ).eval()
        self._config = self._model.config
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True, use_fast=False)

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
        elif self._use_auto_device_map:
            eval_logger.info("Using auto device map for tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self.modality = modality

    @property
    def config(self):
        """Return the model configuration."""
        return self._config

    @property
    def tokenizer(self):
        """Return the tokenizer."""
        return self._tokenizer

    @property
    def model(self):
        """Return the unwrapped model."""
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def batch_size(self) -> int:
        """Return the batch size per GPU."""
        return self.batch_size_per_gpu

    @property
    def device(self) -> torch.device:
        """Return the device."""
        return self._device

    @property
    def rank(self) -> int:
        """Return the process rank."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Return the world size."""
        return self._world_size

    def flatten(self, input: List[List]) -> List:
        """Flatten a nested list."""
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate responses for a list of requests.

        Args:
            requests: List of Instance objects containing generation requests.

        Returns:
            List of generated response strings.
        """
        res: List[str] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

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
                    processed_visuals = [load_image(visual, max_num=self.max_num).to(torch.bfloat16).to(self._device) for visual in visuals]
                    pixel_values = torch.cat(processed_visuals, dim=0)
                    num_patches_list = [v.size(0) for v in processed_visuals]
                    image_tokens = " ".join(["<image>"] * len(processed_visuals))
                    contexts = image_tokens + "\n" + contexts
                else:
                    pixel_values = None
                    num_patches_list = None
                response, _ = self.model.chat(
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
                pixel_values, num_patches_list = load_video(video_path, num_segments=self.num_frame, max_num=1)
                pixel_values = pixel_values.to(torch.bfloat16).to(self._device)
                video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
                question = video_prefix + contexts
                response, _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    gen_kwargs,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )
            res.append(response)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood for requests. Not implemented for InternVL3."""
        raise NotImplementedError("Loglikelihood is not implemented for InternVL3.")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """Generate multi-round responses. Not implemented for InternVL3."""
        raise NotImplementedError("Multi-round generation is not implemented for InternVL3.")
