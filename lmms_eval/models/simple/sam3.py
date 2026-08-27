import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from transformers import Sam3Model, Sam3Processor

    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False
    eval_logger.warning("SAM3 not found. Please install transformers with SAM3 support.")

try:
    from pycocotools import mask as mask_utils

    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    eval_logger.warning("pycocotools not found. Install with: pip install pycocotools")

try:
    from accelerate import Accelerator, DistributedType

    HAS_ACCELERATOR = True
except ImportError:
    HAS_ACCELERATOR = False
    eval_logger.warning("Accelerate not found. Multi-GPU evaluation will not work.")


def mask_to_coco_rle(binary_mask: np.ndarray) -> Dict:
    """Convert a binary mask (H, W) to COCO RLE format via pycocotools.

    Returns ``{"counts": "<rle_string>", "size": [H, W]}``.
    This is the universal COCO RLE representation used by pycocotools,
    COCO evaluation tools, and SAM3 evaluation scripts.
    """
    mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(mask_fortran)
    # pycocotools returns bytes for counts — convert to str for JSON
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


@register_model("sam3")
class SAM3(lmms):
    """SAM3 (Segment Anything Model 3) for text-prompted instance segmentation.

    SAM3 is *not* a text-generation model.  It takes an image and a short text
    prompt and returns segmentation masks, bounding boxes and confidence scores.

    In order to fit into the lmms-eval ``generate_until`` interface the model
    serialises its structured outputs as a JSON string containing **both**
    masks and bounding boxes for every request::

        {
            "masks": [                         # COCO RLE encoded masks
                {"counts": "...", "size": [H, W]},
                ...
            ],
            "boxes": [                         # normalised xyxy bounding boxes
                {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                ...
            ],
            "scores": [0.95, 0.87, ...]        # confidence scores
        }

    Downstream task ``process_results`` functions pick whichever fields they
    need (masks for segmentation tasks, boxes for detection tasks).

    Example
    -------
    ::

        python -m lmms_eval \\
            --model sam3 \\
            --model_args pretrained=facebook/sam3 \\
            --tasks saco_gold_attributes_seg \\
            --output_path ./results
    """

    is_simple = True

    def __init__(
        self,
        pretrained: str = "facebook/sam3",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()

        if not HAS_SAM3:
            raise ImportError("SAM3 is required but not installed.  " "Please install / upgrade transformers with SAM3 support.")
        if not HAS_PYCOCOTOOLS:
            raise ImportError("pycocotools is required for COCO RLE mask encoding.  " "Install with: pip install pycocotools")

        self.pretrained = pretrained
        self.threshold = threshold
        self.mask_threshold = mask_threshold

        eval_logger.info(f"Loading SAM3 model: {pretrained}")
        eval_logger.info(f"threshold={threshold}, mask_threshold={mask_threshold}")

        if kwargs:
            eval_logger.info(f"Additional kwargs: {kwargs}")

        # ---- Distributed / device setup ---------------------------------- #
        accelerator = Accelerator() if HAS_ACCELERATOR else None

        if accelerator is not None and accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
            eval_logger.info(f"Distributed mode: rank {self._rank}/{self._world_size}")
        else:
            if torch.cuda.is_available() and device is not None:
                self._device = torch.device(device)
            else:
                self._device = torch.device("cpu")
            self.device_map = str(self._device)
            self._rank = 0
            self._world_size = 1

        self._accelerator = accelerator

        # ---- Load model & processor -------------------------------------- #
        self._model = Sam3Model.from_pretrained(pretrained).to(self._device).eval()
        self._processor = Sam3Processor.from_pretrained(pretrained)

        eval_logger.info("SAM3 model and processor loaded successfully")

        self.batch_size_per_gpu = int(batch_size)

    # -- Properties expected by the base class / framework ----------------- #

    @property
    def model(self):
        return self._model

    @property
    def processor(self):
        return self._processor

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

    # -- Helpers ----------------------------------------------------------- #

    @staticmethod
    def _to_pil(image) -> Image.Image:
        """Coerce *image* to an RGB PIL Image."""
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise ValueError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _format_results(results: Dict, img_width: int, img_height: int) -> str:
        """Serialise SAM3 post-processed outputs to a JSON string.

        Returns a JSON object with ``masks`` (COCO RLE), ``boxes`` (normalised
        xyxy) and ``scores``.  Downstream task utils pick the fields they need.
        """
        # -- Masks → COCO RLE ---------------------------------------------- #
        raw_masks = results.get("masks", [])
        rle_masks = []
        for mask in raw_masks:
            mask_np = mask.cpu().numpy().astype(np.uint8)
            rle_masks.append(mask_to_coco_rle(mask_np))

        # -- Boxes → normalised xyxy --------------------------------------- #
        raw_boxes = results.get("boxes", [])
        norm_boxes = []
        for box in raw_boxes:
            box_cpu = box.cpu() if torch.is_tensor(box) else box
            norm_boxes.append(
                {
                    "x_min": float(box_cpu[0] / img_width),
                    "y_min": float(box_cpu[1] / img_height),
                    "x_max": float(box_cpu[2] / img_width),
                    "y_max": float(box_cpu[3] / img_height),
                }
            )

        # -- Scores -------------------------------------------------------- #
        raw_scores = results.get("scores", [])
        scores = [float(s) for s in raw_scores]

        return json.dumps({"masks": rle_masks, "boxes": norm_boxes, "scores": scores})

    # -- Core interface methods -------------------------------------------- #

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not applicable for SAM3 — returns dummy values."""
        eval_logger.warning("loglikelihood is not applicable for SAM3")
        return [(0.0, True) for _ in requests]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Run SAM3 inference and return results as JSON strings.

        Each request yields a JSON object containing COCO-RLE masks, normalised
        bounding boxes and confidence scores.  The downstream task
        ``process_results`` function picks whichever fields it needs.
        """
        res: List[str] = []
        empty_result = json.dumps({"masks": [], "boxes": [], "scores": []})

        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="SAM3 Inference",
        )

        for req in requests:
            try:
                # Unpack instance arguments (simple model format)
                doc_id = req.args[3]
                task = req.args[4]
                split = req.args[5]
                doc_data = self.task_dict[task][split][doc_id]

                image = doc_data["image"]
                expression = doc_data["expression"]

                image_pil = self._to_pil(image)

                # Prepare inputs
                inputs = self._processor(
                    images=image_pil,
                    text=expression,
                    return_tensors="pt",
                ).to(self.device)

                # Forward pass
                with torch.no_grad():
                    outputs = self._model(**inputs)

                # Post-process
                results = self._processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.threshold,
                    mask_threshold=self.mask_threshold,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]

                img_w, img_h = image_pil.size
                res.append(self._format_results(results, img_w, img_h))

            except Exception as e:
                eval_logger.error(f"Error processing request: {e}")
                res.append(empty_result)

            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """Not applicable for SAM3."""
        raise NotImplementedError("Multi-round generation is not supported for SAM3.")
