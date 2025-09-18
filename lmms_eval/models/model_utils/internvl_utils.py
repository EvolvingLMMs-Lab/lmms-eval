import logging
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu


eval_logger = logging.getLogger("eval_logger")


def adaptive_keyframe_sampling(video_path: str, num_segments: int, query: str) -> np.ndarray:
    """
    Select frame indices relevant to the textual query using CLIP similarity.
    Falls back to uniform sampling if CLIP loading fails due to PyTorch version issues.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    num_segments : int
        Number of key frames to sample.
    query : str
        Text query describing the desired content.

    Returns
    -------
    np.ndarray
        Sorted array of selected frame indices.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    # Fallback to uniform indices when query is empty
    if not query or not query.strip():
        eval_logger.debug("adaptive_keyframe_sampling: Query is empty, using uniform sampling.")
        return np.linspace(0, total_frames - 1, num_segments, dtype=int)

    try:
        # Try to load CLIP with safetensors support
        from transformers import CLIPModel, CLIPProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Prefer a fast image processor but fall back to the slow version when necessary
        def _load_processor(model_name: str):
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
            try:
                proc = CLIPProcessor.from_pretrained(model_name, use_fast=True)
                if not hasattr(proc.image_processor, "_valid_processor_keys"):
                    raise AttributeError
            except AttributeError:
                logging.warning("⚠️ Fast image processor unsupported; using slow version")
                proc = CLIPProcessor.from_pretrained(model_name, use_fast=False)

            return proc

        # Try multiple approaches to load CLIP safely
        try:
            model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                use_safetensors=True,
            ).to(device)
            processor = _load_processor("openai/clip-vit-base-patch32")
            load_msg = "✓ CLIP loaded with safetensors"
        except Exception as e1:
            try:
                model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch16",
                    use_safetensors=True,
                ).to(device)
                processor = _load_processor("openai/clip-vit-base-patch16")
                load_msg = "✓ CLIP loaded with clip-vit-base-patch16 and safetensors"
            except Exception as e2:
                try:
                    model = CLIPModel.from_pretrained(
                        "openai/clip-vit-base-patch32",
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        trust_remote_code=False,
                    ).to(device)
                    processor = _load_processor("openai/clip-vit-base-patch32")
                    load_msg = "✓ CLIP loaded with float16 and safetensors"
                except Exception as e3:
                    eval_logger.debug("All CLIP loading attempts failed:")
                    eval_logger.debug(f"  Attempt 1 (safetensors): {e1}")
                    eval_logger.debug(f"  Attempt 2 (patch16): {e2}")
                    eval_logger.debug(f"  Attempt 3 (float16): {e3}")
                    eval_logger.debug("Falling back to uniform sampling")
                    return np.linspace(0, total_frames - 1, num_segments, dtype=int)
        try:
            text_inputs = processor(
                text=query,
                return_tensors="pt",
                truncation=True,
                max_length=77,
                return_overflowing_tokens=True,
            ).to(device)
            overflow = text_inputs.pop("overflowing_tokens", None)
            text_inputs.pop("num_truncated_tokens", None)
            if overflow is not None and overflow.numel() > 0:
                logging.warning("⚠️ query truncated to 77 tokens for CLIP")
            with torch.no_grad():
                text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            candidate_indices = np.linspace(0, total_frames - 1, num_segments * 4, dtype=int)
            frames = [Image.fromarray(vr[idx].asnumpy()) for idx in candidate_indices]
            img_inputs = processor(images=frames, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**img_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            scores = (image_features @ text_features.T).squeeze()
            topk = scores.topk(num_segments).indices.cpu().numpy()
            selected = np.sort(candidate_indices[topk])
        except Exception as e:
            eval_logger.debug(f"Adaptive sampling failed after CLIP load: {e}")
            eval_logger.debug("Falling back to uniform sampling")
            return np.linspace(0, total_frames - 1, num_segments, dtype=int)

        eval_logger.debug(load_msg)
        eval_logger.debug(f"\u2713 Adaptive sampling successful: selected {len(selected)} frames")
        return selected
    except ImportError as e:
        eval_logger.debug(f"CLIP dependencies not available: {e}")
        eval_logger.debug("Falling back to uniform sampling")
        return np.linspace(0, total_frames - 1, num_segments, dtype=int)
    except Exception as e:
        eval_logger.debug(f"Adaptive sampling failed with error: {e}")
        eval_logger.debug("Falling back to uniform sampling")
        return np.linspace(0, total_frames - 1, num_segments, dtype=int)
