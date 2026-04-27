from __future__ import annotations

import argparse
import base64
import copy
import importlib
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm
from vllm_omni import Omni

from lmms_eval.tasks.vbvr.vbvr_bench import VBVRBench

try:
    from diffusers.utils import export_to_video
except Exception:  # pragma: no cover
    export_to_video = None


FILE_SPLIT_MAP = {
    "all": None,
    "in_domain": "In-Domain_50",
    "out_of_domain": "Out-of-Domain_50",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wan2.2 with vllm-omni on a local VBVR-Bench checkout.")
    parser.add_argument("--model", required=True, help="Path to the local Wan2.2 checkpoint")
    parser.add_argument("--vbvr-root", required=True, help="Path to the local VBVR-Bench root")
    parser.add_argument("--manifest", default=None, help="Optional path to VBVR-Bench.json; defaults to <vbvr-root>/VBVR-Bench.json")
    parser.add_argument("--output-root", required=True, help="Directory for generated videos and metrics")
    parser.add_argument("--split", choices=sorted(FILE_SPLIT_MAP), default="all")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit after split filtering")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cache-backend", default="cache_dit")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--guidance-scale-2", type=float, default=None)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary-ratio", type=float, default=None)
    parser.add_argument("--flow-shift", type=float, default=None)
    parser.add_argument("--diffusion-batch-size", type=int, default=None)
    parser.add_argument("--request-batch-size", type=int, default=None, help="How many samples to submit in one Omni.generate call.")
    parser.add_argument("--shard-id", type=int, default=0, help="0-based shard index for process-level sample parallelism.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of disjoint shards.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate videos even if the output mp4 already exists")
    parser.add_argument("--skip-generate", action="store_true", help="Only run VBVR scoring on existing videos; skip generation")
    parser.add_argument("--skip-eval", action="store_true", help="Only generate videos; skip VBVR scoring")
    parser.add_argument("--task-specific-only", action="store_true", help="Score only VBVR task-specific rules instead of the default weighted aggregate")
    parser.add_argument("--run-name", default=None, help="Optional name for the evaluation result JSON")
    return parser.parse_args()


def decode_base64_image(data: str) -> Image.Image:
    payload = data.split(",", 1)[-1]
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def parse_task_meta(doc: dict[str, Any]) -> tuple[str, str, str]:
    raw = doc.get("first_frame_path") or doc.get("final_frame_path") or doc.get("prompt_path") or doc.get("ground_truth_video_path") or ""
    parts = [part for part in str(raw).split("/") if part]
    if len(parts) < 3:
        raise ValueError(f"Cannot parse VBVR task meta from path: {raw!r}")
    return parts[0], parts[1], parts[2]


def filtered_docs(
    manifest_path: Path,
    split: str,
    limit: int | None,
    shard_id: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    docs = json.loads(manifest_path.read_text())
    file_split = FILE_SPLIT_MAP[split]
    selected: list[dict[str, Any]] = []
    for doc in docs:
        doc_file_split, _, _ = parse_task_meta(doc)
        if file_split is not None and doc_file_split != file_split:
            continue
        selected.append(doc)
    selected.sort(key=lambda doc: parse_task_meta(doc))
    if limit is not None:
        selected = selected[:limit]
    if num_shards > 1:
        selected = [doc for idx, doc in enumerate(selected) if idx % num_shards == shard_id]
    return selected


def build_sampling_params(omni: Omni, args: argparse.Namespace) -> list[Any]:
    sampling_params_list = copy.deepcopy(list(omni.default_sampling_params_list))
    if not sampling_params_list:
        raise RuntimeError("vllm-omni returned an empty default_sampling_params_list")

    stage0 = sampling_params_list[0]
    boundary_ratio = args.boundary_ratio
    if boundary_ratio is None:
        boundary_ratio = read_model_index_float(args.model, "boundary_ratio")

    values = {
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "guidance_scale_2": args.guidance_scale_2,
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "seed": args.seed,
        "boundary_ratio": boundary_ratio,
        "flow_shift": args.flow_shift,
    }
    for key, value in values.items():
        if value is not None and hasattr(stage0, key):
            setattr(stage0, key, value)
    if hasattr(stage0, "guidance_scale_provided"):
        stage0.guidance_scale_provided = True
    sampling_params_list[0] = stage0
    return sampling_params_list


def read_model_index_float(model_path: str, key: str) -> float | None:
    model_index_path = Path(model_path).expanduser() / "model_index.json"
    if not model_index_path.is_file():
        return None
    try:
        value = json.loads(model_index_path.read_text()).get(key)
    except Exception:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_parallel_config(args: argparse.Namespace) -> dict[str, int]:
    return {
        "pipeline_parallel_size": 1,
        "data_parallel_size": args.data_parallel_size,
        "tensor_parallel_size": args.tensor_parallel_size,
    }


def to_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    import numpy as np
    import torch

    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image output type: {type(image).__name__}")
    if image.ndim == 3 and image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = image.transpose(1, 2, 0)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 1) * 255 if image.max() <= 1.0 else np.clip(image, 0, 255)
        image = image.astype(np.uint8)
    return Image.fromarray(image).convert("RGB")


def normalize_video_frames(frames: Any) -> list[Any]:
    import numpy as np
    import torch

    if isinstance(frames, list):
        if not frames:
            return []
        normalized: list[Any] = []
        for item in frames:
            normalized.extend(normalize_video_frames(item))
        return normalized
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()
    if isinstance(frames, np.ndarray):
        if frames.ndim == 5 and frames.shape[0] == 1:
            return normalize_video_frames(frames[0])
        if frames.ndim == 4:
            return [frames[i] for i in range(frames.shape[0])]
        if frames.ndim == 3:
            return [frames]
    return [frames]


def save_video(frames: Any, output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [to_pil_image(frame) for frame in normalize_video_frames(frames)]
    if export_to_video is not None:
        export_to_video(pil_frames, output_video_path=str(output_path), fps=fps)
        return

    imageio_v2 = importlib.import_module("imageio.v2")
    imageio_v2.mimsave(str(output_path), pil_frames, fps=fps)


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def generation_failures_path(metrics_root: Path, shard_id: int, num_shards: int) -> Path:
    if num_shards <= 1:
        return metrics_root / "generation_failures.json"
    return metrics_root / f"generation_failures_shard_{shard_id:02d}_of_{num_shards:02d}.json"


def generate_videos(
    args: argparse.Namespace,
    docs: list[dict[str, Any]],
    videos_root: Path,
    metrics_root: Path,
) -> None:
    failures: list[dict[str, Any]] = []
    failures_path = generation_failures_path(metrics_root, args.shard_id, args.num_shards)
    if not docs:
        failures_path.write_text(json.dumps(failures, indent=2))
        print("Generation failures: 0")
        print(f"Failure log: {failures_path}")
        return

    omni = Omni(
        model=args.model,
        parallel_config=build_parallel_config(args),
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        cache_backend=args.cache_backend,
        diffusion_batch_size=args.diffusion_batch_size,
    )

    try:
        sampling_params_list = build_sampling_params(omni, args)
        batches = chunked(docs, args.request_batch_size)
        desc = f"Generating shard {args.shard_id + 1}/{args.num_shards}"
        for batch_docs in tqdm(batches, desc=desc, dynamic_ncols=True):
            request_batch: list[dict[str, Any]] = []
            meta_batch: list[tuple[str, str, str, Path]] = []
            for doc in batch_docs:
                file_split, task_name, video_idx = parse_task_meta(doc)
                output_path = videos_root / file_split / task_name / f"{video_idx}.mp4"
                if output_path.exists() and not args.overwrite:
                    continue
                try:
                    prompt = str(doc.get("prompt") or "").strip()
                    image = decode_base64_image(str(doc["first_image"]))
                    request_batch.append({"prompt": prompt, "multi_modal_data": {"image": image}})
                    meta_batch.append((file_split, task_name, video_idx, output_path))
                except Exception as e:  # noqa: BLE001
                    failures.append(
                        {
                            "file_split": file_split,
                            "task_name": task_name,
                            "video_idx": video_idx,
                            "output_path": str(output_path),
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )

            if not request_batch:
                continue

            prompts: list[dict[str, Any]] | dict[str, Any]
            prompts = request_batch if len(request_batch) > 1 else request_batch[0]
            try:
                outputs = omni.generate(prompts, sampling_params_list=sampling_params_list, use_tqdm=False)
            except Exception as e:  # noqa: BLE001
                for file_split, task_name, video_idx, output_path in meta_batch:
                    failures.append(
                        {
                            "file_split": file_split,
                            "task_name": task_name,
                            "video_idx": video_idx,
                            "output_path": str(output_path),
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
                continue

            if len(outputs) != len(meta_batch):
                error_text = f"Expected {len(meta_batch)} outputs, got {len(outputs)}"
                for file_split, task_name, video_idx, output_path in meta_batch:
                    failures.append(
                        {
                            "file_split": file_split,
                            "task_name": task_name,
                            "video_idx": video_idx,
                            "output_path": str(output_path),
                            "error": error_text,
                        }
                    )
                continue

            for (file_split, task_name, video_idx, output_path), result in zip(meta_batch, outputs):
                try:
                    if getattr(result, "error", None):
                        raise RuntimeError(str(result.error))
                    frames = getattr(result, "images", None)
                    if frames is None or (isinstance(frames, list) and not frames):
                        raise RuntimeError("Omni returned no image frames")
                    save_video(frames, output_path, fps=args.fps)
                except Exception as e:  # noqa: BLE001
                    failures.append(
                        {
                            "file_split": file_split,
                            "task_name": task_name,
                            "video_idx": video_idx,
                            "output_path": str(output_path),
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
    finally:
        omni.close()

    failures_path.write_text(json.dumps(failures, indent=2))
    print(f"Generation failures: {len(failures)}")
    print(f"Failure log: {failures_path}")


def main() -> None:
    args = parse_args()
    if args.diffusion_batch_size is None:
        args.diffusion_batch_size = max(1, args.data_parallel_size)
    if args.request_batch_size is None:
        args.request_batch_size = max(1, args.data_parallel_size)
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit("--shard-id must be in [0, num_shards)")
    if args.request_batch_size < 1:
        raise SystemExit("--request-batch-size must be >= 1")

    vbvr_root = Path(args.vbvr_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else vbvr_root / "VBVR-Bench.json"
    output_root = Path(args.output_root).expanduser().resolve()
    videos_root = output_root / "videos"
    metrics_root = output_root / "metrics"
    videos_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    docs = filtered_docs(manifest_path, args.split, args.limit, args.shard_id, args.num_shards)

    print(f"Using manifest: {manifest_path}")
    print(f"Using GT root:  {vbvr_root}")
    print(f"Output root:    {output_root}")
    print(f"Shard:          {args.shard_id + 1}/{args.num_shards}")
    print(f"Samples:        {len(docs)}")

    if not args.skip_generate:
        generate_videos(args, docs, videos_root, metrics_root)

    if args.skip_eval:
        return

    file_split = FILE_SPLIT_MAP[args.split]
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    bench = VBVRBench(gt_base_path=str(vbvr_root), output_path=str(metrics_root))
    bench.evaluate(str(videos_root), name=run_name, split=file_split, task_specific_only=args.task_specific_only)


if __name__ == "__main__":
    main()
