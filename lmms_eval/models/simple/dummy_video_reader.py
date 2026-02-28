import json
import statistics
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null"}:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(ratio * (len(sorted_vals) - 1))
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


@register_model("dummy_video_reader")
class DummyVideoReader(lmms):
    def __init__(
        self,
        response: str = "A",
        read_bytes: int = 65536,
        allow_remote: bool = False,
        fail_on_missing: bool = True,
        decode_num_frames: int = 0,
        decode_fps: Optional[float] = None,
        metrics_output_path: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self._response = str(response).strip() or "A"
        self._read_bytes = _as_int(read_bytes, 65536)
        self._allow_remote = _as_bool(allow_remote)
        self._fail_on_missing = _as_bool(fail_on_missing)
        self._decode_num_frames = max(0, _as_int(decode_num_frames, 0))
        self._decode_fps = _as_optional_float(decode_fps)
        self._metrics_output_path = str(metrics_output_path).strip()
        self._decode_fn: Optional[Callable] = None

        self._resolved_count = 0
        self._local_file_count = 0
        self._remote_count = 0
        self._missing_count = 0
        self._decoded_frame_count = 0

        self._resolve_latencies: List[float] = []
        self._io_latencies: List[float] = []
        self._decode_latencies: List[float] = []
        self._total_latencies: List[float] = []

    def _get_decode_fn(self) -> Callable:
        if self._decode_fn is None:
            from lmms_eval.models.model_utils.load_video import read_video

            self._decode_fn = read_video
        return self._decode_fn

    def _touch_visual(self, visual: str) -> float:
        self._resolved_count += 1

        if visual.startswith("http://") or visual.startswith("https://"):
            self._remote_count += 1
            if not self._allow_remote and self._fail_on_missing:
                raise RuntimeError(f"DummyVideoReader got remote visual path: {visual}")
            return 0.0

        path = Path(visual)
        if not path.exists():
            self._missing_count += 1
            if self._fail_on_missing:
                raise FileNotFoundError(f"DummyVideoReader visual path does not exist: {visual}")
            return 0.0

        self._local_file_count += 1
        if self._decode_num_frames > 0:
            decode_fn = self._get_decode_fn()
            start = time.perf_counter()
            frames = decode_fn(str(path), num_frm=self._decode_num_frames, fps=self._decode_fps)
            elapsed = time.perf_counter() - start
            self._decoded_frame_count += len(frames)
            return elapsed

        read_size = self._read_bytes if self._read_bytes >= 0 else None
        with path.open("rb") as file_obj:
            _ = file_obj.read(read_size)
        return 0.0

    def _summarize_latency(self, values: List[float]) -> dict:
        if not values:
            return {"samples": 0}

        total_s = float(sum(values))
        return {
            "samples": len(values),
            "total_s": total_s,
            "mean_ms": statistics.mean(values) * 1000,
            "p50_ms": _percentile(values, 0.50) * 1000,
            "p95_ms": _percentile(values, 0.95) * 1000,
            "max_ms": max(values) * 1000,
        }

    def _build_metrics_payload(self) -> dict:
        payload = {
            "resolved": self._resolved_count,
            "local": self._local_file_count,
            "remote": self._remote_count,
            "missing": self._missing_count,
            "decoded_frames": self._decoded_frame_count,
            "decode_num_frames": self._decode_num_frames,
            "decode_fps": self._decode_fps,
            "latency": {
                "resolve": self._summarize_latency(self._resolve_latencies),
                "io": self._summarize_latency(self._io_latencies),
                "decode": self._summarize_latency(self._decode_latencies),
                "total": self._summarize_latency(self._total_latencies),
            },
        }
        if payload["latency"]["total"].get("total_s", 0.0) > 0 and payload["latency"]["total"].get("samples", 0) > 0:
            payload["throughput_videos_per_s"] = payload["latency"]["total"]["samples"] / payload["latency"]["total"]["total_s"]
        if payload["latency"]["decode"].get("total_s", 0.0) > 0:
            payload["throughput_decode_frames_per_s"] = payload["decoded_frames"] / payload["latency"]["decode"]["total_s"]
        return payload

    def _maybe_write_metrics(self, payload: dict) -> None:
        if self._metrics_output_path == "":
            return
        output_path = Path(self._metrics_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def generate_until(self, requests) -> List[str]:
        responses = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for context, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            total_start = time.perf_counter()
            doc = self.task_dict[task][split][doc_id]
            resolve_start = time.perf_counter()
            visuals = doc_to_visual(doc)
            resolve_elapsed = time.perf_counter() - resolve_start
            self._resolve_latencies.append(resolve_elapsed)
            if not isinstance(visuals, list):
                visuals = [visuals]

            io_elapsed = 0.0
            decode_elapsed = 0.0
            for visual in visuals:
                if isinstance(visual, str):
                    io_start = time.perf_counter()
                    decode_s = self._touch_visual(visual)
                    io_elapsed += time.perf_counter() - io_start
                    decode_elapsed += decode_s

            self._io_latencies.append(io_elapsed)
            if self._decode_num_frames > 0:
                self._decode_latencies.append(decode_elapsed)
            self._total_latencies.append(time.perf_counter() - total_start)

            responses.append(self._response)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), self._response)
            pbar.update(1)

        pbar.close()
        metrics = self._build_metrics_payload()
        eval_logger.info(
            "DummyVideoReader summary - resolved: {} | local: {} | remote: {} | missing: {}",
            self._resolved_count,
            self._local_file_count,
            self._remote_count,
            self._missing_count,
        )
        total_latency = metrics["latency"]["total"]
        if total_latency.get("samples", 0) > 0:
            eval_logger.info(
                "DummyVideoReader latency total - samples: {} | mean_ms: {:.3f} | p50_ms: {:.3f} | p95_ms: {:.3f}",
                total_latency["samples"],
                total_latency["mean_ms"],
                total_latency["p50_ms"],
                total_latency["p95_ms"],
            )
        if self._decode_num_frames > 0:
            decode_latency = metrics["latency"]["decode"]
            if decode_latency.get("samples", 0) > 0:
                eval_logger.info(
                    "DummyVideoReader decode - frames: {} | mean_ms: {:.3f} | frames_per_s: {:.3f}",
                    self._decoded_frame_count,
                    decode_latency["mean_ms"],
                    metrics.get("throughput_decode_frames_per_s", 0.0),
                )
        self._maybe_write_metrics(metrics)
        return responses

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("DummyVideoReader does not implement loglikelihood")

    def generate_until_multi_round(self, requests) -> List[str]:
        return self.generate_until(requests)
