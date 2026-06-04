"""LlavaOnevision2 inference (transformers 5.x, trust_remote_code).

Registered as ``llava_onevision2``. Targets the NEW checkpoint
``llava_onevision2_8b_final_round_32nodes_mcore_tp8_pp4_hf_new`` whose
bundled remote code (``modeling_llava_onevision2.py``,
``processing_llava_onevision2.py``, ``video_processing_llava_onevision2.py``)
already implements training-aligned ``patch_positions``, RoPE block layout,
and OLD/V2-aligned video preprocessing (frame sampling, smart_resize,
timestamp precision, decord + torchvision BICUBIC resize, per-frame text
expansion).

Preprocessing path is now V2-aligned:
  - Pre-fetches frames via ``qwen_vl_utils.fetch_video`` (same kwargs as
    V2 chat: fps / min_pixels / max_pixels / max_frames).
  - Builds a per-frame content list of timestamp text +
    ``image`` PIL pairs (and per-frame ``subtitle\\n...`` text when the
    task layer embeds ``__SUBTITLE_DATA__:`` JSON, mirroring the V2
    handler).
  - Feeds the resulting PIL list via ``images=...`` (NOT ``videos=...``)
    so the model goes through the image-processor route the sweep
    checkpoint was trained on.

Validated bit-exact on LVBench (1549/1549) and videomme_isub samples
(stripping the ``__SUBTITLE_DATA__:`` prefix + injecting per-frame
subtitles).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages


# ============================================================================
# Codec + subtitle interleaving helpers.
#
# Implements the online cv-preinfer codec path together with the subtitle
# interleaving logic used by ``videommev2_interleaved_subtitle``. No offline
# asset pre-generation is required.
# ============================================================================

_VISION_START = "<|vision_start|>"
_VISION_END = "<|vision_end|>"
_IMAGE_PAD = "<|image_pad|>"


def _filter_and_group_subtitles(subtitle_data: dict) -> dict:
    """Apply subtitle preprocessing controlled by env flags.

    Supported flags:
      - ``LMMS_IL_FILTER_NOISE=1``: drop noise lines (regex blacklist).
      - ``LMMS_IL_MIN_LEN=N``: drop segments with text shorter than N chars.
      - ``LMMS_IL_SENTENCE_GROUP=1``: merge consecutive segments into
        sentence-level chunks (gap-based + punctuation-based).
      - ``LMMS_IL_SG_GAP=<float>``: gap threshold (default 0.5s).
    """
    if not subtitle_data:
        return subtitle_data
    _filter_noise = os.environ.get("LMMS_IL_FILTER_NOISE", "0") == "1"
    try:
        _min_len = int(os.environ.get("LMMS_IL_MIN_LEN", "0"))
    except ValueError:
        _min_len = 0
    _sentence_group = os.environ.get("LMMS_IL_SENTENCE_GROUP", "0") == "1"
    try:
        _group_gap = float(os.environ.get("LMMS_IL_SG_GAP", "0.5"))
    except ValueError:
        _group_gap = 0.5

    if _filter_noise or _min_len > 0:
        import re as _re

        _noise_pats = [
            _re.compile(r"^[\s\W_]*$"),
            _re.compile(r"^\s*[\(\[\{<].*[\)\]\}>]\s*$"),
            _re.compile(
                r"music|applause|laughter|sound effect|silence|background",
                _re.IGNORECASE,
            ),
            _re.compile(r"^[♪♫\u266a\u266b\u266c\u266d\u266e\u266f\s]+$"),
        ]
        kept: dict = {}
        for _k, _v in subtitle_data.items():
            _t = (_v or "").strip()
            if not _t:
                continue
            if _filter_noise:
                _matched_noise = False
                for _p in _noise_pats:
                    if _p.search(_t):
                        _matched_noise = True
                        break
                if _matched_noise:
                    continue
            if _min_len > 0 and len(_t) < _min_len:
                continue
            kept[_k] = _t
        subtitle_data = kept

    if _sentence_group and subtitle_data:
        items = []
        for _k, _v in subtitle_data.items():
            _ss, _ee = _k.split(":")
            items.append((float(_ss), float(_ee), _v.strip()))
        items.sort(key=lambda x: x[0])
        grouped: dict = {}
        if items:
            cur_s, cur_e, cur_t = items[0]
            for ns, ne, nt in items[1:]:
                gap = ns - cur_e
                ends_sent = bool(cur_t) and cur_t[-1] in ".!?。！？"
                if gap > _group_gap or ends_sent:
                    grouped[f"{cur_s:.3f}:{cur_e:.3f}"] = cur_t
                    cur_s, cur_e, cur_t = ns, ne, nt
                else:
                    cur_e = ne
                    cur_t = (cur_t + " " + nt).strip() if cur_t else nt
            grouped[f"{cur_s:.3f}:{cur_e:.3f}"] = cur_t
        subtitle_data = grouped

    return subtitle_data


def _build_sub_after_frame(subtitle_data: dict | None, all_frame_times: list[float]) -> dict[int, list[str]]:
    """Match each subtitle segment to a frame index.

    Modes:
      - ``LMMS_IL_RANGE_OVERLAP=1``: strict containment; segments outside
        all sampled frames are dropped.
      - default: nearest-frame midpoint (100% retention).
    Prefix:
      - ``LMMS_IL_NOPREFIX=1``: append raw subtitle text.
      - default: prefix with ``"<s.s> - <e.e> "``.
    """
    sub_after_frame: dict[int, list[str]] = {}
    if not subtitle_data or not all_frame_times:
        return sub_after_frame
    _use_range = os.environ.get("LMMS_IL_RANGE_OVERLAP", "0") == "1"
    _noprefix = os.environ.get("LMMS_IL_NOPREFIX", "0") == "1"
    for key, text in subtitle_data.items():
        s_str, e_str = key.split(":")
        s, e = float(s_str), float(e_str)
        if _use_range:
            matched = False
            last_idx = -1
            for idx, t_sec in enumerate(all_frame_times):
                if s <= t_sec <= e:
                    last_idx = idx
                    matched = True
            if not matched:
                continue
            best_idx = last_idx
        else:
            mid = 0.5 * (s + e)
            best_idx = 0
            best_dist = abs(all_frame_times[0] - mid)
            for idx in range(1, len(all_frame_times)):
                d = abs(all_frame_times[idx] - mid)
                if d <= best_dist:
                    best_dist = d
                    best_idx = idx
        entry = text if _noprefix else f"{s:.1f} - {e:.1f} {text}"
        sub_after_frame.setdefault(best_idx, []).append(entry)
    return sub_after_frame


def _rewrite_codec_vision_with_subtitles(
    text: str,
    patch_positions: torch.Tensor,
    fps: float,
    decimals: int,
    spatial_merge_size: int,
    sub_after_frame: dict[int, list[str]],
) -> str:
    """Re-emit the chat-template vision span with timestamps + subtitles.

    Replaces everything between the first ``<|vision_start|>`` and the
    last ``<|vision_end|>`` with one block per source frame:

        <T.t seconds><|vision_start|><|image_pad|>*N<|vision_end|>
        subtitle
        <s.s> - <e.e> <text>      # one line per matched subtitle

    The ``patch_positions`` table is in block layout (post 2x2 reorder), so
    the per-frame token count is ``count // (spatial_merge_size**2)``.
    """
    t_values = patch_positions[:, 0]
    unique_t, counts = torch.unique_consecutive(t_values, return_counts=True)
    merge_factor = int(spatial_merge_size) ** 2

    timestamps: list[str] = []
    per_frame_merged_counts: list[int] = []
    all_frame_indices: list[int] = []
    for t_val, count in zip(unique_t.tolist(), counts.tolist()):
        if int(t_val) < 0:
            continue
        token_count = int(count) // merge_factor
        if token_count <= 0:
            continue
        timestamps.append(f"<{float(t_val) / float(fps):.{decimals}f} seconds>")
        per_frame_merged_counts.append(token_count)
        all_frame_indices.append(int(t_val))

    first_vs = text.find(_VISION_START)
    last_ve = text.rfind(_VISION_END)
    if first_vs == -1 or last_ve == -1:
        return text

    parts: list[str] = []
    for frame_idx, count in enumerate(per_frame_merged_counts):
        if frame_idx < len(timestamps):
            parts.append(timestamps[frame_idx])
        parts.append(_VISION_START)
        parts.append(_IMAGE_PAD * count)
        parts.append(_VISION_END)
        parts.append("\n")
        subs = sub_after_frame.get(frame_idx)
        if subs:
            parts.append("subtitle\n" + "\n".join(subs) + "\n")

    tail_start = last_ve + len(_VISION_END)
    if tail_start < len(text) and text[tail_start] == "\n":
        tail_start += 1

    return text[:first_vs] + "".join(parts) + text[tail_start:]


def _frame_times_from_patch_positions(patch_positions: torch.Tensor, fps: float) -> list[float]:
    """Distinct frame timestamps (seconds) from a codec patch_positions table."""
    t_values = patch_positions[:, 0]
    unique_t, _ = torch.unique_consecutive(t_values, return_counts=True)
    return [float(t.item()) / float(fps) for t in unique_t if int(t.item()) >= 0]


# ---------------------------------------------------------------------------
# Offline codec asset loader (LLAVA_CODEC_OFFLINE_ROOT)
# ---------------------------------------------------------------------------


def _sha1_8(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


_OFFLINE_INDEX_CACHE: dict[str, dict[str, str]] = {}


def _build_offline_index(root: str) -> dict[str, str]:
    """Index <stem> -> asset_dir for an offline codec output root.

    The asset dir naming convention is ``<stem>__<sha1_8>`` (preprocess
    script output), with ``_DONE`` sentinel inside.
    """
    out: dict[str, str] = {}
    if not root or not os.path.isdir(root):
        return out
    try:
        for name in os.listdir(root):
            d = os.path.join(root, name)
            if not os.path.isdir(d):
                continue
            if not os.path.exists(os.path.join(d, "_DONE")):
                continue
            stem = name.split("__", 1)[0]
            out.setdefault(stem, d)
    except Exception:
        return out
    return out


def _find_offline_asset_dir(video_url: str, root: str) -> Optional[str]:
    """Resolve offline codec asset directory for a given video URL.

    Strategies: ``<stem>__<sha1_8(video_url)>`` -> ``<stem>`` -> root index.
    """
    if not root or not os.path.isdir(root):
        return None
    stem = Path(video_url).stem
    cand = os.path.join(root, f"{stem}__{_sha1_8(video_url)}")
    if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "_DONE")):
        return cand
    cand = os.path.join(root, stem)
    if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "_DONE")):
        return cand
    idx = _OFFLINE_INDEX_CACHE.get(root)
    if idx is None:
        idx = _build_offline_index(root)
        _OFFLINE_INDEX_CACHE[root] = idx
        eval_logger.info(f"[codec][offline] Built index from {root}: {len(idx)} entries")
    return idx.get(stem)


def _try_load_offline_codec_assets(video_url: str, offline_root: str) -> Optional[dict]:
    """Read pre-generated codec assets (canvas_*.jpg + src_patch_position.npy + meta.json).

    Returns a dict compatible with :func:`process_codec_video` outputs:
    ``{"images": [PIL...], "src_positions": np.ndarray[N,3], "fps": float}``;
    or ``None`` when the asset dir is missing.
    """
    if not offline_root:
        return None
    asset_dir = _find_offline_asset_dir(video_url, offline_root)
    if asset_dir is None:
        eval_logger.warning(f"[codec][offline] MISS {video_url} (root={offline_root})")
        return None

    import glob

    canvas_files = sorted(glob.glob(os.path.join(asset_dir, "canvas_*.jpg")))
    if not canvas_files:
        canvas_files = sorted(glob.glob(os.path.join(asset_dir, "canvas_*.png")))
    if not canvas_files:
        eval_logger.warning(f"[codec][offline] no canvas_*.{{jpg,png}} in {asset_dir}")
        return None
    images = [Image.open(f).convert("RGB") for f in canvas_files]

    pos_path = os.path.join(asset_dir, "src_patch_position.npy")
    if not os.path.exists(pos_path):
        eval_logger.warning(f"[codec][offline] missing src_patch_position.npy in {asset_dir}")
        return None
    src_positions = np.load(pos_path)

    meta_path = os.path.join(asset_dir, "meta.json")
    fps = 30.0
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            v = meta.get("fps")
            if v is None:
                v = (meta.get("config") or {}).get("video_fps")
            if v is None:
                # Probe with cv2 if meta lacks fps (some old assets).
                v = 30.0
            fps = float(v) if v and float(v) > 0 else 30.0
        except Exception as e:
            eval_logger.warning(f"[codec][offline] meta.json read failed {meta_path}: {e}")

    eval_logger.info(
        f"[codec][offline] HIT {video_url} -> {asset_dir} "
        f"({len(images)} canvases, positions={src_positions.shape}, fps={fps})"
    )
    return {"images": images, "src_positions": src_positions, "fps": fps}


# ---------------------------------------------------------------------------
# Online cv-preinfer driver with tuned parameter set
# ---------------------------------------------------------------------------


def _process_codec_video_tuned(video_url: str, cfg) -> dict:
    """Online cv-preinfer driver with the tuned parameter set.

    Differences vs the ckpt-bundled ``process_codec_video`` (which calls
    ``cv-preinfer`` with ``--avoid_keyframes`` and ``max_group_frames=64``):

      * ``--no_avoid_keyframes``  (vs ckpt: ``--avoid_keyframes``)
      * ``--max_group_frames 128`` (vs ckpt: 64)
      * ``--frame_sampling_mode uniform_count``
      * ``--readiness_sum_threshold_mode auto``
      * ``--readiness_norm_sum_threshold 1500000``
      * ``--block_size 2``
      * ``--bitcost_grid sub``

    Result is cached on disk under ``cfg.cache_root`` keyed by
    ``(video_url, max_pixels, target_canvas, ...)``; concurrent workers
    coordinate via a flock-protected sentinel (same pattern as the ckpt's
    ``process_codec_video``).
    """
    import subprocess
    import tempfile
    import shutil

    try:
        import fcntl
    except ImportError:  # pragma: no cover
        fcntl = None  # type: ignore

    cfg.validate()

    # Reuse the ckpt's cache_dir hashing so we don't double-cache.
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cfg.cache_root.mkdir(parents=True, exist_ok=True)
    # Distinct cache key from the ckpt default (avoid_keyframes flag flip).
    raw = (
        f"{video_url}|tc={cfg.target_canvas}|gs={cfg.group_size}"
        f"|ipg={cfg.images_per_group}|patch={cfg.patch}"
        f"|mp={cfg.max_pixels}|mask={cfg.spatial_mask_mode}|tuned=1"
    )
    key = hashlib.md5(raw.encode()).hexdigest()
    out_dir = cfg.cache_root / f"{Path(video_url).stem}_{key}"

    def _load_from(_d: Path) -> dict:
        with open(_d / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        canvas_files = meta.get("canvas_files")
        if not canvas_files:
            import glob as _glob

            canvas_files = sorted(Path(p).name for p in _glob.glob(str(_d / "canvas_*.jpg")))
        images = [Image.open(_d / n).convert("RGB") for n in canvas_files]
        src_positions = np.load(_d / "src_patch_position.npy")
        fps = float(meta.get("fps") or 30.0)
        return {"images": images, "src_positions": src_positions, "fps": fps}

    if (out_dir / "meta.json").exists() and (out_dir / "src_patch_position.npy").exists():
        return _load_from(out_dir)

    lock_path = cfg.cache_root / f".{out_dir.name}.lock"
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        if fcntl is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # Re-check after acquiring lock.
        if (out_dir / "meta.json").exists() and (out_dir / "src_patch_position.npy").exists():
            return _load_from(out_dir)

        # Resolve total frames to clamp num_sampled.
        import cv2 as _cv2

        cap = _cv2.VideoCapture(video_url)
        try:
            total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        total = max(1, total)
        num_sampled = min(cfg.num_sampled_frames(), total)

        tmp_dir = Path(
            tempfile.mkdtemp(
                dir=str(cfg.cache_root),
                prefix=f".tmp_{out_dir.name[:48]}_",
            )
        )
        cmd = [
            "codec-video-prep-legacy-exact",
            "--video",
            video_url,
            "--out_dir",
            str(tmp_dir),
            "--frame_sampling_mode",
            "uniform_count",
            "--num_sampled_frames",
            str(num_sampled),
            "--grouping_mode",
            "readiness",
            "--readiness_sum_threshold_mode",
            "auto",
            "--group_size",
            str(cfg.group_size),
            "--images_per_group",
            str(cfg.images_per_group),
            "--patch",
            str(cfg.patch),
            "--max_pixels",
            str(cfg.max_pixels),
            "--min_group_frames",
            str(cfg.min_group_frames),
            "--max_group_frames",
            "128",
            "--bitcost_grid",
            "sub",
            "--canvas_format",
            "jpg",
        ]
        try:
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=cfg.timeout_seconds,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout)[-2000:]
                raise RuntimeError(f"online cv-preinfer (tuned params) failed rc={result.returncode}: {detail}")
            if out_dir.exists():
                shutil.rmtree(out_dir)
            tmp_dir.rename(out_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        return _load_from(out_dir)
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


@register_model("llava_onevision2")
class Llava_OneVision2(lmms):
    """Trust-remote-code wrapper for the NEW LlavaOnevision2 checkpoint."""

    is_simple = False

    def __init__(
        self,
        pretrained: str = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
        device: str = "cuda",
        device_map: str = "",
        batch_size: int = 1,
        use_cache: bool = True,
        attn_implementation: str = "flash_attention_2",
        # Vision sampling controls (forwarded to processor on each call)
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        fixed_num_frames: int | None = None,
        target_fps: float | None = None,
        # Generation controls
        max_new_tokens: int = 128,
        system_prompt: str = "You are a helpful assistant.",
        torch_dtype: str = "bfloat16",
        # Video frame sampling (used when pre-fetching frames via
        # qwen_vl_utils.fetch_video to mirror V2 chat path).
        fps: float = 1.0,
        messages_format: str = "timestamp",
        timestamp_decimals: int = 1,
        # Video backend: "frames" (default, uniform sampling via qwen_vl_utils)
        # or "codec" (canvas packing via cv-preinfer, requires codec-video-prep + ffmpeg).
        # See the model checkpoint README "Optional: codec video backend".
        video_backend: str = "frames",
        codec_target_canvas: int | None = None,
        codec_group_size: int | None = None,
        codec_images_per_group: int | None = None,
        **kwargs,
    ):
        super().__init__()

        # --- Distributed setup (mirror the existing chat wrapper) --------
        accelerator = Accelerator()
        if accelerator.num_processes > 1 and not device_map:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # --- Load model + processor via trust_remote_code -----------------
        self.pretrained = pretrained
        eval_logger.info(f"[llava_onevision2] Loading from: {pretrained}")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype_t = dtype_map.get(torch_dtype, torch.bfloat16)

        cfg = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            pretrained,
            config=cfg,
            torch_dtype=torch_dtype_t,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        self.model.to(self._device).eval()

        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
        # Override pixel budget on the bundled VideoProcessor + ImageProcessor so
        # CLI args min_pixels/max_pixels actually take effect. The processor's
        # __call__ does NOT accept these per-call, so we mutate the underlying
        # processor instances once at init.
        try:
            vp = self.processor.video_processor
            if vp is not None:
                vp.min_pixels = int(min_pixels)
                vp.max_pixels = int(max_pixels)
        except AttributeError:
            pass
        try:
            ip = self.processor.image_processor
            if ip is not None:
                ip.min_pixels = int(min_pixels)
                ip.max_pixels = int(max_pixels)
                # Qwen2VLImageProcessor also reads `size`{shortest_edge,longest_edge}.
                if hasattr(ip, "size") and isinstance(ip.size, dict):
                    ip.size = dict(ip.size)
                    ip.size["shortest_edge"] = int(min_pixels)
                    ip.size["longest_edge"] = int(max_pixels)
        except AttributeError:
            pass

        # NOTE: There are no runtime monkey-patches here. The bundled
        # trust_remote modules in the checkpoint directory already
        # implement OLD/V2-aligned preprocessing:
        #   - select_frame_indices: torch.linspace(...).round().long()
        #   - smart_resize:         floor/ceil with align_patch_size=28
        #   - format_timestamp:     6-decimal precision (round-trip safe)
        #   - extract_video_frames: decord + smart_resize + torchvision
        #                           BICUBIC + antialias (qwen_vl_utils-aligned)
        #   - _expand_video_block_for_frames: no inter-frame '\n'
        # See ``video_processing_llava_onevision2.py`` and
        # ``processing_llava_onevision2.py`` in the model checkpoint dir.

        # Tokenizer convenience handle.
        try:
            self.tokenizer = self.processor.tokenizer
        except AttributeError:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)

        # --- Save sampling / generation knobs -----------------------------
        self.batch_size = int(batch_size)
        self.use_cache = bool(use_cache)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.max_num_frames = int(max_num_frames)
        self.fixed_num_frames = int(fixed_num_frames) if fixed_num_frames else None
        self.target_fps = float(target_fps) if target_fps else None
        self.max_new_tokens = int(max_new_tokens)
        self.system_prompt = system_prompt
        self.fps = float(fps)
        self.messages_format = str(messages_format)
        self.timestamp_decimals = int(timestamp_decimals)

        # --- Video backend selection --------------------------------------
        # "frames" (default): legacy uniform sampling via qwen_vl_utils.
        # "codec": route raw video URLs to processor(..., video_backend="codec"),
        # which uses cv-preinfer for canvas packing. Requires `codec-video-prep`
        # and ffmpeg to be installed (see model README).
        self.video_backend = str(video_backend).lower()
        if self.video_backend not in ("frames", "codec"):
            raise ValueError(f"video_backend must be 'frames' or 'codec', got {video_backend!r}")
        self.codec_config: dict = {}
        if codec_target_canvas is not None:
            self.codec_config["target_canvas"] = int(codec_target_canvas)
        if codec_group_size is not None:
            self.codec_config["group_size"] = int(codec_group_size)
        if codec_images_per_group is not None:
            self.codec_config["images_per_group"] = int(codec_images_per_group)
        if self.video_backend == "codec" and getattr(self, "_rank", 0) == 0:
            eval_logger.info(
                f"[llava_onevision2] video_backend=codec, codec_config={self.codec_config or '(processor defaults)'}"
            )
        if getattr(self, "_rank", 0) == 0:
            eval_logger.info(
                f"[llava_onevision2] messages_format={self.messages_format}, timestamp_decimals={self.timestamp_decimals}, fps={self.fps}"
            )

        # --- Frame caching (mirror V2) ------------------------------------
        # Cache key includes video_url + decode params so V2/V3 sweeps with
        # the same (path, fps, max_frames, max_pixels) hit the same files.
        self.cache_video_frames = os.getenv("CACHE_VIDEO_FRAMES", "0") == "1"
        self.video_frame_cache_dir = os.getenv("VIDEO_FRAME_CACHE_DIR", "/tmp/video_frame_cache")
        self._cache_format = os.getenv("CACHE_FORMAT", "jpg").lower()
        if self.cache_video_frames and getattr(self, "_rank", 0) == 0:
            eval_logger.info(
                f"[v3] Frame caching enabled: dir={self.video_frame_cache_dir}, format={self._cache_format}"
            )

        # --- Distributed bookkeeping --------------------------------------
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in (
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ), "Unsupported distributed type for this wrapper."
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    # ------------------------------------------------------------------ utils

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def device(self):
        return self._device

    # ---------------------------------------------------- message building

    @staticmethod
    def _find_subtitle_for_timestamp(t: float, subtitle_data: dict) -> str:
        matched: list[str] = []
        for key, text in subtitle_data.items():
            s_str, e_str = key.split(":")
            s, e = float(s_str), float(e_str)
            if s <= t <= e:
                matched.append(f"{s:.1f} - {e:.1f} {text}")
        if not matched:
            return ""
        return "subtitle\n" + "\n".join(matched)

    def _timestamp_decimals_for_task(self, task: str | None) -> int:
        if task and "jump_rope" in task:
            return 2
        return self.timestamp_decimals

    # ─── Frame Caching (V2-compatible cache key) ────────────────────────
    def _frame_cache_key(self, video_url: str) -> str:
        """Deterministic cache key matching V2 wrapper layout."""
        raw = (
            f"{video_url}|nframes=0|fps={self.fps}|max_frames={self.max_num_frames}"
            f"|max_pixels={self.max_pixels}|min_pixels={self.min_pixels}"
        )
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_dir_for(self, video_url: str) -> Path:
        video_name = Path(video_url).stem
        return Path(self.video_frame_cache_dir) / f"{video_name}_{self._frame_cache_key(video_url)}"

    def _load_frames_from_cache(self, cache_dir: Path) -> Optional[Tuple[List[Image.Image], float, List[int]]]:
        meta_path = cache_dir / "meta.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        fps = meta["fps"]
        indices = meta["indices"]
        num_frames = meta["num_frames"]

        npy_path = cache_dir / "frames.npy"
        if npy_path.exists():
            arr = np.load(npy_path)
            images = [Image.fromarray(arr[i]) for i in range(num_frames)]
            return images, fps, indices

        frame_paths = [cache_dir / f"{i:04d}.jpg" for i in range(num_frames)]
        if not all(p.exists() for p in frame_paths):
            return None
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        return images, fps, indices

    def _save_frames_to_cache(
        self,
        cache_dir: Path,
        frames: torch.Tensor,
        fps: float,
        indices: list,
    ) -> None:
        if cache_dir.exists():
            return
        Path(self.video_frame_cache_dir).mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(
            tempfile.mkdtemp(
                dir=self.video_frame_cache_dir,
                prefix=f".tmp_{cache_dir.name[:64]}_",
            )
        )
        try:
            meta = {"fps": fps, "indices": indices, "num_frames": len(indices)}
            (tmp_dir / "meta.json").write_text(json.dumps(meta))
            if self._cache_format == "npy":
                arr = frames.permute(0, 2, 3, 1).numpy().astype(np.uint8)
                np.save(tmp_dir / "frames.npy", arr)
            else:
                for i, frame in enumerate(frames):
                    img = Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
                    img.save(tmp_dir / f"{i:04d}.jpg", quality=95)
            try:
                tmp_dir.rename(cache_dir)
            except OSError:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def _process_video_with_timestamp(
        self,
        video_url: str,
        subtitle_data: dict | None = None,
        task: str | None = None,
    ):
        """Mirror V2 chat ``_process_video_content_with_timestamp``.

        Decodes via qwen-vl-utils ``fetch_video``, builds a per-frame
        content list ``[timestamp, image, (subtitle), ...]`` and
        returns it together with the list of PIL frames (so the caller
        can pass ``images=...`` to the processor — the V2-aligned route).
        """
        from qwen_vl_utils import fetch_video

        # Try cache first.
        cache_hit_images: List[Image.Image] | None = None
        cache_indices: list | None = None
        cache_fps: float | None = None
        cache_dir = None
        if self.cache_video_frames:
            cache_dir = self._cache_dir_for(video_url)
            cached = self._load_frames_from_cache(cache_dir)
            if cached is not None:
                cache_hit_images, cache_fps, cache_indices = cached

        if cache_hit_images is None:
            video_request = {
                "type": "video",
                "video": video_url,
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fixed_num_frames:
                total_frames = self._get_video_total_frames(video_url)
                video_request["nframes"] = min(self.fixed_num_frames, total_frames)
            else:
                video_request["fps"] = self.fps
                video_request["max_frames"] = self.max_num_frames
            video_input, video_metadata = fetch_video(
                video_request,
                return_video_metadata=True,
            )
            frames = video_input  # tensor [T, 3, H, W] uint8
            indices = video_metadata["frames_indices"]
            fps = video_metadata.get("fps", 30.0)
            if not isinstance(indices, list):
                indices = indices.tolist()
            # Persist before merge_size padding so cache stays canonical.
            if self.cache_video_frames and cache_dir is not None:
                try:
                    self._save_frames_to_cache(cache_dir, frames, fps, indices)
                except Exception as e:
                    eval_logger.warning(f"[v3] frame cache save failed: {e}")
            pil_frames = [Image.fromarray(f.permute(1, 2, 0).numpy().astype(np.uint8)) for f in frames]
        else:
            pil_frames = cache_hit_images
            indices = list(cache_indices)
            fps = cache_fps or 30.0

        merge_size = 2
        if len(indices) % merge_size != 0:
            pad = merge_size - len(indices) % merge_size
            indices.extend(indices[-1] for _ in range(pad))
            pil_frames = list(pil_frames) + [pil_frames[-1]] * pad
        timestamps = [idx / fps for idx in indices]

        video_content: list[dict] = []
        pil_images: list = []
        timestamp_decimals = self._timestamp_decimals_for_task(task)
        for img, t in zip(pil_frames, timestamps):
            ts_text = f"<{t:.{timestamp_decimals}f} seconds>"
            video_content.append({"type": "text", "text": ts_text})
            video_content.append({"type": "image", "image": img})
            pil_images.append(img)
            if subtitle_data:
                sub = self._find_subtitle_for_timestamp(t, subtitle_data)
                if sub:
                    video_content.append({"type": "text", "text": sub})
        return video_content, pil_images

    def _get_video_total_frames(self, video_url: str) -> int:
        cap = cv2.VideoCapture(video_url)
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        return max(1, total_frames)

    def _build_messages(self, chat_message: ChatMessages, task: str | None = None):
        """Convert lmms-eval ChatMessages -> HF chat-template list, V2-aligned.

        Mirrors the V2 chat wrapper:
          - ``__SUBTITLE_DATA__:`` text content is stripped from the input
            and parsed; the resulting subtitle dict is interleaved into the
            video content per-frame after the corresponding timestamp tag.
          - ``video`` items are expanded to per-frame timestamp
            text + image pairs via qwen-vl-utils ``fetch_video`` (frames
            backend) or routed as raw URLs to the processor's codec
            backend (``video_backend="codec"``).
          - The caller feeds the returned ``images=...`` to the processor
            (NOT ``videos=``), matching the V2 image-processor route which
            is what the sweep checkpoint was trained on. For codec mode the
            caller instead forwards the collected URL list as ``videos=``.

        Returns: ``(messages, pil_images, video_urls, subtitle_dicts)``.
        ``video_urls`` and ``subtitle_dicts`` are non-empty only when
        ``self.video_backend == "codec"``; they are aligned position-wise
        with the video placeholders in ``messages``.
        """
        from lmms_eval.tasks.videomme.utils import SUBTITLE_DATA_PREFIX

        out_msgs = []
        all_pil_images: list = []
        all_video_urls: list = []
        all_subtitle_dicts: list = []
        if self.system_prompt:
            out_msgs.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
        for message in chat_message.messages:
            content: list[dict] = []
            subtitle_data: dict | None = None
            for c in message.content:
                if c.type == "text":
                    if c.text.startswith(SUBTITLE_DATA_PREFIX):
                        subtitle_data = json.loads(c.text[len(SUBTITLE_DATA_PREFIX) :])
                        continue
                    content.append({"type": "text", "text": c.text})
                elif c.type == "image":
                    content.append({"type": "image", "image": c.url})
                elif c.type == "video":
                    if self.video_backend == "codec":
                        # Codec backend: defer canvas packing to the
                        # processor's cv-preinfer pipeline. Subtitles are
                        # retained here and interleaved per-frame *after*
                        # the processor returns patch_positions (see
                        # ``_codec_call_processor`` for the rewrite).
                        content.append({"type": "video"})
                        all_video_urls.append(c.url)
                        all_subtitle_dicts.append(subtitle_data)
                    elif "timestamp" in self.messages_format:
                        video_content, pil_images = self._process_video_with_timestamp(
                            c.url,
                            subtitle_data=subtitle_data,
                            task=task,
                        )
                        content.extend(video_content)
                        all_pil_images.extend(pil_images)
                    else:
                        # Fall back to bundled video pipeline (non-timestamp)
                        content.append({"type": "video", "video": c.url})
            out_msgs.append({"role": message.role, "content": content})
        return out_msgs, all_pil_images, all_video_urls, all_subtitle_dicts

    # -------------------------------------------------------- codec helpers

    def _codec_call_processor(
        self,
        texts: list[str],
        flat_videos: list[str],
        subtitle_dicts: list[dict | None],
    ):
        """Codec preprocessing with per-frame subtitle interleaving.

        Re-implements the codec branch of
        ``LlavaOnevision2Processor.__call__`` (processing_llava_onevision2.py
        lines ~245-315), but inserts subtitles into the rewritten chat
        template *before* tokenization. The codec assets can be sourced
        either from pre-generated offline files (``LLAVA_CODEC_OFFLINE_ROOT``)
        or from an online ``cv-preinfer`` invocation.

        Args:
            texts: chat-template strings (one per prompt in the batch).
            flat_videos: video URLs flattened across the batch (one per
                ``<|vision_start|><|video_pad|><|vision_end|>`` span).
            subtitle_dicts: per-video subtitle dicts (same length as
                ``flat_videos``); may contain ``None``.

        Returns:
            ``BatchFeature``-shaped dict with ``input_ids``,
            ``attention_mask``, ``pixel_values``, ``image_grid_thw``,
            ``patch_positions``.
        """
        # Import bundled codec helpers from the model checkpoint dir
        # (transformers' dynamic-module loader registers them as a sibling
        # module of processing_llava_onevision2).
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            module_path = f"{self.pretrained}--codec_video_processing_llava_onevision2.process_codec_video"
            process_codec_video = get_class_from_dynamic_module(module_path, self.pretrained)
            CodecConfig = get_class_from_dynamic_module(
                module_path.replace("process_codec_video", "CodecConfig"),
                self.pretrained,
            )
            drop_padding_canvases = get_class_from_dynamic_module(
                module_path.replace("process_codec_video", "drop_padding_canvases"),
                self.pretrained,
            )
            codec_positions_for_processor = get_class_from_dynamic_module(
                module_path.replace("process_codec_video", "codec_positions_for_processor"),
                self.pretrained,
            )
            codec_image_processor_outputs = get_class_from_dynamic_module(
                module_path.replace("process_codec_video", "codec_image_processor_outputs"),
                self.pretrained,
            )
        except Exception:
            # Fallback: model dir is on PYTHONPATH (rare) or already imported.
            from codec_video_processing_llava_onevision2 import (  # type: ignore
                CodecConfig,
                codec_image_processor_outputs,
                codec_positions_for_processor,
                drop_padding_canvases,
                process_codec_video,
            )

        # Build effective codec config (defaults < class-level < per-call).
        cfg_kwargs = dict(self.codec_config)
        cfg_kwargs["max_pixels"] = int(self.max_pixels)
        cfg = CodecConfig(**cfg_kwargs)

        if len(texts) != len(flat_videos):
            # Processor contract is one video per prompt; if mismatched we
            # broadcast a single text across multiple videos (rare for the
            # benchmarks we target, but mirrors the bundled processor).
            if len(texts) == 1 and len(flat_videos) >= 1:
                texts = texts * len(flat_videos)
            else:
                raise ValueError(f"codec backend: got {len(texts)} texts but {len(flat_videos)} videos")
        if len(subtitle_dicts) != len(flat_videos):
            # Pad with None so the index lookup never fails.
            subtitle_dicts = list(subtitle_dicts) + [None] * (len(flat_videos) - len(subtitle_dicts))

        all_pixel_values: list[torch.Tensor] = []
        all_grid_thw: list[torch.Tensor] = []
        all_patch_positions: list[torch.Tensor] = []
        rewritten_texts: list[str] = []

        sms = int(getattr(self.processor, "spatial_merge_size", 2))

        offline_root = os.environ.get("LLAVA_CODEC_OFFLINE_ROOT", "").strip()
        use_online_tuned = os.environ.get("LLAVA_CODEC_ONLINE_TUNED", "0") == "1"

        for idx, (video_url, in_text) in enumerate(zip(flat_videos, texts)):
            offline_payload = _try_load_offline_codec_assets(video_url, offline_root) if offline_root else None
            if offline_payload is not None:
                payload = offline_payload
            elif use_online_tuned:
                payload = _process_codec_video_tuned(video_url, cfg)
            else:
                payload = process_codec_video(video_url, cfg)
            imgs, src_positions, _ = drop_padding_canvases(payload["images"], payload["src_positions"])
            if not imgs:
                raise RuntimeError(f"codec produced no usable canvases for {video_url}")
            image_data = codec_image_processor_outputs(
                self.processor.image_processor,
                imgs,
                max_pixels=int(self.max_pixels),
            )
            image_grid_thw = image_data["image_grid_thw"]
            patch_positions = codec_positions_for_processor(
                src_positions, image_grid_thw, device=image_grid_thw.device
            )
            fps = float(payload["fps"]) if payload.get("fps") else 30.0

            # Build subtitle->frame assignment from the (filtered+grouped)
            # subtitle dict, then rewrite the vision span in `in_text` with
            # per-frame timestamps + interleaved subtitle lines.
            raw_subs = subtitle_dicts[idx] if idx < len(subtitle_dicts) else None
            subs = _filter_and_group_subtitles(raw_subs) if raw_subs else None
            frame_times = _frame_times_from_patch_positions(patch_positions, fps)
            sub_after_frame = _build_sub_after_frame(subs, frame_times)

            rewritten = _rewrite_codec_vision_with_subtitles(
                in_text,
                patch_positions,
                fps=fps,
                decimals=self.timestamp_decimals,
                spatial_merge_size=sms,
                sub_after_frame=sub_after_frame,
            )
            if not getattr(self, "_logged_first_rewritten", False) and getattr(self, "_rank", 0) == 0:
                import re as _re

                ts_samples = _re.findall(r"<\d+\.\d+ seconds>", rewritten)[:5]
                eval_logger.info(
                    f"[llava_onevision2] first rewritten timestamps (decimals={self.timestamp_decimals}): {ts_samples}"
                )
                self._logged_first_rewritten = True

            all_pixel_values.append(image_data["pixel_values"])
            all_grid_thw.append(image_grid_thw)
            all_patch_positions.append(patch_positions)
            rewritten_texts.append(rewritten)

        # Tokenize once.
        encoding = self.tokenizer(
            rewritten_texts,
            padding=True,
            return_tensors="pt",
        )

        # Merge per-canvas image_grid_thw rows into a single ``[N, H, W]`` row
        # so the vision encoder treats the canvases as one ``N``-frame sample.
        # This matches the production simple-path behaviour
        # (lmms-eval-ov2/.../simple/llava_onevision2.py ``merged_grid_thw``)
        # and is required for the vision encoder's ``_build_cu_seqlens`` to
        # group canvases into ``frame_windows_size=4`` attention windows
        # (cross-canvas self-attention) instead of per-canvas isolated groups.
        # When canvases differ in (H, W) we keep the row-wise layout; this is
        # rare for codec output and the pipeline already pads to common dims.
        merged_grid_thw_rows = []
        for grid in all_grid_thw:
            # ``grid`` is per-video, shape ``(num_canvases, 3)`` with each
            # row being ``[1, h, w]``; merge into a single ``[N, h, w]`` row
            # iff all canvases share the same (h, w).
            same_h = bool(torch.all(grid[:, 1] == grid[0, 1]).item())
            same_w = bool(torch.all(grid[:, 2] == grid[0, 2]).item())
            if same_h and same_w:
                n = int(grid.shape[0])
                h = int(grid[0, 1].item())
                w = int(grid[0, 2].item())
                merged_grid_thw_rows.append(
                    torch.tensor([[n, h, w]], dtype=grid.dtype, device=grid.device)
                )
            else:
                merged_grid_thw_rows.append(grid)

        out = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding.get("attention_mask", torch.ones_like(encoding["input_ids"])),
            "pixel_values": torch.cat(all_pixel_values, dim=0),
            "image_grid_thw": torch.cat(merged_grid_thw_rows, dim=0),
            "patch_positions": torch.cat(all_patch_positions, dim=0),
        }
        try:
            from transformers.feature_extraction_utils import BatchFeature

            return BatchFeature(data=out)
        except Exception:
            return out

    # -------------------------------------------------------- main loop

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []

        def _collate(x):
            return x[2], x[2]

        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        n_iter = (len(requests) + self.batch_size - 1) // self.batch_size
        pbar = tqdm(total=n_iter, disable=(self.rank != 0), desc="Model Responding")

        e2e_latency = 0.0
        total_tokens = 0

        # Optional checkpoint cache (env-driven, mirrors existing wrappers)
        ckpt_dir = os.getenv("EVAL_CHECKPOINT_DIR", "")
        ckpt_path = None
        ckpt_cache: dict = {}
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_rank{self.rank}.jsonl")
            if os.path.exists(ckpt_path):
                with open(ckpt_path, "r") as f:
                    for line in f:
                        e = json.loads(line)
                        ckpt_cache[e["doc_id"]] = e["response"]

        for chunk in chunks:
            ctx, doc_to_messages, gen_kwargs_raw, doc_ids, tasks, splits = zip(*chunk)
            task = tasks[0]
            split = splits[0]

            chunk_doc_ids = list(doc_ids)
            if ckpt_cache and all(d in ckpt_cache for d in chunk_doc_ids):
                for d in chunk_doc_ids:
                    res.append(ckpt_cache[d])
                pbar.update(1)
                continue

            # Build chat messages for each doc in chunk.
            try:
                hf_messages_batch = []
                pil_images_batch: List[list] = []
                video_urls_batch: List[list] = []
                subtitle_dicts_batch: List[list] = []
                for did in doc_ids:
                    raw = doc_to_messages[0](self.task_dict[task][split][did])
                    cm = ChatMessages(**{"messages": raw})
                    msgs, pil_imgs, vid_urls, sub_dicts = self._build_messages(cm, task=task)
                    hf_messages_batch.append(msgs)
                    pil_images_batch.append(pil_imgs)
                    video_urls_batch.append(vid_urls)
                    subtitle_dicts_batch.append(sub_dicts)

                texts = [
                    self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                    for m in hf_messages_batch
                ]

                # V2-aligned: per-frame PIL images go via images=, NOT
                # videos=. This matches the V2 chat path and produces
                # bit-exact input_ids on validated samples (LVBench: 100%
                # match across 1549 samples; videomme_isub: matches V2
                # incl. SUBTITLE_DATA stripping + per-frame interleave).
                images_arg = None
                first_pil = pil_images_batch[0] if pil_images_batch else []
                if first_pil:
                    images_arg = first_pil

                proc_kwargs = dict(return_tensors="pt", padding=True)

                if self.video_backend == "codec":
                    # Codec backend: run cv-preinfer + per-frame subtitle
                    # interleaving on this side instead of letting the
                    # bundled processor do it (the bundled codec branch
                    # has no subtitle hook).
                    flat_videos = [u for urls in video_urls_batch for u in urls]
                    flat_subs = [s for ss in subtitle_dicts_batch for s in ss]
                    inputs = self._codec_call_processor(
                        texts=texts,
                        flat_videos=flat_videos,
                        subtitle_dicts=flat_subs,
                    )
                else:
                    inputs = self.processor(
                        text=texts,
                        images=images_arg,
                        videos=None,
                        **proc_kwargs,
                    )
            except Exception as e:
                eval_logger.error(f"[v3] preparing chunk failed: {e}; empty resp")
                for did in chunk_doc_ids:
                    res.append("")
                    if ckpt_path:
                        with open(ckpt_path, "a") as f:
                            f.write(json.dumps({"doc_id": did, "response": ""}) + "\n")
                pbar.update(1)
                continue

            # Move to device.
            inputs = {k: (v.to(self._device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

            # Build generation kwargs.
            user_gk = dict(gen_kwargs_raw[0] or {})
            max_new = int(user_gk.get("max_new_tokens", self.max_new_tokens))
            do_sample = bool(user_gk.get("temperature", 0) and user_gk["temperature"] > 0)
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

            gen_args = dict(inputs)
            # transformers 5.x processor adds mm_token_type_ids which our
            # custom modeling does not consume; drop it to avoid
            # "model_kwargs not used" ValueError from generate().
            gen_args.pop("mm_token_type_ids", None)
            gen_args.update(
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_id,
                max_new_tokens=max_new,
                num_beams=int(user_gk.get("num_beams", 1)),
                do_sample=do_sample,
                use_cache=self.use_cache,
            )
            if do_sample:
                gen_args["temperature"] = float(user_gk["temperature"])
                if user_gk.get("top_p"):
                    gen_args["top_p"] = float(user_gk["top_p"])

            # Generate. We deliberately do NOT swallow OOM / runtime errors:
            # silently writing empty strings on generate failure caused
            # entire benchmarks to be reported as "OK" with garbage scores
            # (~19% on videomme_short) when GPUs were oversubscribed by
            # other workloads. Crash loudly so the sweep records FAILED.
            start = time.time()
            with torch.inference_mode():
                cont = self.model.generate(**gen_args)
            e2e_latency += time.time() - start
            cont = cont[:, inputs["input_ids"].shape[-1] :]
            total_tokens += int(cont.shape[-1]) if cont.ndim > 1 else int(cont.shape[-1])

            text_outs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            for i, txt in enumerate(text_outs):
                res.append(txt)
                self.cache_hook.add_partial("generate_until", (texts[0], user_gk), txt)
                if ckpt_path:
                    did = doc_ids[i] if i < len(doc_ids) else doc_ids[0]
                    with open(ckpt_path, "a") as f:
                        f.write(json.dumps({"doc_id": did, "response": txt}) + "\n")
            pbar.update(1)

        res = re_ords.get_original(res)

        log_metrics(
            total_tokens=total_tokens,
            e2e_latency=e2e_latency,
            avg_speed=total_tokens / e2e_latency if e2e_latency > 0 else 0,
            additional_metrics={"rank": self.rank},
        )
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError

    def loglikelihood(self, requests):
        raise NotImplementedError
