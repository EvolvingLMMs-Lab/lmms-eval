"""
Base Evaluator class for VBVR-Bench.
All task-specific evaluators inherit from this class.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils import (
    compute_frame_difference,
    compute_mse,
    compute_psnr,
    compute_ssim,
    get_frame_count,
    get_video_frames,
    get_video_info,
    linear_score,
    load_image,
    normalize_frame_size,
    weighted_average,
)


class BaseEvaluator(ABC):
    """
    Abstract base class for all VBVR-Bench evaluators.

    Each evaluator implements task-specific evaluation logic and returns
    continuous scores based on rule-based criteria.
    """

    # Default weights for evaluation dimensions
    # These MUST match the dimension names used in evaluate()
    DEFAULT_WEIGHTS = {
        "first_frame_consistency": 0.15,
        "final_frame_accuracy": 0.35,
        "temporal_smoothness": 0.15,
        "visual_quality": 0.10,
        "task_specific": 0.25,
    }

    def __init__(self, device: str = "cuda", task_name: str = ""):
        """
        Initialize base evaluator.

        Args:
            device: Device for computation ('cuda' or 'cpu')
            task_name: Name of the task being evaluated
        """
        self.device = device
        self.task_name = task_name

    def evaluate(self, eval_info: Dict, **kwargs) -> Dict[str, Any]:
        """
        Main evaluation function.

        Args:
            eval_info: Dictionary containing:
                - video_path: Path to the video to evaluate
                - gt_path: Path to ground truth folder
                - gt_video_path: Path to GT video
                - gt_first_frame: Path to GT first frame
                - gt_final_frame: Path to GT final frame
                - prompt: Task prompt (optional)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - score: Overall score (0-1, continuous)
                - dimensions: Dict of dimension -> score
                - details: Additional evaluation details
        """
        result = {"score": 0.0, "dimensions": {}, "details": {}}

        try:
            # Load videos and frames
            video_frames = self._load_video_frames(eval_info["video_path"])
            gt_first_frame = self._load_gt_first_frame(eval_info)
            gt_final_frame = self._load_gt_final_frame(eval_info)
            gt_frames = self._load_gt_video_frames(eval_info)

            # Get GT video info for reference
            gt_video_path = eval_info.get("gt_video_path")
            if gt_video_path and os.path.exists(gt_video_path):
                gt_info = get_video_info(gt_video_path)
                result["details"]["gt_frame_count"] = gt_info["frame_count"]

            result["details"]["video_frame_count"] = len(video_frames)

            # CRITICAL: Normalize video frames to match GT frame size
            # This handles different video resolutions (e.g., 720x1280 vs 1024x1024)
            target_frame = gt_first_frame if gt_first_frame is not None else gt_final_frame
            if target_frame is not None and len(video_frames) > 0:
                if video_frames[0].shape != target_frame.shape:
                    result["details"]["frame_normalization"] = f"{video_frames[0].shape} -> {target_frame.shape}"
                    video_frames = [normalize_frame_size(f, target_frame) for f in video_frames]
                    # Also normalize GT frames if loaded
                    if gt_frames and len(gt_frames) > 0:
                        gt_frames = [normalize_frame_size(f, target_frame) for f in gt_frames]

            # Compute dimension scores using STANDARD dimension names
            # These names MUST match DEFAULT_WEIGHTS keys
            dimensions = {}

            # 1. First frame consistency (weight: 0.15)
            if gt_first_frame is not None and len(video_frames) > 0:
                dimensions["first_frame_consistency"] = self._evaluate_first_frame(video_frames[0], gt_first_frame)
            else:
                dimensions["first_frame_consistency"] = 0.5  # Default if no GT

            # 2. Final frame accuracy (weight: 0.35)
            if gt_final_frame is not None and len(video_frames) > 0:
                dimensions["final_frame_accuracy"] = self._evaluate_final_frame(video_frames[-1], gt_final_frame)
            else:
                dimensions["final_frame_accuracy"] = 0.0  # No GT means we can't evaluate

            # 3. Temporal smoothness (weight: 0.15)
            if len(video_frames) > 1:
                dimensions["temporal_smoothness"] = self._evaluate_temporal_smoothness(video_frames)
            else:
                dimensions["temporal_smoothness"] = 0.5  # Default for single frame

            # 4. Visual quality (weight: 0.10)
            if len(video_frames) > 0:
                dimensions["visual_quality"] = self._evaluate_visual_quality(video_frames)
            else:
                dimensions["visual_quality"] = 0.0

            # 5. Task-specific evaluation (weight: 0.25)
            task_score = self._evaluate_task_specific(video_frames, gt_frames, gt_first_frame, gt_final_frame, eval_info)
            # SAFETY: Clamp task score to [0, 1] range
            task_score = max(0.0, min(1.0, task_score))
            dimensions["task_specific"] = task_score

            if hasattr(self, "_last_task_details"):
                result["details"]["task_specific_details"] = self._last_task_details

            # Optionally focus on task-specific score only
            if kwargs.get("task_specific_only"):
                result["dimensions"] = {"task_specific": task_score}
                result["score"] = task_score
                result["details"]["task_specific_only"] = True
            else:
                # Calculate weighted overall score using standard weights
                result["dimensions"] = dimensions
                result["score"] = self._calculate_overall_score(dimensions)

        except Exception as e:
            result["error"] = str(e)
            result["score"] = 0.0

        return result

    def _load_video_frames(self, video_path: str, max_frames: int = 100) -> List[np.ndarray]:
        """Load frames from the video to evaluate."""
        if not os.path.exists(video_path):
            return []
        return get_video_frames(video_path, max_frames=max_frames)

    def _load_gt_first_frame(self, eval_info: Dict) -> Optional[np.ndarray]:
        """Load ground truth first frame."""
        path = eval_info.get("gt_first_frame")
        if path and os.path.exists(path):
            return load_image(path)
        return None

    def _load_gt_final_frame(self, eval_info: Dict) -> Optional[np.ndarray]:
        """Load ground truth final frame."""
        path = eval_info.get("gt_final_frame")
        if path and os.path.exists(path):
            return load_image(path)
        return None

    def _load_gt_video_frames(self, eval_info: Dict, max_frames: int = 100) -> List[np.ndarray]:
        """Load ground truth video frames."""
        path = eval_info.get("gt_video_path")
        if path and os.path.exists(path):
            return get_video_frames(path, max_frames=max_frames)
        return []

    @staticmethod
    def _resize_to_match(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize frames to match dimensions for comparison.
        Handles padding removal (gray/white/black borders) before resizing.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Tuple of (frame1, frame2) with matching dimensions
        """
        if frame1.shape == frame2.shape:
            return frame1, frame2

        frame2_normalized = normalize_frame_size(frame2, frame1)
        return frame1, frame2_normalized

    def _evaluate_first_frame(self, first_frame: np.ndarray, gt_first_frame: np.ndarray) -> float:
        """
        Evaluate first frame consistency with GT.

        Important: For I2V tasks, the first frame should match the input image.

        Returns:
            Score between 0 and 1
        """
        # Normalize frame size (handles different resolutions and aspect ratios)
        if first_frame.shape != gt_first_frame.shape:
            gt_first_frame = normalize_frame_size(gt_first_frame, first_frame)

        # Compute SSIM
        ssim = compute_ssim(first_frame, gt_first_frame)

        # SSIM typically ranges from 0 to 1, with 1 being identical
        # We apply a threshold-based scoring:
        # SSIM >= 0.95: Perfect (1.0)
        # SSIM >= 0.85: Good (0.8-1.0)
        # SSIM >= 0.70: Acceptable (0.5-0.8)
        # SSIM < 0.70: Poor (0-0.5)

        if ssim >= 0.95:
            return 1.0
        elif ssim >= 0.85:
            return 0.8 + (ssim - 0.85) / 0.10 * 0.2
        elif ssim >= 0.70:
            return 0.5 + (ssim - 0.70) / 0.15 * 0.3
        else:
            return ssim / 0.70 * 0.5

    def _evaluate_final_frame(self, final_frame: np.ndarray, gt_final_frame: np.ndarray) -> float:
        """
        Evaluate final frame accuracy with GT.

        This is a key metric as it measures whether the task was completed correctly.

        Returns:
            Score between 0 and 1
        """
        # Normalize frame size (handles different resolutions and aspect ratios)
        if final_frame.shape != gt_final_frame.shape:
            gt_final_frame = normalize_frame_size(gt_final_frame, final_frame)

        # Compute multiple metrics
        ssim = compute_ssim(final_frame, gt_final_frame)
        psnr = compute_psnr(final_frame, gt_final_frame)

        # Normalize PSNR (typical range 20-50 for images)
        psnr_score = linear_score(psnr, 20, 40)

        # Combine SSIM and PSNR
        # SSIM is more perceptually relevant
        combined_score = 0.7 * ssim + 0.3 * psnr_score

        return combined_score

    def _evaluate_temporal_smoothness(self, frames: List[np.ndarray]) -> float:
        """
        Evaluate temporal smoothness of the video.

        Measures consistency between consecutive frames to detect flickering,
        sudden jumps, or temporal artifacts.

        Returns:
            Score between 0 and 1
        """
        if len(frames) < 2:
            return 1.0

        # Compute frame-to-frame differences
        differences = []
        for i in range(len(frames) - 1):
            diff = compute_frame_difference(frames[i], frames[i + 1])
            differences.append(diff)

        # Calculate statistics
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        max_diff = np.max(differences)

        # Score based on consistency
        # Low variance indicates smooth transitions
        # High max difference indicates potential jumps

        # Variance score (lower is better)
        variance_score = 1.0 - min(1.0, std_diff / 0.1)

        # Max jump score (lower is better)
        jump_score = 1.0 - min(1.0, max_diff / 0.3)

        # Combine scores
        smoothness_score = 0.6 * variance_score + 0.4 * jump_score

        return smoothness_score

    def _evaluate_visual_quality(self, frames: List[np.ndarray]) -> float:
        """
        Evaluate visual quality of the video.

        Measures sharpness, noise levels, and overall image quality.

        Returns:
            Score between 0 and 1
        """
        if len(frames) == 0:
            return 0.0

        quality_scores = []

        for frame in frames[:: max(1, len(frames) // 10)]:  # Sample frames
            # Measure sharpness using Laplacian variance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Normalize sharpness (typical range 0-5000)
            sharpness_score = min(1.0, laplacian_var / 1000)

            # Check for noise using high-frequency content
            # Apply Gaussian blur and measure difference
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise_estimate = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
            noise_score = 1.0 - min(1.0, noise_estimate / 30)

            quality_scores.append(0.6 * sharpness_score + 0.4 * noise_score)

        return np.mean(quality_scores)

    @abstractmethod
    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """
        Task-specific evaluation logic.

        This method must be implemented by each task-specific evaluator.

        Args:
            video_frames: List of frames from the video being evaluated
            gt_frames: List of frames from the ground truth video
            gt_first_frame: Ground truth first frame
            gt_final_frame: Ground truth final frame
            eval_info: Evaluation info dictionary

        Returns:
            Task-specific score between 0 and 1
        """
        pass

    def _calculate_overall_score(self, dimensions: Dict[str, float]) -> float:
        """
        Calculate overall score from dimension scores.

        Always uses the STANDARD weights defined in BaseEvaluator to ensure
        consistent scoring across all evaluators. The dimension names MUST match:
        - first_frame_consistency
        - final_frame_accuracy
        - temporal_smoothness
        - visual_quality
        - task_specific
        """
        # Always use BaseEvaluator's standard weights
        standard_weights = {
            "first_frame_consistency": 0.15,
            "final_frame_accuracy": 0.35,
            "temporal_smoothness": 0.15,
            "visual_quality": 0.10,
            "task_specific": 0.25,
        }
        # Clamp all dimension scores to [0, 1] before calculating weighted average
        clamped_dimensions = {k: max(0.0, min(1.0, v)) for k, v in dimensions.items()}
        return max(0.0, min(1.0, weighted_average(clamped_dimensions, standard_weights)))

    # =========================================================================
    # Utility methods for subclasses
    # =========================================================================

    def get_key_frames(self, frames: List[np.ndarray], gt_frame_count: int) -> List[np.ndarray]:
        """
        Extract key frames aligned with ground truth frame count.

        Args:
            frames: Video frames
            gt_frame_count: Number of frames in GT video

        Returns:
            List of key frames matching GT timing
        """
        if len(frames) == 0:
            return []

        if len(frames) <= gt_frame_count:
            return frames

        # Sample frames to match GT count
        indices = np.linspace(0, len(frames) - 1, gt_frame_count, dtype=int)
        return [frames[i] for i in indices]

    def find_best_matching_frame(self, target_frame: np.ndarray, candidate_frames: List[np.ndarray], start_idx: int = 0) -> Tuple[int, float]:
        """
        Find the frame in candidates that best matches target.

        Args:
            target_frame: Frame to match
            candidate_frames: List of candidate frames
            start_idx: Start searching from this index

        Returns:
            Tuple of (best_index, best_score)
        """
        best_idx = start_idx
        best_score = 0.0

        for i in range(start_idx, len(candidate_frames)):
            frame = candidate_frames[i]
            if frame.shape != target_frame.shape:
                frame = normalize_frame_size(frame, target_frame)

            score = compute_ssim(target_frame, frame)
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx, best_score

    def compute_trajectory_similarity(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], num_samples: int = 10) -> float:
        """
        Compare video trajectory to GT trajectory using sampled frames.

        Args:
            video_frames: Frames from video being evaluated
            gt_frames: Frames from ground truth video
            num_samples: Number of samples to compare

        Returns:
            Trajectory similarity score (0-1)
        """
        if len(video_frames) == 0 or len(gt_frames) == 0:
            return 0.0

        # Sample indices
        video_indices = np.linspace(0, len(video_frames) - 1, num_samples, dtype=int)
        gt_indices = np.linspace(0, len(gt_frames) - 1, num_samples, dtype=int)

        # Compare sampled frames
        similarities = []
        for v_idx, g_idx in zip(video_indices, gt_indices):
            v_frame = video_frames[v_idx]
            g_frame = gt_frames[g_idx]

            if v_frame.shape != g_frame.shape:
                g_frame = normalize_frame_size(g_frame, v_frame)

            sim = compute_ssim(v_frame, g_frame)
            similarities.append(sim)

        return np.mean(similarities)
