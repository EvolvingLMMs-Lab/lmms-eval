"""lmms-eval adapter for OpenDataLab's Apache-2.0 CDM implementation.

Modified to expose a reusable evaluator and bounded per-sample workspace for
MDPBench. See ``cdm/LICENSE`` and ``THIRD_PARTY_NOTICES.md``.
"""

import json
import logging
import os
import shutil
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import ransac

from lmms_eval.tasks.mdpbench.cdm.modules.latex2bbox_color import latex2bbox_color
from lmms_eval.tasks.mdpbench.cdm.modules.visual_matcher import (
    HungarianMatcher,
    SimpleAffineTransform,
)

eval_logger = logging.getLogger("lmms-eval")


class CDM:
    def __init__(self, output_root="./result"):
        """
        Initialize the LaTeX formula evaluator.

        Args:
            output_root (str): Root directory for saving intermediate and final results
        """
        self.output_root = output_root
        self.matcher = HungarianMatcher()

        # Evaluation parameters
        self.max_iter = 3
        self.min_samples = 3
        self.residual_threshold = 25
        self.max_trials = 50

    @staticmethod
    @lru_cache(maxsize=8)
    def gen_color_list(num=10, gap=15):
        """Generate a list of distinct colors for visualization (cached)."""
        num += 1
        single_num = 255 // gap + 1
        max_num = single_num**3
        num = min(num, max_num)
        color_list = []
        for idx in range(num):
            R = idx // single_num**2
            GB = idx % single_num**2
            G = GB // single_num
            B = GB % single_num
            color_list.append((R * gap, G * gap, B * gap))
        return color_list[1:]

    @staticmethod
    def update_inliers(ori_inliers, sub_inliers):
        """Update inliers status based on new RANSAC results"""
        inliers = np.copy(ori_inliers)
        sub_idx = -1
        for idx in range(len(ori_inliers)):
            if ori_inliers[idx] == False:
                sub_idx += 1
                if sub_inliers[sub_idx] == True:
                    inliers[idx] = True
        return inliers

    def _prepare_directories(self, img_id):
        """Create necessary directories for output"""
        os.makedirs(os.path.join(self.output_root, "gt", "bbox"), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, "gt", "vis"), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, "pred", "bbox"), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, "pred", "vis"), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, "vis_match"), exist_ok=True)

    def _generate_bboxes(self, gt_latex, pred_latex, img_id):
        """Generate bounding boxes for both GT and prediction"""
        total_color_list = self.gen_color_list(num=5800)

        for subset, latex in zip(["gt", "pred"], [gt_latex, pred_latex]):
            output_path = os.path.join(self.output_root, subset)
            temp_dir = os.path.join(self.output_root, f"temp_dir_{subset}_{img_id}")
            os.makedirs(temp_dir, exist_ok=True)

            latex2bbox_color((latex, img_id, output_path, temp_dir, total_color_list))
            shutil.rmtree(temp_dir)

    def _load_bboxes(self, img_id):
        """Load generated bounding boxes from files"""
        gt_box_path = os.path.join(self.output_root, "gt", "bbox", f"{img_id}.jsonl")
        pred_box_path = os.path.join(self.output_root, "pred", "bbox", f"{img_id}.jsonl")

        with open(gt_box_path, "r") as f:
            box_gt = [json.loads(line) for line in f if json.loads(line)["bbox"]]

        with open(pred_box_path, "r") as f:
            box_pred = [json.loads(line) for line in f if json.loads(line)["bbox"]]

        return box_gt, box_pred

    def _load_images(self, img_id):
        """Load visualization images"""
        gt_img_path = os.path.join(self.output_root, "gt", "vis", f"{img_id}_base.png")
        pred_img_path = os.path.join(self.output_root, "pred", "vis", f"{img_id}_base.png")
        return Image.open(gt_img_path), Image.open(pred_img_path)

    def _match_boxes(self, box_gt, box_pred, img_gt, img_pred):
        """Perform box matching using Hungarian algorithm and RANSAC"""
        matched_idxes = self.matcher(box_gt, box_pred, img_gt.size, img_pred.size)

        # Prepare matching points
        src, dst = [], []
        for idx1, idx2 in matched_idxes:
            x1min, y1min, x1max, y1max = box_gt[idx1]["bbox"]
            x2min, y2min, x2max, y2max = box_pred[idx2]["bbox"]
            src.append([float((y1min + y1max) / 2), float((x1min + x1max) / 2)])
            dst.append([float((y2min + y2max) / 2), float((x2min + x2max) / 2)])

        src = np.array(src)
        dst = np.array(dst)

        # Apply RANSAC filtering
        if src.shape[0] <= self.min_samples:
            inliers = np.array([True for _ in matched_idxes])
        else:
            inliers = np.array([False for _ in matched_idxes])
            for i in range(self.max_iter):
                if src[inliers == False].shape[0] <= self.min_samples:
                    break
                try:
                    model, inliers_1 = ransac(
                        (src[inliers == False], dst[inliers == False]),
                        SimpleAffineTransform,
                        min_samples=self.min_samples,
                        residual_threshold=self.residual_threshold,
                        max_trials=self.max_trials,
                        random_state=42,
                    )
                except TypeError:
                    # Older scikit-image versions do not support `random_state`.
                    model, inliers_1 = ransac(
                        (src[inliers == False], dst[inliers == False]),
                        SimpleAffineTransform,
                        min_samples=self.min_samples,
                        residual_threshold=self.residual_threshold,
                        max_trials=self.max_trials,
                    )
                if inliers_1 is not None and inliers_1.any():
                    inliers = self.update_inliers(inliers, inliers_1)
                else:
                    break
                if len(inliers[inliers == True]) >= len(matched_idxes):
                    break

        # Filter token mismatches
        for idx, (a, b) in enumerate(matched_idxes):
            if inliers[idx] == True and self.matcher.cost["token"][a, b] == 1:
                inliers[idx] = False

        return matched_idxes, inliers

    def _calculate_metrics(self, box_gt, box_pred, inliers):
        """Calculate evaluation metrics"""
        final_match_num = len(inliers[inliers == True])
        recall = round(final_match_num / len(box_gt), 3)
        precision = round(final_match_num / len(box_pred), 3)
        F1_score = round(2 * final_match_num / (len(box_gt) + len(box_pred)), 3)
        return recall, precision, F1_score

    def _visualize_matches(self, img_gt, img_pred, box_gt, box_pred, matched_idxes, inliers, img_id):
        """Generate and save visualization of matches"""
        gap = 5
        W1, H1 = img_gt.size
        W2, H2 = img_pred.size
        H = H1 + H2 + gap
        W = max(W1, W2)

        # Create base visualization
        vis_img = Image.new("RGB", (W, H), (255, 255, 255))
        vis_img.paste(img_gt, (0, 0))
        vis_img.paste(Image.new("RGB", (W, gap), (120, 120, 120)), (0, H1))
        vis_img.paste(img_pred, (0, H1 + gap))

        # Create match visualization
        match_img = vis_img.copy()
        match_draw = ImageDraw.Draw(match_img)

        gt_matched_idx = {a: flag for (a, b), flag in zip(matched_idxes, inliers)}
        pred_matched_idx = {b: flag for (a, b), flag in zip(matched_idxes, inliers)}

        # Draw GT boxes
        for idx, box in enumerate(box_gt):
            color = "green" if idx in gt_matched_idx and gt_matched_idx[idx] == True else "red"
            x_min, y_min, x_max, y_max = box["bbox"]
            match_draw.rectangle([x_min - 1, y_min - 1, x_max + 1, y_max + 1], fill=None, outline=color, width=2)

        # Draw prediction boxes
        for idx, box in enumerate(box_pred):
            color = "green" if idx in pred_matched_idx and pred_matched_idx[idx] == True else "red"
            x_min, y_min, x_max, y_max = box["bbox"]
            match_draw.rectangle([x_min - 1, y_min - 1 + H1 + gap, x_max + 1, y_max + 1 + H1 + gap], fill=None, outline=color, width=2)

        # Save visualizations
        vis_img.save(os.path.join(self.output_root, "vis_match", f"{img_id}_base.png"))
        match_img.save(os.path.join(self.output_root, "vis_match", f"{img_id}.png"))

    def evaluate(self, gt_latex, pred_latex, img_id):
        """
        Evaluate a single LaTeX formula pair (ground truth vs prediction)

        Args:
            gt_latex (str): Ground truth LaTeX formula
            pred_latex (str): Predicted LaTeX formula
            img_id (str): Unique identifier for this evaluation

        Returns:
            dict: Evaluation metrics (recall, precision, F1_score)
        """

        import time

        try:
            self._prepare_directories(img_id)
            t0 = time.time()
            self._generate_bboxes(gt_latex, pred_latex, img_id)
            t1 = time.time()
            box_gt, box_pred = self._load_bboxes(img_id)
            t2 = time.time()
            img_gt, img_pred = self._load_images(img_id)
            t3 = time.time()
            matched_idxes, inliers = self._match_boxes(box_gt, box_pred, img_gt, img_pred)
            t4 = time.time()
            if (t4 - t0) > 5:
                eval_logger.warning(f"CDM slow [{img_id}]: generate={t1 - t0:.1f}s load_bbox={t2 - t1:.1f}s load_img={t3 - t2:.1f}s match={t4 - t3:.1f}s total={t4 - t0:.1f}s")
        except Exception as e:
            eval_logger.warning(f"CDM evaluation failed for img_id={img_id}: {type(e).__name__}: {e}")
            self._cleanup_files(img_id)
            return {"recall": 0, "precision": 0, "F1_score": 0}

        recall, precision, F1_score = self._calculate_metrics(box_gt, box_pred, inliers)
        self._visualize_matches(img_gt, img_pred, box_gt, box_pred, matched_idxes, inliers, img_id)
        self._cleanup_files(img_id)

        return {
            "recall": recall,
            "precision": precision,
            "F1_score": F1_score,
        }

    def _cleanup_files(self, img_id):
        """Remove per-sample intermediate files to prevent accumulation"""
        for subset in ["gt", "pred"]:
            for subdir in ["bbox", "vis"]:
                for suffix in [".jsonl", "_base.png", ".png"]:
                    path = os.path.join(self.output_root, subset, subdir, f"{img_id}{suffix}")
                    if os.path.exists(path):
                        os.remove(path)
        for suffix in ["_base.png", ".png"]:
            path = os.path.join(self.output_root, "vis_match", f"{img_id}{suffix}")
            if os.path.exists(path):
                os.remove(path)
