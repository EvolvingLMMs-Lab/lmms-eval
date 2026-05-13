"""
Specific evaluators for In-Domain_50 tasks (Part 1).
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils import compute_optical_flow, normalize_frame_size, safe_distance
from .base_evaluator import BaseEvaluator


class StableSortEvaluator(BaseEvaluator):
    """
    G-3: Stable sort objects evaluator.

    Rule-based evaluation:
    - Classification correctness (30%): Shapes correctly grouped by type (same type adjacent)
    - Order correctness (30%): Each group sorted small to large (left to right)
    - Object fidelity (30%): Shape types, sizes, colors preserved from initial frame
    - Layout accuracy (10%): Horizontal alignment (same y-coordinate)
    """

    TASK_WEIGHTS = {"classification": 0.30, "order": 0.30, "fidelity": 0.30, "layout": 0.10}

    def _detect_shapes(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored shapes and return their properties."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Find non-white/non-black areas (colored shapes)
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Skip noise
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Determine shape type by vertex count
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            vertices = len(approx)

            if vertices == 3:
                shape_type = "triangle"
            elif vertices == 4:
                shape_type = "square"
            else:
                shape_type = "circle"

            # Get dominant color at centroid region
            color = frame[max(0, cy - 5) : cy + 5, max(0, cx - 5) : cx + 5].mean(axis=(0, 1))

            shapes.append(
                {
                    "type": shape_type,
                    "center": (cx, cy),
                    "area": area,
                    "color": tuple(color.astype(int).tolist()),
                }
            )

        return shapes

    def _color_distance(self, c1: Tuple, c2: Tuple) -> float:
        """Calculate color distance."""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

    def _group_by_color(self, shapes: List[Dict], threshold: float = 50) -> Dict[str, List[Dict]]:
        """Group shapes by similar color."""
        if not shapes:
            return {}

        groups = {}
        for shape in shapes:
            color = shape["color"]
            matched = False
            for group_color, group_shapes in groups.items():
                if self._color_distance(color, eval(group_color)) < threshold:
                    group_shapes.append(shape)
                    matched = True
                    break
            if not matched:
                groups[str(color)] = [shape]

        return groups

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate stable sort task with rule-based logic."""
        if len(video_frames) < 2:
            return 0.0

        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        # Detect shapes in initial and final frames
        initial_shapes = self._detect_shapes(first_frame)
        final_shapes = self._detect_shapes(last_frame)
        gt_final_shapes = self._detect_shapes(gt_final_frame) if gt_final_frame is not None else []

        scores = {}

        # 1. Classification (30%): Check if shapes are grouped by type/color
        final_groups = self._group_by_color(final_shapes)
        gt_groups = self._group_by_color(gt_final_shapes) if gt_final_shapes else final_groups

        if len(final_shapes) >= 6 and len(final_groups) >= 2:
            # Check if shapes of same color are adjacent (grouped)
            final_sorted = sorted(final_shapes, key=lambda s: s["center"][0])

            # Count color transitions (fewer = better grouping)
            transitions = 0
            for i in range(1, len(final_sorted)):
                if self._color_distance(final_sorted[i]["color"], final_sorted[i - 1]["color"]) > 50:
                    transitions += 1

            # Ideal: 1 transition for 2 groups
            expected_transitions = len(final_groups) - 1
            if transitions <= expected_transitions:
                scores["classification"] = 1.0
            else:
                scores["classification"] = max(0, 1.0 - (transitions - expected_transitions) * 0.3)
        else:
            # Wrong number of shapes
            scores["classification"] = max(0, len(final_shapes) / 6.0) * 0.5

        # 2. Order (30%): Check if each group is sorted small to large (left to right)
        order_score = 0.0
        if final_groups:
            group_scores = []
            for group_shapes in final_groups.values():
                if len(group_shapes) >= 2:
                    # Sort by x-position
                    sorted_by_x = sorted(group_shapes, key=lambda s: s["center"][0])
                    # Check if sizes increase left to right
                    sizes = [s["area"] for s in sorted_by_x]

                    # Count correctly ordered pairs
                    correct_pairs = sum(1 for i in range(len(sizes) - 1) if sizes[i] < sizes[i + 1])
                    total_pairs = len(sizes) - 1
                    group_scores.append(correct_pairs / total_pairs if total_pairs > 0 else 1.0)

            order_score = np.mean(group_scores) if group_scores else 0.5
        scores["order"] = order_score

        # 3. Fidelity (30%): Check if shapes are preserved from initial frame
        fidelity_score = 0.0
        if initial_shapes and final_shapes:
            # Check object count preservation
            count_match = max(0, 1.0 - abs(len(initial_shapes) - len(final_shapes)) / max(len(initial_shapes), 1))

            # Check size preservation (total area should be similar)
            initial_total_area = sum(s["area"] for s in initial_shapes)
            final_total_area = sum(s["area"] for s in final_shapes)
            area_ratio = min(initial_total_area, final_total_area) / max(initial_total_area, final_total_area) if max(initial_total_area, final_total_area) > 0 else 0

            # Check shape type preservation
            initial_types = sorted([s["type"] for s in initial_shapes])
            final_types = sorted([s["type"] for s in final_shapes])
            type_match = sum(1 for a, b in zip(initial_types, final_types) if a == b) / max(len(initial_types), len(final_types), 1)

            fidelity_score = 0.4 * count_match + 0.3 * area_ratio + 0.3 * type_match
        scores["fidelity"] = fidelity_score

        # 4. Layout (10%): Check horizontal alignment
        layout_score = 0.0
        if final_shapes:
            y_coords = [s["center"][1] for s in final_shapes]
            y_variance = np.var(y_coords)
            # Good alignment: variance < 100 pixels
            layout_score = max(0, 1.0 - y_variance / 5000.0)
        scores["layout"] = layout_score

        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class MultiObjectPlacementEvaluator(BaseEvaluator):
    """
    G-5: Multi-object placement evaluator.

    Rule-based evaluation:
    - Color matching (30%): Objects moved to matching color star markers
    - Alignment precision (25%): Object centers aligned with star centers
    - Path optimality (20%): Straight-line movement paths
    - Object fidelity (15%): Shape, size, color preserved
    - Star invariance (10%): Star markers remain stationary
    """

    TASK_WEIGHTS = {"color_matching": 0.30, "alignment": 0.25, "path": 0.20, "fidelity": 0.15, "star_invariance": 0.10}

    def _detect_colored_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored objects (shapes) in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        objects = []

        # Define color ranges for common colors
        color_ranges = {
            "red": [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [180, 255, 255])],
            "blue": [([100, 100, 100], [130, 255, 255])],
            "green": [([35, 100, 100], [85, 255, 255])],
            "yellow": [([20, 100, 100], [35, 255, 255])],
        }

        for color_name, ranges in color_ranges.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 300:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                objects.append({"color": color_name, "center": (cx, cy), "area": area})

        return objects

    def _detect_star_markers(self, frame: np.ndarray) -> List[Dict]:
        """Detect star markers by looking for small star-shaped objects."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        stars = []
        color_ranges = {
            "red": [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [180, 255, 255])],
            "blue": [([100, 100, 100], [130, 255, 255])],
            "green": [([35, 100, 100], [85, 255, 255])],
            "yellow": [([20, 100, 100], [35, 255, 255])],
        }

        for color_name, ranges in color_ranges.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Stars are typically smaller markers
                if area < 100 or area > 2000:
                    continue

                # Check if shape has star-like properties (many vertices)
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                if len(approx) >= 8:  # Star has many vertices
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    stars.append({"color": color_name, "center": (cx, cy), "area": area})

        return stars

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate multi-object placement task."""
        if len(video_frames) < 2 or gt_final_frame is None:
            return 0.0

        scores = {}
        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        # Detect objects and stars in final frames
        gen_objects = self._detect_colored_objects(last_frame)
        gt_objects = self._detect_colored_objects(gt_final_frame)

        # 1. Color matching: Check if objects are at correct color-matched positions
        if gen_objects and gt_objects:
            matched = 0
            for gen_obj in gen_objects:
                for gt_obj in gt_objects:
                    if gen_obj["color"] == gt_obj["color"]:
                        dist = safe_distance(gen_obj["center"], gt_obj["center"])
                        if dist < 30:  # Within 30 pixels
                            matched += 1
                            break
            scores["color_matching"] = matched / max(len(gt_objects), 1)
        else:
            scores["color_matching"] = 0.2  # Detection failed

        # 2. Alignment precision: Compare object positions with GT
        if gen_objects and gt_objects:
            total_dist = 0
            count = 0
            for gen_obj in gen_objects:
                min_dist = float("inf")
                for gt_obj in gt_objects:
                    if gen_obj["color"] == gt_obj["color"]:
                        dist = safe_distance(gen_obj["center"], gt_obj["center"])
                        min_dist = min(min_dist, dist)
                if min_dist < float("inf"):
                    total_dist += min_dist
                    count += 1
            avg_dist = total_dist / count if count > 0 else 100
            scores["alignment"] = max(0, 1.0 - avg_dist / 50.0)
        else:
            scores["alignment"] = 0.2  # Detection failed

        # 3. Path optimality: Analyze motion smoothness
        if len(video_frames) > 2:
            motion_scores = []
            for i in range(1, min(len(video_frames), 10)):
                diff = cv2.absdiff(video_frames[i], video_frames[i - 1])
                motion = np.mean(diff)
                motion_scores.append(motion)
            # Smooth motion should have consistent changes
            if motion_scores:
                variance = np.var(motion_scores)
                scores["path"] = max(0, 1.0 - variance / 1000.0)
            else:
                scores["path"] = 0.2  # Detection failed
        else:
            scores["path"] = 0.2  # Detection failed

        # 4. Fidelity: Check object count and area preservation
        first_objects = self._detect_colored_objects(first_frame)
        if first_objects and gen_objects:
            count_ratio = min(len(gen_objects), len(first_objects)) / max(len(gen_objects), len(first_objects), 1)
            first_area = sum(o["area"] for o in first_objects)
            gen_area = sum(o["area"] for o in gen_objects)
            area_ratio = min(first_area, gen_area) / max(first_area, gen_area, 1)
            scores["fidelity"] = 0.5 * count_ratio + 0.5 * area_ratio
        else:
            scores["fidelity"] = 0.2  # Detection failed

        # 5. Star invariance: Check if star positions are preserved
        first_stars = self._detect_star_markers(first_frame)
        final_stars = self._detect_star_markers(last_frame)
        if first_stars and final_stars:
            preserved = 0
            for fs in first_stars:
                for ls in final_stars:
                    if fs["color"] == ls["color"]:
                        dist = safe_distance(fs["center"], ls["center"])
                        if dist < 20:
                            preserved += 1
                            break
            scores["star_invariance"] = preserved / max(len(first_stars), 1)
        else:
            scores["star_invariance"] = 0.3  # No stars detected, assume OK

        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class TrackObjectMovementEvaluator(BaseEvaluator):
    """
    G-8: Track object movement evaluator.

    Rule-based evaluation:
    - Tracking continuity (30%): Green border follows object B throughout movement
    - Horizontal movement (25%): Object B moves strictly horizontally
    - Alignment precision (20%): Object B aligns with object A's x-coordinate
    - Marker identification (15%): Correct identification of marked objects
    - Object fidelity (10%): Shape, size, color preserved
    """

    # CRITICAL: Alignment is the main success criterion - object must reach red star
    TASK_WEIGHTS = {"tracking": 0.10, "horizontal": 0.10, "alignment": 0.60, "identification": 0.10, "fidelity": 0.10}  # Main criterion - must align with red star

    def _detect_green_border(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect green border marker and return its center."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Find largest green contour (the border)
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def _detect_red_star(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red star marker and return its center."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Find star-shaped contour
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > 2000:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) >= 8:  # Star has many vertices
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        return None

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate object tracking task."""
        if len(video_frames) < 2 or gt_final_frame is None:
            return 0.0

        scores = {}
        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        # 1. Tracking continuity: Check green border presence throughout video
        tracking_scores = []
        for frame in video_frames:
            border_pos = self._detect_green_border(frame)
            if border_pos is not None:
                tracking_scores.append(1.0)
            else:
                tracking_scores.append(0.0)
        scores["tracking"] = np.mean(tracking_scores) if tracking_scores else 0.5

        # 2. Horizontal movement: Track green border y-coordinate stability
        y_positions = []
        for frame in video_frames:
            border_pos = self._detect_green_border(frame)
            if border_pos is not None:
                y_positions.append(border_pos[1])

        if len(y_positions) >= 2:
            y_variance = np.var(y_positions)
            # Good horizontal movement: y variance < 100 pixels
            scores["horizontal"] = max(0, 1.0 - y_variance / 500.0)
        else:
            scores["horizontal"] = 0.2  # Detection failed

        # 3. Alignment precision: Check if green border aligns with red star at the END
        final_border = self._detect_green_border(last_frame)
        red_star = self._detect_red_star(first_frame)
        gt_border = self._detect_green_border(gt_final_frame) if gt_final_frame is not None else None

        if final_border is not None and red_star is not None:
            # The green-bordered object should align with red star's x-coordinate at end
            x_diff = abs(final_border[0] - red_star[0])
            # Strict alignment: must be within 30 pixels
            if x_diff < 30:
                scores["alignment"] = 1.0
            elif x_diff < 60:
                scores["alignment"] = 0.5
            else:
                scores["alignment"] = 0.0  # Not aligned - STRICT failure
        elif gt_border is not None and final_border is not None:
            x_diff = abs(final_border[0] - gt_border[0])
            scores["alignment"] = max(0, 1.0 - x_diff / 50.0)
        else:
            scores["alignment"] = 0.0  # Detection failed - STRICT

        # 4. Marker identification: Check if red star and green border are detected
        has_red_star = red_star is not None
        has_green_border = self._detect_green_border(first_frame) is not None

        if not has_red_star or not has_green_border:
            scores["identification"] = 0.0  # Must have both markers
        else:
            scores["identification"] = 1.0

        # 5. Fidelity: Check if movement was in a straight line (y stays constant)
        if len(y_positions) >= 3:
            # Check y variance - should be very low for straight horizontal movement
            y_variance = np.var(y_positions)
            if y_variance < 50:  # Very straight line
                scores["fidelity"] = 1.0
            elif y_variance < 200:
                scores["fidelity"] = 0.5
            else:
                scores["fidelity"] = 0.0  # Not a straight line
        else:
            scores["fidelity"] = 0.0

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class IdentifyObjectsInRegionEvaluator(BaseEvaluator):
    """
    G-9: Identify objects in region evaluator.

    Rule-based evaluation:
    - Region identification (30%): Correct target region identified
    - Shape identification (30%): Correct target shape type identified
    - Marking completeness (25%): All target objects marked, no extras
    - Border quality (15%): Green border complete and proper
    """

    TASK_WEIGHTS = {"region": 0.30, "shape": 0.30, "completeness": 0.25, "border_quality": 0.15}

    def _detect_green_borders(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect green border markings and return their centers."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))

        return centers

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate identify objects in region task."""
        if len(video_frames) < 1 or gt_final_frame is None:
            return 0.0

        scores = {}
        last_frame = video_frames[-1]

        # Detect green borders in generated and GT frames
        gen_borders = self._detect_green_borders(last_frame)
        gt_borders = self._detect_green_borders(gt_final_frame)

        # 1. Region identification: Check if borders are in correct regions
        # Compare border positions with GT
        if gen_borders and gt_borders:
            matched = 0
            for gb in gen_borders:
                for gtb in gt_borders:
                    dist = np.sqrt((gb[0] - gtb[0]) ** 2 + (gb[1] - gtb[1]) ** 2)
                    if dist < 50:
                        matched += 1
                        break
            scores["region"] = matched / max(len(gt_borders), 1)
        else:
            scores["region"] = 0.5 if not gt_borders else 0.0

        # 2. Shape identification: Compare number of marked objects
        if gt_borders:
            count_diff = abs(len(gen_borders) - len(gt_borders))
            scores["shape"] = max(0, 1.0 - count_diff * 0.3)
        else:
            scores["shape"] = 0.2  # Detection failed

        # 3. Completeness: Check precision and recall
        if gen_borders and gt_borders:
            # Precision: how many generated borders match GT
            precision_matches = 0
            for gb in gen_borders:
                for gtb in gt_borders:
                    dist = np.sqrt((gb[0] - gtb[0]) ** 2 + (gb[1] - gtb[1]) ** 2)
                    if dist < 50:
                        precision_matches += 1
                        break
            precision = precision_matches / len(gen_borders) if gen_borders else 0

            # Recall: how many GT borders are matched
            recall_matches = 0
            for gtb in gt_borders:
                for gb in gen_borders:
                    dist = np.sqrt((gb[0] - gtb[0]) ** 2 + (gb[1] - gtb[1]) ** 2)
                    if dist < 50:
                        recall_matches += 1
                        break
            recall = recall_matches / len(gt_borders) if gt_borders else 0

            scores["completeness"] = 0.5 * precision + 0.5 * recall
        else:
            scores["completeness"] = 0.5 if not gt_borders else 0.0

        # 4. Border quality: Check green pixel coverage
        hsv_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        hsv_gt = cv2.cvtColor(gt_final_frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        green_mask_gen = cv2.inRange(hsv_gen, lower_green, upper_green)
        green_mask_gt = cv2.inRange(hsv_gt, lower_green, upper_green)

        green_overlap = np.sum((green_mask_gen > 0) & (green_mask_gt > 0))
        green_union = np.sum((green_mask_gen > 0) | (green_mask_gt > 0))

        green_iou = green_overlap / green_union if green_union > 0 else 0.5
        scores["border_quality"] = green_iou

        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class GridNumberSequenceEvaluator(BaseEvaluator):
    """
    G-13: Grid number sequence evaluator.

    Rule-based evaluation:
    - Sequence correctness (35%): Visit numbers 1→2→3 in order
    - Path optimality (35%): Shortest path between consecutive targets
    - Movement rules (20%): Only up/down/left/right, no diagonal
    - Completeness (10%): Agent reaches red endpoint after all numbers
    """

    TASK_WEIGHTS = {"sequence": 0.35, "path_optimal": 0.35, "movement": 0.20, "completeness": 0.10}

    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect orange circular agent."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def _detect_endpoint(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red endpoint."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate grid number sequence task."""
        if len(video_frames) < 1 or gt_final_frame is None:
            return 0.0

        scores = {}
        last_frame = video_frames[-1]

        # Track agent positions through video
        agent_positions = []
        for frame in video_frames:
            pos = self._detect_agent(frame)
            if pos is not None:
                agent_positions.append(pos)

        # 1. Sequence correctness: Compare agent trajectory with GT
        gt_agent_positions = []
        for frame in gt_frames:
            pos = self._detect_agent(frame)
            if pos is not None:
                gt_agent_positions.append(pos)

        if agent_positions and gt_agent_positions:
            # Compare final positions
            final_gen = agent_positions[-1]
            final_gt = gt_agent_positions[-1]
            dist = np.sqrt((final_gen[0] - final_gt[0]) ** 2 + (final_gen[1] - final_gt[1]) ** 2)
            scores["sequence"] = max(0, 1.0 - dist / 100.0)
        else:
            scores["sequence"] = 0.1  # Detection failed

        # 2. Path optimality: Compare path length
        if len(agent_positions) >= 2 and len(gt_agent_positions) >= 2:
            gen_path_len = sum(np.sqrt((agent_positions[i][0] - agent_positions[i - 1][0]) ** 2 + (agent_positions[i][1] - agent_positions[i - 1][1]) ** 2) for i in range(1, len(agent_positions)))
            gt_path_len = sum(np.sqrt((gt_agent_positions[i][0] - gt_agent_positions[i - 1][0]) ** 2 + (gt_agent_positions[i][1] - gt_agent_positions[i - 1][1]) ** 2) for i in range(1, len(gt_agent_positions)))

            if gt_path_len > 0:
                ratio = min(gen_path_len, gt_path_len) / max(gen_path_len, gt_path_len)
                scores["path_optimal"] = ratio
            else:
                scores["path_optimal"] = 0.1  # Detection failed
        else:
            scores["path_optimal"] = 0.1  # Detection failed

        # 3. Movement rules: Check for diagonal movements
        if len(agent_positions) >= 2:
            diagonal_count = 0
            for i in range(1, len(agent_positions)):
                dx = abs(agent_positions[i][0] - agent_positions[i - 1][0])
                dy = abs(agent_positions[i][1] - agent_positions[i - 1][1])
                # Diagonal if both dx and dy are significant
                if dx > 10 and dy > 10:
                    diagonal_count += 1
            scores["movement"] = max(0, 1.0 - diagonal_count * 0.2)
        else:
            scores["movement"] = 0.1  # Detection failed

        # 4. Completeness: Check if agent reaches endpoint
        endpoint = self._detect_endpoint(last_frame)
        final_agent = self._detect_agent(last_frame)

        if endpoint is not None and final_agent is not None:
            dist = np.sqrt((endpoint[0] - final_agent[0]) ** 2 + (endpoint[1] - final_agent[1]) ** 2)
            scores["completeness"] = 1.0 if dist < 50 else max(0, 1.0 - dist / 100.0)
        else:
            scores["completeness"] = 0.1  # Detection failed

        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class GridAvoidObstaclesEvaluator(BaseEvaluator):
    """
    G-15: Grid avoid obstacles evaluator.

    CRITICAL RULES:
    1. Yellow dot (agent) must move step by step from blue grid to red grid
    2. Agent must finally reach the red grid (endpoint)
    3. Grid colors (blue start, red end) must NOT change
    4. Avoid black X obstacles
    """

    TASK_WEIGHTS = {"completion": 0.45, "grid_preserved": 0.30, "avoidance": 0.15, "movement": 0.10}  # Agent reaches red endpoint  # Grid colors unchanged  # No collision with obstacles  # Step by step movement

    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect yellow circular agent."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def _detect_obstacles(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect black X obstacles."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Black obstacles
        black_mask = (gray < 50).astype(np.uint8) * 255

        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100 or area > 5000:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            obstacles.append((cx, cy))

        return obstacles

    def _detect_endpoint(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red endpoint."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def _get_grid_cell_colors(self, frame: np.ndarray, grid_size: int = 10) -> Dict[Tuple[int, int], str]:
        """Get the dominant color for each grid cell."""
        h, w = frame.shape[:2]
        cell_h = h // grid_size
        cell_w = w // grid_size

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        cell_colors = {}
        for row in range(grid_size):
            for col in range(grid_size):
                y1 = row * cell_h + 10
                y2 = (row + 1) * cell_h - 10
                x1 = col * cell_w + 10
                x2 = (col + 1) * cell_w - 10

                if y2 <= y1 or x2 <= x1:
                    cell_colors[(row, col)] = "white"
                    continue

                cell_hsv = hsv[y1:y2, x1:x2]
                sat = cell_hsv[:, :, 1].flatten()
                hue = cell_hsv[:, :, 0].flatten()

                # Check if cell is colored (saturated)
                sat_mask = sat > 80
                if np.sum(sat_mask) > 100:
                    dom_hue = np.median(hue[sat_mask])
                    # Classify color
                    if 100 <= dom_hue <= 130:
                        color = "blue"
                    elif dom_hue <= 10 or dom_hue >= 160:
                        color = "red"
                    elif 20 <= dom_hue <= 35:
                        color = "yellow"
                    elif 35 <= dom_hue <= 85:
                        color = "green"
                    else:
                        color = "other"
                    cell_colors[(row, col)] = color
                else:
                    cell_colors[(row, col)] = "white"

        return cell_colors

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate grid obstacle avoidance task.

        CRITICAL RULES:
        1. Agent must reach red endpoint
        2. Grid cell colors must NOT change (check each cell)
        3. Agent must avoid obstacles
        """
        if len(video_frames) < 1 or gt_final_frame is None:
            return 0.0

        scores = {}
        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        # CRITICAL: Check if grid cell colors are preserved
        first_colors = self._get_grid_cell_colors(first_frame)
        final_colors = self._get_grid_cell_colors(last_frame)

        # Count cells that changed color
        # Key rule: blue and red cells should not change to other colors
        # Yellow (agent) can move around
        changed_cells = 0
        total_grid_cells = 0

        for key in first_colors:
            first_color = first_colors[key]
            final_color = final_colors.get(key, "white")

            # Count all colored cells
            if first_color in ["blue", "red"]:
                total_grid_cells += 1
                # Blue/red cells should remain blue/red (or be covered by yellow agent)
                if first_color != final_color and final_color != "yellow":
                    changed_cells += 1

            # Check if a cell that was white/yellow now has a different grid color
            # (This could happen if grid structure changed)
            if first_color in ["white", "yellow"]:
                # If final is blue/red, check if it's a valid reveal (agent moved away from start)
                # For simplicity, count any new blue/red as suspicious
                if final_color in ["blue", "red"]:
                    # This is OK if it's the start cell being revealed
                    pass  # Allow this

        # Also check total number of blue+red cells
        first_br_count = sum(1 for c in first_colors.values() if c in ["blue", "red"])
        final_br_count = sum(1 for c in final_colors.values() if c in ["blue", "red"])

        # If significantly more blue/red cells appeared, grid changed
        if final_br_count > first_br_count + 2:  # Allow up to 2 new cells (start revealed + tolerance)
            changed_cells += final_br_count - first_br_count - 2

        # Grid preservation score - STRICTER
        if changed_cells > 0:
            scores["grid_preserved"] = 0.0
            scores["completion"] = 0.0
            scores["avoidance"] = 0.0
            scores["movement"] = 0.0
            self._last_task_details = scores
            self._last_task_details["cells_changed"] = changed_cells
            return 0.0
        else:
            scores["grid_preserved"] = 1.0

        # Detect obstacles in first frame
        obstacles = self._detect_obstacles(first_frame)

        # Track agent positions through video
        agent_positions = []
        for frame in video_frames:
            pos = self._detect_agent(frame)
            if pos is not None:
                agent_positions.append(pos)

        # 1. Completion: Check if agent reaches endpoint
        endpoint = self._detect_endpoint(last_frame)
        final_agent = self._detect_agent(last_frame)

        if endpoint is not None and final_agent is not None:
            dist = np.sqrt((endpoint[0] - final_agent[0]) ** 2 + (endpoint[1] - final_agent[1]) ** 2)
            # Stricter threshold - must be within 40 pixels
            if dist < 40:
                scores["completion"] = 1.0
            elif dist < 80:
                scores["completion"] = 0.3  # STRICT: Close but not at endpoint
            else:
                scores["completion"] = 0.0  # STRICT: Failed to reach endpoint
        else:
            scores["completion"] = 0.0

        # 2. Obstacle avoidance
        if agent_positions and obstacles:
            collision_count = 0
            for pos in agent_positions:
                for obs in obstacles:
                    dist = np.sqrt((pos[0] - obs[0]) ** 2 + (pos[1] - obs[1]) ** 2)
                    if dist < 30:
                        collision_count += 1
                        break
            scores["avoidance"] = max(0, 1.0 - collision_count / len(agent_positions))
        else:
            scores["avoidance"] = 0.0

        # 3. Movement: Check for step-by-step movement
        if len(agent_positions) >= 2:
            # Check if agent moves step by step (not teleporting)
            large_jumps = 0
            for i in range(1, len(agent_positions)):
                dx = abs(agent_positions[i][0] - agent_positions[i - 1][0])
                dy = abs(agent_positions[i][1] - agent_positions[i - 1][1])
                if dx > 100 or dy > 100:  # Too large a jump
                    large_jumps += 1
            scores["movement"] = max(0, 1.0 - large_jumps * 0.3)
        else:
            scores["movement"] = 0.0

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class GridGoThroughBlockEvaluator(BaseEvaluator):
    """
    G-16: Grid go through block evaluator.

    Rule-based evaluation:
    - Block visit completeness (40%): All blue blocks visited
    - Path optimality (30%): TSP-optimal path through all blocks
    - Completion (20%): Agent reaches red endpoint after visiting all
    - Movement rules (10%): Only up/down/left/right movement
    """

    TASK_WEIGHTS = {"block_visit": 0.40, "path_optimal": 0.30, "completion": 0.20, "movement": 0.10}

    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect orange circular agent."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def _detect_blue_blocks(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect blue blocks to visit."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            blocks.append((cx, cy))

        return blocks

    def _detect_endpoint(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red endpoint."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate grid go through block task."""
        if len(video_frames) < 1 or gt_final_frame is None:
            return 0.0

        scores = {}
        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        # Detect blue blocks in first frame
        blue_blocks = self._detect_blue_blocks(first_frame)

        # Track agent positions through video
        agent_positions = []
        for frame in video_frames:
            pos = self._detect_agent(frame)
            if pos is not None:
                agent_positions.append(pos)

        # 1. Block visit completeness: Check if agent visits all blue blocks
        if agent_positions and blue_blocks:
            visited_blocks = set()
            for pos in agent_positions:
                for i, block in enumerate(blue_blocks):
                    dist = np.sqrt((pos[0] - block[0]) ** 2 + (pos[1] - block[1]) ** 2)
                    if dist < 40:
                        visited_blocks.add(i)
            scores["block_visit"] = len(visited_blocks) / max(len(blue_blocks), 1)
        else:
            scores["block_visit"] = 0.2  # Detection failed

        # 2. Path optimality: Compare path length with GT
        gt_agent_positions = []
        for frame in gt_frames:
            pos = self._detect_agent(frame)
            if pos is not None:
                gt_agent_positions.append(pos)

        if len(agent_positions) >= 2 and len(gt_agent_positions) >= 2:
            gen_path_len = sum(np.sqrt((agent_positions[i][0] - agent_positions[i - 1][0]) ** 2 + (agent_positions[i][1] - agent_positions[i - 1][1]) ** 2) for i in range(1, len(agent_positions)))
            gt_path_len = sum(np.sqrt((gt_agent_positions[i][0] - gt_agent_positions[i - 1][0]) ** 2 + (gt_agent_positions[i][1] - gt_agent_positions[i - 1][1]) ** 2) for i in range(1, len(gt_agent_positions)))

            if gt_path_len > 0:
                ratio = min(gen_path_len, gt_path_len) / max(gen_path_len, gt_path_len)
                scores["path_optimal"] = ratio
            else:
                scores["path_optimal"] = 0.2  # Detection failed
        else:
            scores["path_optimal"] = 0.2  # Detection failed

        # 3. Completion: Check if agent reaches endpoint
        endpoint = self._detect_endpoint(last_frame)
        final_agent = self._detect_agent(last_frame)

        if endpoint is not None and final_agent is not None:
            dist = np.sqrt((endpoint[0] - final_agent[0]) ** 2 + (endpoint[1] - final_agent[1]) ** 2)
            scores["completion"] = 1.0 if dist < 50 else max(0, 1.0 - dist / 100.0)
        else:
            scores["completion"] = 0.2  # Detection failed

        # 4. Movement rules: Check for diagonal movements
        if len(agent_positions) >= 2:
            diagonal_count = 0
            for i in range(1, len(agent_positions)):
                dx = abs(agent_positions[i][0] - agent_positions[i - 1][0])
                dy = abs(agent_positions[i][1] - agent_positions[i - 1][1])
                if dx > 10 and dy > 10:
                    diagonal_count += 1
            scores["movement"] = max(0, 1.0 - diagonal_count * 0.2)
        else:
            scores["movement"] = 0.2  # Detection failed

        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class GridShortestPathEvaluator(BaseEvaluator):
    """
    G-18: Grid shortest path evaluator.

    Rule-based evaluation:
    - Path optimality (50%): Shortest path (Manhattan distance)
    - Completion (25%): Agent reaches endpoint
    - Movement rules (15%): Only up/down/left/right movement
    - Visual fidelity (10%): Agent appearance preserved
    """

    TASK_WEIGHTS = {"path_optimal": 0.50, "completion": 0.25, "movement": 0.15, "fidelity": 0.10}

    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect colored circular agent."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Try multiple colors for agent (pink agent is common)
        for lower, upper in [
            (np.array([150, 50, 50]), np.array([180, 255, 255])),  # Pink (most common agent color)
            (np.array([120, 50, 50]), np.array([160, 255, 255])),  # Purple
            (np.array([10, 100, 100]), np.array([25, 255, 255])),  # Orange
            (np.array([0, 100, 100]), np.array([10, 255, 255])),  # Red
        ]:
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 100 or area > 5000:
                    continue
                # Check circularity
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.6:  # Reasonably circular
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)

        return None

    def _detect_endpoint(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect colored endpoint square."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Try multiple colors for endpoint
        for lower, upper in [
            (np.array([150, 50, 50]), np.array([180, 255, 255])),  # Pink
            (np.array([0, 100, 100]), np.array([10, 255, 255])),  # Red
        ]:
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)

        return None

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate grid shortest path task."""
        if len(video_frames) < 1 or gt_final_frame is None:
            return 0.0

        scores = {}
        last_frame = video_frames[-1]

        # Track agent positions through video
        agent_positions = []
        for frame in video_frames:
            pos = self._detect_agent(frame)
            if pos is not None:
                agent_positions.append(pos)

        # Track GT agent positions
        gt_agent_positions = []
        for frame in gt_frames:
            pos = self._detect_agent(frame)
            if pos is not None:
                gt_agent_positions.append(pos)

        # 1. Path optimality: Compare path length with GT
        if len(agent_positions) >= 2 and len(gt_agent_positions) >= 2:
            gen_path_len = sum(np.sqrt((agent_positions[i][0] - agent_positions[i - 1][0]) ** 2 + (agent_positions[i][1] - agent_positions[i - 1][1]) ** 2) for i in range(1, len(agent_positions)))
            gt_path_len = sum(np.sqrt((gt_agent_positions[i][0] - gt_agent_positions[i - 1][0]) ** 2 + (gt_agent_positions[i][1] - gt_agent_positions[i - 1][1]) ** 2) for i in range(1, len(gt_agent_positions)))

            if gt_path_len > 0:
                # Allow some tolerance for path length
                ratio = min(gen_path_len, gt_path_len) / max(gen_path_len, gt_path_len)
                scores["path_optimal"] = ratio
            else:
                scores["path_optimal"] = 0.2  # Detection failed
        else:
            scores["path_optimal"] = 0.2  # Detection failed

        # 2. Completion: Check if agent reaches endpoint
        endpoint = self._detect_endpoint(last_frame)
        final_agent = self._detect_agent(last_frame)
        gt_endpoint = self._detect_endpoint(gt_final_frame)
        gt_final_agent = self._detect_agent(gt_final_frame)

        if final_agent is not None and gt_final_agent is not None:
            dist = np.sqrt((final_agent[0] - gt_final_agent[0]) ** 2 + (final_agent[1] - gt_final_agent[1]) ** 2)
            scores["completion"] = max(0, 1.0 - dist / 50.0)
        elif endpoint is not None and final_agent is not None:
            dist = np.sqrt((endpoint[0] - final_agent[0]) ** 2 + (endpoint[1] - final_agent[1]) ** 2)
            scores["completion"] = 1.0 if dist < 50 else max(0, 1.0 - dist / 100.0)
        elif len(agent_positions) >= 2 and len(gt_agent_positions) >= 2:
            # If agent detection fails in final frame, use last known position
            # This handles cases where agent merges with endpoint
            gen_last_pos = agent_positions[-1]
            gt_last_pos = gt_agent_positions[-1]
            dist = np.sqrt((gen_last_pos[0] - gt_last_pos[0]) ** 2 + (gen_last_pos[1] - gt_last_pos[1]) ** 2)
            scores["completion"] = max(0, 1.0 - dist / 100.0)
        else:
            scores["completion"] = 0.2  # Detection failed

        # 3. Movement rules: Check for diagonal movements
        if len(agent_positions) >= 2:
            diagonal_count = 0
            for i in range(1, len(agent_positions)):
                dx = abs(agent_positions[i][0] - agent_positions[i - 1][0])
                dy = abs(agent_positions[i][1] - agent_positions[i - 1][1])
                if dx > 10 and dy > 10:
                    diagonal_count += 1
            scores["movement"] = max(0, 1.0 - diagonal_count * 0.2)
        else:
            scores["movement"] = 0.2  # Detection failed

        # 4. Fidelity: Compare agent appearance (use any detected agent through video)
        if final_agent is not None or len(agent_positions) > 0:
            scores["fidelity"] = 1.0
        else:
            scores["fidelity"] = 0.2  # Detection failed

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class MultipleOcclusionsVerticalEvaluator(BaseEvaluator):
    """
    G-21: Multiple occlusions vertical evaluator.

    Rule-based evaluation:
    - Occlusion correctness (35%): Mask properly occludes objects
    - Object permanence (30%): All objects reappear after mask leaves
    - Motion correctness (20%): Mask moves vertically downward
    - Visual consistency (15%): Objects maintain position and attributes
    """

    TASK_WEIGHTS = {"occlusion": 0.35, "permanence": 0.30, "motion": 0.20, "consistency": 0.15}

    def _detect_mask(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect dark rectangular mask."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Dark mask
        dark_mask = (gray < 100).astype(np.uint8) * 255

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Check if rectangular
            if w * h * 0.7 < area:  # Reasonably rectangular
                return (x, y, w, h)

        return None

    def _detect_colored_objects(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect colored objects."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Find non-white/non-black colored areas
        mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))

        # Remove dark areas (mask)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = mask & (gray > 100)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            objects.append((cx, cy))

        return objects

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate multiple occlusions task."""
        if len(video_frames) < 2 or gt_final_frame is None:
            return 0.0

        scores = {}
        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        # Detect objects
        initial_objects = self._detect_colored_objects(first_frame)
        final_objects = self._detect_colored_objects(last_frame)
        gt_final_objects = self._detect_colored_objects(gt_final_frame)

        # 1. Occlusion correctness: Check if there's visual change during video (occlusion happening)
        # Compare first frame with middle frames to detect occlusion
        occlusion_detected = False
        mid_idx = len(video_frames) // 2
        for i in range(max(1, mid_idx - 3), min(len(video_frames) - 1, mid_idx + 3)):
            mid_frame = video_frames[i]
            diff = cv2.absdiff(first_frame, mid_frame)
            diff_sum = np.sum(diff) / (first_frame.shape[0] * first_frame.shape[1] * 3)
            if diff_sum > 5:  # Significant visual change
                occlusion_detected = True
                break

        scores["occlusion"] = 1.0 if occlusion_detected else 0.0

        # 2. Object permanence: Objects should reappear at original positions in final frame
        if initial_objects and final_objects:
            reappeared = 0
            for io in initial_objects:
                for fo in final_objects:
                    dist = np.sqrt((io[0] - fo[0]) ** 2 + (io[1] - fo[1]) ** 2)
                    if dist < 50:  # Same position (with tolerance)
                        reappeared += 1
                        break

            scores["permanence"] = reappeared / len(initial_objects)
        else:
            scores["permanence"] = 0.0 if initial_objects else 0.5

        # 3. Motion correctness: Compare final frame with GT final
        if last_frame.shape == gt_final_frame.shape:
            diff = np.abs(last_frame.astype(float) - gt_final_frame.astype(float)).mean()
            if diff < 20:
                scores["motion"] = 1.0
            elif diff < 40:
                scores["motion"] = 0.5
            else:
                scores["motion"] = 0.0
        else:
            scores["motion"] = 0.0

        # 4. Visual consistency: Final frame should match GT final frame (objects at same positions)
        if final_objects and gt_final_objects:
            # Check if number of objects matches
            count_match = len(final_objects) == len(gt_final_objects)

            # Check if positions match GT
            position_matches = 0
            for gto in gt_final_objects:
                for fo in final_objects:
                    dist = np.sqrt((gto[0] - fo[0]) ** 2 + (gto[1] - fo[1]) ** 2)
                    if dist < 50:
                        position_matches += 1
                        break

            position_score = position_matches / max(len(gt_final_objects), 1)
            scores["consistency"] = 0.5 * (1.0 if count_match else 0.0) + 0.5 * position_score
        else:
            scores["consistency"] = 0.0 if gt_final_objects else 0.5

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class SeparateObjectsSpinningEvaluator(BaseEvaluator):
    """
    G-25: Separate objects with spinning evaluator.

    Evaluates:
    - Rotation correctness (40%): Objects rotated to match target orientation
    - Alignment precision (30%): Objects align with dashed outlines
    - Operation order (20%): Rotate first, then translate
    - Visual fidelity (10%): Shape, color, size preserved
    """

    TASK_WEIGHTS = {"rotation": 0.40, "alignment": 0.30, "order": 0.20, "fidelity": 0.10}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate separate objects spinning task."""
        scores = {}

        if len(video_frames) < 2 or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        gt_last = gt_final_frame

        # Detect objects in first and last frames
        first_objects = self._detect_colored_objects(first_frame)
        last_objects = self._detect_colored_objects(last_frame)
        gt_objects = self._detect_colored_objects(gt_last)

        # 1. Rotation correctness (40%): Check if objects reached correct orientation
        rotation_score = self._evaluate_rotation(last_objects, gt_objects)
        scores["rotation"] = rotation_score

        # 2. Alignment precision (30%): Check if objects align with target outlines
        alignment_score = self._evaluate_alignment(last_frame, gt_last)
        scores["alignment"] = alignment_score

        # 3. Operation order (20%): Check if rotation happens before translation
        order_score = self._evaluate_operation_order(video_frames, first_objects)
        scores["order"] = order_score

        # 4. Visual fidelity (10%): Check if shape, color, size preserved
        fidelity_score = self._evaluate_fidelity(first_objects, last_objects)
        scores["fidelity"] = fidelity_score

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

    def _detect_colored_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect solid colored objects (not dashed outlines)."""
        objects = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color ranges for common object colors
        color_ranges = {
            "red": ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
            "green": ([35, 100, 100], [85, 255, 255], None, None),
            "blue": ([100, 100, 100], [130, 255, 255], None, None),
            "yellow": ([20, 100, 100], [35, 255, 255], None, None),
            "orange": ([10, 100, 100], [20, 255, 255], None, None),
            "purple": ([130, 100, 100], [160, 255, 255], None, None),
        }

        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500:  # Filter small noise
                    continue

                # Get moments and orientation
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Calculate orientation using moments
                if M["mu20"] - M["mu02"] != 0:
                    angle = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
                else:
                    angle = 0

                # Get bounding box
                rect = cv2.minAreaRect(contour)

                objects.append({"color": color_name, "center": (cx, cy), "area": area, "angle": np.degrees(angle), "contour": contour, "rect": rect})

        return objects

    def _evaluate_rotation(self, last_objects: List[Dict], gt_objects: List[Dict]) -> float:
        """Evaluate if objects are rotated to correct orientation."""
        if not last_objects or not gt_objects:
            return 0.0

        total_score = 0.0
        matched = 0

        for gt_obj in gt_objects:
            # Find matching object by color and approximate position
            best_match = None
            best_dist = float("inf")

            for obj in last_objects:
                if obj["color"] == gt_obj["color"]:
                    dist = safe_distance(obj["center"], gt_obj["center"])
                    if dist < best_dist:
                        best_dist = dist
                        best_match = obj

            if best_match:
                matched += 1
                # Compare angles (accounting for symmetry)
                angle_diff = abs(best_match["angle"] - gt_obj["angle"])
                angle_diff = min(angle_diff, 180 - angle_diff)  # Handle symmetry

                # Score based on angle error: <5° = 1.0, 5-10° = 0.8, 10-20° = 0.6, >20° = 0.4
                if angle_diff < 5:
                    total_score += 1.0
                elif angle_diff < 10:
                    total_score += 0.8
                elif angle_diff < 20:
                    total_score += 0.6
                else:
                    total_score += max(0.2, 1.0 - angle_diff / 90)

        return total_score / max(1, len(gt_objects))

    def _evaluate_alignment(self, last_frame: np.ndarray, gt_last: np.ndarray) -> float:
        """Evaluate if objects align with target outlines."""
        # Normalize frame size (handles padding removal + resize)
        if last_frame.shape != gt_last.shape:
            gt_last = normalize_frame_size(gt_last, last_frame)

        # Compare using structural similarity in regions with objects
        gray_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        gray_gt = cv2.cvtColor(gt_last, cv2.COLOR_BGR2GRAY)

        # Find object regions in GT
        _, thresh_gt = cv2.threshold(gray_gt, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0  # STRICT: No contours to compare

        alignment_scores = []
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue

            # Create mask for this region
            mask = np.zeros(gray_gt.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            # Expand mask slightly for alignment tolerance
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # Compare regions
            roi_gen = gray_gen[mask > 0]
            roi_gt = gray_gt[mask > 0]

            if len(roi_gen) > 0 and len(roi_gt) > 0:
                diff = np.mean(np.abs(roi_gen.astype(float) - roi_gt.astype(float)))
                alignment_scores.append(max(0, 1.0 - diff / 128))

        return np.mean(alignment_scores) if alignment_scores else 0.0

    def _evaluate_operation_order(self, frames: List[np.ndarray], first_objects: List[Dict]) -> float:
        """Evaluate if rotation happens before translation."""
        if len(frames) < 5 or not first_objects:
            return 0.0  # STRICT: Not enough data

        # Track object positions and angles through video
        n_samples = min(20, len(frames))
        sample_indices = np.linspace(0, len(frames) - 1, n_samples, dtype=int)

        position_changes = []
        angle_changes = []

        prev_objects = first_objects
        for idx in sample_indices[1:]:
            frame = frames[idx]
            curr_objects = self._detect_colored_objects(frame)

            if not curr_objects:
                continue

            # Match objects and track changes
            for prev_obj in prev_objects:
                best_match = None
                best_dist = float("inf")

                for obj in curr_objects:
                    if obj["color"] == prev_obj["color"]:
                        dist = safe_distance(obj["center"], prev_obj["center"])
                        if dist < best_dist:
                            best_dist = dist
                            best_match = obj

                if best_match:
                    position_changes.append(best_dist)
                    angle_diff = abs(best_match["angle"] - prev_obj["angle"])
                    angle_changes.append(min(angle_diff, 180 - angle_diff))

            prev_objects = curr_objects

        if len(position_changes) < 3:
            return 0.0  # STRICT: Not enough position data

        # Check if rotation (angle changes) happens before translation (position changes)
        # Split into first half and second half
        mid = len(position_changes) // 2

        first_half_angle = np.mean(angle_changes[:mid]) if angle_changes[:mid] else 0
        second_half_angle = np.mean(angle_changes[mid:]) if angle_changes[mid:] else 0
        first_half_pos = np.mean(position_changes[:mid]) if position_changes[:mid] else 0
        second_half_pos = np.mean(position_changes[mid:]) if position_changes[mid:] else 0

        # Ideal: more rotation in first half, more translation in second half
        rotation_first = first_half_angle > second_half_angle
        translation_second = second_half_pos > first_half_pos

        if rotation_first and translation_second:
            return 1.0
        elif rotation_first or translation_second:
            return 0.7
        else:
            return 0.4

    def _evaluate_fidelity(self, first_objects: List[Dict], last_objects: List[Dict]) -> float:
        """Evaluate if shape, color, size are preserved."""
        if not first_objects or not last_objects:
            return 0.5

        preserved_count = 0
        total = len(first_objects)

        for first_obj in first_objects:
            # Find matching object by color
            for last_obj in last_objects:
                if first_obj["color"] == last_obj["color"]:
                    # Check area preservation (within 20%)
                    area_ratio = last_obj["area"] / max(1, first_obj["area"])
                    if 0.8 <= area_ratio <= 1.2:
                        preserved_count += 1
                    break

        return preserved_count / max(1, total)


# Mapping of task names to evaluators
IN_DOMAIN_50_EVALUATORS = {
    "G-3_stable_sort_data-generator": StableSortEvaluator,
    "G-5_multi_object_placement_data-generator": MultiObjectPlacementEvaluator,
    "G-8_track_object_movement_data-generator": TrackObjectMovementEvaluator,
    "G-9_identify_objects_in_region_data-generator": IdentifyObjectsInRegionEvaluator,
    "G-13_grid_number_sequence_data-generator": GridNumberSequenceEvaluator,
    "G-15_grid_avoid_obstacles_data-generator": GridAvoidObstaclesEvaluator,
    "G-16_grid_go_through_block_data-generator": GridGoThroughBlockEvaluator,
    "G-18_grid_shortest_path_data-generator": GridShortestPathEvaluator,
    "G-21_multiple_occlusions_vertical_data-generator": MultipleOcclusionsVerticalEvaluator,
    "G-25_seperate_object_spinning_data-generator": SeparateObjectsSpinningEvaluator,
}
