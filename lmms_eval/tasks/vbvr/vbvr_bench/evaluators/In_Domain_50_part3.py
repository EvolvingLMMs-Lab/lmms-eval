"""
Specific evaluators for In-Domain_50 tasks (Part 3).
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils import normalize_frame_size
from .base_evaluator import BaseEvaluator


class IdentifyAllHollowPointsEvaluator(BaseEvaluator):
    """
    G-158: Identify all hollow points evaluator.

    Rule-based evaluation:
    - Hollow point identification accuracy (40%): Distinguish hollow from solid
    - Marking completeness (30%): All hollow points marked
    - Marking position accuracy (20%): Circles centered on hollow points
    - Visual annotation quality (10%): Red circles proper
    """

    TASK_WEIGHTS = {"identification": 0.40, "completeness": 0.30, "position": 0.20, "annotation": 0.10}

    def _detect_hollow_points(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect hollow (unfilled) circular points."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find circles using Hough transform
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=50)

        hollow_points = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy, r = i[0], i[1], i[2]

                # Check if hollow (center is similar to background)
                # Sample center and edge
                if cy >= r and cy + r < frame.shape[0] and cx >= r and cx + r < frame.shape[1]:
                    center_val = gray[cy, cx]
                    edge_vals = []
                    for angle in range(0, 360, 45):
                        ex = int(cx + r * np.cos(np.radians(angle)))
                        ey = int(cy + r * np.sin(np.radians(angle)))
                        if 0 <= ex < frame.shape[1] and 0 <= ey < frame.shape[0]:
                            edge_vals.append(gray[ey, ex])

                    if edge_vals:
                        edge_avg = np.mean(edge_vals)
                        # Hollow if center is brighter than edge (outline only)
                        if center_val > edge_avg + 30:
                            hollow_points.append((cx, cy))

        return hollow_points

    def _detect_red_markings(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect red circle markings."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))

        return centers

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate identify all hollow points task."""
        scores = {}

        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame

        if last_frame is None or gt_last is None:
            return 0.0

        # Detect red markings
        gen_markings = self._detect_red_markings(last_frame)
        gt_markings = self._detect_red_markings(gt_last)

        # 1. Identification: Compare number of markings
        if gt_markings:
            count_diff = abs(len(gen_markings) - len(gt_markings))
            scores["identification"] = max(0, 1.0 - count_diff * 0.2)
        else:
            scores["identification"] = 0.2  # Detection failed

        # 2. Completeness: Recall - how many GT markings are matched
        if gen_markings and gt_markings:
            matched = 0
            for gtm in gt_markings:
                for gm in gen_markings:
                    dist = np.sqrt((gm[0] - gtm[0]) ** 2 + (gm[1] - gtm[1]) ** 2)
                    if dist < 40:
                        matched += 1
                        break
            scores["completeness"] = matched / len(gt_markings)
        else:
            scores["completeness"] = 0.5 if not gt_markings else 0.0

        # 3. Position accuracy: Average distance between matched markings
        if gen_markings and gt_markings:
            total_dist = 0
            matched_count = 0
            for gm in gen_markings:
                min_dist = float("inf")
                for gtm in gt_markings:
                    dist = np.sqrt((gm[0] - gtm[0]) ** 2 + (gm[1] - gtm[1]) ** 2)
                    min_dist = min(min_dist, dist)
                if min_dist < float("inf"):
                    total_dist += min_dist
                    matched_count += 1

            if matched_count > 0:
                avg_dist = total_dist / matched_count
                scores["position"] = max(0, 1.0 - avg_dist / 50.0)
            else:
                scores["position"] = 0.2  # Detection failed
        else:
            scores["position"] = 0.2  # Detection failed

        # 4. Annotation quality: Red pixel IoU
        hsv_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        hsv_gt = cv2.cvtColor(gt_last, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        red_mask_gen = cv2.inRange(hsv_gen, lower_red1, upper_red1) | cv2.inRange(hsv_gen, lower_red2, upper_red2)
        red_mask_gt = cv2.inRange(hsv_gt, lower_red1, upper_red1) | cv2.inRange(hsv_gt, lower_red2, upper_red2)

        red_overlap = np.sum((red_mask_gen > 0) & (red_mask_gt > 0))
        red_union = np.sum((red_mask_gen > 0) | (red_mask_gt > 0))

        scores["annotation"] = red_overlap / red_union if red_union > 0 else 0.5

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class ConstructConcentricRingEvaluator(BaseEvaluator):
    """
    G-194: Construct concentric ring evaluator.

    Rule-based evaluation:
    - Center alignment accuracy (35%): Both circles centered at image center
    - Circle attribute fidelity (35%): Radius, color, shape preserved
    - Concentric structure correctness (20%): Proper concentric geometry
    - Animation smoothness (10%): Smooth movement
    """

    TASK_WEIGHTS = {"center": 0.35, "fidelity": 0.35, "structure": 0.20, "smoothness": 0.10}

    def _detect_circles(self, frame: np.ndarray) -> List[Dict]:
        """Detect circles in the frame - optimized for concentric ring detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # For concentric rings, we expect 2 circles with the same center
        # Use edge detection to find circle outlines
        edges = cv2.Canny(gray, 50, 150)

        # Use contour detection for more reliable circle finding
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        detected = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # Skip small contours
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 100:  # Skip small perimeters
                continue

            # Check circularity
            circularity = 4 * np.pi * area / (perimeter**2)
            if circularity > 0.6:  # Reasonably circular
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                # Verify the enclosing circle fits well
                if radius > 50:  # Minimum radius for concentric rings
                    detected.append({"center": (int(x), int(y)), "radius": int(radius)})

        # If contour method fails, try Hough circles with stricter params
        if len(detected) < 2:
            # Use stricter parameters to avoid false positives
            for param2 in [50, 40, 30]:
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=param2, minRadius=50, maxRadius=500)
                if circles is not None and len(circles[0]) >= 2:
                    circles = np.uint16(np.around(circles))
                    detected = []
                    for i in circles[0, :]:
                        detected.append({"center": (int(i[0]), int(i[1])), "radius": int(i[2])})
                    break

        # Remove duplicates (circles with very similar centers and radii)
        if len(detected) > 2:
            unique = []
            for d in detected:
                is_dup = False
                for u in unique:
                    center_dist = np.sqrt((d["center"][0] - u["center"][0]) ** 2 + (d["center"][1] - u["center"][1]) ** 2)
                    radius_diff = abs(d["radius"] - u["radius"])
                    if center_dist < 30 and radius_diff < 30:
                        is_dup = True
                        break
                if not is_dup:
                    unique.append(d)
            detected = unique

        return detected

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate construct concentric ring task."""
        scores = {}

        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame

        if last_frame is None or gt_last is None:
            return 0.0

        # Normalize frame size (handles padding removal + resize)
        if last_frame.shape != gt_last.shape:
            gt_last = normalize_frame_size(gt_last, last_frame)

        # Detect circles
        gen_circles = self._detect_circles(last_frame)
        gt_circles = self._detect_circles(gt_last)

        frame_center = (last_frame.shape[1] // 2, last_frame.shape[0] // 2)

        # 1. Center alignment: Check if circles are centered at image center (512, 512)
        if gen_circles:
            center_dists = []
            for c in gen_circles:
                dist = np.sqrt((c["center"][0] - frame_center[0]) ** 2 + (c["center"][1] - frame_center[1]) ** 2)
                center_dists.append(dist)
            avg_dist = np.mean(center_dists)
            # Rule: circles should be within 5 pixels of center for perfect score
            if avg_dist < 5:
                scores["center"] = 1.0
            elif avg_dist < 15:
                scores["center"] = 0.9
            else:
                scores["center"] = max(0, 1.0 - avg_dist / 100.0)
        else:
            # No circles detected - check if any circular content exists
            gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            scores["center"] = 0.3 if np.sum(edges > 0) > 1000 else 0.0

        # 2. Fidelity: Compare circle count and radii with GT
        if gen_circles and gt_circles:
            # Count match - should have 2 circles for concentric rings
            count_match = max(0, 1.0 - abs(len(gen_circles) - len(gt_circles)) / max(len(gt_circles), 1))

            # Radius comparison - radii should be preserved (rule: error <= 3 pixels)
            gen_radii = sorted([c["radius"] for c in gen_circles])
            gt_radii = sorted([c["radius"] for c in gt_circles])

            if len(gen_radii) == len(gt_radii):
                radius_diffs = [abs(g - gt) for g, gt in zip(gen_radii, gt_radii)]
                avg_radius_diff = np.mean(radius_diffs)
                if avg_radius_diff <= 3:
                    radius_match = 1.0
                elif avg_radius_diff <= 8:
                    radius_match = 0.8
                else:
                    radius_match = max(0, 1.0 - avg_radius_diff / 50.0)
            else:
                # Partial match based on common radii
                min_len = min(len(gen_radii), len(gt_radii))
                if min_len > 0:
                    radius_diffs = [abs(g - gt) for g, gt in zip(gen_radii[:min_len], gt_radii[:min_len])]
                    avg_radius_diff = np.mean(radius_diffs)
                    radius_match = max(0, 1.0 - avg_radius_diff / 50.0) * 0.7
                else:
                    radius_match = 0.5

            scores["fidelity"] = 0.5 * count_match + 0.5 * radius_match
        elif gen_circles:
            # Generated has circles but GT doesn't (shouldn't happen for concentric task)
            scores["fidelity"] = 0.3
        else:
            # No circles detected
            scores["fidelity"] = 0.0

        # 3. Concentric structure: Check if circles share same center (rule: centers must coincide)
        if len(gen_circles) >= 2:
            centers = [c["center"] for c in gen_circles]
            # Calculate max distance between any two circle centers
            max_center_dist = 0
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = np.sqrt((centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2)
                    max_center_dist = max(max_center_dist, dist)

            # Rule: for perfect concentric, centers should coincide (error <= 5 pixels)
            if max_center_dist <= 5:
                scores["structure"] = 1.0
            elif max_center_dist <= 10:
                scores["structure"] = 0.9
            else:
                scores["structure"] = max(0, 1.0 - max_center_dist / 50.0)
        elif len(gen_circles) == 1:
            scores["structure"] = 0.2  # Single circle cannot form concentric structure
        else:
            scores["structure"] = 0.0

        # 4. Smoothness: Analyze motion through video (both circles should move simultaneously)
        if len(video_frames) >= 3:
            motion_scores = []
            for i in range(1, min(len(video_frames), 10)):
                diff = cv2.absdiff(video_frames[i], video_frames[i - 1])
                motion = np.mean(diff)
                motion_scores.append(motion)

            if motion_scores:
                variance = np.var(motion_scores)
                # Low variance = smooth, consistent motion
                scores["smoothness"] = max(0.5, 1.0 - variance / 1000.0)
            else:
                scores["smoothness"] = 0.8
        else:
            scores["smoothness"] = 0.8

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class ShapeOutlineFillEvaluator(BaseEvaluator):
    """
    O-10: Shape Outline Fill (Visual Analogy)

    CRITICAL RULES:
    1. First row (top left, top right) objects must remain UNCHANGED
    2. Second row: bottom right should have same shape as bottom left
    3. Bottom right shape should be either filled or have thicker lines (follow pattern)
    """

    TASK_WEIGHTS = {"first_row_preserved": 0.45, "bottom_right_correct": 0.35, "shape_preserved": 0.20}  # Top row unchanged  # D has correct shape and style  # D shape matches C

    def _detect_shapes_in_quadrant(self, frame: np.ndarray, quadrant: str) -> Dict:
        """Detect shape in specified quadrant and determine if filled or outline."""
        h, w = frame.shape[:2]

        if quadrant == "top_left":
            region = frame[: h // 2, : w // 2]
        elif quadrant == "top_right":
            region = frame[: h // 2, w // 2 :]
        elif quadrant == "bottom_left":
            region = frame[h // 2 :, : w // 2]
        else:  # bottom_right
            region = frame[h // 2 :, w // 2 :]

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region

        # Detect shape
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"exists": False}

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)

        # Determine shape type
        approx = cv2.approxPolyDP(largest, 0.04 * perimeter, True)
        vertices = len(approx)

        if vertices == 3:
            shape_type = "triangle"
        elif vertices == 4:
            shape_type = "rectangle"
        elif vertices >= 8:
            shape_type = "circle"
        else:
            shape_type = "polygon"

        # Determine if filled or outline
        # Filled shapes have higher area-to-perimeter ratio
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter * perimeter)
        else:
            compactness = 0

        # Check interior fill by sampling center
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Check if center is dark (filled)
            if 0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1]:
                center_val = gray[cy, cx]
                is_filled = center_val < 128
            else:
                is_filled = area > 1000
        else:
            is_filled = area > 1000

        return {"exists": True, "shape_type": shape_type, "is_filled": is_filled, "area": area, "vertices": vertices}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate visual analogy transformation.

        CRITICAL RULES:
        1. First row (A, B) must remain unchanged
        2. Bottom right (D) should have same shape as bottom left (C)
        3. D should be filled or have thicker lines (follow pattern)
        """

        if not video_frames or gt_final_frame is None:
            return 0.0

        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame

        scores = {}
        h, w = gen_final.shape[:2]

        # 1. CRITICAL: First row (top) must be preserved
        first_top_left = self._detect_shapes_in_quadrant(first_frame, "top_left")
        first_top_right = self._detect_shapes_in_quadrant(first_frame, "top_right")
        final_top_left = self._detect_shapes_in_quadrant(gen_final, "top_left")
        final_top_right = self._detect_shapes_in_quadrant(gen_final, "top_right")

        # Check if top row shapes are preserved (area should be similar)
        if first_top_left["exists"] and first_top_right["exists"]:
            tl_change = abs(final_top_left.get("area", 0) - first_top_left.get("area", 0)) / max(first_top_left.get("area", 1), 1)
            tr_change = abs(final_top_right.get("area", 0) - first_top_right.get("area", 0)) / max(first_top_right.get("area", 1), 1)

            if tl_change > 0.5 or tr_change > 0.5:
                # First row changed significantly
                scores["first_row_preserved"] = 0.0
                scores["bottom_right_correct"] = 0.0
                scores["shape_preserved"] = 0.0
                self._last_task_details = scores
                self._last_task_details["first_row_changed"] = True
                return 0.0
            else:
                scores["first_row_preserved"] = max(0, 1.0 - (tl_change + tr_change) / 2)
        else:
            scores["first_row_preserved"] = 0.0  # STRICT: No shapes detected in first row

        # Analyze bottom row
        gen_c = self._detect_shapes_in_quadrant(gen_final, "bottom_left")
        gen_d = self._detect_shapes_in_quadrant(gen_final, "bottom_right")
        gt_d = self._detect_shapes_in_quadrant(gt_final, "bottom_right")

        # 2. Bottom right (D) should match GT
        gen_d_region = gen_final[h // 2 :, w // 2 :]
        gt_d_region = gt_final[h // 2 :, w // 2 :]

        if gen_d_region.shape == gt_d_region.shape:
            diff = np.abs(gen_d_region.astype(float) - gt_d_region.astype(float)).mean()

            if diff < 15:
                scores["bottom_right_correct"] = 1.0
            elif diff < 30:
                scores["bottom_right_correct"] = 0.5
            else:
                scores["bottom_right_correct"] = 0.1
        else:
            scores["bottom_right_correct"] = 0.0

        # 3. Shape preservation: D should have same shape type as C
        if gen_d["exists"] and gen_c["exists"]:
            if gen_d["shape_type"] == gen_c["shape_type"]:
                scores["shape_preserved"] = 1.0
            else:
                scores["shape_preserved"] = 0.3
        else:
            scores["shape_preserved"] = 0.0

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class ShapeColorThenScaleEvaluator(BaseEvaluator):
    """
    O-12: Shape Color Then Scale (Two-step transformation)

    Task: A→B→C :: D→?→? format - first change color, then change scale.

    Rule-based evaluation:
    1. Two-step rule identification (30%) - Recognize color then scale
    2. First step accuracy (25%) - Color change, size unchanged
    3. Second step accuracy (25%) - Scale change, color maintained
    4. Sequence consistency (20%) - Correct order maintained
    """

    TASK_WEIGHTS = {"two_step_rule": 0.30, "first_step": 0.25, "second_step": 0.25, "sequence": 0.20}

    def _detect_shape_properties(self, frame: np.ndarray) -> Dict:
        """Detect shape color and size."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"exists": False}

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Get dominant color
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest], -1, 255, -1)

        if hsv is not None:
            mean_hue = cv2.mean(hsv[:, :, 0], mask=mask)[0]
            mean_sat = cv2.mean(hsv[:, :, 1], mask=mask)[0]
        else:
            mean_hue = 0
            mean_sat = 0

        return {"exists": True, "area": area, "hue": mean_hue, "saturation": mean_sat}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate two-step color-then-scale transformation."""

        if not video_frames or gt_final_frame is None:
            return 0.0

        scores = {}

        gen_final = video_frames[-1]
        gt_final = gt_final_frame

        # Analyze final state
        gen_props = self._detect_shape_properties(gen_final)
        gt_props = self._detect_shape_properties(gt_final)

        # 1. Two-step rule: Check if final matches GT (STRICT)
        if gen_props["exists"] and gt_props["exists"]:
            # Compare area (scale) - must be within 20%
            area_ratio = min(gen_props["area"], gt_props["area"]) / max(gen_props["area"], gt_props["area"], 1)

            # Compare color (hue)
            hue_diff = abs(gen_props["hue"] - gt_props["hue"])
            hue_diff = min(hue_diff, 180 - hue_diff)  # Circular hue

            # STRICT: both area and color must match closely
            area_ok = area_ratio > 0.7
            color_ok = hue_diff < 20

            if area_ok and color_ok:
                scores["two_step_rule"] = 1.0
            elif area_ok or color_ok:
                scores["two_step_rule"] = 0.3
            else:
                scores["two_step_rule"] = 0.0
        else:
            scores["two_step_rule"] = 0.0

        # 2. First step: Compare with GT final (STRICT)
        # The key is whether the final result matches GT
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            if diff < 20:
                scores["first_step"] = 1.0
            elif diff < 40:
                scores["first_step"] = 0.3
            else:
                scores["first_step"] = 0.0
        else:
            scores["first_step"] = 0.0

        # 3. Second step: Check final scale matches GT
        if gen_props["exists"] and gt_props["exists"]:
            area_ratio = min(gen_props["area"], gt_props["area"]) / max(gen_props["area"], gt_props["area"], 1)
            if area_ratio > 0.8:
                scores["second_step"] = 1.0
            elif area_ratio > 0.6:
                scores["second_step"] = 0.3
            else:
                scores["second_step"] = 0.0
        else:
            scores["second_step"] = 0.0

        # 4. Sequence: Overall consistency with GT
        scores["sequence"] = min(scores["first_step"], scores["second_step"])

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class ShapeOutlineThenMoveEvaluator(BaseEvaluator):
    """
    O-13: Shape Outline Then Move (Two-step transformation)

    Task: A→B→C :: D→?→? format - first change fill/outline style,
    then move vertically.

    Rule-based evaluation:
    1. Two-step rule identification (30%) - Recognize style then position
    2. First step accuracy (25%) - Style change, position at center
    3. Second step accuracy (25%) - Position change, style maintained
    4. Sequence consistency (20%) - Correct order maintained
    """

    TASK_WEIGHTS = {"two_step_rule": 0.30, "first_step": 0.25, "second_step": 0.25, "sequence": 0.20}

    def _get_shape_centroid(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Get centroid of main shape in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                return (cx, cy)

        return None

    def _is_outline_style(self, frame: np.ndarray) -> bool:
        """Check if shape is outline (not filled)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Check interior
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if 0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1]:
                center_val = gray[cy, cx]
                # Outline style: center is bright (not filled)
                return center_val > 200

        return False

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate two-step outline-then-move transformation."""

        if not video_frames or gt_final_frame is None:
            return 0.0

        scores = {}

        gen_final = video_frames[-1]
        gt_final = gt_final_frame

        # Get centroids
        gen_centroid = self._get_shape_centroid(gen_final)
        gt_centroid = self._get_shape_centroid(gt_final)

        # 1. Two-step rule: Compare final position with GT
        if gen_centroid is not None and gt_centroid is not None:
            dist = np.sqrt((gen_centroid[0] - gt_centroid[0]) ** 2 + (gen_centroid[1] - gt_centroid[1]) ** 2)
            scores["two_step_rule"] = max(0, 1.0 - dist / 100.0)
        else:
            scores["two_step_rule"] = 0.2  # Detection failed

        # 2. First step: Style change (outline detection)
        gen_is_outline = self._is_outline_style(gen_final)
        gt_is_outline = self._is_outline_style(gt_final)

        scores["first_step"] = 1.0 if gen_is_outline == gt_is_outline else 0.3

        # 3. Second step: Vertical movement
        if len(video_frames) >= 2:
            first_centroid = self._get_shape_centroid(video_frames[0])
            if first_centroid is not None and gen_centroid is not None:
                dy = abs(gen_centroid[1] - first_centroid[1])
                dx = abs(gen_centroid[0] - first_centroid[0])

                # Vertical movement: dy should be significant, dx minimal
                if dy > 20 and dx < dy:
                    scores["second_step"] = min(1.0, dy / 50.0)
                else:
                    scores["second_step"] = 0.3
            else:
                scores["second_step"] = 0.2  # Detection failed
        else:
            scores["second_step"] = 0.2  # Detection failed

        # 4. Sequence consistency
        scores["sequence"] = (scores["first_step"] + scores["second_step"]) / 2

        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class ShapeScaleThenOutlineEvaluator(BaseEvaluator):
    """
    O-14: Shape Scale Then Outline (Two-step transformation)

    Task: A→B→C :: D→?→? format - first change scale, then change
    from filled to outline.

    Rule-based evaluation:
    1. Two-step rule identification (30%) - Recognize scale then style
    2. First step accuracy (25%) - Scale change, style maintained
    3. Second step accuracy (25%) - Style change to outline
    4. Sequence consistency (20%) - Correct order maintained
    """

    TASK_WEIGHTS = {"two_step_rule": 0.30, "first_step": 0.25, "second_step": 0.25, "sequence": 0.20}

    def _get_shape_area(self, frame: np.ndarray) -> float:
        """Get area of main shape."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            return cv2.contourArea(largest)

        return 0

    def _is_outline_style(self, frame: np.ndarray) -> bool:
        """Check if shape is outline style."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False

        largest = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if 0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1]:
                center_val = gray[cy, cx]
                return center_val > 200

        return False

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate two-step scale-then-outline transformation."""

        if not video_frames or gt_final_frame is None:
            return 0.0

        scores = {}

        gen_final = video_frames[-1]
        gt_final = gt_final_frame

        # Get properties
        gen_area = self._get_shape_area(gen_final)
        gt_area = self._get_shape_area(gt_final)
        gen_outline = self._is_outline_style(gen_final)
        gt_outline = self._is_outline_style(gt_final)

        # 1. Two-step rule: Compare final state with GT (STRICT)
        if gen_area > 0 and gt_area > 0:
            area_ratio = min(gen_area, gt_area) / max(gen_area, gt_area)
            style_match = gen_outline == gt_outline

            # STRICT: Both area and style must match
            if area_ratio > 0.7 and style_match:
                scores["two_step_rule"] = 1.0
            elif area_ratio > 0.5 or style_match:
                scores["two_step_rule"] = 0.3
            else:
                scores["two_step_rule"] = 0.0
        else:
            scores["two_step_rule"] = 0.0

        # 2. First step: Compare with GT final (STRICT)
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            if diff < 20:
                scores["first_step"] = 1.0
            elif diff < 40:
                scores["first_step"] = 0.3
            else:
                scores["first_step"] = 0.0
        else:
            scores["first_step"] = 0.0

        # 3. Second step: Outline style must match GT
        scores["second_step"] = 1.0 if gen_outline == gt_outline else 0.0

        # 4. Sequence consistency
        scores["sequence"] = min(scores["first_step"], scores["second_step"])

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class BallBounceEvaluator(BaseEvaluator):
    """
    O-15: Ball Bounces Given Time

    Task: Given initial position and velocity arrow, predict ball trajectory
    with specified number of bounces off walls.

    Rule-based evaluation:
    1. Bounce count accuracy (30%) - Correct number of bounces
    2. Physics accuracy (35%) - Reflection law (angle in = angle out)
    3. Trajectory completeness (25%) - Full path shown
    4. Animation smoothness (10%) - Fluid motion
    """

    TASK_WEIGHTS = {"bounce_count": 0.30, "physics": 0.35, "trajectory": 0.25, "smoothness": 0.10}

    def _track_ball_positions(self, frames: List[np.ndarray]) -> List[Tuple[float, float]]:
        """Track ball center position across frames."""
        positions = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # Find dark regions (ball is typically dark)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

            # Try HoughCircles first
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=50)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                if len(circles) > 0:
                    x, y, r = circles[0]
                    positions.append((x, y))
                    continue

            # Fallback: use contour centroid
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    positions.append((cx, cy))

        return positions

    def _count_bounces(self, positions: List[Tuple[float, float]]) -> int:
        """Count number of direction changes (bounces)."""
        if len(positions) < 3:
            return 0

        bounces = 0
        for i in range(2, len(positions)):
            dx1 = positions[i - 1][0] - positions[i - 2][0]
            dy1 = positions[i - 1][1] - positions[i - 2][1]
            dx2 = positions[i][0] - positions[i - 1][0]
            dy2 = positions[i][1] - positions[i - 1][1]

            # Check for direction reversal (bounce)
            if (dx1 * dx2 < -5) or (dy1 * dy2 < -5):
                bounces += 1

        return bounces

    def _calculate_motion_smoothness(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate how smooth the motion is."""
        if len(positions) < 2:
            return 1.0

        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            v = np.sqrt(dx**2 + dy**2)
            velocities.append(v)

        if len(velocities) < 2:
            return 1.0

        mean_v = np.mean(velocities)
        if mean_v < 1:
            return 0.5

        std_v = np.std(velocities)
        cv = std_v / mean_v

        return max(0, 1 - cv / 2)

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate ball bounce trajectory prediction."""

        if not video_frames or not gt_frames:
            return 0.0

        scores = {}

        # Track ball positions
        gen_positions = self._track_ball_positions(video_frames)
        gt_positions = self._track_ball_positions(gt_frames)

        # 1. Bounce count
        gen_bounces = self._count_bounces(gen_positions)
        gt_bounces = self._count_bounces(gt_positions)

        if gt_bounces > 0:
            bounce_diff = abs(gen_bounces - gt_bounces)
            scores["bounce_count"] = max(0, 1 - bounce_diff / gt_bounces)
        else:
            scores["bounce_count"] = 1.0 if gen_bounces == 0 else 0.5

        # 2. Physics accuracy: Compare final positions
        if gen_positions and gt_positions:
            gen_final = gen_positions[-1]
            gt_final = gt_positions[-1]
            dist = np.sqrt((gen_final[0] - gt_final[0]) ** 2 + (gen_final[1] - gt_final[1]) ** 2)
            scores["physics"] = max(0, 1.0 - dist / 100.0)
        else:
            scores["physics"] = 0.2  # Detection failed

        # 3. Trajectory completeness
        if gen_positions:
            # Check if trajectory spans reasonable distance
            if len(gen_positions) >= 2:
                total_dist = sum(np.sqrt((gen_positions[i][0] - gen_positions[i - 1][0]) ** 2 + (gen_positions[i][1] - gen_positions[i - 1][1]) ** 2) for i in range(1, len(gen_positions)))
                scores["trajectory"] = min(1.0, total_dist / 200.0)
            else:
                scores["trajectory"] = 0.3
        else:
            scores["trajectory"] = 0.0

        # 4. Smoothness
        scores["smoothness"] = self._calculate_motion_smoothness(gen_positions)

        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class ColorAdditionEvaluator(BaseEvaluator):
    """
    O-16: Color Addition (Additive Color Mixing)

    Task: Two colored balls move toward each other and merge, showing
    additive color mixing in the overlap region.

    Rule-based evaluation:
    1. Additive mixing accuracy (40%) - Correct RGB addition
    2. Movement trajectory (30%) - Equal speed, meet at midpoint
    3. Overlap handling (20%) - Proper blend in overlap region
    4. Visual fidelity (10%) - Ball size, shape preserved
    """

    TASK_WEIGHTS = {"mixing": 0.40, "movement": 0.30, "overlap": 0.20, "fidelity": 0.10}

    def _detect_colored_regions(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored regions and their properties."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        if hsv is None:
            return []

        regions = []

        # Define color ranges (lower saturation threshold for blended colors)
        color_ranges = {
            "red": [([0, 50, 100], [10, 255, 255]), ([160, 50, 100], [180, 255, 255])],
            "green": [([35, 50, 100], [85, 255, 255])],
            "blue": [([100, 50, 100], [140, 255, 255])],  # Extended to include violet/purple
            "yellow": [([20, 50, 100], [35, 255, 255])],
            "cyan": [([85, 50, 100], [100, 255, 255])],
            "magenta": [([140, 50, 100], [160, 255, 255])],
        }

        for color_name, ranges in color_ranges.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                regions.append({"color": color_name, "center": (cx, cy), "area": area})

        return regions

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate additive color mixing animation."""

        if not video_frames or gt_final_frame is None:
            return 0.0

        scores = {}

        gen_final = video_frames[-1]
        gt_final = gt_final_frame

        # Detect colored regions
        gen_regions = self._detect_colored_regions(gen_final)
        gt_regions = self._detect_colored_regions(gt_final)

        # 1. Mixing accuracy: Compare color distribution (non-white regions only)
        if gen_final.shape == gt_final.shape:
            # Only compare non-white regions
            gt_gray = cv2.cvtColor(gt_final, cv2.COLOR_BGR2GRAY)
            non_white_mask = gt_gray < 240

            if np.sum(non_white_mask) > 100:
                gen_masked = gen_final[non_white_mask]
                gt_masked = gt_final[non_white_mask]
                color_diff = np.abs(gen_masked.astype(float) - gt_masked.astype(float)).mean()

                # Stricter threshold for non-background comparison
                if color_diff < 20:
                    scores["mixing"] = 1.0
                elif color_diff < 40:
                    scores["mixing"] = 0.5
                else:
                    scores["mixing"] = 0.0
            else:
                scores["mixing"] = 0.0  # No colored content
        else:
            scores["mixing"] = 0.0  # Detection failed

        # 2. Movement: Check if balls moved toward center
        if len(video_frames) >= 2:
            first_regions = self._detect_colored_regions(video_frames[0])

            if first_regions and gen_regions:
                # Calculate center of mass movement
                first_com = np.mean([r["center"] for r in first_regions], axis=0) if first_regions else (0, 0)
                final_com = np.mean([r["center"] for r in gen_regions], axis=0) if gen_regions else (0, 0)

                frame_center = (gen_final.shape[1] // 2, gen_final.shape[0] // 2)

                # Check if regions moved toward center
                first_dist = np.sqrt((first_com[0] - frame_center[0]) ** 2 + (first_com[1] - frame_center[1]) ** 2)
                final_dist = np.sqrt((final_com[0] - frame_center[0]) ** 2 + (final_com[1] - frame_center[1]) ** 2)

                if first_dist > final_dist:
                    scores["movement"] = min(1.0, (first_dist - final_dist) / 50.0 + 0.5)
                else:
                    scores["movement"] = 0.0
            else:
                scores["movement"] = 0.0  # Detection failed
        else:
            scores["movement"] = 0.0  # Detection failed

        # 3. Overlap handling: Check for blended region
        if gen_regions:
            # Multiple colors or blended = good overlap handling
            unique_colors = set(r["color"] for r in gen_regions)
            scores["overlap"] = min(1.0, len(unique_colors) / 2.0)
        else:
            scores["overlap"] = 0.0  # Detection failed

        # 4. Fidelity: Compare region counts
        if gen_regions and gt_regions:
            count_ratio = min(len(gen_regions), len(gt_regions)) / max(len(gen_regions), len(gt_regions), 1)
            scores["fidelity"] = count_ratio
        else:
            scores["fidelity"] = 0.0  # Detection failed

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class GlassRefractionEvaluator(BaseEvaluator):
    """
    O-18: Glass Refraction (Snell's Law)

    Task: Given incident angle and refractive index, calculate and draw
    the refracted ray according to Snell's law.

    Rule-based evaluation:
    1. Snell's law application (50%) - Correct angle calculation
    2. Refracted ray direction (30%) - Accurate angle
    3. Ray completeness (15%) - Extends to edge
    4. Scene fidelity (5%) - Original elements preserved
    """

    TASK_WEIGHTS = {"snells_law": 0.50, "ray_direction": 0.30, "ray_completeness": 0.15, "scene_fidelity": 0.05}

    def _detect_lines(self, frame: np.ndarray) -> List[Dict]:
        """Detect lines in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

        detected = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                detected.append({"start": (x1, y1), "end": (x2, y2), "angle": angle, "length": length})

        return detected

    def _detect_red_line(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect red line (refracted ray)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        if hsv is None:
            return None

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        lines = cv2.HoughLinesP(red_mask, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=10)

        if lines is not None and len(lines) > 0:
            longest = max(lines, key=lambda l: np.sqrt((l[0][2] - l[0][0]) ** 2 + (l[0][3] - l[0][1]) ** 2))
            x1, y1, x2, y2 = longest[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return {"start": (x1, y1), "end": (x2, y2), "angle": angle, "length": length}

        return None

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate refraction ray drawing accuracy."""

        if not video_frames or gt_final_frame is None:
            return 0.0

        scores = {}

        gen_final = video_frames[-1]
        gt_final = gt_final_frame

        # Detect red lines (refracted rays)
        gen_ray = self._detect_red_line(gen_final)
        gt_ray = self._detect_red_line(gt_final)

        # 1. Snell's law: Compare ray angles
        if gen_ray is not None and gt_ray is not None:
            angle_diff = abs(gen_ray["angle"] - gt_ray["angle"])
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            scores["snells_law"] = max(0, 1.0 - angle_diff / 30.0)
        else:
            scores["snells_law"] = 0.5 if gen_ray is None and gt_ray is None else 0.0

        # 2. Ray direction: Same as Snell's law with looser tolerance
        if gen_ray is not None and gt_ray is not None:
            angle_diff = abs(gen_ray["angle"] - gt_ray["angle"])
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            scores["ray_direction"] = max(0, 1.0 - angle_diff / 45.0)
        else:
            scores["ray_direction"] = 0.2  # Detection failed

        # 3. Ray completeness: Check line length
        if gen_ray is not None and gt_ray is not None:
            length_ratio = min(gen_ray["length"], gt_ray["length"]) / max(gen_ray["length"], gt_ray["length"], 1)
            scores["ray_completeness"] = length_ratio
        elif gen_ray is not None:
            scores["ray_completeness"] = min(1.0, gen_ray["length"] / 100.0)
        else:
            scores["ray_completeness"] = 0.0

        # 4. Scene fidelity: Compare non-ray elements
        if len(video_frames) >= 1:
            first_frame = video_frames[0]
            gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) if len(first_frame.shape) == 3 else first_frame
            gray_final = cv2.cvtColor(gen_final, cv2.COLOR_BGR2GRAY) if len(gen_final.shape) == 3 else gen_final

            if gray_first.shape == gray_final.shape:
                diff = np.abs(gray_first.astype(float) - gray_final.astype(float)).mean()
                # Small diff means scene preserved (only ray added)
                scores["scene_fidelity"] = max(0, 1.0 - diff / 50.0)
            else:
                scores["scene_fidelity"] = 0.2  # Detection failed
        else:
            scores["scene_fidelity"] = 0.2  # Detection failed

        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class MirrorReflectionEvaluator(BaseEvaluator):
    """
    O-19: Mirror Reflection

    Task: Given incident ray angle and mirror reflectivity, draw the
    reflected ray according to reflection law (angle in = angle out).

    Rule-based evaluation:
    1. Reflection angle accuracy (40%) - Incident angle = reflected angle
    2. Symmetry correctness (30%) - Ray symmetric about normal
    3. Ray extension (20%) - Extends to image boundary
    4. Starting point accuracy (10%) - Starts from incident point
    """

    TASK_WEIGHTS = {"reflection_angle": 0.40, "symmetry": 0.30, "ray_extension": 0.20, "starting_point": 0.10}

    def _detect_lines(self, frame: np.ndarray) -> List[Dict]:
        """Detect all lines in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

        detected = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                detected.append({"start": (x1, y1), "end": (x2, y2), "angle": angle, "length": length})

        return detected

    def _detect_colored_line(self, frame: np.ndarray, color: str) -> Optional[Dict]:
        """Detect line of specific color."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        if hsv is None:
            return None

        color_ranges = {
            "red": [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [180, 255, 255])],
            "blue": [([100, 100, 100], [130, 255, 255])],
            "green": [([35, 100, 100], [85, 255, 255])],
        }

        if color not in color_ranges:
            return None

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges[color]:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=10)

        if lines is not None and len(lines) > 0:
            longest = max(lines, key=lambda l: np.sqrt((l[0][2] - l[0][0]) ** 2 + (l[0][3] - l[0][1]) ** 2))
            x1, y1, x2, y2 = longest[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return {"start": (x1, y1), "end": (x2, y2), "angle": angle, "length": length}

        return None

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate mirror reflection ray drawing accuracy."""

        if not video_frames or gt_final_frame is None:
            return 0.0

        scores = {}

        gen_final = video_frames[-1]
        gt_final = gt_final_frame

        # Detect lines
        gen_lines = self._detect_lines(gen_final)
        gt_lines = self._detect_lines(gt_final)

        # Try to detect reflected ray (often red or blue)
        gen_reflected = self._detect_colored_line(gen_final, "red")
        if gen_reflected is None:
            gen_reflected = self._detect_colored_line(gen_final, "blue")

        gt_reflected = self._detect_colored_line(gt_final, "red")
        if gt_reflected is None:
            gt_reflected = self._detect_colored_line(gt_final, "blue")

        # 1. Reflection angle: Compare ray angles
        if gen_reflected is not None and gt_reflected is not None:
            angle_diff = abs(gen_reflected["angle"] - gt_reflected["angle"])
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            scores["reflection_angle"] = max(0, 1.0 - angle_diff / 30.0)
        else:
            scores["reflection_angle"] = 0.2  # Detection failed

        # 2. Symmetry: Check if angles are symmetric about normal
        if gen_lines and len(gen_lines) >= 2:
            # Find incident and reflected rays
            angles = [l["angle"] for l in gen_lines if l["length"] > 50]
            if len(angles) >= 2:
                # Check for angle symmetry
                angles_sorted = sorted(angles)
                angle_spread = angles_sorted[-1] - angles_sorted[0]
                if 60 < angle_spread < 120:
                    scores["symmetry"] = 0.8
                else:
                    scores["symmetry"] = 0.2  # Detection failed
            else:
                scores["symmetry"] = 0.2  # Detection failed
        else:
            scores["symmetry"] = 0.2  # Detection failed

        # 3. Ray extension: Check line length
        if gen_reflected is not None and gt_reflected is not None:
            length_ratio = min(gen_reflected["length"], gt_reflected["length"]) / max(gen_reflected["length"], gt_reflected["length"], 1)
            scores["ray_extension"] = length_ratio
        elif gen_reflected is not None:
            scores["ray_extension"] = min(1.0, gen_reflected["length"] / 100.0)
        else:
            scores["ray_extension"] = 0.2  # Detection failed

        # 4. Starting point: Compare overall structure
        if gen_final.shape == gt_final.shape:
            gray_gen = cv2.cvtColor(gen_final, cv2.COLOR_BGR2GRAY) if len(gen_final.shape) == 3 else gen_final
            gray_gt = cv2.cvtColor(gt_final, cv2.COLOR_BGR2GRAY) if len(gt_final.shape) == 3 else gt_final

            gen_edges = cv2.Canny(gray_gen, 50, 150)
            gt_edges = cv2.Canny(gray_gt, 50, 150)

            edge_overlap = np.sum(gen_edges & gt_edges)
            edge_union = np.sum(gen_edges | gt_edges)

            scores["starting_point"] = edge_overlap / max(edge_union, 1)
        else:
            scores["starting_point"] = 0.2  # Detection failed

        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


# Export mapping for this batch
IN_DOMAIN_50_EVALUATORS_PART3 = {
    "G-158_identify_all_hollow_points_data-generator": IdentifyAllHollowPointsEvaluator,
    "G-194_construct_concentric_ring_data-generator": ConstructConcentricRingEvaluator,
    "O-10_shape_outline_fill_data-generator": ShapeOutlineFillEvaluator,
    "O-12_shape_color_then_scale_data-generator": ShapeColorThenScaleEvaluator,
    "O-13_shape_outline_then_move_data-generator": ShapeOutlineThenMoveEvaluator,
    "O-14_shape_scale_then_outline_data-generator": ShapeScaleThenOutlineEvaluator,
    "O-15_ball_bounces_given_time_data-generator": BallBounceEvaluator,
    "O-16_color_addition_data-generator": ColorAdditionEvaluator,
    "O-18_glass_refraction_data-generator": GlassRefractionEvaluator,
    "O-19_mirror_reflection_data-generator": MirrorReflectionEvaluator,
}
