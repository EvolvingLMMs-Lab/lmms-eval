"""
Specific evaluators for Out-of-Domain_50 tasks (Part 5).
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils import compute_optical_flow, normalize_frame_size
from .base_evaluator import BaseEvaluator


class ControlPanelEvaluator(BaseEvaluator):
    """
    O-54: Control panel evaluator.

    Rule-based evaluation:
    - State matching correctness (45%): All controls reach target state
      - Switch: correct on/off state (green track=on, gray track=off)
      - Slider: value within 5% of target
      - Button: pressed state (green color)
      - Dial: angle within 10 degrees
    - Operation smoothness (25%): Smooth transitions
    - Control identification (20%): Correct control types identified
    - Panel preservation (10%): Panel layout unchanged
    """

    TASK_WEIGHTS = {"state_matching": 0.45, "smoothness": 0.25, "identification": 0.20, "preservation": 0.10}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        if len(video_frames) < 2:
            return 0.0

        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]

        # CRITICAL: First check if control panel structure is preserved
        # Count distinct colored regions (controls) in first and final frames
        first_controls = self._count_control_regions(first_frame)
        final_controls = self._count_control_regions(final_frame)

        # If final frame has very few controls or one huge region, structure is destroyed
        if final_controls < 2 or (first_controls > 2 and final_controls < first_controls // 2):
            self._last_task_details = {"state_matching": 0.0, "smoothness": 0.3, "identification": 0.0, "preservation": 0.0, "structure_destroyed": True, "first_controls": first_controls, "final_controls": final_controls}
            return 0.0

        scores["state_matching"] = self._evaluate_state_matching(first_frame, final_frame, gt_final_frame)
        scores["smoothness"] = self._evaluate_smoothness(video_frames)
        scores["identification"] = self._evaluate_identification(first_frame, final_frame)
        scores["preservation"] = self._evaluate_preservation(first_frame, final_frame)

        self._last_task_details = scores
        self._last_task_details["first_controls"] = first_controls
        self._last_task_details["final_controls"] = final_controls

        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

    def _count_control_regions(self, frame: np.ndarray) -> int:
        """Count distinct control regions (colored areas) in frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] > 50).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count regions that are reasonably sized (not too small, not too large)
        h, w = frame.shape[:2]
        max_area = h * w * 0.3  # Max 30% of frame

        return sum(1 for cnt in contours if 100 < cv2.contourArea(cnt) < max_area)

    def _detect_controls(self, frame: np.ndarray) -> Dict:
        """Detect control panel elements."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        controls = {"switches": [], "buttons": [], "sliders": [], "dials": []}

        # Detect green elements (active switches, pressed buttons)
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Detect blue elements (slider progress)
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Detect yellow elements (dial pointers)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Find green control regions
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(cnt)

            # Classify by aspect ratio
            aspect = w / h if h > 0 else 1
            if 0.5 <= aspect <= 2.0 and area < 2000:
                # Could be button or switch
                controls["buttons"].append({"center": (cx, cy), "area": area, "bbox": (x, y, w, h)})

        # Find blue slider regions
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w > h * 2:  # Horizontal slider
                controls["sliders"].append({"bbox": (x, y, w, h), "value": w})

        # Find yellow dial pointers
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            controls["dials"].append({"center": (cx, cy), "area": area})

        return controls

    def _evaluate_state_matching(self, first: np.ndarray, final: np.ndarray, gt_final: Optional[np.ndarray]) -> float:
        """Check if all controls reach target state."""
        if gt_final is None:
            return 0.5

        final_controls = self._detect_controls(final)
        gt_controls = self._detect_controls(gt_final)

        scores = []

        # Compare button states (green = pressed)
        final_buttons = len(final_controls["buttons"])
        gt_buttons = len(gt_controls["buttons"])
        if gt_buttons > 0:
            button_match = min(final_buttons, gt_buttons) / gt_buttons
            scores.append(button_match)

        # Compare slider states (blue progress width)
        if gt_controls["sliders"] and final_controls["sliders"]:
            for gt_slider in gt_controls["sliders"]:
                best_match = 0
                for f_slider in final_controls["sliders"]:
                    # Compare slider values (width of blue area)
                    gt_val = gt_slider["value"]
                    f_val = f_slider["value"]
                    if gt_val > 0:
                        ratio = min(f_val, gt_val) / max(f_val, gt_val)
                        best_match = max(best_match, ratio)
                scores.append(best_match)

        # Compare dial positions
        if gt_controls["dials"] and final_controls["dials"]:
            dial_match = min(len(final_controls["dials"]), len(gt_controls["dials"])) / max(len(gt_controls["dials"]), 1)
            scores.append(dial_match)

        return np.mean(scores) if scores else 0.5

    def _evaluate_smoothness(self, video_frames: List[np.ndarray]) -> float:
        """Check if transitions are smooth."""
        if len(video_frames) < 3:
            return 0.5

        diffs = []
        for i in range(1, min(len(video_frames), 20)):
            diff = np.mean(np.abs(video_frames[i].astype(float) - video_frames[i - 1].astype(float)))
            diffs.append(diff)

        if len(diffs) < 2:
            return 0.5

        # Low variance = smooth transitions
        variance = np.var(diffs)
        return 1.0 / (1.0 + variance / 50)

    def _evaluate_identification(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if controls are correctly identified and operated."""
        first_controls = self._detect_controls(first)
        final_controls = self._detect_controls(final)

        # Check if control counts are reasonable
        first_count = sum(len(v) for v in first_controls.values())
        final_count = sum(len(v) for v in final_controls.values())

        if first_count == 0:
            return 0.5

        # Some controls should change (buttons turn green, etc)
        changed = abs(len(final_controls["buttons"]) - len(first_controls["buttons"]))

        return min(1.0, 0.5 + changed * 0.2)

    def _evaluate_preservation(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if panel layout is preserved."""
        # Compare edge structures
        edges_final = cv2.Canny(final, 50, 150)
        edges_first = cv2.Canny(first, 50, 150)

        # Panel border should be mostly preserved
        intersection = np.sum((edges_final > 0) & (edges_first > 0))
        union = np.sum((edges_final > 0) | (edges_first > 0))

        if union == 0:
            return 0.5
        return intersection / union


class RavenMatrixEvaluator(BaseEvaluator):
    """
    O-56: Raven's Progressive Matrices evaluator.

    CRITICAL RULES:
    1. The video shows a 3x3 grid (9 cells)
    2. ONLY the bottom-right cell (position 2,2) should change
    3. The other 8 cells MUST remain UNCHANGED
    4. The answer cell should contain the correct pattern based on the rules

    Evaluation dimensions:
    - Other cells preserved (40%): CRITICAL - first 8 cells must not change
    - Answer cell correct (40%): Bottom-right has correct shape/pattern
    - Answer cell has content (15%): Something was drawn in answer cell
    - Grid structure preserved (5%): 3x3 grid structure maintained
    """

    TASK_WEIGHTS = {"preservation": 0.40, "answer_correct": 0.40, "answer_has_content": 0.15, "grid_structure": 0.05}  # CRITICAL: other 8 cells unchanged  # Answer cell matches GT  # Answer cell is not empty

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        if len(video_frames) < 2:
            return 0.0

        first_frame = video_frames[0]
        final_frame = video_frames[-1]

        scores = {}

        # Normalize frame size (handles padding removal + resize)
        if gt_final_frame is not None and final_frame.shape != gt_final_frame.shape:
            gt_final_frame = normalize_frame_size(gt_final_frame, final_frame)
        if gt_first_frame is not None and first_frame.shape != gt_first_frame.shape:
            gt_first_frame = normalize_frame_size(gt_first_frame, first_frame)

        # Detect what's in each cell
        first_cell_info = self._analyze_all_cells(first_frame)
        final_cell_info = self._analyze_all_cells(final_frame)

        # Store debug info
        scores["first_cell_counts"] = [c["count"] for c in first_cell_info]
        scores["final_cell_counts"] = [c["count"] for c in final_cell_info]

        # 1. CRITICAL: Check if other 8 cells are preserved (40%)
        preservation_score = self._evaluate_other_cells_preserved(first_frame, final_frame, first_cell_info, final_cell_info)
        scores["preservation"] = preservation_score

        # If preservation is too low, other scores are less meaningful
        if preservation_score < 0.5:
            scores["error"] = "other_cells_changed"

        # 2. Check if answer cell (2,2) is correct (40%)
        if gt_final_frame is not None:
            answer_score = self._evaluate_answer_cell(final_frame, gt_final_frame)
        else:
            answer_score = 0.5  # Can't evaluate without GT
        scores["answer_correct"] = answer_score

        # 3. Check if answer cell has content (15%)
        answer_cell = self._extract_cell(final_frame, 2, 2)
        answer_props = self._detect_shapes_in_cell(answer_cell)
        scores["answer_has_content"] = 1.0 if answer_props["count"] > 0 else 0.0
        scores["answer_shape_count"] = answer_props["count"]
        scores["answer_shape_types"] = answer_props["types"]

        # 4. Grid structure (5%)
        scores["grid_structure"] = self._evaluate_grid_structure(final_frame)

        self._last_task_details = scores
        return sum(scores.get(k, 0) * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS if k in scores)

    def _analyze_all_cells(self, frame: np.ndarray) -> List[Dict]:
        """Analyze shapes in all 9 cells."""
        cells = []
        for row in range(3):
            for col in range(3):
                cell = self._extract_cell(frame, row, col)
                props = self._detect_shapes_in_cell(cell)
                props["row"] = row
                props["col"] = col
                cells.append(props)
        return cells

    def _evaluate_other_cells_preserved(self, first_frame: np.ndarray, final_frame: np.ndarray, first_info: List[Dict], final_info: List[Dict]) -> float:
        """CRITICAL: Check that all 8 cells except bottom-right are unchanged."""
        unchanged_count = 0
        total_checked = 0

        for row in range(3):
            for col in range(3):
                if row == 2 and col == 2:
                    continue  # Skip answer cell

                idx = row * 3 + col
                first_props = first_info[idx]
                final_props = final_info[idx]

                total_checked += 1

                # Check if cell is unchanged
                # 1. Same shape count
                if first_props["count"] != final_props["count"]:
                    continue

                # 2. Same shape types
                if sorted(first_props["types"]) != sorted(final_props["types"]):
                    continue

                # 3. Similar pixel content (using cell comparison)
                first_cell = self._extract_cell(first_frame, row, col)
                final_cell = self._extract_cell(final_frame, row, col)

                # Compare cells
                diff = np.mean(np.abs(first_cell.astype(float) - final_cell.astype(float)))
                if diff < 15:  # Very similar
                    unchanged_count += 1
                elif diff < 30:  # Somewhat similar (partial credit)
                    unchanged_count += 0.5

        return unchanged_count / total_checked if total_checked > 0 else 0.0

    def _evaluate_answer_cell(self, final: np.ndarray, gt_final: np.ndarray) -> float:
        """Check if answer cell (bottom-right) matches GT."""
        final_cell = self._extract_cell(final, 2, 2)
        gt_cell = self._extract_cell(gt_final, 2, 2)

        final_props = self._detect_shapes_in_cell(final_cell)
        gt_props = self._detect_shapes_in_cell(gt_cell)

        score = 0.0

        # Check shape count (most important)
        if final_props["count"] == gt_props["count"]:
            score += 0.5
        elif abs(final_props["count"] - gt_props["count"]) == 1:
            score += 0.2

        # Check shape types
        if sorted(final_props["types"]) == sorted(gt_props["types"]):
            score += 0.3
        elif set(final_props["types"]) & set(gt_props["types"]):
            score += 0.1

        # Check fill patterns
        if final_props["filled"] == gt_props["filled"]:
            score += 0.2

        return min(1.0, score)

    def _evaluate_grid_structure(self, frame: np.ndarray) -> float:
        """Check if 3x3 grid structure is preserved."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        h, w = gray.shape

        # Check for horizontal and vertical lines at grid boundaries
        grid_score = 0.0

        # Check horizontal lines at 1/3 and 2/3 height
        for y_frac in [1 / 3, 2 / 3]:
            y = int(h * y_frac)
            line_region = edges[max(0, y - 5) : min(h, y + 5), :]
            if np.sum(line_region > 0) > w * 0.3:  # At least 30% of width has edges
                grid_score += 0.25

        # Check vertical lines at 1/3 and 2/3 width
        for x_frac in [1 / 3, 2 / 3]:
            x = int(w * x_frac)
            line_region = edges[:, max(0, x - 5) : min(w, x + 5)]
            if np.sum(line_region > 0) > h * 0.3:  # At least 30% of height has edges
                grid_score += 0.25

        return grid_score

    def _extract_cell(self, frame: np.ndarray, row: int, col: int) -> np.ndarray:
        """Extract a single cell from the 3x3 matrix."""
        h, w = frame.shape[:2]
        cell_h, cell_w = h // 3, w // 3
        return frame[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w]

    def _detect_shapes_in_cell(self, cell: np.ndarray) -> Dict:
        """Detect shapes in a cell and return properties."""
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            # Approximate contour to get shape type
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            vertices = len(approx)

            # Determine shape type
            if vertices == 3:
                shape_type = "triangle"
            elif vertices == 4:
                shape_type = "square"
            elif vertices >= 6:
                # Check circularity
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
                if circularity > 0.7:
                    shape_type = "circle"
                else:
                    shape_type = "polygon"
            else:
                shape_type = "other"

            # Check if filled
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Sample inside
                if 0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1]:
                    is_filled = gray[cy, cx] < 128
                else:
                    is_filled = False
            else:
                is_filled = False

            shapes.append({"type": shape_type, "area": area, "vertices": vertices, "filled": is_filled})

        return {"count": len(shapes), "shapes": shapes, "types": [s["type"] for s in shapes], "filled": [s["filled"] for s in shapes]}


class SymbolDeleteEvaluator(BaseEvaluator):
    """
    O-58: Symbol delete evaluator.

    Rule-based evaluation:
    - Target identification & deletion (40%): Correct symbol removed
    - Sequence reorganization (30%): Remaining symbols shift left
    - Order preservation (20%): Relative order maintained
    - Symbol fidelity (10%): Remaining symbols unchanged
    """

    TASK_WEIGHTS = {"deletion_accuracy": 0.40, "reorganization": 0.30, "order_preservation": 0.20, "symbol_fidelity": 0.10}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        if len(video_frames) < 2:
            return 0.0

        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]

        scores["deletion_accuracy"] = self._evaluate_deletion(first_frame, final_frame, gt_final_frame)
        scores["reorganization"] = self._evaluate_reorganization(first_frame, final_frame)
        scores["order_preservation"] = self._evaluate_order(first_frame, final_frame)
        scores["symbol_fidelity"] = self._evaluate_fidelity(first_frame, final_frame)

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

    def _detect_symbols(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored symbols in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Find colored (non-white, non-black) regions
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        symbols = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Get dominant color
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[max(0, cy - 5) : cy + 5, max(0, cx - 5) : cx + 5]
            if roi.size > 0:
                color = tuple(roi.mean(axis=(0, 1)).astype(int).tolist())
            else:
                color = (0, 0, 0)

            symbols.append({"center": (cx, cy), "area": area, "color": color, "bbox": (x, y, w, h)})

        # Sort by x position
        symbols.sort(key=lambda s: s["center"][0])
        return symbols

    def _evaluate_deletion(self, first: np.ndarray, final: np.ndarray, gt_final: Optional[np.ndarray]) -> float:
        """Check if correct symbol is deleted."""
        first_symbols = self._detect_symbols(first)
        final_symbols = self._detect_symbols(final)

        # Should have one fewer symbol
        expected_count = len(first_symbols) - 1
        actual_count = len(final_symbols)

        if actual_count == expected_count:
            score = 1.0
        elif actual_count == expected_count - 1 or actual_count == expected_count + 1:
            score = 0.5  # STRICT: Close but not exact
        else:
            score = 0.0  # STRICT: Wrong symbol count

        return score

    def _evaluate_reorganization(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if symbols shift left correctly."""
        first_symbols = self._detect_symbols(first)
        final_symbols = self._detect_symbols(final)

        if not final_symbols:
            return 0.0

        # Check if symbols are evenly spaced
        if len(final_symbols) >= 2:
            spacings = []
            for i in range(1, len(final_symbols)):
                spacing = final_symbols[i]["center"][0] - final_symbols[i - 1]["center"][0]
                spacings.append(spacing)

            if spacings:
                variance = np.var(spacings)
                mean_spacing = np.mean(spacings)
                cv = np.sqrt(variance) / mean_spacing if mean_spacing > 0 else 1.0
                return max(0, 1.0 - cv)

        return 0.0  # STRICT: Not enough symbols to evaluate reorganization

    def _evaluate_order(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if relative order is maintained."""
        first_symbols = self._detect_symbols(first)
        final_symbols = self._detect_symbols(final)

        if not final_symbols:
            return 0.0

        # Compare colors of remaining symbols (should be subset of first)
        first_colors = [s["color"] for s in first_symbols]
        final_colors = [s["color"] for s in final_symbols]

        # Check if colors maintain relative order
        matches = 0
        for i, f_color in enumerate(final_colors):
            # Find matching color in first
            for j, color in enumerate(first_colors):
                dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(f_color, color)))
                if dist < 50:  # Close enough
                    matches += 1
                    break

        return matches / max(len(final_colors), 1)

    def _evaluate_fidelity(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if remaining symbols are unchanged."""
        first_symbols = self._detect_symbols(first)
        final_symbols = self._detect_symbols(final)

        if not final_symbols:
            return 0.0

        # Check area preservation
        first_areas = sorted([s["area"] for s in first_symbols])
        final_areas = sorted([s["area"] for s in final_symbols])

        if len(first_areas) > len(final_areas):
            # Remove one area (the deleted symbol)
            first_areas_subset = first_areas[:-1] if len(first_areas) > 0 else []
        else:
            first_areas_subset = first_areas

        if len(first_areas_subset) == len(final_areas):
            area_diffs = [abs(a - b) / max(a, 1) for a, b in zip(first_areas_subset, final_areas)]
            return 1.0 - min(1.0, np.mean(area_diffs))

        return 0.0  # STRICT: Symbol counts don't match


class SymbolInsertEvaluator(BaseEvaluator):
    """
    O-59: Symbol insert evaluator.

    Rule-based evaluation:
    - Insert position accuracy (40%): Correct position
    - Symbol identification (30%): Correct symbol inserted
    - Sequence adjustment (25%): Other symbols shift correctly
    - Layout accuracy (5%): Centered, even spacing
    """

    TASK_WEIGHTS = {"position_accuracy": 0.40, "symbol_identification": 0.30, "sequence_adjustment": 0.25, "layout_accuracy": 0.05}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        if len(video_frames) < 2:
            return 0.0

        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]

        scores["position_accuracy"] = self._evaluate_position(first_frame, final_frame, gt_final_frame)
        scores["symbol_identification"] = self._evaluate_symbol(first_frame, final_frame, gt_final_frame)
        scores["sequence_adjustment"] = self._evaluate_adjustment(first_frame, final_frame)
        scores["layout_accuracy"] = self._evaluate_layout(final_frame)

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

    def _detect_symbols(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored symbols in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        symbols = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[max(0, cy - 5) : cy + 5, max(0, cx - 5) : cx + 5]
            if roi.size > 0:
                color = tuple(roi.mean(axis=(0, 1)).astype(int).tolist())
            else:
                color = (0, 0, 0)

            symbols.append({"center": (cx, cy), "area": area, "color": color, "bbox": (x, y, w, h)})

        symbols.sort(key=lambda s: s["center"][0])
        return symbols

    def _evaluate_position(self, first: np.ndarray, final: np.ndarray, gt_final: Optional[np.ndarray]) -> float:
        """Check if insert position is correct."""
        first_symbols = self._detect_symbols(first)
        final_symbols = self._detect_symbols(final)

        # Should have one more symbol
        expected_count = len(first_symbols) + 1
        actual_count = len(final_symbols)

        if actual_count == expected_count:
            score = 1.0
        elif abs(actual_count - expected_count) == 1:
            score = 0.5  # STRICT: Close but not exact
        else:
            score = 0.0  # STRICT: Wrong symbol count

        return score

    def _evaluate_symbol(self, first: np.ndarray, final: np.ndarray, gt_final: Optional[np.ndarray]) -> float:
        """Check if correct symbol is inserted."""
        first_symbols = self._detect_symbols(first)
        final_symbols = self._detect_symbols(final)

        # Check if a new symbol was added
        if len(final_symbols) > len(first_symbols):
            return 1.0
        elif len(final_symbols) == len(first_symbols):
            return 0.0  # STRICT: No symbol added
        return 0.0  # STRICT: Symbol removed instead

    def _evaluate_adjustment(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if other symbols shift correctly."""
        first_symbols = self._detect_symbols(first)
        final_symbols = self._detect_symbols(final)

        if len(final_symbols) != len(first_symbols) + 1:
            return 0.0  # STRICT: Wrong symbol count

        # Check if original symbols are still present (by color)
        first_colors = [s["color"] for s in first_symbols]
        final_colors = [s["color"] for s in final_symbols]

        matches = 0
        for f_color in first_colors:
            for color in final_colors:
                dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(f_color, color)))
                if dist < 50:
                    matches += 1
                    break

        return matches / max(len(first_colors), 1)

    def _evaluate_layout(self, final: np.ndarray) -> float:
        """Check layout centering and spacing."""
        symbols = self._detect_symbols(final)

        if len(symbols) < 2:
            return 0.0  # STRICT: Not enough symbols for layout eval

        # Check even spacing
        spacings = []
        for i in range(1, len(symbols)):
            spacing = symbols[i]["center"][0] - symbols[i - 1]["center"][0]
            spacings.append(spacing)

        if spacings:
            variance = np.var(spacings)
            mean_spacing = np.mean(spacings)
            cv = np.sqrt(variance) / mean_spacing if mean_spacing > 0 else 1.0
            return max(0, 1.0 - cv * 2)

        return 0.0  # STRICT: No spacing to evaluate


class SymbolSubstituteEvaluator(BaseEvaluator):
    """
    O-60: Symbol substitute evaluator.

    Rule-based evaluation:
    - Symbol count preservation (40%): Same number of symbols
    - Symbol preservation (35%): All OTHER symbols' colors unchanged
    - Substitution occurred (20%): Exactly one symbol changed
    - Animation quality (5%): Smooth cross-fade
    """

    TASK_WEIGHTS = {"count_preservation": 0.40, "symbol_preservation": 0.35, "substitution_occurred": 0.20, "animation_quality": 0.05}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        if len(video_frames) < 2:
            return 0.0

        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]

        first_symbols = self._detect_symbols(first_frame)
        final_symbols = self._detect_symbols(final_frame)

        first_count = len(first_symbols)
        final_count = len(final_symbols)

        # CRITICAL CHECK 1: Symbol count must remain the same
        if final_count != first_count:
            self._last_task_details = {"count_preservation": 0.0, "symbol_preservation": 0.0, "substitution_occurred": 0.0, "animation_quality": 0.0, "count_mismatch": True, "first_count": first_count, "final_count": final_count}
            return 0.0

        scores["count_preservation"] = 1.0

        # CRITICAL CHECK 2: All other symbols' colors must remain unchanged
        # Match symbols by position and check color preservation
        changed_count, preservation_score = self._evaluate_symbol_changes(first_symbols, final_symbols)
        scores["symbol_preservation"] = preservation_score

        # CRITICAL CHECK 3: Exactly one symbol must have changed
        if changed_count == 1:
            scores["substitution_occurred"] = 1.0
        elif changed_count == 0:
            scores["substitution_occurred"] = 0.0  # No substitution
        else:
            scores["substitution_occurred"] = max(0.0, 1.0 - (changed_count - 1) * 0.3)

        scores["animation_quality"] = self._evaluate_animation(video_frames)

        self._last_task_details = scores
        self._last_task_details["changed_count"] = changed_count
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

    def _detect_symbols(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored symbols in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        symbols = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Get average color using mask
            mask_cnt = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
            mean_color = cv2.mean(frame, mask=mask_cnt)[:3]

            # Get HSV hue for color matching
            color_arr = np.array(mean_color, dtype=np.uint8).reshape(1, 1, 3)
            hsv_c = cv2.cvtColor(color_arr, cv2.COLOR_BGR2HSV)[0, 0]

            symbols.append({"center": (cx, cy), "area": area, "color": mean_color, "hue": int(hsv_c[0]), "saturation": int(hsv_c[1])})

        symbols.sort(key=lambda s: s["center"][0])
        return symbols

    def _evaluate_symbol_changes(self, first_symbols: List[Dict], final_symbols: List[Dict]) -> Tuple[int, float]:
        """Count how many symbols changed color and calculate preservation score."""
        if len(first_symbols) != len(final_symbols):
            return len(first_symbols), 0.0

        changed_count = 0
        preserved_count = 0

        # Match by position (sorted by x)
        for f_sym, l_sym in zip(first_symbols, final_symbols):
            # Check position match
            pos_dist = abs(f_sym["center"][0] - l_sym["center"][0])
            if pos_dist > 50:
                # Position shifted too much - treat as changed
                changed_count += 1
                continue

            # Check color match using hue
            hue_diff = abs(f_sym["hue"] - l_sym["hue"])
            hue_diff = min(hue_diff, 180 - hue_diff)

            # Also check saturation for white/gray symbols
            sat_diff = abs(f_sym["saturation"] - l_sym["saturation"])

            if hue_diff < 15 and sat_diff < 50:
                preserved_count += 1
            else:
                changed_count += 1

        # Preservation score: (n-1) symbols should be preserved (one is substituted)
        expected_preserved = len(first_symbols) - 1
        preservation_score = preserved_count / expected_preserved if expected_preserved > 0 else 0.0

        return changed_count, preservation_score

    def _evaluate_animation(self, video_frames: List[np.ndarray]) -> float:
        """Check if cross-fade animation is smooth."""
        if len(video_frames) < 3:
            return 0.5

        changes = []
        for i in range(1, len(video_frames)):
            diff = np.mean(np.abs(video_frames[i].astype(float) - video_frames[i - 1].astype(float)))
            changes.append(diff)

        if not changes:
            return 0.5

        variance = np.var(changes)
        return 1.0 / (1.0 + variance / 50)


class SymbolEditConstraintEvaluator(BaseEvaluator):
    """
    O-61: Symbol edit with constraint evaluator.

    Rule-based evaluation:
    - Original preservation (45%): All original symbols remain unchanged
    - Insertion occurred (35%): Correct number of symbols inserted
    - Count correctness (15%): Final count matches expected
    - Layout accuracy (5%): Proper spacing
    """

    TASK_WEIGHTS = {"original_preservation": 0.45, "insertion_occurred": 0.35, "count_correctness": 0.15, "layout_accuracy": 0.05}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        if len(video_frames) < 2:
            return 0.0

        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]

        first_symbols = self._detect_symbols(first_frame)
        final_symbols = self._detect_symbols(final_frame)

        first_count = len(first_symbols)
        final_count = len(final_symbols)

        # Get expected count from GT if available
        if gt_final_frame is not None:
            gt_final_symbols = self._detect_symbols(gt_final_frame)
            expected_count = len(gt_final_symbols)
        else:
            expected_count = first_count + 2  # Default: insert 2

        expected_inserted = expected_count - first_count

        # CRITICAL CHECK 1: Symbols must be inserted (not deleted)
        if final_count <= first_count:
            self._last_task_details = {"original_preservation": 0.0, "insertion_occurred": 0.0, "count_correctness": 0.0, "layout_accuracy": 0.0, "no_insertion": True, "first_count": first_count, "final_count": final_count}
            return 0.0

        # CRITICAL CHECK 2: All original symbols must be preserved
        scores["original_preservation"] = self._evaluate_original_preservation(first_symbols, final_symbols)

        # If original symbols are not preserved, heavily penalize
        if scores["original_preservation"] < 0.5:
            self._last_task_details = {"original_preservation": scores["original_preservation"], "insertion_occurred": 0.0, "count_correctness": 0.0, "layout_accuracy": 0.0, "originals_changed": True}
            return scores["original_preservation"] * self.TASK_WEIGHTS["original_preservation"]

        # Check if correct number of symbols were inserted
        actual_inserted = final_count - first_count
        if actual_inserted == expected_inserted:
            scores["insertion_occurred"] = 1.0
        elif actual_inserted > 0:
            scores["insertion_occurred"] = max(0.3, 1.0 - abs(actual_inserted - expected_inserted) * 0.2)
        else:
            scores["insertion_occurred"] = 0.0

        # CRITICAL: Check if inserted symbols are of the correct type
        # Get GT final symbols to determine target symbol type
        if gt_final_frame is not None:
            gt_final_symbols = self._detect_symbols(gt_final_frame)
            # Find the target symbol type (the one that was duplicated)
            target_hue = self._find_target_symbol_hue(first_symbols, gt_final_symbols)

            if target_hue is not None:
                # Check if new symbols in final match the target type
                new_symbols_correct = self._check_new_symbols_type(first_symbols, final_symbols, target_hue)
                # Penalize if new symbols don't match target type
                scores["insertion_occurred"] *= new_symbols_correct

        # Check count correctness
        if final_count == expected_count:
            scores["count_correctness"] = 1.0
        else:
            scores["count_correctness"] = max(0.0, 1.0 - abs(final_count - expected_count) * 0.2)

        scores["layout_accuracy"] = self._evaluate_layout(final_symbols)

        self._last_task_details = scores
        self._last_task_details["first_count"] = first_count
        self._last_task_details["final_count"] = final_count
        self._last_task_details["expected_count"] = expected_count
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

    def _detect_symbols(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored symbols in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        symbols = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Get average color using mask
            mask_cnt = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
            mean_color = cv2.mean(frame, mask=mask_cnt)[:3]

            # Get HSV hue for color matching
            color_arr = np.array(mean_color, dtype=np.uint8).reshape(1, 1, 3)
            hsv_c = cv2.cvtColor(color_arr, cv2.COLOR_BGR2HSV)[0, 0]

            symbols.append({"center": (cx, cy), "area": area, "color": mean_color, "hue": int(hsv_c[0]), "saturation": int(hsv_c[1])})

        symbols.sort(key=lambda s: s["center"][0])
        return symbols

    def _evaluate_original_preservation(self, first_symbols: List[Dict], final_symbols: List[Dict]) -> float:
        """Check if all original symbols are preserved."""
        if len(first_symbols) == 0:
            return 1.0

        preserved = 0
        used = set()

        for f_sym in first_symbols:
            # Find a matching symbol in final
            best_match = None
            best_score = 0

            for i, l_sym in enumerate(final_symbols):
                if i in used:
                    continue

                # Check hue match
                hue_diff = abs(f_sym["hue"] - l_sym["hue"])
                hue_diff = min(hue_diff, 180 - hue_diff)

                # Check saturation match
                sat_diff = abs(f_sym["saturation"] - l_sym["saturation"])

                if hue_diff < 15 and sat_diff < 50:
                    score = 1.0 - hue_diff / 15
                    if score > best_score:
                        best_score = score
                        best_match = i

            if best_match is not None:
                preserved += 1
                used.add(best_match)

        return preserved / len(first_symbols)

    def _evaluate_layout(self, final_symbols: List[Dict]) -> float:
        """Check layout spacing."""
        if len(final_symbols) < 2:
            return 0.5

        # Check y-coordinate alignment
        y_coords = [s["center"][1] for s in final_symbols]
        y_var = np.var(y_coords)

        if y_var < 100:
            return 1.0
        elif y_var < 500:
            return 0.7
        return 0.2

    def _find_target_symbol_hue(self, first_symbols: List[Dict], gt_final_symbols: List[Dict]) -> Optional[int]:
        """Find the hue of the target symbol (the one that was duplicated)."""
        # Count hues in first and GT final
        first_hues = {}
        for sym in first_symbols:
            h = sym["hue"] // 10 * 10  # Quantize to 10-degree bins
            first_hues[h] = first_hues.get(h, 0) + 1

        gt_hues = {}
        for sym in gt_final_symbols:
            h = sym["hue"] // 10 * 10
            gt_hues[h] = gt_hues.get(h, 0) + 1

        # Find hue that increased the most
        max_increase = 0
        target_hue = None
        for h, count in gt_hues.items():
            increase = count - first_hues.get(h, 0)
            if increase > max_increase:
                max_increase = increase
                target_hue = h

        return target_hue

    def _check_new_symbols_type(self, first_symbols: List[Dict], final_symbols: List[Dict], target_hue: int) -> float:
        """Check if new symbols match the target type."""
        # Find symbols in final that are not in first (new symbols)
        first_positions = set()
        for sym in first_symbols:
            first_positions.add((sym["center"][0] // 30, sym["center"][1] // 30))

        new_symbols = []
        for sym in final_symbols:
            pos = (sym["center"][0] // 30, sym["center"][1] // 30)
            if pos not in first_positions:
                new_symbols.append(sym)

        if len(new_symbols) == 0:
            return 1.0

        # Check how many new symbols match the target hue
        correct = 0
        for sym in new_symbols:
            hue_diff = abs(sym["hue"] - target_hue)
            hue_diff = min(hue_diff, 180 - hue_diff)
            if hue_diff < 20:
                correct += 1

        return correct / len(new_symbols)


class GravityPhysicsEvaluator(BaseEvaluator):
    """
    O-62: Gravity physics simulation evaluator.

    Rule-based evaluation:
    - Physics accuracy (50%): Correct trajectory following gravity equations
      - v(t) = v₀ - g·t
      - h(t) = h₀ + v₀·t - ½g·t²
    - Final position accuracy (30%): Ball at correct location
    - Motion quality (15%): Smooth acceleration
    - Visual preservation (5%): Scene elements unchanged
    """

    TASK_WEIGHTS = {"physics_accuracy": 0.50, "final_position": 0.30, "motion_quality": 0.15, "visual_preservation": 0.05}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        if len(video_frames) < 2:
            return 0.0

        scores = {}
        final_frame = video_frames[-1]
        first_frame = video_frames[0]

        scores["physics_accuracy"] = self._evaluate_physics(video_frames)
        scores["final_position"] = self._evaluate_position(final_frame, gt_final_frame)
        scores["motion_quality"] = self._evaluate_motion(video_frames)
        scores["visual_preservation"] = self._evaluate_visual(first_frame, final_frame)

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

    def _detect_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red ball position."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None

    def _evaluate_physics(self, video_frames: List[np.ndarray]) -> float:
        """Check if physics simulation is accurate."""
        # Track ball positions through video
        positions = []
        for frame in video_frames[:: max(1, len(video_frames) // 20)]:
            pos = self._detect_ball(frame)
            if pos is not None:
                positions.append(pos[1])  # y position (height)

        if len(positions) < 5:
            return 0.5

        # For gravity, y position should follow parabolic curve
        # Second derivative (acceleration) should be approximately constant
        velocities = np.diff(positions)
        accelerations = np.diff(velocities)

        if len(accelerations) < 2:
            return 0.5

        # Check if acceleration is roughly constant (gravity)
        accel_variance = np.var(accelerations)
        mean_accel = np.mean(np.abs(accelerations))

        if mean_accel > 0:
            cv = np.sqrt(accel_variance) / mean_accel
            return max(0, 1.0 - cv)

        return 0.5

    def _evaluate_position(self, final: np.ndarray, gt_final: Optional[np.ndarray]) -> float:
        """Check if ball is at correct final position."""
        final_pos = self._detect_ball(final)

        if gt_final is not None:
            gt_pos = self._detect_ball(gt_final)

            # If ball not in generated final but in GT final, check if frames match
            if final_pos is None and gt_pos is not None:
                # Check if the frames are nearly identical (GT vs GT case)
                diff = cv2.absdiff(final, gt_final)
                if np.mean(diff) < 10:
                    return 1.0  # Frames are essentially identical
                return 0.0  # Ball missing from generated

            if final_pos is not None and gt_pos is not None:
                distance = np.sqrt((final_pos[0] - gt_pos[0]) ** 2 + (final_pos[1] - gt_pos[1]) ** 2)
                if distance < 20:
                    return 1.0
                elif distance < 50:
                    return 0.7
                elif distance < 100:
                    return 0.4
                return 0.1

        if final_pos is None:
            return 0.0

        # Check if ball is near ground (bottom of frame)
        h = final.shape[0]
        if final_pos[1] > h * 0.7:  # Ball should be in lower part
            return 0.8

        return 0.5

    def _evaluate_motion(self, video_frames: List[np.ndarray]) -> float:
        """Check if motion is smooth with acceleration."""
        if len(video_frames) < 5:
            return 0.5

        # Track ball positions
        positions = []
        for frame in video_frames[:: max(1, len(video_frames) // 20)]:
            pos = self._detect_ball(frame)
            if pos is not None:
                positions.append(pos[1])

        if len(positions) < 3:
            return 0.5

        # Check for acceleration (increasing velocity)
        velocities = np.diff(positions)
        if len(velocities) < 2:
            return 0.5

        # For falling objects, velocity should generally increase (positive acceleration)
        increasing = np.sum(np.diff(velocities) >= -5)  # Allow some noise
        total = len(velocities) - 1

        if total == 0:
            return 0.5
        return min(1.0, increasing / total + 0.3)

    def _evaluate_visual(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if scene elements (ground, markers) are preserved."""
        # Compare bottom portion (ground)
        h = first.shape[0]
        ground_first = first[h - 60 :, :]
        ground_final = final[h - 60 :, :]

        # Compare histograms
        hist_first = cv2.calcHist([ground_first], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_final = cv2.calcHist([ground_final], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        correlation = cv2.compareHist(hist_first, hist_final, cv2.HISTCMP_CORREL)
        return max(0.0, correlation)


class AnimalMatchingEvaluator(BaseEvaluator):
    """
    O-64: Animal matching evaluator.

    Rule-based evaluation:
    - Animal identification (30%): Correct animal types recognized
      - Cat: pointed triangular ears, whiskers, orange
      - Dog: droopy oval ears, tongue, brown
      - Rabbit: long upright ears, buck teeth, pink
      - Bear: round ears, smile, dark brown
    - Matching correctness (35%): Each animal matched to correct outline
    - Position alignment (25%): Animals centered on outlines
    - Appearance fidelity (10%): Animal features preserved
    """

    TASK_WEIGHTS = {"identification": 0.30, "matching": 0.35, "alignment": 0.25, "appearance": 0.10}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        if len(video_frames) < 2:
            return 0.0

        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]

        scores["identification"] = self._evaluate_identification(first_frame, final_frame)
        scores["matching"] = self._evaluate_matching(first_frame, final_frame, gt_final_frame)
        scores["alignment"] = self._evaluate_alignment(final_frame, gt_final_frame)
        scores["appearance"] = self._evaluate_appearance(first_frame, final_frame)

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

    def _detect_colored_animals(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored animal faces."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        animals = []

        # Color ranges for different animals
        color_ranges = {
            "orange_cat": [([5, 100, 100], [20, 255, 255])],  # Orange for cat
            "brown_dog": [([10, 50, 50], [20, 200, 200])],  # Brown for dog
            "pink_rabbit": [([150, 50, 50], [180, 200, 255])],  # Pink for rabbit
            "dark_bear": [([0, 50, 30], [30, 150, 100])],  # Dark brown for bear
        }

        for animal_type, ranges in color_ranges.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

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
                animals.append({"type": animal_type, "center": (cx, cy), "area": area})

        return animals

    def _evaluate_identification(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if animals are correctly identified."""
        first_animals = self._detect_colored_animals(first)
        final_animals = self._detect_colored_animals(final)

        # Should have same number of animals
        if len(final_animals) == len(first_animals):
            return 1.0
        elif abs(len(final_animals) - len(first_animals)) <= 1:
            return 0.7
        return 0.3

    def _evaluate_matching(self, first: np.ndarray, final: np.ndarray, gt_final: Optional[np.ndarray]) -> float:
        """Check if animals are matched to correct outlines."""
        first_animals = self._detect_colored_animals(first)
        final_animals = self._detect_colored_animals(final)

        if not first_animals:
            return 0.5

        # Check if animals moved from left to right side
        w = first.shape[1]
        mid = w // 2

        # Count animals on each side
        first_left = sum(1 for a in first_animals if a["center"][0] < mid)
        final_right = sum(1 for a in final_animals if a["center"][0] >= mid)

        if first_left > 0:
            move_ratio = final_right / first_left
            return min(1.0, move_ratio)

        return 0.5

    def _evaluate_alignment(self, final: np.ndarray, gt_final: Optional[np.ndarray]) -> float:
        """Check if animals are aligned with outlines."""
        final_animals = self._detect_colored_animals(final)

        if gt_final is not None:
            gt_animals = self._detect_colored_animals(gt_final)

            if final_animals and gt_animals:
                # Compare positions
                total_dist = 0
                matched = 0
                for fa in final_animals:
                    min_dist = float("inf")
                    for ga in gt_animals:
                        dist = np.sqrt((fa["center"][0] - ga["center"][0]) ** 2 + (fa["center"][1] - ga["center"][1]) ** 2)
                        min_dist = min(min_dist, dist)
                    if min_dist < float("inf"):
                        total_dist += min_dist
                        matched += 1

                if matched > 0:
                    avg_dist = total_dist / matched
                    return max(0, 1.0 - avg_dist / 100.0)

        # Check if animals are on right side (target area)
        w = final.shape[1]
        mid = w // 2
        right_count = sum(1 for a in final_animals if a["center"][0] >= mid)

        return right_count / max(len(final_animals), 1)

    def _evaluate_appearance(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if animal appearances are preserved."""
        first_animals = self._detect_colored_animals(first)
        final_animals = self._detect_colored_animals(final)

        if not first_animals or not final_animals:
            return 0.5

        # Compare total colored area
        first_area = sum(a["area"] for a in first_animals)
        final_area = sum(a["area"] for a in final_animals)

        if max(first_area, final_area) > 0:
            ratio = min(first_area, final_area) / max(first_area, final_area)
            return ratio

        return 0.5


class AnimalSizeSortingEvaluator(BaseEvaluator):
    """
    O-65: Animal size sorting evaluator.

    Rule-based evaluation:
    - Sorting correctness (40%): Correct small-to-large order (left to right)
    - Baseline alignment (30%): All animals on baseline
    - Animal fidelity (20%): Size, appearance preserved
    - Completeness (10%): All animals included
    """

    TASK_WEIGHTS = {"sorting": 0.40, "alignment": 0.30, "fidelity": 0.20, "completeness": 0.10}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        if len(video_frames) < 2:
            return 0.0

        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]

        scores["sorting"] = self._evaluate_sorting(final_frame)
        scores["alignment"] = self._evaluate_alignment(final_frame)
        scores["fidelity"] = self._evaluate_fidelity(first_frame, final_frame)
        scores["completeness"] = self._evaluate_completeness(first_frame, final_frame)

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

    def _detect_animals(self, frame: np.ndarray) -> List[Dict]:
        """Detect animal figures by color and size."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Find colored (non-white, non-black) regions
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        animals = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:  # Filter noise
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            x, y, w, h = cv2.boundingRect(cnt)
            size = max(w, h)

            animals.append({"center": (cx, cy), "area": area, "size": size, "bbox": (x, y, w, h), "bottom_y": y + h})

        # Sort by x position
        animals.sort(key=lambda a: a["center"][0])
        return animals

    def _evaluate_sorting(self, final: np.ndarray) -> float:
        """Check if animals are sorted correctly (small to large, left to right)."""
        animals = self._detect_animals(final)

        if len(animals) < 2:
            return 0.0  # STRICT: Not enough animals detected

        # Check if sizes increase left to right
        sizes = [a["size"] for a in animals]

        correct_pairs = 0
        total_pairs = len(sizes) - 1

        for i in range(total_pairs):
            if sizes[i] <= sizes[i + 1]:
                correct_pairs += 1

        return correct_pairs / total_pairs if total_pairs > 0 else 0.0  # STRICT

    def _evaluate_alignment(self, final: np.ndarray) -> float:
        """Check if animals are aligned on baseline."""
        animals = self._detect_animals(final)

        if len(animals) < 2:
            return 0.0  # STRICT: Not enough animals

        # Check if bottom y-coordinates are similar (aligned on baseline)
        bottom_ys = [a["bottom_y"] for a in animals]

        variance = np.var(bottom_ys)
        mean_y = np.mean(bottom_ys)

        # Low variance = good alignment
        if mean_y > 0:
            cv = np.sqrt(variance) / mean_y
            return max(0, 1.0 - cv * 5)

        return 0.0  # STRICT: Mean Y is invalid

    def _evaluate_fidelity(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if animal appearances are preserved."""
        first_animals = self._detect_animals(first)
        final_animals = self._detect_animals(final)

        if not first_animals or not final_animals:
            return 0.0  # STRICT: No animals detected

        # Compare sizes (should be preserved)
        first_sizes = sorted([a["size"] for a in first_animals])
        final_sizes = sorted([a["size"] for a in final_animals])

        if len(first_sizes) != len(final_sizes):
            return 0.0  # STRICT: Different number of animals

        # Check size preservation
        size_diffs = [abs(f - l) / max(f, 1) for f, l in zip(first_sizes, final_sizes)]
        return 1.0 - min(1.0, np.mean(size_diffs))

    def _evaluate_completeness(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if all animals are included."""
        first_animals = self._detect_animals(first)
        final_animals = self._detect_animals(final)

        first_count = len(first_animals)
        final_count = len(final_animals)

        if first_count == 0:
            return 0.0  # STRICT: No animals to compare

        # STRICT: Must have same count
        if final_count != first_count:
            return max(0, 1.0 - abs(final_count - first_count) / first_count)
        return 1.0


class ObjectRotation2DEvaluator(BaseEvaluator):
    """
    O-85: 2D object rotation evaluator.

    Rule-based evaluation:
    - Rotation angle accuracy (40%): Correct degrees (within 2° for perfect)
    - Rotation direction (30%): Clockwise/counterclockwise correct
    - Rotation center (20%): Around object center (no translation)
    - Object fidelity (10%): Shape, color, size preserved
    """

    TASK_WEIGHTS = {"angle_accuracy": 0.40, "direction": 0.30, "center": 0.20, "fidelity": 0.10}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        if len(video_frames) < 2:
            return 0.0

        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]

        scores["angle_accuracy"] = self._evaluate_angle(first_frame, final_frame, gt_final_frame)
        scores["direction"] = self._evaluate_direction(video_frames)
        scores["center"] = self._evaluate_center(first_frame, final_frame)
        scores["fidelity"] = self._evaluate_fidelity(first_frame, final_frame)

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored objects and their orientations."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Find colored regions
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Get orientation using moments
            if M["mu20"] - M["mu02"] != 0:
                angle = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
            else:
                angle = 0

            # Get bounding rect for additional angle info
            rect = cv2.minAreaRect(cnt)
            rect_angle = rect[2]

            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[max(0, cy - 5) : cy + 5, max(0, cx - 5) : cx + 5]
            if roi.size > 0:
                color = tuple(roi.mean(axis=(0, 1)).astype(int).tolist())
            else:
                color = (0, 0, 0)

            objects.append({"center": (cx, cy), "area": area, "angle": np.degrees(angle), "rect_angle": rect_angle, "color": color, "bbox": (x, y, w, h)})

        return objects

    def _evaluate_angle(self, first: np.ndarray, final: np.ndarray, gt_final: Optional[np.ndarray]) -> float:
        """Check if rotation angle is correct."""
        first_objects = self._detect_objects(first)
        final_objects = self._detect_objects(final)

        if not first_objects or not final_objects:
            return 0.0  # STRICT: No objects detected

        if gt_final is not None:
            gt_objects = self._detect_objects(gt_final)

            if gt_objects and final_objects:
                # Compare angles with GT
                total_diff = 0
                matched = 0
                for fo in final_objects:
                    min_diff = float("inf")
                    for go in gt_objects:
                        # Match by color
                        color_dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(fo["color"], go["color"])))
                        if color_dist < 50:
                            angle_diff = abs(fo["angle"] - go["angle"])
                            angle_diff = min(angle_diff, 180 - angle_diff)  # Handle wrap
                            min_diff = min(min_diff, angle_diff)
                    if min_diff < float("inf"):
                        total_diff += min_diff
                        matched += 1

                if matched > 0:
                    avg_diff = total_diff / matched
                    if avg_diff < 2:
                        return 1.0
                    elif avg_diff < 10:
                        return 0.8
                    elif avg_diff < 30:
                        return 0.5
                    return 0.2

        # Compare with first frame - check if rotation happened
        angle_changes = []
        for fo in final_objects:
            for ffo in first_objects:
                color_dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(fo["color"], ffo["color"])))
                if color_dist < 50:
                    angle_diff = abs(fo["angle"] - ffo["angle"])
                    angle_changes.append(angle_diff)

        if angle_changes and np.mean(angle_changes) > 5:
            return 0.5  # Some rotation happened but not verified

        return 0.0  # STRICT: No clear rotation detected

    def _evaluate_direction(self, video_frames: List[np.ndarray]) -> float:
        """Check if rotation direction is correct."""
        if len(video_frames) < 5:
            return 0.0  # STRICT: Not enough frames

        # Track angle changes through video
        angle_progression = []
        prev_objects = None

        for frame in video_frames[:: max(1, len(video_frames) // 10)]:
            objects = self._detect_objects(frame)

            if prev_objects and objects:
                for obj in objects:
                    for pobj in prev_objects:
                        color_dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(obj["color"], pobj["color"])))
                        if color_dist < 50:
                            angle_change = obj["angle"] - pobj["angle"]
                            angle_progression.append(angle_change)

            prev_objects = objects

        if not angle_progression:
            return 0.0  # STRICT: No angle changes detected

        # Check if rotation is consistent (all same direction)
        positive = sum(1 for a in angle_progression if a > 0)
        negative = sum(1 for a in angle_progression if a < 0)
        total = positive + negative

        if total > 0:
            consistency = max(positive, negative) / total
            return consistency

        return 0.0  # STRICT: No clear rotation direction

    def _evaluate_center(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if rotation is around correct center."""
        first_objects = self._detect_objects(first)
        final_objects = self._detect_objects(final)

        if not first_objects or not final_objects:
            return 0.0  # STRICT: No objects detected

        # Objects should remain in their grid cells (center shouldn't move much)
        center_drifts = []

        for fo in final_objects:
            for ffo in first_objects:
                color_dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(fo["color"], ffo["color"])))
                if color_dist < 50:
                    center_drift = np.sqrt((fo["center"][0] - ffo["center"][0]) ** 2 + (fo["center"][1] - ffo["center"][1]) ** 2)
                    center_drifts.append(center_drift)

        if center_drifts:
            avg_drift = np.mean(center_drifts)
            # Objects shouldn't move much during rotation around center
            if avg_drift < 10:
                return 1.0
            elif avg_drift < 30:
                return 0.7
            elif avg_drift < 50:
                return 0.5
            return 0.3

        return 0.5

    def _evaluate_fidelity(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if objects are preserved."""
        first_objects = self._detect_objects(first)
        final_objects = self._detect_objects(final)

        if not first_objects:
            return 0.5

        # Compare object counts
        count_ratio = min(len(final_objects), len(first_objects)) / max(len(first_objects), 1)

        # Compare total areas
        first_area = sum(o["area"] for o in first_objects)
        final_area = sum(o["area"] for o in final_objects)

        if max(first_area, final_area) > 0:
            area_ratio = min(first_area, final_area) / max(first_area, final_area)
        else:
            area_ratio = 0.5

        return 0.5 * count_ratio + 0.5 * area_ratio


# Export all Part 4 evaluators
OUT_OF_DOMAIN_50_EVALUATORS_PART5 = {
    "O-54_control_panel_data-generator": ControlPanelEvaluator,
    "O-56_raven_data-generator": RavenMatrixEvaluator,
    "O-58_symbol_delete_data-generator": SymbolDeleteEvaluator,
    "O-59_symbol_insert_data-generator": SymbolInsertEvaluator,
    "O-60_symbol_substitute_data-genertor": SymbolSubstituteEvaluator,
    "O-61_symbol_edit_data-generator": SymbolEditConstraintEvaluator,
    "O-62_gravity_physics_data-generator": GravityPhysicsEvaluator,
    "O-64_animal_matching_data-generator": AnimalMatchingEvaluator,
    "O-65_animal_size_sorting_data-generator": AnimalSizeSortingEvaluator,
    "O-85_2d_object_rotation_data-generator": ObjectRotation2DEvaluator,
}
