"""
Specific evaluators for In-Domain_50 tasks (Part 5).
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils import normalize_frame_size, safe_distance
from .base_evaluator import BaseEvaluator


class GridShiftEvaluator(BaseEvaluator):
    """
    O-36: Grid Shift

    Task: Move all colored blocks in NxN grid simultaneously in specified
    direction (up/down/left/right) by specified steps.

    Key evaluation criteria:
    1. Direction correctness (30%) - All blocks move correct direction
    2. Step accuracy (30%) - Exact number of steps moved
    3. Synchronization (20%) - All blocks move together
    4. Position precision (15%) - Final positions correct
    5. Completeness (5%) - All blocks moved, properties preserved
    """

    def __init__(self, device: str = "cuda", task_name: str = ""):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {"direction_correctness": 0.30, "step_accuracy": 0.30, "synchronization": 0.20, "position_precision": 0.15, "completeness": 0.05}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate grid shift movement."""

        if not video_frames or gt_final_frame is None:
            return 0.0

        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame

        # Normalize frame size (handles padding removal + resize)
        if gen_final.shape != gt_final.shape:
            gt_final = normalize_frame_size(gt_final, gen_final)

        # Detect colored blocks in first and final frames
        first_blocks = self._detect_colored_blocks(first_frame)
        gen_final_blocks = self._detect_colored_blocks(gen_final)
        gt_final_blocks = self._detect_colored_blocks(gt_final)

        scores = {}

        # CRITICAL: First check if blocks are preserved (completeness)
        # If blocks change, the whole task fails
        completeness_score = self._evaluate_completeness(first_blocks, gen_final_blocks)

        # Also check pattern preservation
        pattern_score = self._evaluate_block_pattern_preservation(first_frame, gen_final, first_blocks, gen_final_blocks)

        # Combine: blocks must be preserved AND patterns must be unchanged
        block_preserved = min(completeness_score, pattern_score) > 0.5
        scores["completeness"] = min(completeness_score, pattern_score)

        # If blocks are NOT preserved, all other scores should be 0
        if not block_preserved:
            scores["direction_correctness"] = 0.0
            scores["step_accuracy"] = 0.0
            scores["synchronization"] = 0.0
            scores["position_precision"] = 0.0
        else:
            # 1. Direction correctness (30%): Check if blocks moved in correct direction
            direction_score = self._evaluate_direction(first_blocks, gen_final_blocks, gt_final_blocks)
            scores["direction_correctness"] = direction_score

            # 2. Step accuracy (30%): Check if blocks moved correct number of steps
            step_score = self._evaluate_step_accuracy(first_blocks, gen_final_blocks, gt_final_blocks, gen_final)
            scores["step_accuracy"] = step_score

            # 3. Synchronization (20%): Check if all blocks moved together
            sync_score = self._evaluate_synchronization(video_frames)
            scores["synchronization"] = sync_score

            # 4. Position precision (15%): Check final block positions
            position_score = self._evaluate_position_precision(gen_final_blocks, gt_final_blocks)
            scores["position_precision"] = position_score

        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)

    def _detect_colored_blocks(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored blocks in the frame."""
        blocks = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Define color ranges for common block colors (lower saturation threshold)
        color_ranges = {
            "red": ([0, 50, 50], [10, 255, 255], [160, 50, 50], [180, 255, 255]),
            "green": ([35, 50, 50], [85, 255, 255], None, None),
            "blue": ([100, 50, 50], [130, 255, 255], None, None),
            "yellow": ([20, 50, 50], [35, 255, 255], None, None),
            "orange": ([10, 50, 50], [20, 255, 255], None, None),
            "purple": ([130, 50, 50], [160, 255, 255], None, None),
            "cyan": ([85, 50, 50], [100, 255, 255], None, None),
        }

        detected_centers = set()  # Avoid duplicates

        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 200:  # Filter noise
                    continue

                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Avoid duplicates
                    center_key = (cx // 20, cy // 20)
                    if center_key in detected_centers:
                        continue
                    detected_centers.add(center_key)

                    x, y, w, h = cv2.boundingRect(contour)

                    blocks.append({"color": color_name, "center": (cx, cy), "bbox": (x, y, w, h), "area": area})

        # Also detect gray/neutral blocks (low saturation, medium value)
        if not blocks:
            # Look for non-white, non-black regions
            non_white = ((gray > 50) & (gray < 220)).astype(np.uint8) * 255
            contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500 or area > 50000:  # Filter noise and background
                    continue

                # Check if roughly square (block-like)
                x, y, w, h = cv2.boundingRect(contour)
                aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 10
                if aspect > 2:  # Not square enough
                    continue

                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    blocks.append({"color": "gray", "center": (cx, cy), "bbox": (x, y, w, h), "area": area})

        return blocks

    def _evaluate_direction(self, first_blocks: List[Dict], gen_blocks: List[Dict], gt_blocks: List[Dict]) -> float:
        """Evaluate if blocks moved in correct direction."""
        if not first_blocks or not gen_blocks or not gt_blocks:
            return 0.0

        # Calculate expected movement direction from GT
        gt_movements = []
        for fb in first_blocks:
            # Find matching GT block by color
            for gtb in gt_blocks:
                if fb["color"] == gtb["color"]:
                    dx = float(gtb["center"][0]) - float(fb["center"][0])
                    dy = float(gtb["center"][1]) - float(fb["center"][1])
                    gt_movements.append((dx, dy))
                    break

        if not gt_movements:
            return 0.5

        # Determine expected direction
        avg_dx = np.mean([m[0] for m in gt_movements])
        avg_dy = np.mean([m[1] for m in gt_movements])

        # Calculate actual movement
        gen_movements = []
        for fb in first_blocks:
            for gb in gen_blocks:
                if fb["color"] == gb["color"]:
                    dx = float(gb["center"][0]) - float(fb["center"][0])
                    dy = float(gb["center"][1]) - float(fb["center"][1])
                    gen_movements.append((dx, dy))
                    break

        if not gen_movements:
            return 0.0

        actual_dx = np.mean([m[0] for m in gen_movements])
        actual_dy = np.mean([m[1] for m in gen_movements])

        # Check direction match
        direction_match = 0.0

        # Check horizontal direction
        if avg_dx != 0:
            if np.sign(actual_dx) == np.sign(avg_dx):
                direction_match += 0.5
        else:
            if abs(actual_dx) < 10:  # No horizontal movement expected
                direction_match += 0.5

        # Check vertical direction
        if avg_dy != 0:
            if np.sign(actual_dy) == np.sign(avg_dy):
                direction_match += 0.5
        else:
            if abs(actual_dy) < 10:  # No vertical movement expected
                direction_match += 0.5

        return direction_match

    def _evaluate_step_accuracy(self, first_blocks: List[Dict], gen_blocks: List[Dict], gt_blocks: List[Dict], frame: np.ndarray) -> float:
        """Evaluate if blocks moved correct number of steps."""
        if not first_blocks or not gen_blocks or not gt_blocks:
            return 0.0

        # Estimate grid cell size
        h, w = frame.shape[:2]
        # Assume 4-12 grid, estimate cell size
        estimated_cell_size = w / 8  # Average estimate

        # Calculate expected displacement from GT
        gt_displacements = []
        for fb in first_blocks:
            for gtb in gt_blocks:
                if fb["color"] == gtb["color"]:
                    dx = abs(float(gtb["center"][0]) - float(fb["center"][0]))
                    dy = abs(float(gtb["center"][1]) - float(fb["center"][1]))
                    gt_displacements.append(max(dx, dy))
                    break

        # Calculate actual displacement
        gen_displacements = []
        for fb in first_blocks:
            for gb in gen_blocks:
                if fb["color"] == gb["color"]:
                    dx = abs(float(gb["center"][0]) - float(fb["center"][0]))
                    dy = abs(float(gb["center"][1]) - float(fb["center"][1]))
                    gen_displacements.append(max(dx, dy))
                    break

        if not gt_displacements or not gen_displacements:
            return 0.0

        avg_gt_disp = np.mean(gt_displacements)
        avg_gen_disp = np.mean(gen_displacements)

        if avg_gt_disp < 1:
            return 1.0 if avg_gen_disp < estimated_cell_size * 0.5 else 0.5

        # Calculate step difference
        ratio = avg_gen_disp / avg_gt_disp

        if 0.8 <= ratio <= 1.2:
            return 1.0
        elif 0.5 <= ratio <= 1.5:
            return 0.7
        elif 0.3 <= ratio <= 2.0:
            return 0.4
        else:
            return 0.2

    def _evaluate_synchronization(self, frames: List[np.ndarray]) -> float:
        """Check if all blocks move synchronously."""
        if len(frames) < 3:
            return 0.5

        # Track block positions through video
        n_samples = min(10, len(frames))
        sample_indices = np.linspace(0, len(frames) - 1, n_samples, dtype=int)

        all_positions = []
        for idx in sample_indices:
            blocks = self._detect_colored_blocks(frames[idx])
            if blocks:
                positions = [b["center"] for b in blocks]
                all_positions.append(positions)

        if len(all_positions) < 3:
            return 0.5

        # Check if all blocks move together (similar displacement at each frame)
        sync_scores = []
        for i in range(1, len(all_positions)):
            if len(all_positions[i]) != len(all_positions[i - 1]):
                continue

            displacements = []
            for j in range(len(all_positions[i])):
                dx = all_positions[i][j][0] - all_positions[i - 1][j][0]
                dy = all_positions[i][j][1] - all_positions[i - 1][j][1]
                displacements.append((dx, dy))

            if len(displacements) > 1:
                # Check variance in displacements
                dx_var = np.var([d[0] for d in displacements])
                dy_var = np.var([d[1] for d in displacements])

                # Low variance means synchronized movement
                max_var = max(dx_var, dy_var)
                if max_var < 100:
                    sync_scores.append(1.0)
                elif max_var < 500:
                    sync_scores.append(0.7)
                else:
                    sync_scores.append(0.3)

        return np.mean(sync_scores) if sync_scores else 0.5

    def _evaluate_position_precision(self, gen_blocks: List[Dict], gt_blocks: List[Dict]) -> float:
        """Evaluate final position accuracy."""
        if not gen_blocks or not gt_blocks:
            return 0.0

        matched_scores = []

        for gtb in gt_blocks:
            best_dist = float("inf")
            for gb in gen_blocks:
                if gb["color"] == gtb["color"]:
                    dist = safe_distance(gb["center"], gtb["center"])
                    best_dist = min(best_dist, dist)

            if best_dist < float("inf"):
                # Score based on distance
                if best_dist < 10:
                    matched_scores.append(1.0)
                elif best_dist < 30:
                    matched_scores.append(0.8)
                elif best_dist < 50:
                    matched_scores.append(0.5)
                else:
                    matched_scores.append(max(0.1, 1.0 - best_dist / 100))

        return np.mean(matched_scores) if matched_scores else 0.0

    def _evaluate_completeness(self, first_blocks: List[Dict], gen_blocks: List[Dict]) -> float:
        """Evaluate if all blocks are preserved with same colors."""
        if not first_blocks:
            return 0.0

        if not gen_blocks:
            return 0.0

        # Check if same number of blocks
        if len(gen_blocks) != len(first_blocks):
            return 0.0  # Block count changed - STRICT failure

        # Check if all block colors are preserved
        first_colors = sorted([b["color"] for b in first_blocks])
        gen_colors = sorted([b["color"] for b in gen_blocks])

        if first_colors != gen_colors:
            return 0.0  # Block colors changed - STRICT failure

        return 1.0  # All blocks preserved with same colors

    def _evaluate_block_pattern_preservation(self, first_frame: np.ndarray, gen_final: np.ndarray, first_blocks: List[Dict], gen_blocks: List[Dict]) -> float:
        """Check if block patterns/content remain unchanged during shift."""
        if not first_blocks or not gen_blocks:
            return 0.0

        # For each block in first frame, extract its appearance
        # and compare with corresponding block in final frame
        preservation_scores = []

        for fb in first_blocks:
            # Find matching block by color in gen_blocks
            matching_gb = None
            for gb in gen_blocks:
                if gb["color"] == fb["color"]:
                    matching_gb = gb
                    break

            if matching_gb is None:
                preservation_scores.append(0.0)
                continue

            # Extract block regions
            fx, fy, fw, fh = fb["bbox"]
            gx, gy, gw, gh = matching_gb["bbox"]

            # Get block regions
            first_region = first_frame[fy : fy + fh, fx : fx + fw]
            gen_region = gen_final[gy : gy + gh, gx : gx + gw]

            # Resize to same size for comparison
            if first_region.size > 0 and gen_region.size > 0:
                target_size = (max(fw, gw), max(fh, gh))
                first_resized = cv2.resize(first_region, target_size)
                gen_resized = cv2.resize(gen_region, target_size)

                # Compare patterns
                diff = np.abs(first_resized.astype(float) - gen_resized.astype(float)).mean()

                if diff < 30:  # Very similar
                    preservation_scores.append(1.0)
                elif diff < 60:
                    preservation_scores.append(0.5)
                else:
                    preservation_scores.append(0.0)  # Pattern changed
            else:
                preservation_scores.append(0.0)

        return np.mean(preservation_scores) if preservation_scores else 0.0


class LightSequenceEvaluator(BaseEvaluator):
    """
    O-37: Light Sequence State Control

    Task: Modify light states according to spatial/mathematical rules.
    Lights can be on (gold) or off (gray). 6 rule types.

    Key evaluation criteria:
    1. Rule understanding (35%) - Correct interpretation of rule type
    2. Position identification (30%) - Correct light positions identified
    3. State transition (25%) - Correct on/off states
    4. Visual quality (10%) - Colors and glow effects
    """

    def __init__(self, device: str = "cuda", task_name: str = ""):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {"rule_understanding": 0.35, "position_identification": 0.30, "state_transition": 0.25, "visual_quality": 0.10}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate light sequence state control - RULE-BASED comparison."""
        if not video_frames or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        gt_first = gt_first_frame

        # Normalize frame size (handles padding removal + resize)
        if gen_final.shape != gt_final.shape:
            gen_final = normalize_frame_size(gen_final, gt_final)

        scores = {}

        # RULE-BASED: Detect light positions from GT first frame, then check states
        # Step 1: Find all light positions from GT first frame
        light_positions = self._detect_all_light_positions(gt_first)

        if len(light_positions) == 0:
            # Fallback to pixel comparison if detection fails
            final_diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            scores["rule_understanding"] = 1.0 if final_diff < 10 else 0.0
            scores["position_identification"] = 1.0 if final_diff < 10 else 0.0
            scores["state_transition"] = 1.0 if final_diff < 10 else 0.0
            scores["visual_quality"] = 1.0 if final_diff < 15 else 0.0
            self._last_task_details = scores
            return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)

        # Step 2: Detect which lights are ON in GT final (expected states)
        gt_on_states = self._get_light_states(gt_final, light_positions)

        # Step 3: Detect which lights are ON in generated final
        gen_on_states = self._get_light_states(gen_final, light_positions)

        # Step 4: Compare states - STRICT rule-based
        # Count matching states
        matching_states = sum(1 for g, e in zip(gen_on_states, gt_on_states) if g == e)
        total_lights = len(light_positions)

        state_accuracy = matching_states / total_lights if total_lights > 0 else 0

        # All scores depend on state accuracy
        # If states don't match, the rule was not followed correctly
        if state_accuracy == 1.0:  # Perfect match
            scores["rule_understanding"] = 1.0
            scores["position_identification"] = 1.0
            scores["state_transition"] = 1.0
            scores["visual_quality"] = 1.0
        elif state_accuracy >= 0.8:  # Minor errors
            scores["rule_understanding"] = 0.5
            scores["position_identification"] = 0.5
            scores["state_transition"] = 0.5
            scores["visual_quality"] = 0.8
        else:  # Wrong states
            scores["rule_understanding"] = 0.0
            scores["position_identification"] = 0.0
            scores["state_transition"] = 0.0
            scores["visual_quality"] = 0.3

        self._last_task_details = scores
        self._last_task_details["gt_on_states"] = str(gt_on_states)
        self._last_task_details["gen_on_states"] = str(gen_on_states)
        self._last_task_details["state_accuracy"] = state_accuracy

        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)

    def _detect_all_light_positions(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect all light positions (both on and off) from the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find non-white regions (lights are colored, background is white)
        non_white = (gray < 250).astype(np.uint8) * 255

        contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        positions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Filter noise
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    positions.append((cx, cy))

        # Sort by x position (left to right)
        positions.sort(key=lambda p: p[0])
        return positions

    def _get_light_states(self, frame: np.ndarray, positions: List[Tuple[int, int]]) -> List[bool]:
        """Get ON/OFF state for each light position."""
        states = []
        for cx, cy in positions:
            # Sample color at light center
            y1, y2 = max(0, cy - 10), min(frame.shape[0], cy + 10)
            x1, x2 = max(0, cx - 10), min(frame.shape[1], cx + 10)
            region = frame[y1:y2, x1:x2]

            if region.size > 0:
                mean_color = np.mean(region, axis=(0, 1))
                b, g, r = mean_color
                # Gold/yellow: high R, high G, low B
                is_on = r > 180 and g > 100 and b < 150
                states.append(is_on)
            else:
                states.append(False)

        return states

    def _detect_light_states(self, frame: np.ndarray) -> List[Dict]:
        """Detect lights and their on/off states using non-white region detection."""
        lights = []

        # Find all non-white regions (lights are colored, background is white)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        non_white = (gray < 250).astype(np.uint8) * 255

        contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:  # Skip very small regions
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Get average color in the region
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_color = cv2.mean(frame, mask=mask)[:3]

            b_val, g_val, r_val = mean_color[0], mean_color[1], mean_color[2]

            # Determine if ON (gold/orange/yellow) or OFF (gray)
            # Gold/orange: high R, high G, low B (BGR format)
            # Gray: similar R, G, B values with low saturation
            color_diff = max(abs(r_val - g_val), abs(g_val - b_val), abs(r_val - b_val))

            # ON lights have high color difference (gold/orange is saturated)
            # OFF lights have low color difference (gray is desaturated)
            is_on = (r_val > 200 and g_val > 150 and b_val < 100) or (r_val > 220 and g_val > 180) or (color_diff > 50 and r_val > 180)  # Saturated warm color

            lights.append({"center": (cx, cy), "area": area, "is_on": is_on, "color": mean_color})

        # Sort by x position (left to right)
        lights.sort(key=lambda l: l["center"][0])
        return lights

    def _detect_lights_by_color(self, frame: np.ndarray) -> List[Dict]:
        """Fallback detection by color."""
        lights = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Gold/yellow/orange detection - expanded range
        # Hue: 10-45 for yellow/gold/orange
        lower_gold = np.array([10, 80, 120])
        upper_gold = np.array([45, 255, 255])
        gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)

        # Also detect by RGB for gold colors that may not be well captured in HSV
        # Gold: RGB(255,215,0), RGB(255,165,0)
        b, g, r = cv2.split(frame)
        rgb_gold_mask = ((r > 180) & (g > 100) & (b < 150)).astype(np.uint8) * 255
        gold_mask = cv2.bitwise_or(gold_mask, rgb_gold_mask)

        # Gray detection
        lower_gray = np.array([0, 0, 60])
        upper_gray = np.array([180, 60, 200])
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)

        # Find gold lights (ON)
        contours, _ = cv2.findContours(gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Lower threshold
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    lights.append({"center": (cx, cy), "is_on": True, "area": area})

        # Find gray lights (OFF)
        contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Lower threshold
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter**2)
                    if circularity > 0.4:  # More lenient circularity
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            # Check not already added as gold
                            is_new = all(abs(l["center"][0] - cx) > 15 for l in lights)
                            if is_new:
                                lights.append({"center": (cx, cy), "is_on": False, "area": area})

        lights.sort(key=lambda l: l["center"][0])
        return lights

    def _evaluate_rule_understanding(self, gen_lights: List[Dict], gt_lights: List[Dict]) -> float:
        """Evaluate if the rule was correctly understood and applied."""
        if not gt_lights:
            return 0.5

        # Get on/off pattern
        gt_pattern = [l["is_on"] for l in gt_lights]
        gen_pattern = [l["is_on"] for l in gen_lights]

        if len(gen_pattern) != len(gt_pattern):
            # Different number of lights
            return max(0.0, max(0, 1.0 - abs(len(gen_pattern) - len(gt_pattern)) / len(gt_pattern)))

        # Compare patterns
        matches = sum(1 for g, gt in zip(gen_pattern, gt_pattern) if g == gt)
        return matches / len(gt_pattern)

    def _evaluate_position_identification(self, gen_lights: List[Dict], gt_lights: List[Dict]) -> float:
        """Evaluate if lights are in correct positions."""
        if not gt_lights or not gen_lights:
            return 0.0 if not gen_lights else 0.5

        # Compare positions
        position_scores = []
        for gt_l in gt_lights:
            best_dist = float("inf")
            for gen_l in gen_lights:
                dist = safe_distance(gt_l["center"], gen_l["center"])
                best_dist = min(best_dist, dist)

            if best_dist < 20:
                position_scores.append(1.0)
            elif best_dist < 50:
                position_scores.append(0.7)
            else:
                position_scores.append(max(0.2, 1.0 - best_dist / 100))

        return np.mean(position_scores) if position_scores else 0.0

    def _evaluate_state_transition(self, gen_lights: List[Dict], gt_lights: List[Dict]) -> float:
        """Evaluate on/off state accuracy."""
        if not gt_lights or not gen_lights:
            return 0.0

        # Match lights by position and compare states
        correct_states = 0
        total = len(gt_lights)

        for gt_l in gt_lights:
            # Find closest generated light
            best_match = None
            best_dist = float("inf")

            for gen_l in gen_lights:
                dist = safe_distance(gt_l["center"], gen_l["center"])
                if dist < best_dist:
                    best_dist = dist
                    best_match = gen_l

            if best_match and best_dist < 50:
                if best_match["is_on"] == gt_l["is_on"]:
                    correct_states += 1

        return correct_states / total if total > 0 else 0.0

    def _evaluate_light_visual_quality(self, gen_frame: np.ndarray, gt_frame: np.ndarray, gen_lights: List[Dict], gt_lights: List[Dict]) -> float:
        """Evaluate visual quality of lights (renamed to avoid conflict with base class)."""
        if not gen_lights:
            return 0.0

        # Check if ON lights have gold/yellow color
        on_lights = [l for l in gen_lights if l.get("is_on", False)]
        off_lights = [l for l in gen_lights if not l.get("is_on", False)]

        quality_scores = []

        # Check ON lights are gold/yellow
        for light in on_lights:
            if "color" in light:
                r, g, b = light["color"][2], light["color"][1], light["color"][0]
                # Gold should be high R, high G, low B
                if r > 180 and g > 150 and b < 150:
                    quality_scores.append(1.0)
                elif r > 150 and g > 120:
                    quality_scores.append(0.7)
                else:
                    quality_scores.append(0.3)

        return np.mean(quality_scores) if quality_scores else 0.5


class MajorityColorEvaluator(BaseEvaluator):
    """
    O-38: Majority Color Identification

    CRITICAL RULES:
    1. All objects size and shape must NOT change
    2. All objects should change to the SAME color (majority color from first frame)
    3. Final frame should have only ONE color (the majority color)
    """

    def __init__(self, device: str = "cuda", task_name: str = ""):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {"shapes_preserved": 0.30, "single_color": 0.55, "correct_majority": 0.15}  # Same number of shapes  # Only one color in final - MOST IMPORTANT  # Correct majority color

    def _count_total_shapes(self, frame: np.ndarray) -> int:
        """Count total number of colored shapes."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        sat_mask = (sat > 50).astype(np.uint8) * 255
        contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len([c for c in contours if cv2.contourArea(c) > 200])

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate majority color identification.

        CRITICAL RULES:
        1. Shape count must be preserved
        2. All shapes should become ONE color (majority)
        """

        if not video_frames or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame

        # Normalize frame size (handles padding removal + resize)
        if gen_final.shape != gt_final.shape:
            gt_final = normalize_frame_size(gt_final, gen_final)

        scores = {}

        # 1. CRITICAL: Shape count must be preserved
        first_shape_count = self._count_total_shapes(first_frame)
        final_shape_count = self._count_total_shapes(gen_final)

        if first_shape_count > 0:
            count_change = abs(final_shape_count - first_shape_count) / first_shape_count
            if count_change > 0.5:
                # Shapes changed significantly
                scores["shapes_preserved"] = 0.0
            else:
                scores["shapes_preserved"] = max(0, 1.0 - count_change)
        else:
            scores["shapes_preserved"] = 0.0

        # 2. Final frame should have only ONE color
        gen_final_colors = self._count_shapes_by_color(gen_final)

        if len(gen_final_colors) == 0:
            scores["single_color"] = 0.0
        elif len(gen_final_colors) == 1:
            scores["single_color"] = 1.0
        else:
            # Multiple colors - penalize
            total_shapes = sum(gen_final_colors.values())
            max_color_count = max(gen_final_colors.values())
            scores["single_color"] = max_color_count / total_shapes * 0.5  # Max 0.5 if not single

        # 3. Check if correct majority color
        initial_colors = self._count_shapes_by_color(first_frame)
        gt_final_colors = self._count_shapes_by_color(gt_final)

        if initial_colors and gt_final_colors:
            majority_color = max(initial_colors.items(), key=lambda x: x[1])[0]

            if majority_color in gen_final_colors:
                scores["correct_majority"] = 1.0
            else:
                scores["correct_majority"] = 0.0
        else:
            scores["correct_majority"] = 0.0

        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)

    def _count_shapes_by_color(self, frame: np.ndarray) -> Dict[str, int]:
        """Count shapes by color."""
        color_counts = {}
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        color_ranges = {
            "red": ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
            "green": ([35, 100, 100], [85, 255, 255], None, None),
            "blue": ([100, 100, 100], [130, 255, 255], None, None),
            "yellow": ([20, 100, 100], [35, 255, 255], None, None),
            "orange": ([10, 100, 100], [20, 255, 255], None, None),
            "purple": ([130, 100, 100], [160, 255, 255], None, None),
            "cyan": ([85, 100, 100], [100, 255, 255], None, None),
            "pink": ([140, 50, 100], [170, 255, 255], None, None),
        }

        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:  # Filter noise
                    count += 1

            if count > 0:
                color_counts[color_name] = count

        return color_counts

    def _evaluate_majority_identification(self, initial_colors: Dict[str, int], gen_colors: Dict[str, int], gt_colors: Dict[str, int]) -> float:
        """Evaluate if correct majority color was identified."""
        if not gt_colors:
            return 0.0  # STRICT: No GT to compare

        # Find majority color in GT (should be only one color remaining)
        gt_majority = max(gt_colors.items(), key=lambda x: x[1], default=("none", 0))

        # Find dominant color in generated
        if not gen_colors:
            return 0.0

        gen_dominant = max(gen_colors.items(), key=lambda x: x[1], default=("none", 0))

        # Check if same color
        if gt_majority[0] == gen_dominant[0]:
            return 1.0

        # Check if it's close (could be color detection variance)
        if gen_dominant[1] > 0:
            return 0.3

        return 0.0

    def _evaluate_non_majority_removal(self, gen_colors: Dict[str, int], gt_colors: Dict[str, int]) -> float:
        """Evaluate if non-majority colors were removed."""
        if not gt_colors:
            return 0.5

        # GT should have only one color (majority)
        gt_color_count = len([c for c, n in gt_colors.items() if n > 0])
        gen_color_count = len([c for c, n in gen_colors.items() if n > 0])

        if gt_color_count == 1:
            if gen_color_count == 1:
                return 1.0
            elif gen_color_count == 2:
                return 0.6
            else:
                return max(0.2, 1.0 - gen_color_count * 0.2)

        return 0.5

    def _evaluate_majority_preservation(self, gen_colors: Dict[str, int], gt_colors: Dict[str, int]) -> float:
        """Evaluate if all majority color shapes were preserved."""
        if not gt_colors:
            return 0.5

        gt_majority = max(gt_colors.items(), key=lambda x: x[1], default=("none", 0))

        if gt_majority[0] in gen_colors:
            gen_count = gen_colors[gt_majority[0]]
            gt_count = gt_majority[1]

            if gt_count == 0:
                return 0.5

            ratio = gen_count / gt_count

            if 0.9 <= ratio <= 1.1:
                return 1.0
            elif 0.7 <= ratio <= 1.3:
                return 0.7
            else:
                return max(0.2, ratio if ratio < 1 else 2 - ratio)

        return 0.0

    def _evaluate_visual_consistency(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate visual cleanliness."""
        # Check if background is mostly white
        gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        white_ratio = np.sum(gray > 240) / gray.size

        if white_ratio > 0.7:
            return 1.0
        elif white_ratio > 0.5:
            return 0.7
        else:
            return 0.4


class RotationPuzzleEvaluator(BaseEvaluator):
    """
    O-44: Rotation Puzzle (Pipe Connection)

    Task: Rotate L-shaped pipe tiles in 2x2 grid to connect all pipes
    into continuous path.

    Key evaluation criteria:
    1. Path connection (40%) - All pipes connected
    2. Rotation accuracy (30%) - Correct 90° rotations
    3. Position preservation (20%) - Tiles stay in place
    4. Alignment precision (10%) - Pipe openings aligned
    """

    def __init__(self, device: str = "cuda", task_name: str = ""):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {"path_connection": 0.40, "rotation_accuracy": 0.30, "position_preservation": 0.20, "alignment_precision": 0.10}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate rotation puzzle solution."""

        if not video_frames or gt_final_frame is None:
            return 0.0

        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame

        # Normalize frame size (handles padding removal + resize)
        if gen_final.shape != gt_final.shape:
            gt_final = normalize_frame_size(gt_final, gen_final)

        scores = {}

        # 1. Path connection (40%): Check if pipes form connected path
        connection_score = self._evaluate_path_connection(gen_final, gt_final)
        scores["path_connection"] = connection_score

        # 2. Rotation accuracy (30%): Check pipe orientations match GT
        rotation_score = self._evaluate_rotation_accuracy(gen_final, gt_final)
        scores["rotation_accuracy"] = rotation_score

        # 3. Position preservation (20%): Check tiles are in correct positions
        position_score = self._evaluate_position_preservation(first_frame, gen_final)
        scores["position_preservation"] = position_score

        # 4. Alignment precision (10%): Check pipe openings align
        alignment_score = self._evaluate_alignment_precision(gen_final, gt_final)
        scores["alignment_precision"] = alignment_score

        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)

    def _detect_blue_pipes(self, frame: np.ndarray) -> np.ndarray:
        """Detect blue pipe regions."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Blue pipe color
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        return blue_mask

    def _evaluate_path_connection(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if pipes form a connected path."""
        gen_blue = self._detect_blue_pipes(gen_frame)
        gt_blue = self._detect_blue_pipes(gt_frame)

        if np.sum(gt_blue > 0) == 0:
            return 0.5

        # IoU of blue regions
        intersection = np.sum((gen_blue > 0) & (gt_blue > 0))
        union = np.sum((gen_blue > 0) | (gt_blue > 0))

        if union > 0:
            iou = intersection / union
            return iou

        return 0.5

    def _evaluate_rotation_accuracy(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if pipe orientations match GT."""
        # Divide frame into 2x2 quadrants and compare pipe orientations
        h, w = gen_frame.shape[:2]

        quadrant_scores = []
        for row in range(2):
            for col in range(2):
                y1, y2 = row * h // 2, (row + 1) * h // 2
                x1, x2 = col * w // 2, (col + 1) * w // 2

                gen_quad = gen_frame[y1:y2, x1:x2]
                gt_quad = gt_frame[y1:y2, x1:x2]

                gen_blue = self._detect_blue_pipes(gen_quad)
                gt_blue = self._detect_blue_pipes(gt_quad)

                if np.sum(gt_blue > 0) > 0:
                    intersection = np.sum((gen_blue > 0) & (gt_blue > 0))
                    gt_area = np.sum(gt_blue > 0)

                    quadrant_scores.append(intersection / gt_area)

        return np.mean(quadrant_scores) if quadrant_scores else 0.5

    def _evaluate_position_preservation(self, first_frame: np.ndarray, gen_final: np.ndarray) -> float:
        """Evaluate if tiles stayed in their grid positions."""
        # Detect tile boundaries in both frames
        h, w = first_frame.shape[:2]

        # Check if 2x2 grid structure is preserved
        # Look for vertical and horizontal dividing lines

        gen_gray = cv2.cvtColor(gen_final, cv2.COLOR_BGR2GRAY)
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # Check center regions for grid lines
        center_y = h // 2
        center_x = w // 2

        # Sample around center for grid line detection
        gen_center_h = gen_gray[center_y - 5 : center_y + 5, :]
        gen_center_v = gen_gray[:, center_x - 5 : center_x + 5]

        first_center_h = first_gray[center_y - 5 : center_y + 5, :]
        first_center_v = first_gray[:, center_x - 5 : center_x + 5]

        # Compare patterns
        h_diff = np.mean(np.abs(gen_center_h.astype(float) - first_center_h.astype(float)))
        v_diff = np.mean(np.abs(gen_center_v.astype(float) - first_center_v.astype(float)))

        avg_diff = (h_diff + v_diff) / 2

        if avg_diff < 20:
            return 1.0
        elif avg_diff < 50:
            return 0.7
        else:
            return max(0.3, 1.0 - avg_diff / 100)

    def _evaluate_alignment_precision(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate pipe opening alignment."""
        gen_blue = self._detect_blue_pipes(gen_frame)
        gt_blue = self._detect_blue_pipes(gt_frame)

        # Check alignment at quadrant boundaries
        h, w = gen_frame.shape[:2]

        # Horizontal boundary (middle row)
        gen_h_boundary = gen_blue[h // 2 - 10 : h // 2 + 10, :]
        gt_h_boundary = gt_blue[h // 2 - 10 : h // 2 + 10, :]

        # Vertical boundary (middle column)
        gen_v_boundary = gen_blue[:, w // 2 - 10 : w // 2 + 10]
        gt_v_boundary = gt_blue[:, w // 2 - 10 : w // 2 + 10]

        h_match = np.sum((gen_h_boundary > 0) & (gt_h_boundary > 0)) / max(1, np.sum(gt_h_boundary > 0))
        v_match = np.sum((gen_v_boundary > 0) & (gt_v_boundary > 0)) / max(1, np.sum(gt_v_boundary > 0))

        return (h_match + v_match) / 2


class SequenceCompletionEvaluator(BaseEvaluator):
    """
    O-45: Sequence Completion

    Task: Observe pattern in sequence (numbers/shapes/colors/directions)
    and replace ? with correct next element.

    Key evaluation criteria:
    1. Sequence type identification (35%) - Correct pattern type
    2. Element calculation (35%) - Correct value computed
    3. Element rendering (20%) - Visual consistency
    4. Sequence integrity (10%) - Complete sequence valid
    """

    def __init__(self, device: str = "cuda", task_name: str = ""):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {"sequence_type_identification": 0.35, "element_calculation": 0.35, "element_rendering": 0.20, "sequence_integrity": 0.10}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate sequence completion accuracy - RULE-BASED."""

        if not video_frames or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        gt_first = gt_first_frame

        # Normalize frame size (handles padding removal + resize)
        if gen_final.shape != gt_final.shape:
            gen_final = normalize_frame_size(gen_final, gt_final)

        scores = {}

        # RULE-BASED: Detect sequence elements and check if answer is correct
        # Step 1: Detect elements in GT first (the pattern)
        gt_first_elements = self._detect_sequence_elements(gt_first)

        # Step 2: Detect elements in GT final (includes correct answer)
        gt_final_elements = self._detect_sequence_elements(gt_final)

        # Step 3: Detect elements in generated final
        gen_final_elements = self._detect_sequence_elements(gen_final)

        # The answer should be the rightmost element in GT final
        # that wasn't in GT first (or the ? position)
        gt_answer_color = None
        if len(gt_final_elements) > len(gt_first_elements):
            gt_answer_color = gt_final_elements[-1][2]  # Color of last element
        elif len(gt_final_elements) > 0:
            gt_answer_color = gt_final_elements[-1][2]

        # Check if generated has the correct answer
        gen_answer_color = None
        answer_added = len(gen_final_elements) >= len(gt_final_elements)
        if len(gen_final_elements) > 0:
            gen_answer_color = gen_final_elements[-1][2]

        # 1. Sequence type identification (35%): Did they add an element?
        if not answer_added:
            scores["sequence_type_identification"] = 0.0  # No answer added
        elif gt_answer_color is not None and gen_answer_color is not None:
            # Check if answer color matches
            color_diff = np.sqrt(sum((a - b) ** 2 for a, b in zip(gt_answer_color, gen_answer_color)))
            if color_diff < 50:
                scores["sequence_type_identification"] = 1.0
            elif color_diff < 100:
                scores["sequence_type_identification"] = 0.3
            else:
                scores["sequence_type_identification"] = 0.0
        else:
            scores["sequence_type_identification"] = 0.0

        # 2. Element calculation (35%): Is the answer color correct?
        if not answer_added:
            scores["element_calculation"] = 0.0
        elif gt_answer_color is not None and gen_answer_color is not None:
            color_diff = np.sqrt(sum((a - b) ** 2 for a, b in zip(gt_answer_color, gen_answer_color)))
            if color_diff < 50:
                scores["element_calculation"] = 1.0
            elif color_diff < 100:
                scores["element_calculation"] = 0.3
            else:
                scores["element_calculation"] = 0.0
        else:
            scores["element_calculation"] = 0.0

        # 3. Element rendering (20%): Is the element count correct?
        if len(gen_final_elements) == len(gt_final_elements):
            scores["element_rendering"] = 1.0
        elif abs(len(gen_final_elements) - len(gt_final_elements)) == 1:
            scores["element_rendering"] = 0.5
        else:
            scores["element_rendering"] = 0.0

        # 4. Sequence integrity (10%): Are original elements preserved?
        if len(gt_first_elements) > 0:
            # Check if first N-1 elements match
            preserved = 0
            for i, (_, _, gt_color) in enumerate(gt_first_elements[:-1]):  # Exclude ? position
                if i < len(gen_final_elements):
                    gen_color = gen_final_elements[i][2]
                    color_diff = np.sqrt(sum((a - b) ** 2 for a, b in zip(gt_color, gen_color)))
                    if color_diff < 80:
                        preserved += 1

            if len(gt_first_elements) > 1:
                scores["sequence_integrity"] = preserved / (len(gt_first_elements) - 1)
            else:
                scores["sequence_integrity"] = 1.0
        else:
            scores["sequence_integrity"] = 0.0  # STRICT: No GT elements

        self._last_task_details = scores
        self._last_task_details["gt_first_count"] = len(gt_first_elements)
        self._last_task_details["gt_final_count"] = len(gt_final_elements)
        self._last_task_details["gen_final_count"] = len(gen_final_elements)

        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)

    def _detect_sequence_elements(self, frame: np.ndarray) -> List[Tuple[int, int, Tuple]]:
        """Detect sequence elements (colored shapes OR numbers) with their colors."""
        elements = []

        # Method 1: Try detecting saturated (colored) regions first
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] > 50).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Filter noise
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Get dominant color
                    mask_cnt = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(frame, mask=mask_cnt)[:3]

                    elements.append((cx, cy, mean_color))

        # Method 2: If no colored elements found, try detecting dark elements (numbers/text)
        if len(elements) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Group nearby contours (digits of same number may be separate)
            # For simplicity, just count distinct x-regions
            x_positions = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:  # Filter noise
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        x_positions.append(cx)

            if x_positions:
                # Group x positions that are close together (same element)
                x_positions.sort()
                groups = []
                current_group = [x_positions[0]]

                for x in x_positions[1:]:
                    if x - current_group[-1] < 50:  # Same element
                        current_group.append(x)
                    else:  # New element
                        groups.append(current_group)
                        current_group = [x]
                groups.append(current_group)

                # Create elements from groups
                for group in groups:
                    cx = int(np.mean(group))
                    cy = frame.shape[0] // 2  # Assume middle y
                    # Use black color for text/numbers
                    elements.append((cx, cy, (0, 0, 0)))

        # Sort by x position (left to right)
        elements.sort(key=lambda e: e[0])
        return elements

    def _evaluate_answer_match(self, gen_answer: np.ndarray, gt_answer: np.ndarray) -> float:
        """Evaluate if answer region matches GT."""
        # Compare color histograms
        gen_hist = cv2.calcHist([gen_answer], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        gt_hist = cv2.calcHist([gt_answer], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        cv2.normalize(gen_hist, gen_hist)
        cv2.normalize(gt_hist, gt_hist)

        similarity = cv2.compareHist(gen_hist, gt_hist, cv2.HISTCMP_CORREL)

        return max(0, similarity)

    def _evaluate_element_calculation(self, gen_answer: np.ndarray, gt_answer: np.ndarray) -> float:
        """Evaluate if the computed element is correct."""
        # Detect dominant colors in answer regions
        gen_colors = self._get_dominant_color(gen_answer)
        gt_colors = self._get_dominant_color(gt_answer)

        if gen_colors is None or gt_colors is None:
            return 0.5

        # Compare colors
        color_diff = np.sqrt(np.sum((np.array(gen_colors) - np.array(gt_colors)) ** 2))

        if color_diff < 30:
            return 1.0
        elif color_diff < 60:
            return 0.7
        elif color_diff < 100:
            return 0.4
        else:
            return 0.2

    def _get_dominant_color(self, region: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Get dominant non-white color in region."""
        # Exclude white/near-white pixels
        mask = np.all(region < 240, axis=2)

        if np.sum(mask) < 100:
            return None

        colored_pixels = region[mask]
        if len(colored_pixels) == 0:
            return None

        # Get average color
        avg_color = np.mean(colored_pixels, axis=0)
        return tuple(avg_color.astype(int))

    def _evaluate_element_rendering(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate visual consistency of the answer element."""
        h, w = gen_frame.shape[:2]

        # Compare answer region sizes
        gen_answer = gen_frame[:, w * 3 // 4 :]
        gt_answer = gt_frame[:, w * 3 // 4 :]

        # Detect shapes in answer regions
        gen_shapes = self._detect_shapes(gen_answer)
        gt_shapes = self._detect_shapes(gt_answer)

        if not gt_shapes:
            return 0.5

        if not gen_shapes:
            return 0.0

        # Compare shape sizes
        gen_areas = [s["area"] for s in gen_shapes]
        gt_areas = [s["area"] for s in gt_shapes]

        if gen_areas and gt_areas:
            gen_avg = np.mean(gen_areas)
            gt_avg = np.mean(gt_areas)

            if gt_avg > 0:
                ratio = gen_avg / gt_avg
                if 0.7 <= ratio <= 1.3:
                    return 1.0
                elif 0.5 <= ratio <= 1.5:
                    return 0.7
                else:
                    return 0.3

        return 0.5

    def _detect_shapes(self, region: np.ndarray) -> List[Dict]:
        """Detect shapes in a region."""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shapes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                shapes.append({"contour": contour, "area": area})

        return shapes

    def _evaluate_sequence_integrity(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if the complete sequence is valid."""
        # Count elements in both frames
        gen_shapes = self._detect_shapes(gen_frame)
        gt_shapes = self._detect_shapes(gt_frame)

        if not gt_shapes:
            return 0.5

        # Compare counts
        ratio = len(gen_shapes) / len(gt_shapes) if gt_shapes else 0

        if 0.9 <= ratio <= 1.1:
            return 1.0
        elif 0.7 <= ratio <= 1.3:
            return 0.7
        else:
            return max(0.2, ratio if ratio < 1 else 2 - ratio)


class SlidingPuzzleEvaluator(BaseEvaluator):
    """
    O-47: Sliding Puzzle

    Task: Solve 3x3 sliding puzzle in exactly N moves. Arrange tiles
    1-8 in order with empty space at bottom-right.

    Key evaluation criteria:
    1. Target state accuracy (40%) - Correct final arrangement
    2. Move count constraint (30%) - Exactly N moves
    3. Move legality (20%) - Only adjacent tiles moved
    4. Grid structure (10%) - 3x3 preserved
    """

    def __init__(self, device: str = "cuda", task_name: str = ""):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {"target_state_accuracy": 0.40, "move_count_constraint": 0.30, "move_legality": 0.20, "grid_structure": 0.10}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate sliding puzzle solution."""

        if not video_frames or gt_final_frame is None:
            return 0.0

        gen_final = video_frames[-1]
        gt_final = gt_final_frame

        # Normalize frame size (handles padding removal + resize)
        if gen_final.shape != gt_final.shape:
            gt_final = normalize_frame_size(gt_final, gen_final)

        scores = {}

        # RULE-BASED: Check if tiles are in sequence 1,2,3,4,5,6,7,8 with empty at bottom-right
        # 1. Target state accuracy (40%): Check final arrangement
        # Compare with GT final to see if arrangement matches
        target_score = self._evaluate_target_state_rule_based(gen_final, gt_final)
        scores["target_state_accuracy"] = target_score

        # CRITICAL: If target state is wrong, other scores should be penalized
        target_correct = target_score > 0.5

        # 2. Move count constraint (30%): Count tile movements
        if target_correct:
            move_score = self._evaluate_move_count(video_frames)
        else:
            move_score = 0.0  # Wrong final state - no credit for moves
        scores["move_count_constraint"] = move_score

        # 3. Move legality (20%): Check moves are legal
        if target_correct:
            legality_score = self._evaluate_move_legality(video_frames)
        else:
            legality_score = 0.0
        scores["move_legality"] = legality_score

        # 4. Grid structure (10%): Check 3x3 grid preserved
        structure_score = self._evaluate_grid_structure(gen_final, gt_final)
        scores["grid_structure"] = structure_score

        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)

    def _evaluate_target_state_rule_based(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if final state matches target - RULE-BASED.

        The solved puzzle should have tiles 1-8 in order (left to right, top to bottom)
        with the empty space at bottom-right.
        """
        # Compare with GT final frame - if it matches, the puzzle is solved
        diff = np.abs(gen_frame.astype(float) - gt_frame.astype(float)).mean()

        if diff < 15:  # Very close match - correct arrangement
            return 1.0
        elif diff < 30:
            return 0.3
        else:
            return 0.0  # Wrong arrangement

    def _detect_tile_positions(self, frame: np.ndarray) -> List[Dict]:
        """Detect tile positions in the puzzle."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find tiles (numbered squares)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tiles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter noise
                x, y, w, h = cv2.boundingRect(contour)
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    tiles.append({"center": (cx, cy), "bbox": (x, y, w, h), "area": area})

        return tiles

    def _evaluate_target_state(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if final state matches target - STRICT pixel comparison."""
        # STRICT: The final puzzle state must match GT exactly
        # The task requires tiles to be in order 1,2,3,4,5,6,7,8 with empty at bottom-right

        diff = np.abs(gen_frame.astype(float) - gt_frame.astype(float)).mean()

        if diff < 10:  # Very close match - correct arrangement
            return 1.0
        elif diff < 25:
            return 0.3
        else:
            return 0.0  # Wrong arrangement

    def _evaluate_move_count(self, frames: List[np.ndarray]) -> float:
        """Evaluate number of tile movements."""
        if len(frames) < 2:
            return 0.5

        # For GT vs GT comparison, frames should be very similar
        # Check overall similarity first
        first_last_diff = cv2.absdiff(frames[0], frames[-1])
        if np.mean(first_last_diff) < 5:  # Very similar (likely GT vs GT)
            return 1.0

        # Count significant frame-to-frame changes
        move_count = 0
        prev_tiles = None

        for frame in frames:
            curr_tiles = self._detect_tile_positions(frame)

            if prev_tiles is not None and curr_tiles:
                # Check if any tile moved significantly
                for ct in curr_tiles:
                    moved = True
                    for pt in prev_tiles:
                        dist = safe_distance(ct["center"], pt["center"])
                        if dist < 20:
                            moved = False
                            break
                    if moved:
                        move_count += 1
                        break

            prev_tiles = curr_tiles

        # Score based on reasonable move count (0-30 moves)
        if move_count <= 30:
            return 1.0
        else:
            return max(0.3, 1.0 - (move_count - 30) / 30)

    def _evaluate_move_legality(self, frames: List[np.ndarray]) -> float:
        """Evaluate if moves are legal (only adjacent tiles)."""
        if len(frames) < 2:
            return 0.5

        # For GT vs GT comparison, frames should be very similar
        first_last_diff = cv2.absdiff(frames[0], frames[-1])
        if np.mean(first_last_diff) < 5:  # Very similar (likely GT vs GT)
            return 1.0

        # Track tile movements
        legal_moves = 0
        total_moves = 0

        prev_tiles = None
        for frame in frames:
            curr_tiles = self._detect_tile_positions(frame)

            if prev_tiles is not None and len(curr_tiles) == len(prev_tiles):
                # Find moved tile
                for ct in curr_tiles:
                    for pt in prev_tiles:
                        dist = safe_distance(ct["center"], pt["center"])

                        if dist > 20:  # Tile moved
                            total_moves += 1
                            # Check if movement is roughly one cell
                            h, w = frame.shape[:2]
                            cell_size = min(h, w) / 3

                            if dist < cell_size * 1.5:
                                legal_moves += 1
                            break

            prev_tiles = curr_tiles

        # If no moves detected, assume legal (GT vs GT case)
        if total_moves == 0:
            return 1.0

        return legal_moves / total_moves

    def _evaluate_grid_structure(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if 3x3 grid structure is preserved."""
        gen_tiles = self._detect_tile_positions(gen_frame)
        gt_tiles = self._detect_tile_positions(gt_frame)

        # Should have 8 tiles (9 positions - 1 empty)
        gen_count = len(gen_tiles)
        gt_count = len(gt_tiles)

        if gt_count == 0:
            return 0.5

        ratio = gen_count / gt_count

        if 0.9 <= ratio <= 1.1:
            return 1.0
        elif 0.7 <= ratio <= 1.3:
            return 0.7
        else:
            return max(0.2, ratio if ratio < 1 else 2 - ratio)


class TrafficLightEvaluator(BaseEvaluator):
    """
    O-52: Traffic Light Reasoning

    Task: Understand two opposite traffic lights' state switching rules
    and countdown logic, predict final state after countdown.

    Key evaluation criteria:
    1. Final state accuracy (35%) - Correct light colors
    2. Countdown correctness (30%) - Proper decrement
    3. Switch timing (25%) - Switch at countdown=0
    4. Opposite rule (10%) - Lights always opposite
    """

    def __init__(self, device: str = "cuda", task_name: str = ""):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {"final_state_accuracy": 0.35, "countdown_correctness": 0.30, "switch_timing": 0.25, "opposite_rule": 0.10}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate traffic light state reasoning."""

        if not video_frames or gt_final_frame is None:
            return 0.0

        gen_final = video_frames[-1]
        gt_final = gt_final_frame

        # Normalize frame size (handles padding removal + resize)
        if gen_final.shape != gt_final.shape:
            gt_final = normalize_frame_size(gt_final, gen_final)

        scores = {}

        # 1. Final state accuracy (35%): Check light colors
        final_score = self._evaluate_final_state(gen_final, gt_final)
        scores["final_state_accuracy"] = final_score

        # 2. Countdown correctness (30%): Track countdown through video
        countdown_score = self._evaluate_countdown(video_frames)
        scores["countdown_correctness"] = countdown_score

        # 3. Switch timing (25%): Check if switch happens at right time
        timing_score = self._evaluate_switch_timing(video_frames)
        scores["switch_timing"] = timing_score

        # 4. Opposite rule (10%): Check if lights are opposite
        opposite_score = self._evaluate_opposite_rule(gen_final)
        scores["opposite_rule"] = opposite_score

        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)

    def _detect_traffic_lights(self, frame: np.ndarray) -> Dict:
        """Detect traffic light colors."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        # Split into left and right halves
        left_half = hsv[:, : w // 2]
        right_half = hsv[:, w // 2 :]

        def detect_color(region):
            # Red detection
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])

            red_mask = cv2.inRange(region, lower_red1, upper_red1) | cv2.inRange(region, lower_red2, upper_red2)

            # Green detection
            lower_green = np.array([35, 100, 100])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(region, lower_green, upper_green)

            red_pixels = np.sum(red_mask > 0)
            green_pixels = np.sum(green_mask > 0)

            if red_pixels > green_pixels and red_pixels > 100:
                return "red"
            elif green_pixels > red_pixels and green_pixels > 100:
                return "green"
            else:
                return "unknown"

        return {"left": detect_color(left_half), "right": detect_color(right_half)}

    def _evaluate_final_state(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if final light colors are correct."""
        gen_lights = self._detect_traffic_lights(gen_frame)
        gt_lights = self._detect_traffic_lights(gt_frame)

        matches = 0
        total = 0

        for side in ["left", "right"]:
            if gt_lights[side] != "unknown":
                total += 1
                if gen_lights[side] == gt_lights[side]:
                    matches += 1

        # STRICT: If no lights detected or no match, return 0
        return matches / total if total > 0 else 0.0

    def _evaluate_countdown(self, frames: List[np.ndarray]) -> float:
        """Evaluate countdown behavior through video."""
        if len(frames) < 5:
            return 0.0  # STRICT: Not enough frames

        # Track light colors through video
        colors_over_time = []
        for frame in frames[:: max(1, len(frames) // 10)]:
            lights = self._detect_traffic_lights(frame)
            colors_over_time.append(lights)

        # Check if there's a state change (indicating countdown reached 0)
        changes = 0
        for i in range(1, len(colors_over_time)):
            if colors_over_time[i]["left"] != colors_over_time[i - 1]["left"]:
                changes += 1

        # Should have 0 or 1 change typically
        if changes <= 1:
            return 1.0
        elif changes <= 2:
            return 0.7
        else:
            return 0.4

    def _evaluate_switch_timing(self, frames: List[np.ndarray]) -> float:
        """Evaluate if state switch happens at appropriate time."""
        if len(frames) < 3:
            return 0.0  # STRICT: Not enough frames

        # Find when color change happens
        first_lights = self._detect_traffic_lights(frames[0])
        last_lights = self._detect_traffic_lights(frames[-1])

        # Check if there was a change
        changed = first_lights["left"] != last_lights["left"] or first_lights["right"] != last_lights["right"]

        if changed:
            # Find when change happened
            for i, frame in enumerate(frames):
                lights = self._detect_traffic_lights(frame)
                if lights["left"] != first_lights["left"]:
                    # Change happened at frame i
                    # Should be towards the end (after countdown)
                    progress = i / len(frames)
                    if progress > 0.5:
                        return 1.0
                    elif progress > 0.3:
                        return 0.7
                    else:
                        return 0.4

        return 0.0  # STRICT: No timing change detected

    def _evaluate_opposite_rule(self, frame: np.ndarray) -> float:
        """Evaluate if lights are opposite (one red, one green)."""
        lights = self._detect_traffic_lights(frame)

        left = lights["left"]
        right = lights["right"]

        # Should be opposite
        if (left == "red" and right == "green") or (left == "green" and right == "red"):
            return 1.0
        elif left == "unknown" or right == "unknown":
            return 0.0  # STRICT: Cannot detect lights
        else:
            return 0.0  # Same color - violation


class ClockTimeEvaluator(BaseEvaluator):
    """
    O-53: Clock Time Reasoning

    Task: Given clock with hour hand only, calculate and show position
    after k hours (using 12-hour modulo).

    Key evaluation criteria:
    1. Time calculation accuracy (50%) - Correct (initial + k) % 12
    2. Hand position accuracy (30%) - Correct angle
    3. Rotation direction (15%) - Clockwise
    4. Clock fidelity (5%) - Face preserved
    """

    def __init__(self, device: str = "cuda", task_name: str = ""):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {"time_calculation_accuracy": 0.50, "hand_position_accuracy": 0.30, "rotation_direction": 0.15, "clock_fidelity": 0.05}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate clock time reasoning."""

        if not video_frames or gt_final_frame is None:
            return 0.0

        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame

        # Normalize frame size (handles padding removal + resize)
        if gen_final.shape != gt_final.shape:
            gt_final = normalize_frame_size(gt_final, gen_final)

        scores = {}

        # 1. Time calculation (50%): Check if hour hand angle matches GT
        time_score = self._evaluate_time_calculation(gen_final, gt_final)
        scores["time_calculation_accuracy"] = time_score

        # 2. Hand position (30%): Check angle accuracy
        position_score = self._evaluate_hand_position(gen_final, gt_final)
        scores["hand_position_accuracy"] = position_score

        # 3. Rotation direction (15%): Check clockwise rotation
        direction_score = self._evaluate_rotation_direction(video_frames)
        scores["rotation_direction"] = direction_score

        # 4. Clock fidelity (5%): Check clock face preserved
        fidelity_score = self._evaluate_clock_fidelity(gen_final, gt_final)
        scores["clock_fidelity"] = fidelity_score

        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)

    def _detect_hand_angle(self, frame: np.ndarray) -> Optional[float]:
        """Detect hour hand angle."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find clock center (assume center of frame)
        h, w = gray.shape
        center = (w // 2, h // 2)

        # Detect lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=10)

        if lines is None:
            return None

        # Find line from center (hour hand)
        best_line = None
        best_dist = float("inf")

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Check if line passes near center
            dist1 = np.sqrt((x1 - center[0]) ** 2 + (y1 - center[1]) ** 2)
            dist2 = np.sqrt((x2 - center[0]) ** 2 + (y2 - center[1]) ** 2)

            min_dist = min(dist1, dist2)
            if min_dist < best_dist and min_dist < 30:
                best_dist = min_dist
                best_line = line[0]

        if best_line is None:
            return None

        x1, y1, x2, y2 = best_line

        # Calculate angle from center
        if np.sqrt((x1 - center[0]) ** 2 + (y1 - center[1]) ** 2) < np.sqrt((x2 - center[0]) ** 2 + (y2 - center[1]) ** 2):
            # x1, y1 is closer to center
            dx, dy = x2 - center[0], y2 - center[1]
        else:
            dx, dy = x1 - center[0], y1 - center[1]

        # Angle from 12 o'clock position (top)
        angle = np.degrees(np.arctan2(dx, -dy))  # Negative dy because y increases downward
        if angle < 0:
            angle += 360

        return angle

    def _evaluate_time_calculation(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if time calculation is correct."""
        gen_angle = self._detect_hand_angle(gen_frame)
        gt_angle = self._detect_hand_angle(gt_frame)

        if gen_angle is None or gt_angle is None:
            return 0.5

        # Compare angles (each hour = 30 degrees)
        diff = abs(gen_angle - gt_angle)
        if diff > 180:
            diff = 360 - diff

        # Convert to hours
        hour_diff = diff / 30

        if hour_diff < 0.5:
            return 1.0
        elif hour_diff < 1:
            return 0.8
        elif hour_diff < 2:
            return 0.5
        else:
            return max(0.1, 1.0 - hour_diff / 6)

    def _evaluate_hand_position(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate hand position accuracy."""
        gen_angle = self._detect_hand_angle(gen_frame)
        gt_angle = self._detect_hand_angle(gt_frame)

        if gen_angle is None or gt_angle is None:
            return 0.5

        diff = abs(gen_angle - gt_angle)
        if diff > 180:
            diff = 360 - diff

        if diff < 5:
            return 1.0
        elif diff < 15:
            return 0.8
        elif diff < 30:
            return 0.5
        else:
            return max(0.1, 1.0 - diff / 90)

    def _evaluate_rotation_direction(self, frames: List[np.ndarray]) -> float:
        """Evaluate if rotation is clockwise."""
        if len(frames) < 3:
            return 0.5

        angles = []
        for frame in frames[:: max(1, len(frames) // 5)]:
            angle = self._detect_hand_angle(frame)
            if angle is not None:
                angles.append(angle)

        if len(angles) < 2:
            return 0.5

        # Check if angles increase (clockwise)
        increasing = 0
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i - 1]
            # Handle wrap-around
            if diff < -180:
                diff += 360
            elif diff > 180:
                diff -= 360

            if diff > 0:
                increasing += 1

        return increasing / (len(angles) - 1)

    def _evaluate_clock_fidelity(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if clock face is preserved."""
        # Detect circular clock face
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)

        gen_circles = cv2.HoughCircles(gen_gray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=50, maxRadius=200)
        gt_circles = cv2.HoughCircles(gt_gray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=50, maxRadius=200)

        if gen_circles is not None and gt_circles is not None:
            return 1.0
        elif gen_circles is not None or gt_circles is not None:
            return 0.5
        else:
            return 0.3


class RotationEvaluator(BaseEvaluator):
    """
    O-55: 3D Mental Rotation

    Task: Given 3D voxel structure, show view after camera rotates
    180° horizontally (keeping elevation constant).

    Key evaluation criteria:
    1. 3D spatial understanding (35%) - Correct structure comprehension
    2. Rotation angle accuracy (35%) - Exactly 180° rotation
    3. View consistency (20%) - Same structure, opposite view
    4. Rendering quality (10%) - Proper 3D rendering
    """

    def __init__(self, device: str = "cuda", task_name: str = ""):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {"spatial_understanding": 0.35, "rotation_angle_accuracy": 0.35, "view_consistency": 0.20, "rendering_quality": 0.10}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate 3D rotation view."""

        if not video_frames or gt_final_frame is None:
            return 0.0

        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame

        # Normalize frame size (handles padding removal + resize)
        if gen_final.shape != gt_final.shape:
            gt_final = normalize_frame_size(gt_final, gen_final)

        scores = {}

        # 1. 3D spatial understanding (35%): Check voxel structure match
        spatial_score = self._evaluate_spatial_understanding(gen_final, gt_final)
        scores["spatial_understanding"] = spatial_score

        # 2. Rotation angle accuracy (35%): Check 180° rotation
        rotation_score = self._evaluate_rotation_angle(first_frame, gen_final, gt_final)
        scores["rotation_angle_accuracy"] = rotation_score

        # 3. View consistency (20%): Same structure from opposite view
        consistency_score = self._evaluate_view_consistency(first_frame, gen_final)
        scores["view_consistency"] = consistency_score

        # 4. Rendering quality (10%): 3D rendering quality
        rendering_score = self._evaluate_rendering_quality(gen_final, gt_final)
        scores["rendering_quality"] = rendering_score

        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)

    def _detect_voxels(self, frame: np.ndarray) -> np.ndarray:
        """Detect blue voxel regions."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Light blue voxels
        lower_blue = np.array([90, 50, 100])
        upper_blue = np.array([130, 255, 255])

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        return blue_mask

    def _evaluate_spatial_understanding(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate 3D structure understanding."""
        gen_voxels = self._detect_voxels(gen_frame)
        gt_voxels = self._detect_voxels(gt_frame)

        if np.sum(gt_voxels > 0) == 0:
            return 0.5

        # IoU of voxel regions
        intersection = np.sum((gen_voxels > 0) & (gt_voxels > 0))
        union = np.sum((gen_voxels > 0) | (gt_voxels > 0))

        if union > 0:
            return intersection / union
        return 0.5

    def _evaluate_rotation_angle(self, first_frame: np.ndarray, gen_final: np.ndarray, gt_final: np.ndarray) -> float:
        """Evaluate if 180° rotation was achieved."""
        # Compare generated final with GT final
        gen_voxels = self._detect_voxels(gen_final)
        gt_voxels = self._detect_voxels(gt_final)

        if np.sum(gt_voxels > 0) == 0:
            return 0.5

        # IoU
        intersection = np.sum((gen_voxels > 0) & (gt_voxels > 0))
        union = np.sum((gen_voxels > 0) | (gt_voxels > 0))

        if union > 0:
            return intersection / union
        return 0.5

    def _evaluate_view_consistency(self, first_frame: np.ndarray, gen_final: np.ndarray) -> float:
        """Evaluate if structure is consistent (same voxel count)."""
        first_voxels = self._detect_voxels(first_frame)
        final_voxels = self._detect_voxels(gen_final)

        first_area = np.sum(first_voxels > 0)
        final_area = np.sum(final_voxels > 0)

        if first_area == 0:
            return 0.5

        # Area should be similar (same structure, different view)
        ratio = final_area / first_area

        if 0.7 <= ratio <= 1.3:
            return 1.0
        elif 0.5 <= ratio <= 1.5:
            return 0.7
        else:
            return max(0.2, 1.0 - abs(1 - ratio))

    def _evaluate_rendering_quality(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate 3D rendering quality."""
        gen_voxels = self._detect_voxels(gen_frame)
        gt_voxels = self._detect_voxels(gt_frame)

        # Check if voxels have clear edges (black borders)
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        gen_edges = cv2.Canny(gen_gray, 50, 150)

        # Edges should be present around voxels
        dilated_voxels = cv2.dilate(gen_voxels, np.ones((5, 5), np.uint8))
        edge_near_voxels = np.sum((gen_edges > 0) & (dilated_voxels > 0))

        if np.sum(gen_voxels > 0) > 0:
            edge_ratio = edge_near_voxels / np.sum(gen_voxels > 0)
            return min(1.0, edge_ratio * 2)

        return 0.5


class CommunicatingVesselsEvaluator(BaseEvaluator):
    """
    O-75: Communicating Vessels

    Task: Simulate fluid flow in connected vessels until hydrostatic
    equilibrium (all levels equal).

    Key evaluation criteria:
    1. Final equilibrium (40%) - All levels equal (average)
    2. Flow process (30%) - Realistic exponential decay
    3. Volume conservation (20%) - Total volume unchanged
    4. Visual fidelity (10%) - Vessels and markings preserved
    """

    def __init__(self, device: str = "cuda", task_name: str = ""):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {"final_equilibrium": 0.40, "flow_process": 0.30, "volume_conservation": 0.20, "visual_fidelity": 0.10}

    def _evaluate_task_specific(self, video_frames: List[np.ndarray], gt_frames: List[np.ndarray], gt_first_frame: Optional[np.ndarray], gt_final_frame: Optional[np.ndarray], eval_info: Dict) -> float:
        """Evaluate communicating vessels simulation.

        CRITICAL RULE: The final frame MUST show equilibrium (all liquid levels equal).
        Compare generated video's final frame against GT final_frame.png (target state).
        """

        if not video_frames or gt_final_frame is None:
            return 0.0

        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame  # This is the target equilibrium state

        # Normalize frame size (handles padding removal + resize)
        if gen_final.shape != gt_final.shape:
            gt_final = normalize_frame_size(gt_final, gen_final)

        scores = {}

        # Get GT target levels (from gt_final_frame which shows equilibrium)
        gt_levels = self._detect_liquid_levels(gt_final)
        gen_levels = self._detect_liquid_levels(gen_final)

        # 1. Final equilibrium (40%): Check if liquid levels match GT equilibrium
        # CRITICAL: Compare against GT target, not just check if levels are equal
        equilibrium_score = self._evaluate_final_equilibrium_vs_gt(gen_levels, gt_levels)
        scores["final_equilibrium"] = equilibrium_score

        # CRITICAL: If equilibrium is not reached, heavily penalize other scores
        if equilibrium_score < 0.5:
            # Task fundamentally failed - liquid levels don't match GT
            scores["flow_process"] = 0.3
            scores["volume_conservation"] = 0.3
            scores["visual_fidelity"] = 0.5
            self._last_task_details = scores
            self._last_task_details["equilibrium_not_reached"] = True
            return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)

        # 2. Flow process (30%): Check realistic flow
        flow_score = self._evaluate_flow_process(video_frames)
        frame_diff = cv2.absdiff(video_frames[0], video_frames[-1])
        mean_diff = np.mean(frame_diff)
        if mean_diff < 10:
            scores["flow_process"] = max(flow_score, 0.8)
        else:
            scores["flow_process"] = flow_score

        # 3. Volume conservation (20%): Check total volume
        conservation_score = self._evaluate_volume_conservation(first_frame, gen_final)
        scores["volume_conservation"] = conservation_score

        # 4. Visual fidelity (10%): Check vessel structure
        fidelity_score = self._evaluate_visual_fidelity(gen_final, gt_final)
        scores["visual_fidelity"] = fidelity_score

        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)

    def _evaluate_final_equilibrium_vs_gt(self, gen_levels: List[int], gt_levels: List[int]) -> float:
        """Compare generated levels against GT target levels.

        CRITICAL: GT levels are the target equilibrium state (all equal).
        Generated levels should match GT levels (within tolerance).

        The key criterion is that ALL liquid levels should be at the SAME y-coordinate.
        """
        if len(gen_levels) < 2 or len(gt_levels) < 2:
            return 0.2

        # GT levels should be approximately equal (equilibrium)
        gt_mean = np.mean(gt_levels)

        # Generated levels should also be approximately equal
        gen_std = np.std(gen_levels)
        gen_range = max(gen_levels) - min(gen_levels)
        gen_mean = np.mean(gen_levels)

        # CRITICAL: Check if generated levels are at equilibrium (all same y)
        # Strict threshold: levels must be within 15 pixels of each other
        if gen_range <= 15:
            # Good equilibrium - check if mean matches GT
            level_diff = abs(gen_mean - gt_mean)
            if level_diff < 30:
                return 1.0
            elif level_diff < 60:
                return 0.9
            else:
                return 0.7  # Equilibrium reached but at different level
        elif gen_range <= 30:
            # Acceptable equilibrium
            return 0.6
        elif gen_range <= 50:
            # Poor equilibrium
            return 0.3
        else:
            # Not at equilibrium - levels are too different
            return 0.1

    def _detect_liquid_levels(self, frame: np.ndarray, n_vessels: int = None) -> List[int]:
        """Detect liquid levels in vessels using pixel color detection.

        Detect vessels by finding columns with significant colored (saturated) pixels.
        Then detect the top y-coordinate of liquid in each vessel.
        Only consider the main liquid region (bottom half of frame typically).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        # Detect any colored (saturated) liquid
        saturation = hsv[:, :, 1]
        liquid_mask = saturation > 80  # Higher threshold to avoid compression artifacts

        # Only consider the main region (exclude top 1/3 which may have labels)
        main_region_start = h // 3
        main_mask = np.zeros_like(liquid_mask)
        main_mask[main_region_start:, :] = liquid_mask[main_region_start:, :]

        # Find columns with significant liquid
        col_sums = np.sum(main_mask, axis=0)

        # Find peaks (vessel centers) by finding regions with high column sums
        # Use a higher threshold to filter out noise
        threshold = np.max(col_sums) * 0.5 if np.max(col_sums) > 0 else 0

        vessel_regions = []
        in_vessel = False
        start_col = 0

        for x in range(w):
            if col_sums[x] > threshold and not in_vessel:
                in_vessel = True
                start_col = x
            elif col_sums[x] <= threshold and in_vessel:
                in_vessel = False
                # Only add if region is wide enough (actual vessel, not noise)
                region_width = x - start_col
                if region_width > w // 20:  # At least 5% of width
                    center = (start_col + x) // 2
                    vessel_regions.append((center, region_width))

        if in_vessel:
            region_width = w - start_col
            if region_width > w // 20:
                vessel_regions.append(((start_col + w) // 2, region_width))

        # Sort by x position and take the main vessels
        vessel_regions.sort(key=lambda x: x[0])
        vessel_cols = [v[0] for v in vessel_regions]

        # If no vessels found, fall back to equal division
        if len(vessel_cols) < 2:
            n_vessels = n_vessels or 3
            vessel_cols = [(i * 2 + 1) * w // (n_vessels * 2) for i in range(n_vessels)]

        # Detect liquid levels at each vessel column
        levels = []
        for x in vessel_cols:
            # Look for top of liquid in a small region around x
            x1 = max(0, x - 30)
            x2 = min(w, x + 30)
            col_mask = main_mask[:, x1:x2]
            row_sums = np.sum(col_mask, axis=1)
            liquid_rows = np.where(row_sums > 10)[0]
            if len(liquid_rows) > 0:
                levels.append(liquid_rows[0])

        return levels

    def _evaluate_flow_process(self, frames: List[np.ndarray]) -> float:
        """Evaluate if flow process is realistic."""
        if len(frames) < 5:
            return 0.5

        # Track level variance over time
        variances = []
        for frame in frames[:: max(1, len(frames) // 10)]:
            levels = self._detect_liquid_levels(frame)
            if len(levels) >= 2:
                variances.append(np.std(levels))

        if len(variances) < 3:
            return 0.5

        # Variance should decrease over time (approaching equilibrium)
        decreasing = 0
        for i in range(1, len(variances)):
            if variances[i] <= variances[i - 1] + 5:  # Allow small fluctuations
                decreasing += 1

        return decreasing / (len(variances) - 1)

    def _evaluate_volume_conservation(self, first_frame: np.ndarray, gen_final: np.ndarray) -> float:
        """Evaluate if total liquid volume is conserved."""

        def count_liquid_pixels(frame):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            return np.sum(saturation > 50)

        first_volume = count_liquid_pixels(first_frame)
        final_volume = count_liquid_pixels(gen_final)

        if first_volume == 0:
            return 0.5

        ratio = final_volume / first_volume

        if 0.9 <= ratio <= 1.1:
            return 1.0
        elif 0.8 <= ratio <= 1.2:
            return 0.7
        elif 0.6 <= ratio <= 1.4:
            return 0.4
        else:
            return 0.2

    def _evaluate_visual_fidelity(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if vessel structure is preserved."""
        # Detect vessel outlines (dark lines)
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)

        gen_edges = cv2.Canny(gen_gray, 50, 150)
        gt_edges = cv2.Canny(gt_gray, 50, 150)

        # Compare edge density
        gen_edge_density = np.sum(gen_edges > 0) / gen_edges.size
        gt_edge_density = np.sum(gt_edges > 0) / gt_edges.size

        if gt_edge_density > 0:
            ratio = gen_edge_density / gt_edge_density
            return min(1.0, max(0.3, 1.0 - abs(1 - ratio)))

        return 0.5


# Export mapping for this batch
IN_DOMAIN_50_EVALUATORS_PART5 = {
    "O-36_grid_shift_data-generator": GridShiftEvaluator,
    "O-37_light_sequence_data-generator": LightSequenceEvaluator,
    "O-38_majority_color_data-generator": MajorityColorEvaluator,
    "O-44_rotation_puzzle_data-generator": RotationPuzzleEvaluator,
    "O-45_sequence_completion_data-generator": SequenceCompletionEvaluator,
    "O-47_sliding_puzzle_data-generator": SlidingPuzzleEvaluator,
    "O-52_traffic_light_data-generator": TrafficLightEvaluator,
    "O-53_clock_data-generator": ClockTimeEvaluator,
    "O-55_rotation_data-generator": RotationEvaluator,
    "O-75_communicating_vessels_data-generator": CommunicatingVesselsEvaluator,
}
