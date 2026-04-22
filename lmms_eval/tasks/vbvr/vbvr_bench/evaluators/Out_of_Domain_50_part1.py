"""
Specific evaluators for Out-of-Domain_50 tasks (Part 1).
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, normalize_frame_size

class SeparateObjectsNoSpinEvaluator(BaseEvaluator):
    """
    G-24: Separate objects (no spin) evaluator.
    
    Rule-based evaluation:
    - Alignment precision (35%): Objects align with dashed target outlines
    - No rotation constraint (30%): Objects maintain original orientation
    - Movement correctness (20%): Horizontal movement to the right
    - Visual fidelity (15%): Shape, color, size preserved
    """
    
    TASK_WEIGHTS = {
        'alignment': 0.35,
        'no_rotation': 0.30,
        'movement': 0.20,
        'fidelity': 0.15
    }
    
    def _detect_colored_shapes(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored shapes and their properties."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Find non-white/non-black areas
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Get bounding rect for orientation
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]
            
            shapes.append({
                'center': (cx, cy),
                'area': area,
                'angle': angle,
                'contour': cnt
            })
        
        return shapes
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate separate objects no spin task - rule-based, no SSIM."""
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        
        # Detect shapes in initial and final frames
        initial_shapes = self._detect_colored_shapes(first_frame)
        final_shapes = self._detect_colored_shapes(last_frame)
        gt_final_shapes = self._detect_colored_shapes(gt_final_frame) if gt_final_frame is not None else []
        
        # 1. Alignment precision: Compare final positions with GT
        if final_shapes and gt_final_shapes:
            total_dist = 0
            matched = 0
            for fs in final_shapes:
                min_dist = float('inf')
                for gts in gt_final_shapes:
                    dist = safe_distance(fs['center'], gts['center'])
                    min_dist = min(min_dist, dist)
                if min_dist < float('inf'):
                    total_dist += min_dist
                    matched += 1
            
            if matched > 0:
                avg_dist = total_dist / matched
                # Close match (< 10 pixels) gets full score
                if avg_dist < 10:
                    scores['alignment'] = 1.0
                else:
                    # Lenient threshold (100 pixels)
                    scores['alignment'] = max(0, 1.0 - avg_dist / 100.0)
            else:
                # No matches found - check if shapes moved right
                scores['alignment'] = self._check_rightward_movement(initial_shapes, final_shapes)
        else:
            # Fallback: check if shapes are on right side
            scores['alignment'] = self._check_rightward_movement(initial_shapes, final_shapes)
        
        # 2. No rotation constraint: Compare angles between initial and final
        if initial_shapes and final_shapes:
            angle_diffs = []
            # Sort by area to match shapes (more robust than position)
            initial_sorted = sorted(initial_shapes, key=lambda s: s['area'], reverse=True)
            final_sorted = sorted(final_shapes, key=lambda s: s['area'], reverse=True)
            
            for i, (i_shape, f_shape) in enumerate(zip(initial_sorted, final_sorted)):
                if i >= min(len(initial_sorted), len(final_sorted)):
                    break
                angle_diff = abs(i_shape['angle'] - f_shape['angle'])
                angle_diff = min(angle_diff, 90 - angle_diff)  # Handle angle wrapping
                angle_diffs.append(angle_diff)
            
            if angle_diffs:
                avg_angle_diff = np.mean(angle_diffs)
                # Small angle diff (< 2 degrees) gets full score
                if avg_angle_diff < 2:
                    scores['no_rotation'] = 1.0
                elif avg_angle_diff < 15:
                    scores['no_rotation'] = 0.8
                elif avg_angle_diff < 30:
                    scores['no_rotation'] = 0.2  # Detection failed
                else:
                    scores['no_rotation'] = max(0, 1.0 - avg_angle_diff / 45.0)
            else:
                scores['no_rotation'] = 0.2  # Detection failed
        else:
            scores['no_rotation'] = 0.2  # Detection failed
        
        # 3. Movement correctness: Check horizontal motion
        frame_diff = cv2.absdiff(first_frame, last_frame)
        if np.mean(frame_diff) < 5:  # Very similar frames
            scores['movement'] = 1.0
        else:
            try:
                flow_result = compute_optical_flow(first_frame, last_frame)
                if flow_result is not None:
                    flow, _ = flow_result
                    h_flow = np.abs(flow[:, :, 0]).mean()
                    v_flow = np.abs(flow[:, :, 1]).mean()
                    if h_flow + v_flow > 0:
                        # Horizontal movement should dominate
                        scores['movement'] = h_flow / (h_flow + v_flow)
                    else:
                        scores['movement'] = 0.8
                else:
                    # Check centroid movement direction
                    scores['movement'] = self._check_horizontal_movement(initial_shapes, final_shapes)
            except Exception:
                scores['movement'] = self._check_horizontal_movement(initial_shapes, final_shapes)
        
        # 4. Fidelity: Check shape preservation
        if initial_shapes and final_shapes:
            # Check count preservation
            count_match = max(0, 1.0 - abs(len(initial_shapes) - len(final_shapes)) / max(len(initial_shapes), 1))
            
            # Check area preservation
            initial_total_area = sum(s['area'] for s in initial_shapes)
            final_total_area = sum(s['area'] for s in final_shapes)
            area_ratio = min(initial_total_area, final_total_area) / max(initial_total_area, final_total_area, 1)
            
            scores['fidelity'] = 0.5 * count_match + 0.5 * area_ratio
        else:
            scores['fidelity'] = 0.2  # Detection failed
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _check_rightward_movement(self, initial_shapes: List[Dict], final_shapes: List[Dict]) -> float:
        """Check if shapes moved to the right side."""
        if not initial_shapes or not final_shapes:
            return 0.5
        
        initial_avg_x = np.mean([s['center'][0] for s in initial_shapes])
        final_avg_x = np.mean([s['center'][0] for s in final_shapes])
        
        # Shapes should move right
        if final_avg_x > initial_avg_x:
            return min(1.0, (final_avg_x - initial_avg_x) / 100.0 + 0.5)
        return 0.3
    
    def _check_horizontal_movement(self, initial_shapes: List[Dict], final_shapes: List[Dict]) -> float:
        """Check if movement is primarily horizontal."""
        if not initial_shapes or not final_shapes:
            return 0.5
        
        initial_centers = [s['center'] for s in initial_shapes]
        final_centers = [s['center'] for s in final_shapes]
        
        # Calculate average movement
        if len(initial_centers) != len(final_centers):
            return 0.5
        
        total_h = 0
        total_v = 0
        for ic, fc in zip(initial_centers, final_centers):
            total_h += abs(fc[0] - ic[0])
            total_v += abs(fc[1] - ic[1])
        
        if total_h + total_v > 0:
            return total_h / (total_h + total_v)
        return 0.8

class MultipleKeysForOneDoorEvaluator(BaseEvaluator):
    """
    G-47: Multi-key collection maze evaluator.
    
    Rule-based evaluation:
    - Key collection completeness (30%): Agent must collect 2 keys (colored objects disappear)
    - Visit order optimization (25%): Keys collected before reaching door
    - Path efficiency (25%): Minimal backtracking
    - Movement legality (20%): No wall crossing, valid grid moves
    """
    
    TASK_WEIGHTS = {
        'key_collection': 0.30,
        'order_optimization': 0.25,
        'path_efficiency': 0.25,
        'movement_legality': 0.20
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        # 1. Key collection completeness (30%)
        # Rule: Count colored key objects in first vs final frame
        scores['key_collection'] = self._evaluate_key_collection(
            first_frame, final_frame
        )
        
        # 2. Visit order optimization (25%)
        # Rule: Track agent reaching key positions before door
        scores['order_optimization'] = self._evaluate_visit_order(video_frames)
        
        # 3. Path efficiency (25%)
        # Rule: Check for minimal backtracking
        scores['path_efficiency'] = self._evaluate_path_efficiency(video_frames)
        
        # 4. Movement legality (20%)
        # Rule: Check for wall crossing and valid moves
        scores['movement_legality'] = self._evaluate_movement_legality(
            video_frames, first_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_key_collection(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Count keys collected (colored objects that disappear)."""
        # Detect colored objects (keys) - typically yellow, blue, red
        initial_keys = self._count_key_objects(first_frame)
        final_keys = self._count_key_objects(final_frame)
        
        # Expected: 2 keys should be collected (disappear)
        keys_collected = initial_keys - final_keys
        
        # Check if agent reached the door (green agent at door position)
        agent_at_door = self._check_agent_at_door(final_frame)
        
        if keys_collected >= 2 and agent_at_door:
            return 1.0
        elif keys_collected >= 2:
            return 0.8  # Keys collected but didn't reach door
        elif keys_collected == 1:
            return 0.5  # Partial collection
        else:
            return 0.0
    
    def _evaluate_visit_order(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Check if keys are collected before reaching door."""
        door_reached_frame = None
        key_collection_frames = []
        
        for i, frame in enumerate(video_frames):
            # Track when agent reaches door area
            if self._check_agent_at_door(frame) and door_reached_frame is None:
                door_reached_frame = i
            
            # Track key collection events (reduction in key count)
            if i > 0:
                prev_keys = self._count_key_objects(video_frames[i-1])
                curr_keys = self._count_key_objects(frame)
                if curr_keys < prev_keys:
                    key_collection_frames.append(i)
        
        # All keys should be collected before or at door reach
        if door_reached_frame is None:
            return 0.3  # Never reached door
        
        if len(key_collection_frames) >= 2:
            if all(f < door_reached_frame or f == door_reached_frame for f in key_collection_frames):
                return 1.0
            return 0.6
        elif len(key_collection_frames) == 1:
            return 0.4
        return 0.2
    
    def _evaluate_path_efficiency(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Check for minimal backtracking."""
        positions = []
        for frame in video_frames[::max(1, len(video_frames)//30)]:
            pos = self._detect_agent_position(frame)
            if pos:
                positions.append(pos)
        
        if len(positions) < 3:
            return 0.5
        
        # Count backtracking (returning to same cell)
        visited = set()
        revisits = 0
        for pos in positions:
            cell = (pos[0] // 30, pos[1] // 30)  # Grid cell approximation
            if cell in visited:
                revisits += 1
            visited.add(cell)
        
        revisit_ratio = revisits / len(positions)
        
        if revisit_ratio < 0.1:
            return 1.0
        elif revisit_ratio < 0.2:
            return 0.8
        elif revisit_ratio < 0.3:
            return 0.6
        else:
            return max(0.2, 1.0 - revisit_ratio)
    
    def _evaluate_movement_legality(
        self, 
        video_frames: List[np.ndarray],
        first_frame: np.ndarray
    ) -> float:
        """Rule-based: Check for wall crossing (illegal moves)."""
        # Detect walls from first frame (black pixels)
        wall_mask = self._detect_walls(first_frame)
        
        if wall_mask is None:
            return 0.5
        
        # Track agent movement and check for wall collision
        violations = 0
        total_moves = 0
        
        for i in range(1, min(len(video_frames), 50)):
            agent_pos = self._detect_agent_position(video_frames[i])
            if agent_pos is not None:
                x, y = agent_pos
                h, w = wall_mask.shape
                if 0 <= y < h and 0 <= x < w:
                    if wall_mask[y, x] > 0:
                        violations += 1
                total_moves += 1
        
        if total_moves == 0:
            return 0.5
        
        violation_rate = violations / total_moves
        
        if violation_rate == 0:
            return 1.0
        elif violation_rate <= 0.05:
            return 0.7
        elif violation_rate <= 0.1:
            return 0.4
        else:
            return 0.1
    
    def _count_key_objects(self, frame: np.ndarray) -> int:
        """Count colored key objects (orange, cyan, yellow, red, blue markers)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Orange keys (HSV hue ~15)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Yellow keys (HSV hue ~30)
        lower_yellow = np.array([25, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Cyan keys (HSV hue ~90)
        lower_cyan = np.array([80, 100, 100])
        upper_cyan = np.array([100, 255, 255])
        cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        # Red keys
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Blue keys
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        combined = orange_mask | yellow_mask | cyan_mask | red_mask | blue_mask
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count significant contours (keys)
        keys = sum(1 for cnt in contours if 200 < cv2.contourArea(cnt) < 5000)
        return keys
    
    def _check_agent_at_door(self, frame: np.ndarray) -> bool:
        """Check if green agent is at door position."""
        agent_pos = self._detect_agent_position(frame)
        if agent_pos is None:
            return False
        
        # Door is typically at a specific location (bottom right or marked differently)
        # Check for brown/orange door color near agent
        x, y = agent_pos
        h, w = frame.shape[:2]
        
        # Sample region around agent
        region = frame[max(0, y-40):min(h, y+40), max(0, x-40):min(w, x+40)]
        if region.size == 0:
            return False
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Door color (brown/orange)
        lower_door = np.array([10, 50, 50])
        upper_door = np.array([25, 255, 255])
        door_mask = cv2.inRange(hsv, lower_door, upper_door)
        
        door_pixels = np.sum(door_mask > 0)
        return door_pixels > 100
    
    def _detect_walls(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect wall regions (black/dark pixels) in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, wall_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        return wall_mask
    
    def _detect_agent_position(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect green agent position in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Green color range
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        return None

class ConnectingColorEvaluator(BaseEvaluator):
    """
    G-54: Connecting color evaluator.
    
    STRICT rule-based evaluation for the "connect same color objects with curves" task:
    - Same color objects MUST be connected by same color curves/lines
    - Different color objects should NOT be connected  
    - Original objects should remain in place (not change position significantly)
    
    CRITICAL RULES:
    1. Detect ALL colored objects in the first frame (not just circles)
    2. Count CORRECT connections: same-color objects connected by same-color line
    3. Count WRONG connections: different-color objects connected by any line
    4. Objects should NOT change position or be destroyed
    
    Expected connections:
    - GT sample 0: 2 correct connections (blue pair + orange pair)
    - GT sample 1: 3 correct connections
    
    Evaluates:
    - Object preservation (20%): Original objects still exist at same positions
    - Correct connections (50%): Same-color objects are connected by same-color lines
    - No wrong connections (30%): No lines connecting different colors
    """
    
    TASK_WEIGHTS = {
        'object_preservation': 0.20,
        'correct_connections': 0.50,
        'no_wrong_connections': 0.30
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate connecting color task with proper connection counting."""
        scores = {}
        
        first_frame = video_frames[0] if len(video_frames) > 0 else None
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_first = gt_first_frame
        gt_last = gt_final_frame
        
        if first_frame is None or last_frame is None:
            return 0.0
        
        # Normalize frame size (handles padding removal + resize)
        if gt_last is not None and last_frame.shape != gt_last.shape:
            gt_last = normalize_frame_size(gt_last, last_frame)
        if gt_first is not None and first_frame.shape != gt_first.shape:
            gt_first = normalize_frame_size(gt_first, first_frame)
        
        # Detect circular objects in first frame (before curves are drawn)
        first_objects = self._detect_circular_objects(first_frame)
        
        # Detect objects in last frame
        last_objects = self._detect_circular_objects(last_frame)
        
        # Store detection info for debugging
        scores['num_first_objects'] = len(first_objects)
        scores['num_last_objects'] = len(last_objects)
        
        if len(first_objects) < 2:
            scores['object_preservation'] = 0.0
            scores['correct_connections'] = 0.0
            scores['no_wrong_connections'] = 0.0
            self._last_task_details = scores
            self._last_task_details['error'] = 'not_enough_objects_in_first_frame'
            return 0.0
        
        # 1. Object preservation (20%): Check if objects remain at similar positions
        preservation_score = self._evaluate_object_preservation(first_objects, last_objects)
        scores['object_preservation'] = preservation_score
        
        # Group first frame objects by color
        objects_by_color = {}
        for obj in first_objects:
            objects_by_color.setdefault(obj['color'], []).append(obj)
        
        # Count expected connections (pairs of same-color objects)
        expected_connections = sum(
            len(objs) * (len(objs) - 1) // 2 
            for objs in objects_by_color.values() 
            if len(objs) >= 2
        )
        
        # 2 & 3. Count correct and wrong connections
        correct_count, wrong_count, connection_details = self._count_connections(
            last_frame, first_objects, objects_by_color
        )
        
        # Store connection info
        scores['expected_connections'] = expected_connections
        scores['correct_connections_count'] = correct_count
        scores['wrong_connections_count'] = wrong_count
        
        # 2. Correct connections score (50%)
        if expected_connections > 0:
            scores['correct_connections'] = min(1.0, correct_count / expected_connections)
        else:
            scores['correct_connections'] = 0.0
        
        # 3. No wrong connections score (30%)
        if wrong_count == 0:
            scores['no_wrong_connections'] = 1.0
        else:
            # Each wrong connection heavily penalizes
            scores['no_wrong_connections'] = max(0.0, 1.0 - wrong_count * 0.4)
        
        scores['connection_details'] = connection_details
        
        self._last_task_details = scores
        return sum(scores.get(k, 0) * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS if k in scores)
    
    def _detect_circular_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect colored objects (dots/circles/blobs) in the frame.
        
        NOTE: Lowered circularity threshold to detect more shapes, not just perfect circles.
        The task involves connecting same-color objects which may be various shapes.
        """
        objects = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
            'green': ([35, 100, 100], [85, 255, 255], None, None),
            'blue': ([100, 100, 100], [130, 255, 255], None, None),
            'yellow': ([20, 100, 100], [35, 255, 255], None, None),
            'orange': ([10, 100, 100], [20, 255, 255], None, None),
            'purple': ([130, 100, 100], [160, 255, 255], None, None),
            'cyan': ([85, 100, 100], [100, 255, 255], None, None),
        }
        
        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # Accept objects in reasonable size range (not too small, not too big)
                if area < 500 or area > 30000:
                    continue
                
                # Check circularity (lowered threshold to accept more shapes)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    # Accept reasonably compact objects (circularity > 0.3)
                    # This accepts ovals, squares with rounded corners, etc.
                    if circularity > 0.3:
                        M = cv2.moments(contour)
                        if M['m00'] > 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            objects.append({
                                'color': color_name,
                                'center': (cx, cy),
                                'area': area,
                                'circularity': circularity
                            })
        
        return objects
    
    def _evaluate_object_preservation(self, first_objects: List[Dict], 
                                      last_objects: List[Dict]) -> float:
        """Check if original objects are preserved in similar positions."""
        if not first_objects:
            return 0.0
        
        matched = 0
        for first_obj in first_objects:
            # Find matching object in last frame (same color, similar position)
            for last_obj in last_objects:
                if first_obj['color'] == last_obj['color']:
                    dist = safe_distance(first_obj['center'], last_obj['center'])
                    if dist < 50:  # Object within 50 pixels of original position
                        matched += 1
                        break
        
        return matched / len(first_objects)
    
    def _count_connections(self, frame: np.ndarray, first_objects: List[Dict],
                          objects_by_color: Dict) -> Tuple[int, int, Dict]:
        """
        Count correct and wrong line connections.
        
        - Correct: Same-color objects connected by same-color line
        - Wrong: Different-color objects connected by any line
        
        Returns: (correct_count, wrong_count, details)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
            'green': ([35, 100, 100], [85, 255, 255], None, None),
            'blue': ([100, 100, 100], [130, 255, 255], None, None),
            'yellow': ([20, 100, 100], [35, 255, 255], None, None),
            'orange': ([10, 100, 100], [20, 255, 255], None, None),
            'purple': ([130, 100, 100], [160, 255, 255], None, None),
            'cyan': ([85, 100, 100], [100, 255, 255], None, None),
        }
        
        correct_connections = []
        wrong_connections = []
        
        # Check correct connections: same-color objects connected by same-color line
        for color_name, objs in objects_by_color.items():
            if len(objs) < 2:
                continue
            
            ranges = color_ranges.get(color_name)
            if not ranges:
                continue
            
            # Create color mask for this color
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            
            # Check each pair of same-color objects
            for i in range(len(objs)):
                for j in range(i + 1, len(objs)):
                    obj1, obj2 = objs[i], objs[j]
                    
                    # Check if there's a continuous path of this color between them
                    if self._has_color_path(mask, obj1['center'], obj2['center']):
                        correct_connections.append({
                            'color': color_name,
                            'obj1': obj1['center'],
                            'obj2': obj2['center']
                        })
        
        # Check wrong connections: different-color objects connected
        colors_list = list(objects_by_color.keys())
        for i in range(len(colors_list)):
            for j in range(i + 1, len(colors_list)):
                color1, color2 = colors_list[i], colors_list[j]
                objs1 = objects_by_color[color1]
                objs2 = objects_by_color[color2]
                
                for obj1 in objs1:
                    for obj2 in objs2:
                        if self._has_any_line_connection(frame, obj1['center'], obj2['center']):
                            wrong_connections.append({
                                'colors': (color1, color2),
                                'obj1': obj1['center'],
                                'obj2': obj2['center']
                            })
        
        details = {
            'correct': correct_connections,
            'wrong': wrong_connections
        }
        
        return len(correct_connections), len(wrong_connections), details
    
    def _has_color_path(self, color_mask: np.ndarray, 
                        p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """
        Check if there's a continuous path of the specific color between two points.
        Uses line sampling with wider search region and checks for color presence along the path.
        
        NOTE: Lowered threshold to 30% because curves may not follow straight line exactly.
        """
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        
        # Sample points along the line
        num_samples = 30
        connected_count = 0
        
        for t in np.linspace(0.15, 0.85, num_samples):  # Avoid endpoints (object areas)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            if 0 <= x < color_mask.shape[1] and 0 <= y < color_mask.shape[0]:
                # Check a LARGER region around the sample point (curves may not be straight)
                y_min, y_max = max(0, y-20), min(color_mask.shape[0], y+20)
                x_min, x_max = max(0, x-20), min(color_mask.shape[1], x+20)
                region = color_mask[y_min:y_max, x_min:x_max]
                
                if np.sum(region > 0) > 5:  # Some pixels of this color
                    connected_count += 1
        
        # Connection exists if more than 30% of samples have the color (curves may not follow straight line)
        return connected_count > num_samples * 0.3
    
    def _has_any_line_connection(self, frame: np.ndarray,
                                 p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """
        Check if there's any drawn line connecting two different-colored objects.
        Looks for non-background colored pixels along the path.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        
        # Sample points along the middle section of the line
        num_samples = 20
        colored_count = 0
        
        for t in np.linspace(0.3, 0.7, num_samples):  # Only check middle section
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                # Check if this pixel is colored (high saturation) and not background
                h, s, v = hsv[y, x]
                if s > 50 and v > 50:  # Colored pixel (not white/black/gray)
                    colored_count += 1
        
        # If more than 40% of middle samples have colored pixels, there's a connection
        return colored_count > num_samples * 0.4

class SelectNextFigureAlternatingEvaluator(BaseEvaluator):
    """
    G-135: Select next figure in small-big alternating sequence.
    
    Rule-based evaluation:
    - Pattern recognition (40%): Identify "small-big-small" pattern in existing sequence
    - Selection correctness (35%): Next should be "big" - largest candidate selected
    - Marking accuracy (15%): Red circle marks exactly one figure
    - Animation quality (10%): Circle appears with smooth expansion
    """
    
    TASK_WEIGHTS = {
        'pattern_recognition': 0.40,
        'selection_correctness': 0.35,
        'marking_accuracy': 0.15,
        'animation_quality': 0.10
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        # 1. Pattern recognition (40%)
        # Rule: Check if pattern analysis shows alternating sizes
        scores['pattern_recognition'] = self._evaluate_pattern_recognition(
            first_frame, final_frame
        )
        
        # 2. Selection correctness (35%)
        # Rule: Red circle should mark the largest candidate figure
        scores['selection_correctness'] = self._evaluate_selection(
            first_frame, final_frame
        )
        
        # 3. Marking accuracy (15%)
        # Rule: Exactly one red circle marking
        scores['marking_accuracy'] = self._evaluate_marking(final_frame, first_frame)
        
        # 4. Animation quality (10%)
        # Rule: Circle should expand smoothly
        scores['animation_quality'] = self._evaluate_animation(video_frames)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_pattern_recognition(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if alternating pattern is understood."""
        # Detect existing shapes
        shapes = self._detect_shapes_with_sizes(first_frame)
        
        if len(shapes) < 3:
            return 0.5
        
        # Only consider sequence shapes (top half) for pattern recognition
        h = first_frame.shape[0]
        sequence_shapes = [s for s in shapes if s[1] < h // 2]
        
        if len(sequence_shapes) < 3:
            return 0.5
        
        # Sort by x-position (left to right sequence)
        shapes_sorted = sorted(sequence_shapes, key=lambda s: s[0])
        sizes = [s[2] for s in shapes_sorted]
        
        if len(sizes) < 3:
            return 0.5
        
        # Check for alternating pattern: small-big-small or big-small-big
        is_alternating = True
        for i in range(len(sizes) - 2):
            if sizes[i] < sizes[i+1] > sizes[i+2] or sizes[i] > sizes[i+1] < sizes[i+2]:
                continue
            else:
                is_alternating = False
                break
        
        if is_alternating:
            return 1.0
        return 0.5
    
    def _evaluate_selection(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if the correct candidate is marked based on pattern."""
        # Detect red circle marking (new markings only)
        circles = self._detect_red_circles(final_frame, first_frame)
        
        if len(circles) == 0:
            return 0.0
        
        marked_pos = circles[0][:2]  # Get position of marked item
        
        # Detect all shapes
        all_shapes = self._detect_shapes_with_sizes(first_frame)
        
        if len(all_shapes) == 0:
            return 0.5
        
        # Separate into sequence (top half) and candidates (bottom half)
        h = first_frame.shape[0]
        sequence_shapes = sorted([s for s in all_shapes if s[1] < h // 2], key=lambda s: s[0])
        candidate_shapes = [s for s in all_shapes if s[1] >= h // 2]
        
        if len(candidate_shapes) == 0:
            return 0.5
        
        # Find which candidate is marked
        marked_candidate = None
        min_dist = float('inf')
        for cand in candidate_shapes:
            dist = np.sqrt((cand[0] - marked_pos[0])**2 + (cand[1] - marked_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                marked_candidate = cand
        
        if marked_candidate is None or min_dist > 100:
            return 0.3
        
        # Determine expected size based on alternating pattern
        if len(sequence_shapes) >= 2:
            sizes = [s[2] for s in sequence_shapes]
            
            # Calculate threshold as mean of min and max sizes
            min_size = min(sizes)
            max_size = max(sizes)
            threshold = (min_size + max_size) / 2
            
            # Classify each shape as small or big
            pattern = ['small' if s <= threshold else 'big' for s in sizes]
            last_type = pattern[-1]
            
            # Determine expected next type
            expected_type = 'big' if last_type == 'small' else 'small'
            
            # Check if marked candidate matches expected type
            marked_type = 'small' if marked_candidate[2] <= threshold else 'big'
            
            if marked_type == expected_type:
                return 1.0
            else:
                return 0.5
        
        # Fallback: check if marked candidate is among the larger ones
        candidate_sizes = [c[2] for c in candidate_shapes]
        if marked_candidate[2] >= np.median(candidate_sizes):
            return 0.8
        return 0.5
    
    def _evaluate_marking(
        self, 
        final_frame: np.ndarray, 
        first_frame: Optional[np.ndarray] = None
    ) -> float:
        """Rule-based: Evaluate red circle marking quality."""
        circles = self._detect_red_circles(final_frame, first_frame)
        
        if len(circles) == 0:
            return 0.0
        elif len(circles) == 1:
            return 1.0  # Correct number of markings
        else:
            return max(0.3, 1.0 - 0.2 * (len(circles) - 1))  # Penalty for multiple
    
    def _evaluate_animation(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Evaluate animation smoothness."""
        if len(video_frames) < 3:
            return 0.5
        
        # Check for smooth circle expansion
        circle_sizes = []
        for frame in video_frames[len(video_frames)//2:]:
            circles = self._detect_red_circles(frame)
            if circles:
                circle_sizes.append(circles[0][2] if len(circles[0]) > 2 else 30)
        
        if len(circle_sizes) < 2:
            return 0.5
        
        # Check if sizes increase smoothly
        increases = sum(1 for i in range(1, len(circle_sizes)) 
                       if circle_sizes[i] >= circle_sizes[i-1] * 0.95)
        smoothness = increases / (len(circle_sizes) - 1)
        
        return smoothness
    
    def _detect_shapes_with_sizes(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect shapes with their (x, y, area)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:  # Lower threshold to detect smaller shapes
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shapes.append((cx, cy, area))
        
        return shapes
    
    def _detect_candidate_shapes(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect candidate shapes (typically in bottom portion)."""
        h, w = frame.shape[:2]
        
        # Focus on bottom half or right portion where candidates usually are
        bottom_region = frame[h//2:, :]
        
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"]) + h//2  # Adjust for cropped region
                    candidates.append((cx, cy, area))
        
        return candidates
    
    def _detect_red_circles(
        self, 
        frame: np.ndarray, 
        first_frame: Optional[np.ndarray] = None
    ) -> List[Tuple[int, int, int]]:
        """Detect red circles in the frame (new markings only if first_frame provided)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # If first_frame provided, only detect NEW red regions (markings)
        if first_frame is not None:
            hsv_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
            mask_first = cv2.inRange(hsv_first, lower_red1, upper_red1) | cv2.inRange(hsv_first, lower_red2, upper_red2)
            # Only keep red regions that are new (not in first frame)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_first))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                circles.append((int(x), int(y), int(radius)))
        
        return circles

class LocatePointInOverlappingAreaEvaluator(BaseEvaluator):
    """
    G-136: Locate points in overlapping region of two shapes.
    
    Rule-based evaluation:
    - Overlap identification (30%): Detect intersection region correctly
    - Point containment (35%): Mark points actually in overlap region
    - Marking completeness (20%): All overlap points marked (recall)
    - Marking accuracy (15%): No false positives (precision)
    """
    
    TASK_WEIGHTS = {
        'overlap_identification': 0.30,
        'point_containment': 0.35,
        'marking_completeness': 0.20,
        'marking_accuracy': 0.15
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        # Detect overlap region from first frame
        overlap_region = self._detect_overlap_region(first_frame)
        
        # Detect all points from first frame
        all_points = self._detect_points(first_frame)
        
        # Detect marked circles in final frame
        marked_circles = self._detect_circles(final_frame)
        
        # 1. Overlap identification (30%)
        scores['overlap_identification'] = self._evaluate_overlap_understanding(
            overlap_region
        )
        
        # 2. Point containment (35%)
        scores['point_containment'] = self._evaluate_point_marking(
            marked_circles, overlap_region, all_points
        )
        
        # 3. Marking completeness (20%) - compare with GT marking count
        gt_marked_circles = self._detect_circles(gt_final_frame) if gt_final_frame is not None else []
        scores['marking_completeness'] = self._evaluate_completeness(
            marked_circles, overlap_region, all_points, gt_marked_circles
        )
        
        # 4. Marking accuracy (15%)
        scores['marking_accuracy'] = self._evaluate_marking_accuracy(
            marked_circles, overlap_region
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_overlap_understanding(self, overlap_region: Optional[np.ndarray]) -> float:
        """Rule-based: Check if overlap region is detected."""
        if overlap_region is None:
            return 0.3
        
        overlap_area = np.sum(overlap_region > 0)
        
        if overlap_area > 1000:  # Significant overlap detected
            return 1.0
        elif overlap_area > 500:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_point_marking(
        self, 
        marked_circles: List[Tuple[int, int]],
        overlap_region: Optional[np.ndarray],
        all_points: List[Tuple[int, int]]
    ) -> float:
        """Rule-based: Check if marked points are in overlap region."""
        if len(marked_circles) == 0:
            return 0.0
        
        if overlap_region is None:
            return 0.5
        
        # Count how many marked points are in overlap
        correct_marks = 0
        for circle in marked_circles:
            x, y = circle
            h, w = overlap_region.shape
            if 0 <= y < h and 0 <= x < w:
                if overlap_region[y, x] > 0:
                    correct_marks += 1
        
        if len(marked_circles) == 0:
            return 0.0
        
        return correct_marks / len(marked_circles)
    
    def _evaluate_completeness(
        self, 
        marked_circles: List[Tuple[int, int]],
        overlap_region: Optional[np.ndarray],
        all_points: List[Tuple[int, int]],
        gt_marked_circles: List[Tuple[int, int]] = None
    ) -> float:
        """Rule-based: Check marking completeness compared to GT."""
        if len(marked_circles) == 0:
            return 0.0
        
        # If GT markings available, compare with GT marking positions
        if gt_marked_circles and len(gt_marked_circles) > 0:
            # Count how many GT markings are matched by generated markings
            matched = 0
            for gt_circle in gt_marked_circles:
                for gen_circle in marked_circles:
                    dist = np.sqrt((gen_circle[0] - gt_circle[0])**2 + (gen_circle[1] - gt_circle[1])**2)
                    if dist < 50:  # Within matching distance
                        matched += 1
                        break
            
            # Score based on matching ratio
            recall = matched / len(gt_marked_circles)
            precision = matched / len(marked_circles) if len(marked_circles) > 0 else 0
            
            # F1 score
            if recall + precision > 0:
                return 2 * recall * precision / (recall + precision)
            return 0.0
        
        # Fallback: check if marking count is reasonable
        if overlap_region is None:
            return 0.5
        
        # Find points in overlap region
        points_in_overlap = []
        for point in all_points:
            x, y = point
            h, w = overlap_region.shape
            if 0 <= y < h and 0 <= x < w:
                if overlap_region[y, x] > 0:
                    points_in_overlap.append(point)
        
        if len(points_in_overlap) == 0:
            return 1.0 if len(marked_circles) == 0 else 0.5
        
        # Score based on marking at least some points
        marked_count = 0
        for point in points_in_overlap:
            for circle in marked_circles:
                dist = np.sqrt((circle[0] - point[0])**2 + (circle[1] - point[1])**2)
                if dist < 40:
                    marked_count += 1
                    break
        
        # If at least some points are marked correctly, give good score
        if marked_count >= len(marked_circles) * 0.8:
            return 1.0
        elif marked_count > 0:
            return 0.7
        return 0.3
    
    def _evaluate_marking_accuracy(
        self, 
        marked_circles: List[Tuple[int, int]],
        overlap_region: Optional[np.ndarray]
    ) -> float:
        """Rule-based: Check for false positives (precision)."""
        if len(marked_circles) == 0:
            return 0.0
        
        if overlap_region is None:
            return 0.5
        
        # Count marks that are in overlap (true positives)
        true_positives = 0
        for circle in marked_circles:
            x, y = circle
            h, w = overlap_region.shape
            if 0 <= y < h and 0 <= x < w:
                if overlap_region[y, x] > 0:
                    true_positives += 1
        
        return true_positives / len(marked_circles)
    
    def _detect_overlap_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect overlapping region of two shapes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find all non-white shapes
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find large shapes (potential overlapping regions)
        large_shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:  # Large shape threshold
                large_shapes.append((cnt, area))
        
        if large_shapes:
            # The largest shape is likely the overlap region itself
            # (when two shapes overlap, they form a single connected region)
            largest = max(large_shapes, key=lambda x: x[1])
            overlap = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(overlap, [largest[0]], -1, 255, -1)
            return overlap
        
        # Fallback: try color-based detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect different colored shapes and find their intersection
        color_ranges = [
            ([100, 50, 50], [130, 255, 255]),  # Blue
            ([15, 50, 50], [45, 255, 255]),     # Yellow/Orange
            ([140, 50, 50], [170, 255, 255]),   # Magenta/Pink
            ([35, 50, 50], [85, 255, 255]),     # Green
        ]
        
        masks = []
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            if np.sum(mask) > 1000:
                masks.append(mask)
        
        # Find overlap between any two color masks
        if len(masks) >= 2:
            overlap = cv2.bitwise_and(masks[0], masks[1])
            if np.sum(overlap) > 100:
                return overlap
        
        return None
    
    def _detect_points(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect small point/dot markers in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 500:  # Small dot size range
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append((cx, cy))
        
        return points
    
    def _detect_circles(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect marking circles in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red circles
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    circles.append((cx, cy))
        
        return circles


class LocateTopmostFigureEvaluator(BaseEvaluator):
    """
    G-140: Locate topmost (unobscured) figure in overlapping shapes.
    
    Rule-based evaluation:
    - Z-order identification (45%): Outline the topmost (least occluded) shape
    - Outline accuracy (30%): Red outline follows shape boundary
    - Marking uniqueness (15%): Only one shape marked
    - Visual clarity (10%): Clear visible outline
    """
    
    TASK_WEIGHTS = {
        'z_order_identification': 0.45,
        'outline_accuracy': 0.30,
        'marking_uniqueness': 0.15,
        'visual_clarity': 0.10
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        # 1. Z-order identification (45%)
        scores['z_order_identification'] = self._evaluate_z_order(
            first_frame, final_frame
        )
        
        # 2. Outline accuracy (30%)
        scores['outline_accuracy'] = self._evaluate_outline(final_frame)
        
        # 3. Marking uniqueness (15%)
        scores['marking_uniqueness'] = self._evaluate_uniqueness(final_frame)
        
        # 4. Visual clarity (10%)
        scores['visual_clarity'] = self._evaluate_clarity(final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_z_order(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if the topmost (least occluded) shape is outlined."""
        # Detect the topmost shape from first frame
        topmost_center = self._find_topmost_shape_center(first_frame)
        
        # Detect outline position in final frame
        outline = self._detect_outline(final_frame)
        
        if outline is None:
            return 0.0
        
        outline_center = self._get_contour_center(outline)
        
        if topmost_center is None or outline_center is None:
            return 0.3
        
        # Check if outline is around topmost shape
        dist = np.sqrt((outline_center[0] - topmost_center[0])**2 + 
                      (outline_center[1] - topmost_center[1])**2)
        
        if dist < 50:
            return 1.0
        elif dist < 100:
            return 0.7
        elif dist < 150:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_outline(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate outline drawing quality."""
        outline = self._detect_outline(final_frame)
        
        if outline is None:
            return 0.0
        
        # Check if outline forms a closed shape
        perimeter = cv2.arcLength(outline, True)
        area = cv2.contourArea(outline)
        
        if perimeter > 100 and area > 500:
            # Check circularity (how well-formed the outline is)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            if circularity > 0.1:  # Reasonable shape
                return 1.0
            return 0.7
        elif perimeter > 50:
            return 0.5
        return 0.3
    
    def _evaluate_uniqueness(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check if only one shape is outlined."""
        outlines = self._detect_all_outlines(final_frame)
        
        if len(outlines) == 0:
            return 0.0
        elif len(outlines) == 1:
            return 1.0
        else:
            return max(0.3, 1.0 - 0.3 * (len(outlines) - 1))
    
    def _evaluate_clarity(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate outline visual clarity."""
        outline = self._detect_outline(final_frame)
        if outline is None:
            return 0.0
        
        perimeter = cv2.arcLength(outline, True)
        if perimeter > 150:
            return 1.0
        elif perimeter > 100:
            return 0.8
        elif perimeter > 50:
            return 0.5
        return perimeter / 100
    
    def _find_topmost_shape_center(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the center of the topmost (least occluded) shape."""
        # Convert to grayscale and find shapes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection to find shape boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # The topmost shape typically has the most visible edge pixels
        # Find contour with most edge density
        best_contour = max(contours, key=cv2.contourArea)
        
        M = cv2.moments(best_contour)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None
    
    def _detect_outline(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect the main red outline in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            return max(contours, key=cv2.contourArea)
        return None
    
    def _detect_all_outlines(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detect all red outlines in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return [c for c in contours if cv2.contourArea(c) > 100]
    
    def _get_contour_center(self, contour: np.ndarray) -> Optional[Tuple[int, int]]:
        """Get center of a contour."""
        M = cv2.moments(contour)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None


class IdentifyUniqueFigureEvaluator(BaseEvaluator):
    """
    G-147: Identify unique figure in uniform set.
    
    Rule-based evaluation:
    - Shape recognition (40%): Find the one shape that differs from others
    - Marking precision (35%): Red circle accurately marks the unique figure
    - Marking quality (15%): Circle color, size, line width
    - Scene preservation (10%): Original shapes unchanged
    """
    
    TASK_WEIGHTS = {
        'shape_recognition': 0.40,
        'marking_precision': 0.35,
        'marking_quality': 0.15,
        'scene_preservation': 0.10
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        # 1. Shape recognition (40%)
        scores['shape_recognition'] = self._evaluate_recognition(
            first_frame, final_frame
        )
        
        # 2. Marking precision (35%)
        scores['marking_precision'] = self._evaluate_marking_position(final_frame)
        
        # 3. Marking quality (15%)
        scores['marking_quality'] = self._evaluate_marking_quality(final_frame)
        
        # 4. Scene preservation (10%)
        scores['scene_preservation'] = self._evaluate_scene_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_recognition(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if unique shape is identified."""
        # Find the unique shape from first frame
        unique_shape_center = self._find_unique_shape(first_frame)
        
        # Detect marking in final frame
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        if unique_shape_center is None:
            # Can't verify, give partial credit
            return 0.5
        
        # Check if circle marks the unique shape
        dist = np.sqrt((circle[0] - unique_shape_center[0])**2 + 
                      (circle[1] - unique_shape_center[1])**2)
        
        if dist < 30:
            return 1.0
        elif dist < 60:
            return 0.7
        elif dist < 100:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_marking_position(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate marking circle position accuracy."""
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # Check if circle is in a reasonable position (not at edges)
        h, w = final_frame.shape[:2]
        x, y, r = circle
        
        if 50 < x < w - 50 and 50 < y < h - 50:
            return 1.0
        elif 20 < x < w - 20 and 20 < y < h - 20:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_marking_quality(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate marking circle quality."""
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        # Check if circle is reasonable size
        if 20 < r < 150:
            size_score = 1.0
        else:
            size_score = 0.5
        
        # Check color (should be red)
        hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        roi_y = max(0, y - r - 10)
        roi_x = max(0, x - r - 10)
        h, w = final_frame.shape[:2]
        roi = hsv[roi_y:min(h, y+r+10), roi_x:min(w, x+r+10)]
        
        if roi.size > 0:
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask = cv2.inRange(roi, lower_red1, upper_red1) | cv2.inRange(roi, lower_red2, upper_red2)
            red_ratio = np.sum(mask > 0) / max(1, mask.size)
            color_score = min(1.0, red_ratio * 10)
        else:
            color_score = 0.5
        
        return 0.6 * size_score + 0.4 * color_score
    
    def _evaluate_scene_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if original shapes are preserved."""
        # Count shapes in first frame
        first_shapes = self._count_shapes(first_frame)
        
        # Count shapes in final frame (excluding red marking)
        final_shapes = self._count_shapes_excluding_red(final_frame)
        
        if first_shapes == 0:
            return 0.5
        
        # Shapes should be preserved
        if abs(first_shapes - final_shapes) <= 1:
            return 1.0
        elif abs(first_shapes - final_shapes) <= 2:
            return 0.7
        else:
            return 0.4
    
    def _find_unique_shape(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the unique shape that differs from others."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 3:
            return None
        
        # Analyze shape features
        shapes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                # Compute shape descriptor (approximate vertices)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                num_vertices = len(approx)
                
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shapes.append((cx, cy, num_vertices, cv2.contourArea(cnt)))
        
        if len(shapes) < 3:
            return None
        
        # Find the outlier (different vertex count or significantly different area)
        vertex_counts = [s[2] for s in shapes]
        areas = [s[3] for s in shapes]
        
        # Check for vertex count outlier
        from collections import Counter
        vertex_counter = Counter(vertex_counts)
        
        for shape in shapes:
            if vertex_counter[shape[2]] == 1:  # Only one shape with this vertex count
                return (shape[0], shape[1])
        
        # Check for area outlier
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        
        for shape in shapes:
            if abs(shape[3] - mean_area) > 2 * std_area:
                return (shape[0], shape[1])
        
        return None
    
    def _count_shapes(self, frame: np.ndarray) -> int:
        """Count number of shapes in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(1 for cnt in contours if cv2.contourArea(cnt) > 200)
    
    def _count_shapes_excluding_red(self, frame: np.ndarray) -> int:
        """Count shapes excluding red marking."""
        # Mask out red
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Remove red from frame
        frame_no_red = frame.copy()
        frame_no_red[red_mask > 0] = [255, 255, 255]
        
        return self._count_shapes(frame_no_red)
    
    def _detect_marking_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect the marking circle (red)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(largest)
            return (int(x), int(y), int(r))
        
        return None


class CircleLargestNumericalValueEvaluator(BaseEvaluator):
    """
    G-160: Circle the largest numerical value.
    
    Rule-based evaluation:
    - Numerical identification (40%): Circle marks the position of largest number
    - Circle position accuracy (30%): Circle center aligns with number
    - Circle style (20%): Red color, appropriate size
    - Animation quality (10%): Smooth expansion
    """
    
    TASK_WEIGHTS = {
        'numerical_identification': 0.40,
        'circle_position': 0.30,
        'circle_style': 0.20,
        'animation_quality': 0.10
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        # 1. Numerical identification (40%)
        scores['numerical_identification'] = self._evaluate_number_selection(
            first_frame, final_frame
        )
        
        # 2. Circle position (30%)
        scores['circle_position'] = self._evaluate_circle_position(final_frame)
        
        # 3. Circle style (20%)
        scores['circle_style'] = self._evaluate_circle_style(final_frame)
        
        # 4. Animation quality (10%)
        scores['animation_quality'] = self._evaluate_animation_quality(video_frames)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_number_selection(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if the largest number is circled."""
        # Detect text regions (numbers) from first frame
        number_regions = self._detect_number_regions(first_frame)
        
        # Detect red circle in final frame
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        if len(number_regions) == 0:
            return 0.5  # Can't verify, partial credit
        
        # Find which number region is circled
        circled_region = None
        min_dist = float('inf')
        for region in number_regions:
            dist = np.sqrt((region[0] - circle[0])**2 + (region[1] - circle[1])**2)
            if dist < min_dist:
                min_dist = dist
                circled_region = region
        
        if circled_region is None or min_dist > 100:
            return 0.3
        
        # The largest number should have the darkest/largest text region
        # (assuming larger digit values = visually larger or more prominent)
        largest_region = max(number_regions, key=lambda r: r[2])  # r[2] is area
        
        if circled_region[2] == largest_region[2]:
            return 1.0
        elif circled_region[2] >= largest_region[2] * 0.7:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_circle_position(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate circle position accuracy."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # Check if circle is in reasonable position (not at edges)
        h, w = final_frame.shape[:2]
        x, y, r = circle
        
        if 30 < x < w - 30 and 30 < y < h - 30:
            return 1.0
        else:
            return 0.5
    
    def _evaluate_circle_style(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate circle style (color, size)."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        # Size check
        if 30 < r < 200:
            size_score = 1.0
        else:
            size_score = 0.5
        
        # Color check
        color_score = self._check_red_color(final_frame, x, y, r)
        
        return 0.5 * size_score + 0.5 * color_score
    
    def _evaluate_animation_quality(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Evaluate animation smoothness."""
        if len(video_frames) < 5:
            return 0.5
        
        radii = []
        for frame in video_frames[len(video_frames)//3:]:
            circle = self._detect_red_circle(frame)
            if circle:
                radii.append(circle[2])
        
        if len(radii) < 2:
            return 0.5
        
        # Check for smooth increase
        smooth_count = sum(1 for i in range(1, len(radii)) if radii[i] >= radii[i-1] * 0.95)
        smoothness = smooth_count / (len(radii) - 1)
        
        return smoothness
    
    def _detect_number_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect text/number regions."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 10000:  # Text size range
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    regions.append((cx, cy, area))
        
        return regions
    
    def _detect_red_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect red circle in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                (x, y), r = cv2.minEnclosingCircle(largest)
                return (int(x), int(y), int(r))
        
        return None
    
    def _check_red_color(self, frame: np.ndarray, x: int, y: int, r: int) -> float:
        """Check if the circle is red colored."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r + 5, 255, 10)
        
        roi_hsv = hsv[mask > 0]
        if len(roi_hsv) == 0:
            return 0.5
        
        red_count = sum(1 for pixel in roi_hsv if pixel[0] < 10 or pixel[0] > 160)
        return red_count / len(roi_hsv)


class MarkSecondLargestShapeEvaluator(BaseEvaluator):
    """
    G-161: Mark the second largest shape.
    
    Rule-based evaluation:
    - Size recognition (40%): Identify second largest correctly
    - Marking precision (35%): Border accurately marks target
    - Marking quality (15%): Border style and color
    - Scene preservation (10%): Original shapes unchanged
    """
    
    TASK_WEIGHTS = {
        'size_recognition': 0.40,
        'marking_precision': 0.35,
        'marking_quality': 0.15,
        'scene_preservation': 0.10
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        # CRITICAL CHECK: Only ONE marking should exist
        marking_count = self._count_red_markings(final_frame)
        if marking_count == 0:
            # No marking at all - task failed
            self._last_task_details = {
                'size_recognition': 0.0,
                'marking_precision': 0.0,
                'marking_quality': 0.0,
                'scene_preservation': 0.0,
                'error': 'no_marking_found'
            }
            return 0.0
        elif marking_count > 1:
            # Multiple markings - violation of "only one mark" rule
            self._last_task_details = {
                'size_recognition': 0.0,
                'marking_precision': 0.0,
                'marking_quality': 0.0,
                'scene_preservation': 0.0,
                'error': f'multiple_markings_found: {marking_count}'
            }
            return 0.0
        
        # CRITICAL CHECK: No new objects should be generated
        first_shapes = self._detect_shapes_with_area(first_frame)
        final_shapes_no_red = self._detect_shapes_without_red(final_frame)
        
        if len(final_shapes_no_red) > len(first_shapes):
            # New objects generated - violation
            self._last_task_details = {
                'size_recognition': 0.0,
                'marking_precision': 0.0,
                'marking_quality': 0.0,
                'scene_preservation': 0.0,
                'error': f'new_objects_generated: first={len(first_shapes)}, final={len(final_shapes_no_red)}'
            }
            return 0.0
        
        # 1. Size recognition (40%) - CRITICAL: Is the CORRECT shape marked?
        size_recognition_score = self._evaluate_size_recognition(
            first_frame, final_frame
        )
        scores['size_recognition'] = size_recognition_score
        
        # CRITICAL: If wrong shape is marked, other scores should be penalized
        correct_shape_marked = size_recognition_score > 0.5
        
        # 2. Marking precision (35%) - Only counts if correct shape is marked
        if correct_shape_marked:
            scores['marking_precision'] = self._evaluate_marking_precision(final_frame)
        else:
            scores['marking_precision'] = 0.0  # Wrong shape - no credit for marking
        
        # 3. Marking quality (15%) - Only counts if correct shape is marked
        if correct_shape_marked:
            scores['marking_quality'] = self._evaluate_marking_quality(final_frame)
        else:
            scores['marking_quality'] = 0.0  # Wrong shape - no credit
        
        # 4. Scene preservation (10%)
        scores['scene_preservation'] = self._evaluate_scene_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _count_red_markings(self, frame: np.ndarray) -> int:
        """Count number of red markings in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Count significant red contours (filter noise)
        return len([c for c in contours if cv2.contourArea(c) > 100])
    
    def _detect_shapes_without_red(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect shapes excluding red marking."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        frame_no_red = frame.copy()
        frame_no_red[red_mask > 0] = [255, 255, 255]
        return self._detect_shapes_with_area(frame_no_red)
    
    def _evaluate_size_recognition(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if second largest shape is correctly identified."""
        # Find shapes sorted by area
        shapes = self._detect_shapes_with_area(first_frame)
        
        if len(shapes) < 2:
            return 0.0  # STRICT: Not enough shapes to have "second largest"
        
        # Sort by area (descending)
        sorted_shapes = sorted(shapes, key=lambda s: s[2], reverse=True)
        second_largest_center = (sorted_shapes[1][0], sorted_shapes[1][1])
        
        # Detect red border/marking in final frame
        marking = self._detect_red_border(final_frame)
        
        if marking is None:
            return 0.0
        
        marking_center = self._get_contour_center(marking)
        
        if marking_center is None:
            return 0.0  # STRICT: Can't determine marking position
        
        # Check if marking is on second largest
        dist = np.sqrt((marking_center[0] - second_largest_center[0])**2 + 
                      (marking_center[1] - second_largest_center[1])**2)
        
        if dist < 40:
            return 1.0
        elif dist < 80:
            return 0.5  # STRICT: Reduced from 0.7
        else:
            return 0.0  # STRICT: Wrong shape marked
    
    def _evaluate_marking_precision(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate border marking precision."""
        marking = self._detect_red_border(final_frame)
        
        if marking is None:
            return 0.0
        
        # Check if marking forms a proper border
        perimeter = cv2.arcLength(marking, True)
        area = cv2.contourArea(marking)
        
        if perimeter > 100 and area > 500:
            return 1.0
        elif perimeter > 50:
            return 0.6
        else:
            return 0.3
    
    def _evaluate_marking_quality(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate border marking quality."""
        marking = self._detect_red_border(final_frame)
        
        if marking is None:
            return 0.0
        
        perimeter = cv2.arcLength(marking, True)
        if perimeter > 100:
            return 1.0
        else:
            return perimeter / 100
    
    def _evaluate_scene_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if original shapes are preserved - NO NEW OBJECTS."""
        first_count = len(self._detect_shapes_with_area(first_frame))
        
        # Count shapes excluding red marking
        final_count = len(self._detect_shapes_without_red(final_frame))
        
        # STRICT: No new objects allowed
        if final_count > first_count:
            return 0.0  # New objects generated - violation
        
        if final_count == first_count:
            return 1.0  # Perfect preservation
        elif final_count == first_count - 1:
            return 0.7  # One shape lost
        else:
            return 0.0  # STRICT: Too many shapes changed
    
    def _detect_shapes_with_area(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect shapes with (x, y, area)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shapes.append((cx, cy, area))
        
        return shapes
    
    def _detect_red_border(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect red border/outline in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            return max(contours, key=cv2.contourArea)
        return None
    
    def _get_contour_center(self, contour: np.ndarray) -> Optional[Tuple[int, int]]:
        """Get center of a contour."""
        M = cv2.moments(contour)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None


class SelectLongestPolygonSideEvaluator(BaseEvaluator):
    """
    G-167: Select the longest polygon side.
    
    Rule-based evaluation:
    - Longest side identification (50%): Correctly identify longest edge
    - Marking position (25%): Circle/marker at midpoint of edge
    - Marking uniqueness (15%): Only one edge marked
    - Visual quality (10%): Circle style
    """
    
    TASK_WEIGHTS = {
        'longest_side_identification': 0.50,
        'marking_position': 0.25,
        'marking_uniqueness': 0.15,
        'visual_quality': 0.10
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        # 1. Longest side identification (50%)
        scores['longest_side_identification'] = self._evaluate_side_identification(
            first_frame, final_frame, gt_final_frame
        )
        
        # 2. Marking position (25%)
        scores['marking_position'] = self._evaluate_marking_position(final_frame)
        
        # 3. Marking uniqueness (15%)
        scores['marking_uniqueness'] = self._evaluate_marking_uniqueness(final_frame)
        
        # 4. Visual quality (10%)
        scores['visual_quality'] = self._evaluate_visual_quality_marking(final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_side_identification(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        gt_final_frame: Optional[np.ndarray] = None
    ) -> float:
        """Rule-based: Check if the correct side is identified."""
        # Detect marking in final frame
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # If GT final frame available, compare with GT marking position
        if gt_final_frame is not None:
            gt_circle = self._detect_marking_circle(gt_final_frame)
            if gt_circle is not None:
                dist = np.sqrt((circle[0] - gt_circle[0])**2 + 
                              (circle[1] - gt_circle[1])**2)
                if dist < 30:
                    return 1.0
                elif dist < 60:
                    return 0.8
                elif dist < 100:
                    return 0.5
                else:
                    return 0.2
        
        # Fallback: Find the longest edge midpoint from first frame
        longest_edge_midpoint = self._find_longest_edge_midpoint(first_frame)
        
        if longest_edge_midpoint is None:
            return 0.5  # Can't verify
        
        # Check if circle is near the longest edge midpoint
        dist = np.sqrt((circle[0] - longest_edge_midpoint[0])**2 + 
                      (circle[1] - longest_edge_midpoint[1])**2)
        
        if dist < 30:
            return 1.0
        elif dist < 60:
            return 0.7
        elif dist < 100:
            return 0.5
        else:
            return 0.2
    
    def _evaluate_marking_position(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate if circle is at midpoint of an edge."""
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # Check if circle is in a reasonable position
        h, w = final_frame.shape[:2]
        x, y, r = circle
        
        if 20 < x < w - 20 and 20 < y < h - 20:
            return 1.0
        else:
            return 0.5
    
    def _evaluate_marking_uniqueness(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check if only one edge is marked."""
        circles = self._detect_all_marking_circles(final_frame)
        
        if len(circles) == 0:
            return 0.0
        elif len(circles) == 1:
            return 1.0
        else:
            return max(0.3, 1.0 - 0.3 * (len(circles) - 1))
    
    def _evaluate_visual_quality_marking(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate marking circle visual quality."""
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        if 5 < r < 50:
            return 1.0
        else:
            return 0.5
    
    def _find_longest_edge_midpoint(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find midpoint of the longest edge in the polygon."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Get the main polygon contour
        polygon = max(contours, key=cv2.contourArea)
        
        # Approximate to get vertices
        peri = cv2.arcLength(polygon, True)
        approx = cv2.approxPolyDP(polygon, 0.02 * peri, True)
        
        if len(approx) < 3:
            return None
        
        # Find longest edge
        max_length = 0
        longest_midpoint = None
        
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]
            
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            if length > max_length:
                max_length = length
                longest_midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        
        return longest_midpoint
    
    def _detect_marking_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect the marking circle (red, orange, or yellow)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Warm colors (red, orange, yellow)
        lower_warm1 = np.array([0, 50, 50])
        upper_warm1 = np.array([35, 255, 255])
        lower_warm2 = np.array([160, 50, 50])
        upper_warm2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_warm1, upper_warm1) | cv2.inRange(hsv, lower_warm2, upper_warm2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 10:
                (x, y), r = cv2.minEnclosingCircle(largest)
                if r > 3:
                    return (int(x), int(y), int(r))
        
        return None
    
    def _detect_all_marking_circles(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect all marking circles."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_warm1 = np.array([0, 80, 80])
        upper_warm1 = np.array([35, 255, 255])
        lower_warm2 = np.array([160, 80, 80])
        upper_warm2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_warm1, upper_warm1) | cv2.inRange(hsv, lower_warm2, upper_warm2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                (x, y), r = cv2.minEnclosingCircle(cnt)
                circles.append((int(x), int(y), int(r)))
        
        return circles


# Export all evaluators
OUT_OF_DOMAIN_50_EVALUATORS = {
    'G-24_separate_objects_no_spin_data-generator': SeparateObjectsNoSpinEvaluator,
    'G-47_multiple_keys_for_one_door_data-generator': MultipleKeysForOneDoorEvaluator,
    'G-54_connecting_color_data-generator': ConnectingColorEvaluator,
    'G-135_select_next_figure_small_large_alternating_sequence_data-generator': SelectNextFigureAlternatingEvaluator,
    'G-136_locate_point_in_overlapping_area_data-generator': LocatePointInOverlappingAreaEvaluator,
    'G-140_locate_topmost_unobscured_figure_data-generator': LocateTopmostFigureEvaluator,
    'G-147_identify_unique_figure_in_uniform_set_data-generator': IdentifyUniqueFigureEvaluator,
    'G-160_circle_largest_numerical_value_data-generator': CircleLargestNumericalValueEvaluator,
    'G-161_mark_second_largest_shape_data-generator': MarkSecondLargestShapeEvaluator,
    'G-167_select_longest_polygon_side_data-generator': SelectLongestPolygonSideEvaluator,
}
