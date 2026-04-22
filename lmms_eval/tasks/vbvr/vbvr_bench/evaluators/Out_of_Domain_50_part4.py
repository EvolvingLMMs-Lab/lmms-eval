"""
Specific evaluators for Out-of-Domain_50 tasks (Part 4).
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator




class SymbolDeletionEvaluator(BaseEvaluator):
    """
    O-5: Symbol deletion evaluator.
    
    Rule-based evaluation:
    - Target identification & deletion (40%): Correct symbol (with red border) removed
    - Symbol preservation (35%): All OTHER symbols' colors remain unchanged
    - Sequence order preservation (15%): Remaining symbols in order
    - Layout & alignment (10%): Centered, evenly spaced
    """
    
    TASK_WEIGHTS = {
        'deletion_accuracy': 0.40,
        'symbol_preservation': 0.35,
        'order_preservation': 0.15,
        'layout_alignment': 0.10
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
        
        # Get symbols with colors
        first_symbols = self._detect_symbols_with_color(first_frame)
        final_symbols = self._detect_symbols_with_color(final_frame)
        
        first_count = len(first_symbols)
        final_count = len(final_symbols)
        
        # CRITICAL CHECK 1: Exactly one symbol must be deleted
        if final_count != first_count - 1:
            # Task failed - wrong number of symbols deleted
            if final_count >= first_count:
                # No deletion or symbols added
                self._last_task_details = {
                    'deletion_accuracy': 0.0,
                    'symbol_preservation': 0.0,
                    'order_preservation': 0.0,
                    'layout_alignment': 0.0,
                    'no_deletion': True
                }
                return 0.0
            elif final_count < first_count - 1:
                # Too many deleted
                self._last_task_details = {
                    'deletion_accuracy': 0.1,
                    'symbol_preservation': 0.0,
                    'order_preservation': 0.0,
                    'layout_alignment': 0.0,
                    'too_many_deleted': True
                }
                return 0.04  # 0.1 * 0.4
        
        # Check deletion accuracy (was the correct symbol deleted?)
        scores['deletion_accuracy'] = self._evaluate_deletion(first_symbols, final_symbols, first_frame)
        
        # CRITICAL: Check if other symbols' colors are preserved
        scores['symbol_preservation'] = self._evaluate_symbol_preservation(first_symbols, final_symbols, first_frame)
        
        # If symbol preservation is very low, it means colors changed significantly
        if scores['symbol_preservation'] < 0.5:
            scores['order_preservation'] = 0.0
            scores['layout_alignment'] = 0.0
            self._last_task_details = scores
            self._last_task_details['colors_changed'] = True
            return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
        
        scores['order_preservation'] = self._evaluate_order(first_symbols, final_symbols)
        scores['layout_alignment'] = self._evaluate_layout(final_symbols)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_symbols_with_color(self, frame: np.ndarray) -> List[Dict]:
        """Detect symbols with their centers and average colors."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        symbols = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 200 < area < 20000:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Get average color
                    mask_cnt = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(frame, mask=mask_cnt)[:3]
                    # Get HSV hue for color matching
                    color_arr = np.array(mean_color, dtype=np.uint8).reshape(1, 1, 3)
                    hsv = cv2.cvtColor(color_arr, cv2.COLOR_BGR2HSV)[0, 0]
                    symbols.append({
                        'center': (cx, cy),
                        'color': mean_color,
                        'hue': int(hsv[0]),
                        'area': area
                    })
        
        return sorted(symbols, key=lambda s: s['center'][0])  # Sort by x
    
    def _detect_red_border(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect the red border marking the target symbol."""
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
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    return (cx, cy)
        return None
    
    def _evaluate_deletion(self, first_symbols: List[Dict], final_symbols: List[Dict], first_frame: np.ndarray) -> float:
        """Check if exactly one symbol is deleted (the one with red border)."""
        first_count = len(first_symbols)
        final_count = len(final_symbols)
        
        if final_count != first_count - 1:
            if final_count == first_count:
                return 0.2  # Nothing deleted
            elif final_count < first_count - 1:
                return 0.1  # Too many deleted
            else:
                return 0.0  # Symbols added
        
        # Check if the deleted symbol was the one with red border
        red_border_pos = self._detect_red_border(first_frame)
        if red_border_pos is None:
            return 0.7  # Can't verify which was marked
        
        # Find which symbol was deleted
        first_centers = [s['center'] for s in first_symbols]
        final_centers = [s['center'] for s in final_symbols]
        
        # Find the deleted symbol by checking which first symbol is missing
        deleted_idx = None
        for i, fc in enumerate(first_centers):
            # Check if this symbol has a match in final
            has_match = False
            for fnc in final_centers:
                # Allow some position shift due to re-centering
                if abs(fc[0] - fnc[0]) < 100 and abs(fc[1] - fnc[1]) < 50:
                    has_match = True
                    break
            if not has_match:
                deleted_idx = i
                break
        
        if deleted_idx is None:
            return 0.5
        
        # Check if deleted symbol was near the red border
        deleted_center = first_centers[deleted_idx]
        dist_to_red = np.sqrt((deleted_center[0] - red_border_pos[0])**2 + 
                              (deleted_center[1] - red_border_pos[1])**2)
        
        if dist_to_red < 50:
            return 1.0  # Correct symbol deleted
        elif dist_to_red < 100:
            return 0.6
        else:
            return 0.3  # Wrong symbol deleted
    
    def _evaluate_symbol_preservation(self, first_symbols: List[Dict], final_symbols: List[Dict], first_frame: np.ndarray) -> float:
        """CRITICAL: Check if all OTHER symbols' colors remain unchanged."""
        if len(final_symbols) == 0:
            return 0.0
        
        # Find which symbol was marked for deletion (has red border)
        red_border_pos = self._detect_red_border(first_frame)
        
        # Get colors of non-deleted symbols from first frame
        expected_symbols = []
        for sym in first_symbols:
            # Skip the marked symbol
            if red_border_pos:
                dist = np.sqrt((sym['center'][0] - red_border_pos[0])**2 + 
                              (sym['center'][1] - red_border_pos[1])**2)
                if dist < 50:
                    continue
            expected_symbols.append(sym)
        
        if len(expected_symbols) == 0 or len(final_symbols) == 0:
            return 0.5
        
        # STRICT: Each expected symbol must have EXACTLY one matching symbol in final
        # Use greedy matching with strict hue threshold
        matched = 0
        used = set()
        for exp_sym in expected_symbols:
            exp_hue = exp_sym['hue']
            best_match = None
            best_diff = float('inf')
            for i, act_sym in enumerate(final_symbols):
                if i in used:
                    continue
                act_hue = act_sym['hue']
                
                # Calculate hue difference with proper wrapping
                hue_diff = abs(exp_hue - act_hue)
                hue_diff = min(hue_diff, 180 - hue_diff)
                
                if hue_diff < best_diff:
                    best_diff = hue_diff
                    best_match = i
            
            # STRICT threshold: hue must be within 15 degrees
            if best_match is not None and best_diff < 15:
                matched += 1
                used.add(best_match)
        
        # If not all symbols are matched, return low score
        preservation_ratio = matched / len(expected_symbols) if expected_symbols else 0.0
        
        # If less than all expected symbols are preserved, heavily penalize
        if preservation_ratio < 1.0:
            return preservation_ratio * 0.5  # Scale down
        
        return 1.0
    
    def _evaluate_order(self, first_symbols: List[Dict], final_symbols: List[Dict]) -> float:
        """Check if remaining symbols maintain order."""
        if len(final_symbols) < 2:
            return 0.5
        
        # Check if x-coordinates maintain relative order
        final_x = [s['center'][0] for s in final_symbols]
        is_ordered = all(final_x[i] < final_x[i+1] for i in range(len(final_x)-1))
        
        return 1.0 if is_ordered else 0.5
    
    def _evaluate_layout(self, final_symbols: List[Dict]) -> float:
        """Check horizontal alignment."""
        if len(final_symbols) < 2:
            return 0.5
        
        y_coords = [s['center'][1] for s in final_symbols]
        y_var = np.var(y_coords)
        
        if y_var < 100:
            return 1.0
        elif y_var < 500:
            return 0.7
        else:
            return 0.4


class GeometricTransformationEvaluator(BaseEvaluator):
    """
    O-6: 2D geometric transformation (rotation) evaluator.
    
    Rule-based evaluation:
    - Rotation center correctness (30%): Rotate around marked point
    - Rotation angle accuracy (35%): Shape aligns with target outline
    - Position alignment precision (25%): Overlap with target
    - Shape fidelity (10%): Size and shape preserved
    """
    
    TASK_WEIGHTS = {
        'rotation_center': 0.30,
        'rotation_angle': 0.35,
        'position_alignment': 0.25,
        'shape_fidelity': 0.10
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
        
        scores['rotation_center'] = self._evaluate_rotation_center(video_frames)
        scores['rotation_angle'] = self._evaluate_rotation_angle(first_frame, final_frame)
        scores['position_alignment'] = self._evaluate_position(first_frame, final_frame)
        scores['shape_fidelity'] = self._evaluate_shape_fidelity(first_frame, final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_rotation_center(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Check if rotation is around the correct center point."""
        # Track shape center across frames
        centers = []
        for frame in video_frames[::max(1, len(video_frames)//10)]:
            center = self._find_shape_center(frame)
            if center:
                centers.append(center)
        
        if len(centers) < 3:
            return 0.0  # STRICT: Not enough centers detected
        
        # Check if centers follow circular path
        # Calculate variance of distances from a potential center
        all_x = [c[0] for c in centers]
        all_y = [c[1] for c in centers]
        
        avg_x = np.mean(all_x)
        avg_y = np.mean(all_y)
        
        distances = [np.sqrt((c[0] - avg_x)**2 + (c[1] - avg_y)**2) for c in centers]
        dist_var = np.var(distances)
        
        # Low variance means consistent rotation around a center
        if dist_var < 100:
            return 1.0
        elif dist_var < 500:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_rotation_angle(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if rotation angle is correct."""
        # Detect target outline in first frame
        target = self._detect_target_outline(first_frame)
        
        # Detect shape in final frame
        shape = self._detect_main_shape(final_frame)
        
        if target is None or shape is None:
            return 0.5
        
        # Compare shape orientation with target
        target_angle = self._get_contour_angle(target)
        shape_angle = self._get_contour_angle(shape)
        
        angle_diff = abs(target_angle - shape_angle)
        angle_diff = min(angle_diff, 180 - angle_diff)  # Handle wrap-around
        
        if angle_diff < 10:
            return 1.0
        elif angle_diff < 20:
            return 0.8
        elif angle_diff < 45:
            return 0.5
        else:
            return 0.2
    
    def _evaluate_position(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if shape aligns with target outline."""
        # Get target and shape positions
        target_center = self._find_target_center(first_frame)
        shape_center = self._find_shape_center(final_frame)
        
        if target_center is None or shape_center is None:
            return 0.5
        
        dist = np.sqrt((target_center[0] - shape_center[0])**2 + 
                      (target_center[1] - shape_center[1])**2)
        
        if dist < 30:
            return 1.0
        elif dist < 60:
            return 0.7
        elif dist < 100:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_shape_fidelity(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if shape is preserved."""
        first_shape = self._detect_main_shape(first_frame)
        final_shape = self._detect_main_shape(final_frame)
        
        if first_shape is None or final_shape is None:
            return 0.5
        
        # Compare areas
        first_area = cv2.contourArea(first_shape)
        final_area = cv2.contourArea(final_shape)
        
        if first_area == 0:
            return 0.5
        
        area_ratio = final_area / first_area
        
        if 0.8 < area_ratio < 1.2:
            return 1.0
        elif 0.6 < area_ratio < 1.4:
            return 0.7
        else:
            return 0.4
    
    def _find_shape_center(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find center of main shape."""
        shape = self._detect_main_shape(frame)
        if shape is None:
            return None
        
        M = cv2.moments(shape)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None
    
    def _detect_main_shape(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect the main (colored) shape."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect colored (not gray) regions
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            return max(contours, key=cv2.contourArea)
        return None
    
    def _detect_target_outline(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect target outline (dashed or dotted)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            return max(contours, key=cv2.contourArea)
        return None
    
    def _find_target_center(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find center of target outline."""
        target = self._detect_target_outline(frame)
        if target is None:
            return None
        
        M = cv2.moments(target)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None
    
    def _get_contour_angle(self, contour: np.ndarray) -> float:
        """Get orientation angle of contour."""
        if len(contour) < 5:
            return 0.0
        
        try:
            ellipse = cv2.fitEllipse(contour)
            return ellipse[2]  # angle
        except Exception:
            return 0.0


class ShapeScalingAnalogyEvaluator(BaseEvaluator):
    """
    O-9: Shape scaling analogy evaluator.
    
    Rule-based evaluation:
    - Element preservation (40%): A, B, C (left-up, right-up, left-bottom) remain UNCHANGED
    - Scaling ratio correctness (35%): D size follows A→B trend (larger/smaller)
    - Shape type matching (20%): D has same shape type as C
    - Position correctness (5%): D is in bottom-right quadrant
    """
    
    TASK_WEIGHTS = {
        'element_preservation': 0.40,
        'scaling_ratio': 0.35,
        'shape_type_matching': 0.20,
        'position_correctness': 0.05
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
        
        # Get shapes in each quadrant
        first_shapes = self._get_quadrant_shapes_detailed(first_frame)
        final_shapes = self._get_quadrant_shapes_detailed(final_frame)
        
        # 1. CRITICAL: Check if A, B, C are preserved (unchanged)
        scores['element_preservation'] = self._evaluate_element_preservation(
            first_shapes, final_shapes
        )
        
        # If elements are not preserved, heavily penalize
        if scores['element_preservation'] < 0.5:
            self._last_task_details = {
                'element_preservation': scores['element_preservation'],
                'scaling_ratio': 0.0,
                'shape_type_matching': 0.0,
                'position_correctness': 0.0,
                'elements_changed': True
            }
            return scores['element_preservation'] * self.TASK_WEIGHTS['element_preservation']
        
        # 2. Check scaling ratio
        scores['scaling_ratio'] = self._evaluate_scaling_ratio(first_shapes, final_shapes)
        
        # 3. Check shape type matching
        scores['shape_type_matching'] = self._evaluate_shape_type(first_shapes, final_shapes)
        
        # 4. Check position
        scores['position_correctness'] = self._evaluate_position(final_shapes)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _get_quadrant_shapes_detailed(self, frame: np.ndarray) -> Dict:
        """Get shapes in each quadrant with detailed info."""
        h, w = frame.shape[:2]
        
        quadrants = {
            'A': frame[:h//2, :w//2],      # Top-left
            'B': frame[:h//2, w//2:],      # Top-right
            'C': frame[h//2:, :w//2],      # Bottom-left
            'D': frame[h//2:, w//2:]       # Bottom-right
        }
        
        result = {}
        for name, region in quadrants.items():
            shapes = self._detect_shapes_detailed(region)
            if shapes:
                # Take the largest shape
                largest = max(shapes, key=lambda s: s['area'])
                result[name] = largest
        
        return result
    
    def _detect_shapes_detailed(self, region: np.ndarray) -> List[Dict]:
        """Detect shapes with detailed info (area, vertices, color)."""
        if region.size == 0:
            return []
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # Detect colored shapes
        mask = hsv[:, :, 1] > 30
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Get shape type by vertices
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                    vertices = len(approx)
                    # Get color
                    mask_cnt = np.zeros(region.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(region, mask=mask_cnt)[:3]
                    color_arr = np.array(mean_color, dtype=np.uint8).reshape(1, 1, 3)
                    hsv_c = cv2.cvtColor(color_arr, cv2.COLOR_BGR2HSV)[0, 0]
                    
                    shapes.append({
                        'center': (cx, cy),
                        'area': area,
                        'vertices': vertices,
                        'hue': int(hsv_c[0]),
                        'color': mean_color
                    })
        
        return shapes
    
    def _evaluate_element_preservation(self, first_shapes: Dict, final_shapes: Dict) -> float:
        """Check if A, B, C remain unchanged."""
        preserved = 0
        total = 0
        
        for quadrant in ['A', 'B', 'C']:
            if quadrant not in first_shapes:
                continue
            total += 1
            
            if quadrant not in final_shapes:
                continue  # Shape disappeared - bad
            
            first_s = first_shapes[quadrant]
            final_s = final_shapes[quadrant]
            
            # Check area similarity
            area_ratio = final_s['area'] / first_s['area'] if first_s['area'] > 0 else 0
            if 0.7 < area_ratio < 1.3:
                # Check color similarity
                hue_diff = abs(first_s['hue'] - final_s['hue'])
                hue_diff = min(hue_diff, 180 - hue_diff)
                if hue_diff < 20:
                    preserved += 1
        
        return preserved / total if total > 0 else 0.0
    
    def _evaluate_scaling_ratio(self, first_shapes: Dict, final_shapes: Dict) -> float:
        """Check if D follows the A→B scaling trend."""
        # Get A, B sizes from first frame (or final for A, B since they should be unchanged)
        a_size = first_shapes.get('A', {}).get('area', 0)
        b_size = first_shapes.get('B', {}).get('area', 0)
        c_size = final_shapes.get('C', {}).get('area', 0)
        d_size = final_shapes.get('D', {}).get('area', 0)
        
        if a_size == 0 or c_size == 0:
            return 0.5
        
        if d_size == 0:
            return 0.0  # No D shape generated
        
        # Determine trend: is B larger or smaller than A?
        ab_ratio = b_size / a_size if a_size > 0 else 1
        
        # D should follow the same trend relative to C
        cd_ratio = d_size / c_size if c_size > 0 else 1
        
        # Check if trend direction matches
        ab_trend = "larger" if ab_ratio > 1.1 else ("smaller" if ab_ratio < 0.9 else "same")
        cd_trend = "larger" if cd_ratio > 1.1 else ("smaller" if cd_ratio < 0.9 else "same")
        
        if ab_trend != cd_trend:
            return 0.2  # Wrong trend direction
        
        # Check if ratio is similar
        ratio_diff = abs(ab_ratio - cd_ratio) / max(ab_ratio, cd_ratio, 1)
        
        if ratio_diff < 0.15:
            return 1.0
        elif ratio_diff < 0.3:
            return 0.7
        elif ratio_diff < 0.5:
            return 0.5
        else:
            return 0.3
    
    def _evaluate_shape_type(self, first_shapes: Dict, final_shapes: Dict) -> float:
        """Check if D has the same shape type as C."""
        c_shape = first_shapes.get('C', {})
        d_shape = final_shapes.get('D', {})
        
        if not c_shape or not d_shape:
            return 0.5
        
        c_vertices = c_shape.get('vertices', 0)
        d_vertices = d_shape.get('vertices', 0)
        
        # Same number of vertices = same shape type
        if c_vertices == d_vertices:
            return 1.0
        elif abs(c_vertices - d_vertices) <= 1:
            return 0.7
        else:
            return 0.3
    
    def _evaluate_position(self, final_shapes: Dict) -> float:
        """Check if D exists in bottom-right quadrant."""
        if 'D' in final_shapes and final_shapes['D'].get('area', 0) > 0:
            return 1.0
        return 0.0


class ShapeColorThenMoveEvaluator(BaseEvaluator):
    """
    O-11: Shape color then move evaluator.
    
    Rule-based evaluation:
    - First row preservation (40%): A, B, C in top row MUST remain unchanged
    - Second row completion (35%): D stays, E and F are added with B's color
    - Color accuracy (20%): E and F have B's color
    - Shape count (5%): Final should have 6 shapes (3 top + 3 bottom)
    """
    
    TASK_WEIGHTS = {
        'first_row_preservation': 0.40,
        'second_row_completion': 0.35,
        'color_accuracy': 0.20,
        'shape_count': 0.05
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
        
        h, w = first_frame.shape[:2]
        
        # Detect shapes in first and final frames
        first_shapes = self._detect_shapes_with_info(first_frame)
        final_shapes = self._detect_shapes_with_info(final_frame)
        
        # 1. CRITICAL: First row (top half) must be preserved
        scores['first_row_preservation'] = self._evaluate_first_row_preservation(
            first_shapes, final_shapes, h
        )
        
        # If first row is not preserved, heavily penalize
        if scores['first_row_preservation'] < 0.5:
            self._last_task_details = {
                'first_row_preservation': scores['first_row_preservation'],
                'second_row_completion': 0.0,
                'color_accuracy': 0.0,
                'shape_count': 0.0,
                'first_row_destroyed': True
            }
            return scores['first_row_preservation'] * self.TASK_WEIGHTS['first_row_preservation']
        
        # 2. Second row should have 3 shapes (D, E, F)
        scores['second_row_completion'] = self._evaluate_second_row(
            first_shapes, final_shapes, h
        )
        
        # 3. Color accuracy - E and F should have B's color
        scores['color_accuracy'] = self._evaluate_color_accuracy(
            first_shapes, final_shapes, h, w
        )
        
        # 4. Shape count
        scores['shape_count'] = self._evaluate_shape_count(final_shapes)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_shapes_with_info(self, frame: np.ndarray) -> List[Dict]:
        """Detect shapes with position and color info."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = hsv[:, :, 1] > 30
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter by area - shapes should be reasonably sized
            if 1000 < area < 20000:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    mask_cnt = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(frame, mask=mask_cnt)[:3]
                    color_arr = np.array(mean_color, dtype=np.uint8).reshape(1, 1, 3)
                    hsv_c = cv2.cvtColor(color_arr, cv2.COLOR_BGR2HSV)[0, 0]
                    shapes.append({
                        'center': (cx, cy),
                        'hue': int(hsv_c[0]),
                        'area': area
                    })
        return shapes
    
    def _evaluate_first_row_preservation(self, first_shapes: List[Dict], 
                                         final_shapes: List[Dict], h: int) -> float:
        """Check if first row (A, B, C) is preserved."""
        # Get shapes in top half
        first_top = [s for s in first_shapes if s['center'][1] < h // 2]
        final_top = [s for s in final_shapes if s['center'][1] < h // 2]
        
        if len(first_top) == 0:
            return 0.5
        
        # Should have same number of shapes in top row
        if len(final_top) < len(first_top):
            return 0.1  # Shapes removed from top row
        
        # Check if hues are preserved
        first_hues = sorted([s['hue'] for s in first_top])
        final_hues = sorted([s['hue'] for s in final_top[:len(first_top)]])
        
        matched = 0
        for fh in first_hues:
            for i, fnlh in enumerate(final_hues):
                hue_diff = abs(fh - fnlh)
                hue_diff = min(hue_diff, 180 - hue_diff)
                if hue_diff < 20:
                    matched += 1
                    break
        
        return matched / len(first_hues) if first_hues else 0.0
    
    def _evaluate_second_row(self, first_shapes: List[Dict], 
                             final_shapes: List[Dict], h: int) -> float:
        """Check if second row has D, E, F."""
        # Get shapes in bottom half
        first_bottom = [s for s in first_shapes if s['center'][1] >= h // 2]
        final_bottom = [s for s in final_shapes if s['center'][1] >= h // 2]
        
        # First frame should have 1 shape (D), final should have 3 (D, E, F)
        if len(first_bottom) == 0:
            return 0.5
        
        if len(final_bottom) >= 3:
            return 1.0
        elif len(final_bottom) == 2:
            return 0.6
        elif len(final_bottom) == 1:
            return 0.3
        else:
            return 0.0
    
    def _evaluate_color_accuracy(self, first_shapes: List[Dict], 
                                 final_shapes: List[Dict], h: int, w: int) -> float:
        """Check if E and F have B's color."""
        # Find B's color (top-middle shape in first frame)
        first_top = [s for s in first_shapes if s['center'][1] < h // 2]
        b_shapes = [s for s in first_top if w // 3 < s['center'][0] < 2 * w // 3]
        
        if len(b_shapes) == 0:
            return 0.5
        
        b_hue = b_shapes[0]['hue']
        
        # Get E and F (middle and right shapes in bottom row of final frame)
        final_bottom = [s for s in final_shapes if s['center'][1] >= h // 2]
        ef_shapes = [s for s in final_bottom if s['center'][0] > w // 3]
        
        if len(ef_shapes) == 0:
            return 0.0
        
        # Check if E and F have B's color
        correct = 0
        for s in ef_shapes:
            hue_diff = abs(s['hue'] - b_hue)
            hue_diff = min(hue_diff, 180 - hue_diff)
            if hue_diff < 20:
                correct += 1
        
        return correct / len(ef_shapes) if ef_shapes else 0.0
    
    def _evaluate_shape_count(self, final_shapes: List[Dict]) -> float:
        """Check if final frame has correct number of shapes (6)."""
        if len(final_shapes) == 6:
            return 1.0
        elif len(final_shapes) >= 5:
            return 0.7
        elif len(final_shapes) >= 4:
            return 0.4
        else:
            return 0.2
    
class ConstructionStackEvaluator(BaseEvaluator):
    """
    O-22: Construction stack (block stacking) evaluator.
    
    CRITICAL RULES:
    1. TARGET (right stack) should remain UNCHANGED from first to last frame
    2. RESULT (left stack in final) should match TARGET (right stack in first)
    3. SOURCE blocks (left side in first frame) should be moved to create the result
    4. Animation should show actual block movement
    
    Evaluation dimensions:
    - Target preservation (25%): Right stack unchanged
    - Final state correctness (40%): Left stack matches target pattern
    - Source changed (20%): Left side actually changed from original
    - Movement detection (15%): Visible block movement animation
    """
    
    TASK_WEIGHTS = {
        'target_preservation': 0.25,
        'final_state': 0.40,
        'source_changed': 0.20,
        'movement_detection': 0.15
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
        h, w = first_frame.shape[:2]
        
        # Detect blocks in different regions
        left_boundary = w // 3
        right_boundary = 2 * w // 3
        
        # Target region (right side)
        first_right = first_frame[:, right_boundary:]
        final_right = final_frame[:, right_boundary:]
        first_target_blocks = self._detect_all_blocks_fine_region(first_right)
        final_target_blocks = self._detect_all_blocks_fine_region(final_right)
        
        # Source region (left side) 
        first_left = first_frame[:, :left_boundary]
        final_left = final_frame[:, :left_boundary]
        first_source_blocks = self._detect_all_blocks_fine_region(first_left)
        final_result_blocks = self._detect_all_blocks_fine_region(final_left)
        
        # Store debug info
        scores['first_target_count'] = len(first_target_blocks)
        scores['final_target_count'] = len(final_target_blocks)
        scores['first_source_count'] = len(first_source_blocks)
        scores['final_result_count'] = len(final_result_blocks)
        
        # 1. Target preservation (25%): Right stack must remain unchanged
        target_preservation = self._evaluate_target_preservation_v2(
            first_target_blocks, final_target_blocks
        )
        scores['target_preservation'] = target_preservation
        
        # If target changed significantly, the task failed
        if target_preservation < 0.3:
            scores['final_state'] = 0.0
            scores['source_changed'] = 0.0
            scores['movement_detection'] = 0.0
            scores['error'] = 'target_stack_changed'
            self._last_task_details = scores
            return target_preservation * self.TASK_WEIGHTS['target_preservation']
        
        # 2. Final state (40%): Result stack should match target pattern
        final_state_score = self._evaluate_final_state_v2(
            first_target_blocks, final_result_blocks
        )
        scores['final_state'] = final_state_score
        
        # 3. Source changed (20%): Left side should have changed
        source_changed_score = self._evaluate_source_changed(
            first_source_blocks, final_result_blocks, first_target_blocks
        )
        scores['source_changed'] = source_changed_score
        
        # 4. Movement detection (15%): Visible movement in video
        movement_score = self._detect_block_movement(video_frames, left_boundary, right_boundary)
        scores['movement_detection'] = movement_score
        
        # STRICT: If no movement detected, fail
        if movement_score < 0.2:
            scores['error'] = 'no_movement_detected'
            self._last_task_details = scores
            return 0.0
        
        self._last_task_details = scores
        return sum(scores.get(k, 0) * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS if k in scores)
    
    def _evaluate_target_preservation_v2(self, first_target: List[Dict], 
                                         final_target: List[Dict]) -> float:
        """Check if target stack (right side) remained unchanged."""
        if len(first_target) == 0:
            return 0.0  # No target blocks = can't evaluate
        
        # Block count must be identical
        if len(first_target) != len(final_target):
            return 0.0
        
        # Sort by y position and compare colors
        first_sorted = sorted(first_target, key=lambda b: b['y'])
        final_sorted = sorted(final_target, key=lambda b: b['y'])
        
        matched = 0
        for fb, lb in zip(first_sorted, final_sorted):
            hue_diff = abs(fb['hue'] - lb['hue'])
            hue_diff = min(hue_diff, 180 - hue_diff)
            if hue_diff < 30:
                matched += 1
        
        return matched / len(first_target)
    
    def _evaluate_final_state_v2(self, target_blocks: List[Dict], 
                                  result_blocks: List[Dict]) -> float:
        """Check if result stack matches target pattern (colors in correct order)."""
        if len(target_blocks) == 0:
            return 0.0
        
        # STRICT: Block count must match
        if len(result_blocks) != len(target_blocks):
            count_diff = abs(len(result_blocks) - len(target_blocks))
            if count_diff == 1:
                return 0.1  # One block off
            return 0.0
        
        # Sort by y position (bottom to top)
        target_sorted = sorted(target_blocks, key=lambda b: b['y'], reverse=True)
        result_sorted = sorted(result_blocks, key=lambda b: b['y'], reverse=True)
        
        matched = 0
        for target_b, result_b in zip(target_sorted, result_sorted):
            hue_diff = abs(target_b['hue'] - result_b['hue'])
            hue_diff = min(hue_diff, 180 - hue_diff)
            if hue_diff < 25:
                matched += 1
        
        ratio = matched / len(target_sorted)
        
        if ratio >= 0.9:
            return 1.0
        elif ratio >= 0.7:
            return 0.6
        elif ratio >= 0.5:
            return 0.3
        else:
            return 0.0
    
    def _evaluate_source_changed(self, first_source: List[Dict], 
                                 final_result: List[Dict],
                                 target_blocks: List[Dict]) -> float:
        """Check if the source (left side) changed to create the result."""
        if len(first_source) == 0 and len(final_result) == 0:
            return 0.0  # Nothing changed
        
        # Compare colors - result should be different from original source
        # but should match target
        first_hues = sorted([b['hue'] for b in first_source])
        final_hues = sorted([b['hue'] for b in final_result])
        target_hues = sorted([b['hue'] for b in target_blocks])
        
        # Check if final matches target more than original
        target_match = self._compare_hue_lists(final_hues, target_hues)
        source_match = self._compare_hue_lists(first_hues, target_hues)
        
        if target_match > source_match:
            return 1.0  # Good: result is closer to target than source was
        elif target_match > 0.5:
            return 0.5  # Partial: result somewhat matches target
        else:
            return 0.2
    
    def _compare_hue_lists(self, hues1: List[int], hues2: List[int]) -> float:
        """Compare two sorted lists of hues."""
        if not hues1 or not hues2:
            return 0.0
        if len(hues1) != len(hues2):
            return 0.2
        
        matched = 0
        for h1, h2 in zip(hues1, hues2):
            diff = abs(h1 - h2)
            diff = min(diff, 180 - diff)
            if diff < 25:
                matched += 1
        
        return matched / len(hues1)
    
    def _detect_block_movement(self, frames: List[np.ndarray], 
                               left_boundary: int, right_boundary: int) -> float:
        """
        Detect if there's visible block movement in the video.
        
        For construction stack task, movement happens primarily in the LEFT region
        where blocks are being stacked. We check both left and middle regions.
        """
        if len(frames) < 3:
            return 0.0
        
        # Sample frames to detect movement
        sample_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
        sample_indices = [min(i, len(frames)-1) for i in sample_indices]
        
        # Look for changes in the LEFT region (where blocks are stacked)
        left_region_changes = 0
        prev_left = None
        prev_middle = None
        
        for idx in sample_indices:
            frame = frames[idx]
            # Extract left region (where stacking happens)
            left = frame[:, :left_boundary]
            middle = frame[:, left_boundary:right_boundary]
            
            if prev_left is not None:
                left_diff = np.mean(np.abs(left.astype(float) - prev_left.astype(float)))
                middle_diff = np.mean(np.abs(middle.astype(float) - prev_middle.astype(float)))
                # Either region having significant change counts
                if left_diff > 3 or middle_diff > 3:
                    left_region_changes += 1
            
            prev_left = left.copy()
            prev_middle = middle.copy()
        
        # Score based on detected changes
        if left_region_changes >= 3:
            return 1.0
        elif left_region_changes >= 2:
            return 0.7
        elif left_region_changes >= 1:
            return 0.4
        else:
            return 0.0
    
    def _detect_all_blocks_fine_region(self, region: np.ndarray) -> List[Dict]:
        """
        Detect colored blocks in a specific region.
        
        Uses lower saturation threshold (30) to detect blocks that might be less saturated.
        Blocks in construction stack are typically small colored squares (~2000-2500 area).
        
        NOTE: When blocks are stacked and touching, they merge into a single contour.
        We estimate the number of blocks by area/expected_block_area.
        """
        if region.size == 0:
            return []
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Detect saturated (colored) regions - lower threshold for better detection
        mask = hsv[:, :, 1] > 30
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        expected_single_block_area = 2200  # Typical single block area
        min_block_area = 500  # Filter out noise smaller than this
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise (blocks should be at least 500 area)
            if area < min_block_area:
                continue
            
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Get average hue
            mask_cnt = np.zeros(region.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
            mean_hsv = cv2.mean(hsv, mask=mask_cnt)[:3]
            
            # Estimate number of blocks from area
            # A single block is ~2000-2500 area
            # If area is large, it might be multiple stacked blocks
            estimated_block_count = max(1, int(round(area / expected_single_block_area)))
            
            if estimated_block_count == 1:
                blocks.append({
                    'x': cx,
                    'y': cy,
                    'hue': int(mean_hsv[0]),
                    'area': area
                })
            else:
                # Create multiple "virtual" blocks for a stacked column
                # Distribute them vertically based on bounding box
                x, y, w, h = cv2.boundingRect(cnt)
                block_height = h / estimated_block_count
                for i in range(estimated_block_count):
                    block_cy = int(y + (i + 0.5) * block_height)
                    blocks.append({
                        'x': cx,
                        'y': block_cy,
                        'hue': int(mean_hsv[0]),
                        'area': expected_single_block_area
                    })
        
        return blocks
    
    def _detect_all_blocks_fine(self, frame: np.ndarray) -> List[Dict]:
        """Detect all colored blocks with finer granularity (lower area threshold)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect saturated (colored) regions
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Lower threshold to catch smaller blocks, but filter out noise
            if 200 < area < 15000:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Get average color using mask
                    mask_cnt = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(frame, mask=mask_cnt)[:3]
                    # Get hue
                    color_arr = np.array(mean_color, dtype=np.uint8).reshape(1, 1, 3)
                    hsv_color = cv2.cvtColor(color_arr, cv2.COLOR_BGR2HSV)[0, 0]
                    blocks.append({
                        'x': cx,
                        'y': cy,
                        'color': mean_color,
                        'hue': int(hsv_color[0]),
                        'area': area
                    })
        
        return blocks
    
    def _detect_blocks_with_color(self, region: np.ndarray) -> List[Dict]:
        """Detect colored blocks with center and average color."""
        if region.size == 0:
            return []
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Detect saturated (colored) regions
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 10000:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Get average color using mask
                    mask_cnt = np.zeros(region.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(region, mask=mask_cnt)[:3]
                    blocks.append({'center': (cx, cy), 'color': mean_color, 'area': area})
        
        return blocks
    
    def _evaluate_steps(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Evaluate if step count is reasonable."""
        # Count significant frame changes
        changes = 0
        for i in range(1, min(len(video_frames), 50)):
            diff = np.mean(np.abs(
                video_frames[i].astype(float) - video_frames[i-1].astype(float)
            ))
            if diff > 5:
                changes += 1
        
        # Reasonable number of moves
        if changes < 20:
            return 1.0
        elif changes < 40:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_movement(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Evaluate movement smoothness."""
        if len(video_frames) < 3:
            return 0.5
        
        diffs = []
        for i in range(1, min(len(video_frames), 20)):
            diff = np.mean(np.abs(
                video_frames[i].astype(float) - video_frames[i-1].astype(float)
            ))
            diffs.append(diff)
        
        if len(diffs) < 2:
            return 0.5
        
        variance = np.var(diffs)
        return 1.0 / (1.0 + variance / 100)
    
    def _detect_blocks(self, region: np.ndarray) -> List[Tuple[int, int, Tuple[int, int, int]]]:
        """Detect colored blocks with (x, y, color)."""
        if region.size == 0:
            return []
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Detect saturated (colored) regions
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 10000:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    color = tuple(int(c) for c in region[cy, cx])
                    blocks.append((cx, cy, color))
        
        return blocks

class MoveObjectsToTargetEvaluator(BaseEvaluator):
    """
    O-27: Move 2 Objects to 2 Targets
    
    Task: Animate two colored balls (pink and blue) sliding to their 
    matching colored target rings simultaneously.
    
    Rule-based evaluation:
    1. Color matching (50%) - Pink to pink, blue to blue
    2. Path and motion (15%) - Straight line, smooth
    3. Movement synchronization (20%) - Start/end together
    4. Visual completeness (15%) - Balls and targets preserved
    """
    
    TASK_WEIGHTS = {
        'color_matching': 0.50,
        'path_motion': 0.15,
        'synchronization': 0.20,
        'completeness': 0.15
    }
    
    def _find_color_centers(self, frame: np.ndarray) -> Dict[str, Optional[Tuple[float, float]]]:
        """Find centers of pink and blue objects."""
        if len(frame.shape) != 3:
            return {'pink': None, 'blue': None}
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        centers = {}
        
        # Pink detection
        lower_pink = np.array([140, 50, 80])
        upper_pink = np.array([170, 255, 255])
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        if np.sum(pink_mask > 0) > 100:
            coords = np.where(pink_mask > 0)
            centers['pink'] = (float(np.mean(coords[1])), float(np.mean(coords[0])))
        else:
            centers['pink'] = None
        
        # Blue detection
        lower_blue = np.array([100, 80, 80])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        if np.sum(blue_mask > 0) > 100:
            coords = np.where(blue_mask > 0)
            centers['blue'] = (float(np.mean(coords[1])), float(np.mean(coords[0])))
        else:
            centers['blue'] = None
        
        return centers
    
    def _analyze_motion_smoothness(self, frames: List[np.ndarray]) -> float:
        """Analyze if motion is smooth and continuous."""
        if len(frames) < 3:
            return 0.5
        
        trajectories = {'pink': [], 'blue': []}
        
        for frame in frames:
            centers = self._find_color_centers(frame)
            for color in ['pink', 'blue']:
                if centers[color] is not None:
                    trajectories[color].append(centers[color])
        
        smoothness_scores = []
        for color in ['pink', 'blue']:
            points = trajectories[color]
            if len(points) < 3:
                continue
            
            disps = []
            for i in range(1, len(points)):
                dx = points[i][0] - points[i-1][0]
                dy = points[i][1] - points[i-1][1]
                disps.append(np.sqrt(dx**2 + dy**2))
            
            if len(disps) < 2:
                continue
            
            mean_disp = np.mean(disps)
            std_disp = np.std(disps)
            
            if mean_disp < 0.5:
                smoothness_scores.append(0.4)
            else:
                smoothness_scores.append(max(0.3, 1 - (std_disp / mean_disp)))
        
        return float(np.mean(smoothness_scores)) if smoothness_scores else 0.5
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate dual object movement to targets."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # 1. Color matching: Compare final positions
        gen_centers = self._find_color_centers(gen_final)
        gt_centers = self._find_color_centers(gt_final)
        
        total_dist = 0
        count = 0
        for color in ['pink', 'blue']:
            if gen_centers[color] is not None and gt_centers[color] is not None:
                dx = gen_centers[color][0] - gt_centers[color][0]
                dy = gen_centers[color][1] - gt_centers[color][1]
                dist = np.sqrt(dx**2 + dy**2)
                total_dist += dist
                count += 1
        
        if count > 0:
            avg_dist = total_dist / count
            scores['color_matching'] = max(0, 1.0 - avg_dist / 50.0)
        else:
            scores['color_matching'] = 0.3
        
        # 2. Path motion: Analyze smoothness
        scores['path_motion'] = self._analyze_motion_smoothness(video_frames)
        
        # 3. Synchronization: Check if both objects move together
        if len(video_frames) >= 3:
            first_centers = self._find_color_centers(video_frames[0])
            mid_centers = self._find_color_centers(video_frames[len(video_frames)//2])
            
            pink_moved = False
            blue_moved = False
            
            if first_centers['pink'] is not None and mid_centers['pink'] is not None:
                pink_dist = np.sqrt((mid_centers['pink'][0] - first_centers['pink'][0])**2 + 
                                   (mid_centers['pink'][1] - first_centers['pink'][1])**2)
                pink_moved = pink_dist > 10
            
            if first_centers['blue'] is not None and mid_centers['blue'] is not None:
                blue_dist = np.sqrt((mid_centers['blue'][0] - first_centers['blue'][0])**2 + 
                                   (mid_centers['blue'][1] - first_centers['blue'][1])**2)
                blue_moved = blue_dist > 10
            
            # Both should move together
            if pink_moved and blue_moved:
                scores['synchronization'] = 1.0
            elif pink_moved or blue_moved:
                scores['synchronization'] = 0.2  # Detection failed
            else:
                scores['synchronization'] = 0.3
        else:
            scores['synchronization'] = 0.2  # Detection failed
        
        # 4. Completeness: Check objects are preserved
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            scores['completeness'] = max(0, 1.0 - diff / 100.0)
        else:
            scores['completeness'] = 0.2  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

class MazePathfindingEvaluator(BaseEvaluator):
    """
    O-39: Maze pathfinding evaluator.
    
    Rule-based evaluation:
    - Path validity (45%): No wall crossing, continuous path
    - Path completeness (30%): Start to end, all marked
    - Navigation accuracy (20%): Adjacent moves only
    - Element preservation (5%): Maze structure unchanged
    """
    
    TASK_WEIGHTS = {
        'path_validity': 0.45,
        'path_completeness': 0.30,
        'navigation_accuracy': 0.20,
        'element_preservation': 0.05
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
        
        scores['path_validity'] = self._evaluate_path_validity(first_frame, final_frame)
        scores['path_completeness'] = self._evaluate_path_completeness(first_frame, final_frame)
        scores['navigation_accuracy'] = self._evaluate_navigation(first_frame, final_frame)
        scores['element_preservation'] = self._evaluate_preservation(first_frame, final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_path_validity(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if path is valid (no wall crossing)."""
        # Detect path markers (orange/yellow)
        path_mask = self._detect_path_markers(final_frame)
        
        # Detect walls (black)
        wall_mask = self._detect_walls(first_frame)
        
        if path_mask is None or wall_mask is None:
            return 0.5
        
        # Check path doesn't cross walls
        overlap = cv2.bitwise_and(path_mask, wall_mask)
        overlap_pixels = np.sum(overlap > 0)
        path_pixels = np.sum(path_mask > 0)
        
        if path_pixels == 0:
            return 0.3
        
        violation_ratio = overlap_pixels / path_pixels
        
        if violation_ratio < 0.05:
            return 1.0
        elif violation_ratio < 0.1:
            return 0.7
        else:
            return 0.3
    
    def _evaluate_path_completeness(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if path is complete from start to end."""
        # Detect start (green) and end (red flag)
        start_pos = self._find_start_position(first_frame)
        end_pos = self._find_end_position(first_frame)
        
        # Detect path markers
        path_mask = self._detect_path_markers(final_frame)
        
        if start_pos is None or end_pos is None or path_mask is None:
            return 0.5
        
        # Check if path reaches start and end
        path_near_start = self._check_path_near_position(path_mask, start_pos)
        path_near_end = self._check_path_near_position(path_mask, end_pos)
        
        if path_near_start and path_near_end:
            return 1.0
        elif path_near_start or path_near_end:
            return 0.6
        else:
            return 0.3
    
    def _evaluate_navigation(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if moves are valid (adjacent only)."""
        # Check path continuity
        path_mask = self._detect_path_markers(final_frame)
        
        if path_mask is None:
            return 0.5
        
        # Count connected components (should be 1 for continuous path)
        num_labels, _ = cv2.connectedComponents(path_mask)
        
        if num_labels == 2:  # 1 background + 1 path
            return 1.0
        elif num_labels <= 4:
            return 0.6
        else:
            return 0.3
    
    def _evaluate_preservation(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if maze structure is preserved."""
        # Compare wall regions
        first_walls = self._detect_walls(first_frame)
        final_walls = self._detect_walls(final_frame)
        
        if first_walls is None or final_walls is None:
            return 0.5
        
        # Compare wall preservation
        intersection = np.sum((first_walls > 0) & (final_walls > 0))
        first_total = np.sum(first_walls > 0)
        
        if first_total == 0:
            return 0.5
        
        preservation_ratio = intersection / first_total
        
        if preservation_ratio > 0.9:
            return 1.0
        elif preservation_ratio > 0.7:
            return 0.7
        else:
            return 0.4
    
    def _detect_path_markers(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect orange/yellow path markers."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([30, 255, 255])
        
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        return mask
    
    def _detect_walls(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect maze walls (black/dark)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, walls = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        return walls
    
    def _find_start_position(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find green start marker."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None
    
    def _find_end_position(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find red end marker (flag)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None
    
    def _check_path_near_position(self, path_mask: np.ndarray, pos: Tuple[int, int]) -> bool:
        """Check if path reaches near a position."""
        x, y = pos
        h, w = path_mask.shape
        
        # Check in a radius around position
        radius = 30
        y_min = max(0, y - radius)
        y_max = min(h, y + radius)
        x_min = max(0, x - radius)
        x_max = min(w, x + radius)
        
        region = path_mask[y_min:y_max, x_min:x_max]
        return np.sum(region > 0) > 50


class ObjectSubtractionEvaluator(BaseEvaluator):
    """
    O-43: Object subtraction (deletion) evaluator.
    
    Rule-based evaluation:
    - Object identification accuracy (40%): Correct objects identified
    - Deletion completeness (30%): Objects fully removed
    - Preserved object fidelity (20%): Remaining objects unchanged
    - Selective deletion accuracy (10%): No extra deletions
    """
    
    TASK_WEIGHTS = {
        'identification': 0.40,
        'deletion_completeness': 0.30,
        'preservation': 0.20,
        'selective_accuracy': 0.10
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
        
        # Count objects
        first_count = self._count_objects(first_frame)
        final_count = self._count_objects(final_frame)
        
        # Get expected final count from GT if available
        expected_final_count = None
        if gt_final_frame is not None:
            expected_final_count = self._count_objects(gt_final_frame)
        
        # CRITICAL: Check if correct number of objects remain
        # Should remove exactly the right number, not all objects
        if final_count == 0 and first_count > 1:
            # All objects removed - WRONG
            self._last_task_details = {
                'identification': 0.0,
                'deletion_completeness': 0.0,
                'preservation': 0.0,
                'selective_accuracy': 0.0,
                'all_objects_removed': True,
                'first_count': first_count,
                'final_count': final_count
            }
            return 0.0
        
        if final_count > first_count:
            # Objects added - WRONG
            self._last_task_details = {
                'identification': 0.0,
                'deletion_completeness': 0.0,
                'preservation': 0.0,
                'selective_accuracy': 0.0,
                'objects_added': True,
                'first_count': first_count,
                'final_count': final_count
            }
            return 0.0
        
        # Check if final count matches expected
        if expected_final_count is not None:
            if final_count != expected_final_count:
                # Wrong number of objects remaining
                self._last_task_details = {
                    'identification': 0.0,
                    'deletion_completeness': 0.0,
                    'preservation': 0.0,
                    'selective_accuracy': 0.0,
                    'wrong_count': True,
                    'first_count': first_count,
                    'final_count': final_count,
                    'expected_count': expected_final_count
                }
                return 0.0
        
        scores['identification'] = self._evaluate_identification(first_frame, final_frame)
        scores['deletion_completeness'] = self._evaluate_deletion(first_frame, final_frame)
        scores['preservation'] = self._evaluate_preservation(first_frame, final_frame)
        scores['selective_accuracy'] = self._evaluate_selective(first_frame, final_frame)
        
        self._last_task_details = scores
        self._last_task_details['first_count'] = first_count
        self._last_task_details['final_count'] = final_count
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_identification(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if correct objects are identified for deletion."""
        first_count = self._count_objects(first_frame)
        final_count = self._count_objects(final_frame)
        
        # Some objects should be deleted
        if final_count < first_count:
            return 1.0
        elif final_count == first_count:
            return 0.3  # Nothing deleted
        else:
            return 0.2  # Objects added (wrong)
    
    def _evaluate_deletion(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if objects are completely deleted."""
        first_count = self._count_objects(first_frame)
        final_count = self._count_objects(final_frame)
        
        deleted = first_count - final_count
        
        if deleted >= 1:
            return 1.0
        else:
            return 0.3
    
    def _evaluate_preservation(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if remaining objects are preserved."""
        # Compare color distributions
        first_colors = self._get_object_colors(first_frame)
        final_colors = self._get_object_colors(final_frame)
        
        if len(final_colors) == 0:
            return 0.5
        
        # Check if remaining colors exist in original
        preserved = 0
        for color in final_colors:
            for orig_color in first_colors:
                if self._colors_similar(color, orig_color):
                    preserved += 1
                    break
        
        return preserved / max(len(final_colors), 1)
    
    def _evaluate_selective(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check for correct selective deletion."""
        first_count = self._count_objects(first_frame)
        final_count = self._count_objects(final_frame)
        
        # Should have fewer objects but not zero
        if 0 < final_count < first_count:
            return 1.0
        elif final_count == 0:
            return 0.3  # All deleted
        else:
            return 0.4
    
    def _count_objects(self, frame: np.ndarray) -> int:
        """Count colored objects."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect saturated (colored) regions
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return sum(1 for cnt in contours if cv2.contourArea(cnt) > 500)
    
    def _get_object_colors(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Get colors of objects."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        colors = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if 0 <= cy < frame.shape[0] and 0 <= cx < frame.shape[1]:
                        colors.append(tuple(int(c) for c in frame[cy, cx]))
        
        return colors
    
    def _colors_similar(self, c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> bool:
        """Check if two colors are similar."""
        diff = np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
        return diff < 60


class ShapeSorterEvaluator(BaseEvaluator):
    """
    O-46: Shape sorter evaluator.
    
    RULE: Colored shapes on left should be moved to cover the outlines on right.
    - Outlines are line drawings (low saturation) on the right side
    - Colored shapes (high saturation) start on left and should end on right
    - Final frame: outlines covered by matching colored shapes
    - No new objects should appear
    """
    
    TASK_WEIGHTS = {
        'shapes_moved_to_right': 0.50,  # Colored shapes should be on right
        'left_side_cleared': 0.30,       # Left side should have no colored shapes
        'no_new_shapes': 0.20            # No new shapes created
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
        h, w = first_frame.shape[:2]
        
        # Detect colored shapes (high saturation) in first and final frames
        first_left_shapes = self._detect_colored_shapes(first_frame[:, :w//2])
        first_right_shapes = self._detect_colored_shapes(first_frame[:, w//2:])
        final_left_shapes = self._detect_colored_shapes(final_frame[:, :w//2])
        final_right_shapes = self._detect_colored_shapes(final_frame[:, w//2:])
        
        # Total colored shapes in first frame (on left side)
        first_colored_count = len(first_left_shapes)
        
        # 1. CRITICAL: Colored shapes should be on right in final frame
        if first_colored_count == 0:
            scores['shapes_moved_to_right'] = 0.5
        else:
            # Check if all shapes moved to right
            if len(final_right_shapes) >= first_colored_count:
                scores['shapes_moved_to_right'] = 1.0
            elif len(final_right_shapes) >= first_colored_count - 1:
                scores['shapes_moved_to_right'] = 0.7
            else:
                scores['shapes_moved_to_right'] = len(final_right_shapes) / first_colored_count
        
        # 2. CRITICAL: Left side should be cleared of colored shapes
        if first_colored_count == 0:
            scores['left_side_cleared'] = 0.5
        else:
            if len(final_left_shapes) == 0:
                scores['left_side_cleared'] = 1.0
            else:
                # Penalize for shapes remaining on left
                remaining_ratio = len(final_left_shapes) / first_colored_count
                scores['left_side_cleared'] = max(0, 1.0 - remaining_ratio)
        
        # 3. No new shapes should be created
        total_first = first_colored_count + len(first_right_shapes)
        total_final = len(final_left_shapes) + len(final_right_shapes)
        
        if total_final <= total_first + 1:  # Allow 1 shape tolerance
            scores['no_new_shapes'] = 1.0
        else:
            # Penalize for new shapes
            new_shapes = total_final - total_first
            scores['no_new_shapes'] = max(0, 1.0 - new_shapes * 0.3)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_colored_shapes(self, region: np.ndarray) -> List[Dict]:
        """Detect colored (high saturation) shapes in region."""
        if region.size == 0:
            return []
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # High saturation indicates colored shapes (not outlines)
        mask = hsv[:, :, 1] > 80
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    shapes.append({'center': (cx, cy), 'area': area})
        
        return shapes
    
    def _detect_shapes_old(self, region: np.ndarray) -> List[Tuple[int, int]]:
        """Detect shapes with their centers."""
        if region.size == 0:
            return []
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 300:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shapes.append((cx, cy))
        
        return shapes


class SymmetryCompletionEvaluator(BaseEvaluator):
    """
    O-49: Symmetry completion evaluator.
    
    Rule-based evaluation:
    - Block preservation (40%): Original blocks unchanged, total count correct
    - Symmetry accuracy (35%): Filled blocks create left-right symmetry
    - Fill correctness (20%): Blocks filled at correct symmetric positions
    - Color consistency (5%): New blocks match original block colors
    """
    
    TASK_WEIGHTS = {
        'block_preservation': 0.40,
        'symmetry_accuracy': 0.35,
        'fill_correctness': 0.20,
        'color_consistency': 0.05
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
        
        # Detect filled blocks
        first_blocks = self._detect_filled_blocks(first_frame)
        final_blocks = self._detect_filled_blocks(final_frame)
        
        # 1. Block preservation - original blocks should remain, correct total count
        scores['block_preservation'] = self._evaluate_block_preservation(
            first_blocks, final_blocks, gt_first_frame, gt_final_frame
        )
        
        # If blocks are completely changed, penalize heavily
        if scores['block_preservation'] < 0.3:
            self._last_task_details = {
                'block_preservation': scores['block_preservation'],
                'symmetry_accuracy': 0.0,
                'fill_correctness': 0.0,
                'color_consistency': 0.0,
                'blocks_changed': True
            }
            return scores['block_preservation'] * self.TASK_WEIGHTS['block_preservation']
        
        # 2. Symmetry accuracy
        scores['symmetry_accuracy'] = self._evaluate_symmetry(final_blocks, final_frame)
        
        # 3. Fill correctness
        scores['fill_correctness'] = self._evaluate_fill_correctness(
            first_blocks, final_blocks, final_frame
        )
        
        # 4. Color consistency
        scores['color_consistency'] = self._evaluate_color_consistency(
            first_blocks, final_blocks
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_filled_blocks(self, frame: np.ndarray) -> List[Dict]:
        """Detect filled (colored) blocks."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 50000:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    # Get color
                    mask_cnt = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(frame, mask=mask_cnt)[:3]
                    color_arr = np.array(mean_color, dtype=np.uint8).reshape(1, 1, 3)
                    hsv_c = cv2.cvtColor(color_arr, cv2.COLOR_BGR2HSV)[0, 0]
                    blocks.append({
                        'center': (cx, cy),
                        'color': mean_color,
                        'hue': int(hsv_c[0]),
                        'area': area
                    })
        return blocks
    
    def _evaluate_block_preservation(
        self, 
        first_blocks: List[Dict], 
        final_blocks: List[Dict],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray]
    ) -> float:
        """Check if original blocks are preserved and total count is correct."""
        first_count = len(first_blocks)
        final_count = len(final_blocks)
        
        # Get expected final count from GT if available
        if gt_final_frame is not None:
            gt_final_blocks = self._detect_filled_blocks(gt_final_frame)
            expected_count = len(gt_final_blocks)
        else:
            # Estimate: should add blocks to make symmetric
            expected_count = first_count + (first_count // 2)  # Rough estimate
        
        # Check count
        if final_count < first_count:
            return 0.0  # Blocks were removed - bad
        
        count_diff = abs(final_count - expected_count)
        if count_diff == 0:
            count_score = 1.0
        elif count_diff <= 2:
            count_score = 0.7
        else:
            count_score = 0.3
        
        # Check if original blocks' colors are preserved
        first_hues = sorted([b['hue'] for b in first_blocks])
        
        # Find matching hues in final
        matched = 0
        final_hues = [b['hue'] for b in final_blocks]
        used = set()
        for fh in first_hues:
            for i, fnlh in enumerate(final_hues):
                if i in used:
                    continue
                hue_diff = abs(fh - fnlh)
                hue_diff = min(hue_diff, 180 - hue_diff)
                if hue_diff < 20:
                    matched += 1
                    used.add(i)
                    break
        
        color_preservation = matched / len(first_hues) if first_hues else 0.0
        
        return (count_score + color_preservation) / 2
    
    def _evaluate_symmetry(self, final_blocks: List[Dict], final_frame: np.ndarray) -> float:
        """Check if blocks form left-right symmetry."""
        if len(final_blocks) == 0:
            return 0.0
        
        h, w = final_frame.shape[:2]
        center_x = w // 2
        
        # Group blocks by their y-coordinate (rows)
        rows = {}
        for block in final_blocks:
            y = block['center'][1]
            # Quantize y to group nearby blocks
            row_key = y // 40
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append(block)
        
        # For each row, check if blocks are symmetric around center
        symmetric_rows = 0
        total_rows = len(rows)
        
        for row_key, row_blocks in rows.items():
            # Get x positions relative to center
            x_positions = [b['center'][0] - center_x for b in row_blocks]
            
            # Check if for each x, there's a -x
            is_symmetric = True
            for x in x_positions:
                if x == 0:
                    continue  # Center block
                has_mirror = any(abs(x + other_x) < 30 for other_x in x_positions)
                if not has_mirror:
                    is_symmetric = False
                    break
            
            if is_symmetric:
                symmetric_rows += 1
        
        return symmetric_rows / total_rows if total_rows > 0 else 0.0
    
    def _evaluate_fill_correctness(
        self, 
        first_blocks: List[Dict], 
        final_blocks: List[Dict],
        final_frame: np.ndarray
    ) -> float:
        """Check if new blocks are filled at correct symmetric positions."""
        h, w = final_frame.shape[:2]
        center_x = w // 2
        
        # Find new blocks (in final but not in first)
        first_positions = set()
        for b in first_blocks:
            # Quantize position
            pos_key = (b['center'][0] // 30, b['center'][1] // 30)
            first_positions.add(pos_key)
        
        new_blocks = []
        for b in final_blocks:
            pos_key = (b['center'][0] // 30, b['center'][1] // 30)
            if pos_key not in first_positions:
                new_blocks.append(b)
        
        if len(new_blocks) == 0:
            return 0.5  # No new blocks - might be okay if already symmetric
        
        # Check if each new block has a corresponding original block on the other side
        correct_fills = 0
        for new_block in new_blocks:
            new_x = new_block['center'][0]
            new_y = new_block['center'][1]
            
            # Calculate mirror position
            mirror_x = 2 * center_x - new_x
            
            # Check if there's an original block at the mirror position
            has_mirror = False
            for orig_block in first_blocks:
                orig_x = orig_block['center'][0]
                orig_y = orig_block['center'][1]
                if abs(orig_x - mirror_x) < 30 and abs(orig_y - new_y) < 30:
                    has_mirror = True
                    break
            
            if has_mirror:
                correct_fills += 1
        
        return correct_fills / len(new_blocks) if new_blocks else 0.5
    
    def _evaluate_color_consistency(
        self, 
        first_blocks: List[Dict], 
        final_blocks: List[Dict]
    ) -> float:
        """Check if new blocks use consistent colors."""
        first_hues = set(b['hue'] for b in first_blocks)
        final_hues = [b['hue'] for b in final_blocks]
        
        # Check how many final block hues are similar to first block hues
        consistent = 0
        for fh in final_hues:
            for oh in first_hues:
                hue_diff = abs(fh - oh)
                hue_diff = min(hue_diff, 180 - hue_diff)
                if hue_diff < 20:
                    consistent += 1
                    break
        
        return consistent / len(final_hues) if final_hues else 0.5
        
        return max(0.0, correlation)
    
    def _evaluate_preservation(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if left side is preserved."""
        h, w = first_frame.shape[:2]
        
        first_left = first_frame[:, :w//2]
        final_left = final_frame[:, :w//2]
        
        # Compare
        diff = np.mean(np.abs(first_left.astype(float) - final_left.astype(float)))
        
        if diff < 10:
            return 1.0
        elif diff < 30:
            return 0.7
        else:
            return 0.4
    
    def _count_filled_cells(self, region: np.ndarray) -> int:
        """Count filled (dark) cells."""
        if region.size == 0:
            return 0
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return np.sum(gray < 100)


# Export all Part 3 evaluators
OUT_OF_DOMAIN_50_EVALUATORS_PART4 = {
    'O-5_symbol_deletion_data-generator': SymbolDeletionEvaluator,
    'O-6_2d_geometric_transformation_data-generator': GeometricTransformationEvaluator,
    'O-9_shape_scaling_data-generator': ShapeScalingAnalogyEvaluator,
    'O-11_shape_color_then_move_data-generator': ShapeColorThenMoveEvaluator,
    'O-22_construction_stack_data-generator': ConstructionStackEvaluator,
    'O-27_move_2_object_to_2_target_data-generator': MoveObjectsToTargetEvaluator,
    'O-39_maze_data-generator': MazePathfindingEvaluator,
    'O-43_object_subtraction_data-generator': ObjectSubtractionEvaluator,
    'O-46_shape_sorter_data-generator': ShapeSorterEvaluator,
    'O-49_symmetry_completion_data-generator': SymmetryCompletionEvaluator,
}
