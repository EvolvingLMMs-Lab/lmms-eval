"""
Specific evaluators for Out-of-Domain_50 tasks (Part 3).
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import normalize_frame_size


class SelectLeftmostShapeEvaluator(BaseEvaluator):
    """
    G-219: Select leftmost shape evaluator.
    
    Rule-based evaluation:
    - Position identification correctness (45%): Find shape with smallest x
    - Marking precision (30%): Circle accurately marks target
    - Marking quality (15%): Red circle quality
    - Scene preservation (10%): Original shapes unchanged
    """
    
    TASK_WEIGHTS = {
        'position_identification': 0.45,
        'marking_precision': 0.30,
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
        
        # 1. Position identification (45%)
        scores['position_identification'] = self._evaluate_position_id(
            first_frame, final_frame
        )
        
        # 2. Marking precision (30%)
        scores['marking_precision'] = self._evaluate_marking_precision(
            first_frame, final_frame
        )
        
        # 3. Marking quality (15%)
        scores['marking_quality'] = self._evaluate_marking_quality(final_frame)
        
        # 4. Scene preservation (10%)
        scores['scene_preservation'] = self._evaluate_scene_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_position_id(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if leftmost shape (smallest x) is identified."""
        # Find leftmost shape
        leftmost = self._find_leftmost_shape(first_frame)
        
        # Detect circle marking
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        if leftmost is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - leftmost[0])**2 + (circle[1] - leftmost[1])**2)
        
        if dist < 40:
            return 1.0
        elif dist < 80:
            return 0.7
        elif dist < 120:
            return 0.4
        else:
            return 0.0
    
    def _evaluate_marking_precision(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Evaluate marking precision."""
        circle = self._detect_red_circle(final_frame)
        leftmost = self._find_leftmost_shape(first_frame)
        
        if circle is None:
            return 0.0
        if leftmost is None:
            return 0.0
        
        dist = np.sqrt((circle[0] - leftmost[0])**2 + (circle[1] - leftmost[1])**2)
        return max(0.0, 1.0 - dist / 60)
    
    def _evaluate_marking_quality(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate red circle quality."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        if 30 < r < 150:
            return 1.0
        else:
            return 0.5
    
    def _evaluate_scene_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if shapes are preserved."""
        first_count = self._count_shapes(first_frame)
        
        # Remove red marking
        hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        final_no_red = final_frame.copy()
        final_no_red[red_mask > 0] = [255, 255, 255]
        
        final_count = self._count_shapes(final_no_red)
        
        if abs(first_count - final_count) <= 1:
            return 1.0
        elif abs(first_count - final_count) <= 2:
            return 0.7
        else:
            return 0.1
    
    def _find_leftmost_shape(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the leftmost shape (smallest x)."""
        shapes = self._detect_shapes(frame)
        
        if len(shapes) == 0:
            return None
        
        leftmost = min(shapes, key=lambda s: s[0])
        return leftmost
    
    def _detect_shapes(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect shapes with their centers."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 300:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shapes.append((cx, cy))
        
        return shapes
    
    def _count_shapes(self, frame: np.ndarray) -> int:
        """Count shapes."""
        return len(self._detect_shapes(frame))
    
    def _detect_red_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect red circle."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                (x, y), r = cv2.minEnclosingCircle(largest)
                return (int(x), int(y), int(r))
        
        return None


class OutlineInnermostSquareEvaluator(BaseEvaluator):
    """
    G-221: Outline innermost square evaluator.
    
    Rule-based evaluation for concentric squares centered at canvas:
    - Concentric structure preservation (40%): Squares remain concentric at canvas center
    - Color preservation (35%): Colors on all sides (上下左右) remain the same
    - Blue outline addition (20%): Blue outline added around innermost square
    - Element preservation (5%): Original squares unchanged
    """
    
    TASK_WEIGHTS = {
        'concentric_structure': 0.40,
        'color_preservation': 0.35,
        'outline_addition': 0.20,
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
        
        h, w = first_frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Detect concentric squares in first frame (GT reference)
        gt_squares = self._detect_concentric_squares(first_frame)
        
        # 1. Check concentric structure preservation (40%)
        scores['concentric_structure'] = self._evaluate_concentric_structure(
            first_frame, final_frame, center_x, center_y
        )
        
        # If structure is completely broken, return early with low score
        if scores['concentric_structure'] < 0.3:
            self._last_task_details = {
                'concentric_structure': scores['concentric_structure'],
                'color_preservation': 0.0,
                'outline_addition': 0.0,
                'element_preservation': 0.0,
                'structure_broken': True
            }
            return 0.0
        
        # 2. Check color preservation (35%) - colors on all 4 sides should match
        scores['color_preservation'] = self._evaluate_color_preservation(
            first_frame, final_frame, center_x, center_y
        )
        
        # 3. Check blue outline addition (20%)
        scores['outline_addition'] = self._evaluate_outline_addition(
            first_frame, final_frame, center_x, center_y
        )
        
        # 4. Element preservation (5%)
        scores['element_preservation'] = self._evaluate_element_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_concentric_squares(self, frame: np.ndarray) -> List[Dict]:
        """Detect concentric squares by scanning from center outward."""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        squares = []
        prev_color = None
        current_dist = 0
        
        # Scan horizontally from center to right edge
        for x in range(center_x, w):
            color = tuple(frame[center_y, x])
            if prev_color is not None:
                color_diff = sum(abs(int(c1) - int(c2)) for c1, c2 in zip(color, prev_color))
                if color_diff > 30:  # Color transition = square boundary
                    squares.append({
                        'distance': x - center_x,
                        'color': prev_color
                    })
            prev_color = color
        
        # Add the outermost square
        if prev_color is not None:
            squares.append({
                'distance': w - center_x,
                'color': prev_color
            })
        
        return squares
    
    def _evaluate_concentric_structure(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        center_x: int,
        center_y: int
    ) -> float:
        """Check if concentric square structure is preserved."""
        h, w = first_frame.shape[:2]
        
        # Sample colors at multiple distances from center in all 4 directions
        # For concentric squares, colors at same distance should be similar
        distances = [50, 100, 150, 200, 250, 300, 350, 400]
        
        matches = 0
        total = 0
        
        for dist in distances:
            if dist >= min(center_x, center_y, w - center_x, h - center_y):
                continue
            
            # Get colors in 4 directions for first frame
            first_colors = []
            final_colors = []
            
            # Right
            first_colors.append(tuple(first_frame[center_y, min(center_x + dist, w-1)]))
            final_colors.append(tuple(final_frame[center_y, min(center_x + dist, w-1)]))
            # Left
            first_colors.append(tuple(first_frame[center_y, max(center_x - dist, 0)]))
            final_colors.append(tuple(final_frame[center_y, max(center_x - dist, 0)]))
            # Down
            first_colors.append(tuple(first_frame[min(center_y + dist, h-1), center_x]))
            final_colors.append(tuple(final_frame[min(center_y + dist, h-1), center_x]))
            # Up
            first_colors.append(tuple(first_frame[max(center_y - dist, 0), center_x]))
            final_colors.append(tuple(final_frame[max(center_y - dist, 0), center_x]))
            
            # Check if colors in final frame match first frame (ignoring blue outline)
            for fc, fnc in zip(first_colors, final_colors):
                total += 1
                # Allow for blue outline (high B, low G, low R)
                is_blue = fnc[0] > 200 and fnc[1] < 50 and fnc[2] < 50
                if is_blue:
                    matches += 1  # Blue outline is acceptable
                else:
                    color_diff = sum(abs(int(c1) - int(c2)) for c1, c2 in zip(fc, fnc))
                    if color_diff < 50:
                        matches += 1
        
        if total == 0:
            return 0.5
        
        return matches / total
    
    def _evaluate_color_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        center_x: int,
        center_y: int
    ) -> float:
        """Check if colors on all 4 sides (上下左右) are preserved."""
        h, w = first_frame.shape[:2]
        
        # Sample at multiple distances
        distances = [100, 200, 300, 400]
        
        preserved = 0
        total = 0
        
        for dist in distances:
            if dist >= min(center_x, center_y, w - center_x, h - center_y):
                continue
            
            # Get first frame colors at 4 directions
            first_right = tuple(first_frame[center_y, min(center_x + dist, w-1)])
            first_left = tuple(first_frame[center_y, max(center_x - dist, 0)])
            first_down = tuple(first_frame[min(center_y + dist, h-1), center_x])
            first_up = tuple(first_frame[max(center_y - dist, 0), center_x])
            
            # Get final frame colors
            final_right = tuple(final_frame[center_y, min(center_x + dist, w-1)])
            final_left = tuple(final_frame[center_y, max(center_x - dist, 0)])
            final_down = tuple(final_frame[min(center_y + dist, h-1), center_x])
            final_up = tuple(final_frame[max(center_y - dist, 0), center_x])
            
            # Check if 4 sides have same color in first frame (concentric property)
            first_colors = [first_right, first_left, first_down, first_up]
            final_colors = [final_right, final_left, final_down, final_up]
            
            # For each direction, check if color is preserved
            for fc, fnc in zip(first_colors, final_colors):
                total += 1
                # Ignore blue outline
                is_blue = fnc[0] > 200 and fnc[1] < 50 and fnc[2] < 50
                if is_blue:
                    preserved += 1
                else:
                    color_diff = sum(abs(int(c1) - int(c2)) for c1, c2 in zip(fc, fnc))
                    if color_diff < 50:
                        preserved += 1
        
        if total == 0:
            return 0.5
        
        return preserved / total
    
    def _evaluate_outline_addition(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        center_x: int,
        center_y: int
    ) -> float:
        """Check if blue outline was added around innermost square."""
        # Count blue pixels in first vs final
        first_blue = self._count_blue_pixels(first_frame)
        final_blue = self._count_blue_pixels(final_frame)
        
        blue_increase = final_blue - first_blue
        
        if blue_increase < 500:
            return 0.0  # No blue outline added
        
        # Check if blue outline is near center (around innermost square)
        blue_mask = self._get_blue_mask(final_frame)
        blue_points = np.where(blue_mask > 0)
        
        if len(blue_points[0]) == 0:
            return 0.0
        
        # Calculate average distance of blue pixels from center
        avg_y = np.mean(blue_points[0])
        avg_x = np.mean(blue_points[1])
        
        dist_from_center = np.sqrt((avg_x - center_x)**2 + (avg_y - center_y)**2)
        
        # Blue outline should be near center (innermost square)
        h, w = final_frame.shape[:2]
        max_dist = min(w, h) / 2
        
        # Closer to center = better
        if dist_from_center < max_dist * 0.3:
            return 1.0
        elif dist_from_center < max_dist * 0.5:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_element_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Check if overall structure is preserved."""
        # Compare histograms (excluding blue)
        first_hist = cv2.calcHist([first_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        final_hist = cv2.calcHist([final_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        correlation = cv2.compareHist(first_hist, final_hist, cv2.HISTCMP_CORREL)
        return max(0.0, correlation)
    
    def _count_blue_pixels(self, frame: np.ndarray) -> int:
        """Count pure blue pixels."""
        b, g, r = cv2.split(frame)
        blue_mask = (b > 200) & (g < 50) & (r < 50)
        return int(np.sum(blue_mask))
    
    def _get_blue_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get mask of blue pixels."""
        b, g, r = cv2.split(frame)
        return ((b > 200) & (g < 50) & (r < 50)).astype(np.uint8) * 255

class MarkTangentPointEvaluator(BaseEvaluator):
    """
    G-222: Mark tangent point of circles evaluator.
    
    Rule-based evaluation:
    - External tangent circle pair identification (40%): Correct pair found
    - Tangent point calculation accuracy (40%): Precise point location
    - Marking position accuracy (15%): Mark centered on tangent point
    - Visual annotation quality (5%): Black circle proper
    """
    
    TASK_WEIGHTS = {
        'pair_id': 0.40,
        'calculation': 0.40,
        'position': 0.15,
        'annotation': 0.05
    }
    
    def _detect_circles(self, frame: np.ndarray) -> List[Dict]:
        """Detect circles in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 30,
                                    param1=50, param2=30, minRadius=20, maxRadius=200)
        
        detected = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                detected.append({
                    'center': (i[0], i[1]),
                    'radius': i[2]
                })
        
        return detected
    
    def _find_tangent_pairs(self, circles: List[Dict]) -> List[Tuple[int, int, Tuple[float, float]]]:
        """Find externally tangent circle pairs and their tangent points."""
        tangent_pairs = []
        
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                c1, c2 = circles[i], circles[j]
                
                # Convert to float to avoid overflow
                c1x, c1y = float(c1['center'][0]), float(c1['center'][1])
                c2x, c2y = float(c2['center'][0]), float(c2['center'][1])
                r1, r2 = float(c1['radius']), float(c2['radius'])
                
                dist = np.sqrt((c1x - c2x)**2 + (c1y - c2y)**2)
                
                # Check if externally tangent (distance ≈ r1 + r2)
                expected_dist = r1 + r2
                if abs(dist - expected_dist) < 10:  # Tolerance
                    # Calculate tangent point
                    t = r1 / (r1 + r2) if (r1 + r2) > 0 else 0.5
                    tx = c1x + t * (c2x - c1x)
                    ty = c1y + t * (c2y - c1y)
                    
                    tangent_pairs.append((i, j, (tx, ty)))
        
        return tangent_pairs
    
    def _detect_black_marking(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect black circle marking."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Black marking
        black_mask = (gray < 50).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30 or area > 1000:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.5:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    return (cx, cy)
        
        return None
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate mark tangent point task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect markings
        gen_marking = self._detect_black_marking(last_frame)
        gt_marking = self._detect_black_marking(gt_last)
        
        # Detect circles and find tangent pairs
        gen_circles = self._detect_circles(last_frame)
        tangent_pairs = self._find_tangent_pairs(gen_circles)
        
        # 1. Pair identification: Check if marking is near a tangent point
        if gen_marking is not None and tangent_pairs:
            near_tangent = False
            for _, _, tangent_point in tangent_pairs:
                dist = np.sqrt((gen_marking[0] - tangent_point[0])**2 + 
                              (gen_marking[1] - tangent_point[1])**2)
                if dist < 30:
                    near_tangent = True
                    break
            scores['pair_id'] = 1.0 if near_tangent else 0.3
        else:
            scores['pair_id'] = 0.2  # Detection failed
        
        # 2. Calculation accuracy: Compare marking positions
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['calculation'] = max(0, 1.0 - dist / 30.0)
        else:
            scores['calculation'] = 0.2  # Detection failed
        
        # 3. Position accuracy: Same with looser tolerance
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['position'] = max(0, 1.0 - dist / 50.0)
        else:
            scores['position'] = 0.2  # Detection failed
        
        # 4. Annotation quality: Black pixel IoU
        gray_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        gray_gt = cv2.cvtColor(gt_last, cv2.COLOR_BGR2GRAY)
        
        black_mask_gen = (gray_gen < 50).astype(np.uint8)
        black_mask_gt = (gray_gt < 50).astype(np.uint8)
        
        black_overlap = np.sum((black_mask_gen > 0) & (black_mask_gt > 0))
        black_union = np.sum((black_mask_gen > 0) | (black_mask_gt > 0))
        
        scores['annotation'] = black_overlap / black_union if black_union > 0 else 0.5
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class HighlightHorizontalLinesEvaluator(BaseEvaluator):
    """
    G-223: Highlight horizontal lines evaluator.
    
    Rule-based evaluation:
    - Horizontal line identification accuracy (40%): All horizontal lines found
    - Marking completeness (30%): All horizontal lines marked
    - Marking position accuracy (20%): Circles centered on line midpoints
    - Visual annotation quality (10%): Black circles proper
    """
    
    TASK_WEIGHTS = {
        'identification': 0.40,
        'completeness': 0.30,
        'position': 0.20,
        'annotation': 0.10
    }
    
    def _detect_horizontal_lines(self, frame: np.ndarray) -> List[Dict]:
        """Detect horizontal line segments using contour analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find non-white regions (colored lines)
        non_white = (gray < 250).astype(np.uint8) * 255
        contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        horizontal_lines = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            
            # Horizontal line: width >> height (aspect > 5)
            if aspect > 5:
                midpoint = (x + w // 2, y + h // 2)
                horizontal_lines.append({
                    'start': (x, y),
                    'end': (x + w, y),
                    'midpoint': midpoint,
                    'length': w
                })
        
        return horizontal_lines
    
    def _detect_black_markings(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect black circle markings."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        black_mask = (gray < 50).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Black circles can be large (up to 50000 area for big marking circles)
            if area < 30:
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append((cx, cy))
        
        return centers
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate highlight horizontal lines task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Normalize frame size (handles padding removal + resize)
        if last_frame.shape != gt_last.shape:
            gt_last = normalize_frame_size(gt_last, last_frame)
        
        # Detect markings
        gen_markings = self._detect_black_markings(last_frame)
        gt_markings = self._detect_black_markings(gt_last)
        
        # Detect horizontal lines in both frames
        gen_lines = self._detect_horizontal_lines(last_frame)
        gt_lines = self._detect_horizontal_lines(gt_last)
        
        # Count expected horizontal lines (lines with y1 = y2)
        expected_horizontal_count = len([l for l in gt_lines if abs(l['start'][1] - l['end'][1]) < 10])
        
        # 1. Identification: Check if markings are on horizontal lines (40%)
        # Rule: Must correctly identify horizontal lines (y1 = y2) vs vertical (x1 = x2)
        if gen_markings and gen_lines:
            on_line_count = 0
            for marking in gen_markings:
                for line in gen_lines:
                    dist = np.sqrt((marking[0] - line['midpoint'][0])**2 + 
                                  (marking[1] - line['midpoint'][1])**2)
                    if dist < 80:  # More lenient threshold
                        on_line_count += 1
                        break
            scores['identification'] = on_line_count / len(gen_markings) if gen_markings else 0.0
        else:
            # No markings - score based on whether horizontal lines exist
            scores['identification'] = 0.5 if expected_horizontal_count == 0 else 0.0
        
        # 2. Completeness: Compare marking counts (30%)
        # Rule: All horizontal lines must be marked (recall = 100%)
        if gt_markings:
            count_diff = abs(len(gen_markings) - len(gt_markings))
            # Exact match or very close gets full score
            if count_diff == 0:
                scores['completeness'] = 1.0
            elif count_diff == 1:
                scores['completeness'] = 0.7
            else:
                scores['completeness'] = max(0.3, 1.0 - count_diff * 0.2)
        else:
            # No GT markings means no horizontal lines expected
            scores['completeness'] = 1.0 if len(gen_markings) == 0 else 0.5
        
        # 3. Position accuracy: Compare marking positions with GT
        if gen_markings and gt_markings:
            matched = 0
            total_dist = 0
            for gm in gen_markings:
                min_dist = float('inf')
                for gtm in gt_markings:
                    dist = np.sqrt((gm[0] - gtm[0])**2 + (gm[1] - gtm[1])**2)
                    min_dist = min(min_dist, dist)
                # Very close match (< 10 pixels) counts as perfect match
                if min_dist < 10:
                    matched += 1
                    total_dist += 0
                elif min_dist < 80:  # More lenient threshold
                    matched += 1
                    total_dist += min_dist
            
            if matched > 0:
                avg_dist = total_dist / matched
                scores['position'] = max(0, 1.0 - avg_dist / 40.0)
            else:
                scores['position'] = 0.3
        else:
            scores['position'] = 0.2  # Detection failed
        
        # 4. Annotation quality: Black pixel IoU
        gray_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        gray_gt = cv2.cvtColor(gt_last, cv2.COLOR_BGR2GRAY)
        
        black_mask_gen = (gray_gen < 50).astype(np.uint8)
        black_mask_gt = (gray_gt < 50).astype(np.uint8)
        
        black_overlap = np.sum((black_mask_gen > 0) & (black_mask_gt > 0))
        black_union = np.sum((black_mask_gen > 0) | (black_mask_gt > 0))
        
        scores['annotation'] = black_overlap / black_union if black_union > 0 else 0.5
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

class AddBordersToUnborderedEvaluator(BaseEvaluator):
    """
    G-240: Add borders to unbordered shapes evaluator.
    
    Rule-based evaluation:
    - Border identification accuracy (40%): Identify shapes without borders
    - Border addition correctness (35%): Add black borders correctly
    - Border appearance quality (15%): Border style and width
    - Scene preservation (10%): Original attributes unchanged
    """
    
    TASK_WEIGHTS = {
        'border_identification': 0.40,
        'border_addition': 0.35,
        'border_appearance': 0.15,
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
        
        # First, detect colorful shapes and check which ones have borders
        first_shapes = self._detect_colorful_shapes(first_frame)
        first_bordered = self._count_shapes_with_black_border(first_frame, first_shapes)
        
        # Check if all shapes already have borders in the first frame
        # If so, no change is needed and the task is already complete
        if len(first_shapes) > 0 and first_bordered == len(first_shapes):
            self._last_task_details = {
                'border_identification': 1.0,
                'border_addition': 1.0,
                'border_appearance': 1.0,
                'scene_preservation': 1.0,
                'all_already_bordered': True,
                'first_shapes': len(first_shapes),
                'first_bordered': first_bordered
            }
            return 1.0  # Task already complete
        
        # Check if borders were added by looking at dark pixel increase
        # (Borders are BLACK lines around colorful shapes, so they add dark pixels)
        first_dark = self._count_dark_edge_pixels(first_frame)
        final_dark = self._count_dark_edge_pixels(final_frame)
        dark_increase = final_dark - first_dark
        
        # Also check frame difference as secondary metric
        frame_diff = np.mean(np.abs(first_frame.astype(float) - final_frame.astype(float)))
        
        # If no dark pixel increase AND no frame difference, task not completed
        # Note: borders are thin black lines, so frame_diff can be small even with borders
        if dark_increase < 500 and frame_diff < 1:
            # No meaningful change - task not completed - ALL scores 0
            self._last_task_details = {
                'border_identification': 0.0,
                'border_addition': 0.0,
                'border_appearance': 0.0,
                'scene_preservation': 0.0,
                'no_change_detected': True,
                'dark_increase': int(dark_increase),
                'frame_diff': float(frame_diff)
            }
            return 0.0
        
        # Need significant increase in dark pixels for borders
        # Borders are BLACK lines, so should add at least 1000 dark pixels
        if dark_increase < 1000:
            self._last_task_details = {
                'border_identification': 0.0,
                'border_addition': 0.0,
                'border_appearance': 0.0,
                'scene_preservation': 0.5,
                'no_borders_added': True,
                'dark_pixel_increase': int(dark_increase)
            }
            return 0.0
        
        # Detect colorful shapes (几何体) in both frames
        final_shapes = self._detect_colorful_shapes(final_frame)
        
        # Count shapes with black borders
        final_bordered = self._count_shapes_with_black_border(final_frame, final_shapes)
        new_borders = final_bordered - first_bordered
        
        # CRITICAL CHECK: Shape count should remain approximately the same
        # Adding borders should NOT create new shapes
        shape_count_change = len(final_shapes) - len(first_shapes)
        if shape_count_change > 2:
            # Too many new shapes created - model is not just adding borders
            self._last_task_details = {
                'border_identification': 0.0,
                'border_addition': 0.0,
                'border_appearance': 0.0,
                'scene_preservation': 0.0,
                'too_many_new_shapes': True,
                'first_shapes': int(len(first_shapes)),
                'final_shapes': int(len(final_shapes)),
                'shape_count_change': int(shape_count_change)
            }
            return 0.0
        
        # 1. Border identification (40%): Check if unbordered shapes were identified
        scores['border_identification'] = self._evaluate_border_id(
            first_shapes, first_bordered, final_bordered
        )
        
        # 2. Border addition (35%): Check if black borders were added correctly
        scores['border_addition'] = self._evaluate_border_addition(
            first_frame, final_frame, dark_increase
        )
        
        # 3. Border appearance (15%): Check border quality (black, clean lines)
        scores['border_appearance'] = self._evaluate_border_appearance(final_frame)
        
        # 4. Scene preservation (10%): Check shapes preserved
        scores['scene_preservation'] = self._evaluate_scene_preservation(
            first_frame, final_frame, first_shapes, final_shapes
        )
        
        self._last_task_details = scores
        self._last_task_details['first_shapes'] = int(len(first_shapes))
        self._last_task_details['final_shapes'] = int(len(final_shapes))
        self._last_task_details['first_bordered'] = int(first_bordered)
        self._last_task_details['final_bordered'] = int(final_bordered)
        self._last_task_details['dark_increase'] = int(dark_increase)
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_colorful_shapes(self, frame: np.ndarray) -> List[Dict]:
        """Detect colorful geometric shapes (几何体) in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect saturated (colorful) regions - shapes are colorful
        saturation_mask = hsv[:, :, 1] > 50
        value_mask = hsv[:, :, 2] > 50
        colorful_mask = (saturation_mask & value_mask).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(colorful_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Skip small noise
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(cnt)
            
            shapes.append({
                'contour': cnt,
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area
            })
        
        return shapes
    
    def _count_shapes_with_black_border(self, frame: np.ndarray, shapes: List[Dict]) -> int:
        """Count how many shapes have black borders around them."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bordered_count = 0
        
        for shape in shapes:
            x, y, w, h = shape['bbox']
            
            # Expand bbox slightly to check for border
            margin = 5
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            # Check for dark pixels (black border) around the shape
            # Look at the perimeter region
            perimeter_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.rectangle(perimeter_mask, (x1, y1), (x2, y2), 255, margin * 2)
            cv2.rectangle(perimeter_mask, (x, y), (x + w, y + h), 0, -1)
            
            # Count dark pixels in perimeter
            perimeter_region = gray[perimeter_mask > 0]
            if len(perimeter_region) > 0:
                dark_pixels = np.sum(perimeter_region < 50)
                dark_ratio = dark_pixels / len(perimeter_region)
                
                # If significant dark pixels around shape, it has a border
                if dark_ratio > 0.1:
                    bordered_count += 1
        
        return bordered_count
    
    def _evaluate_border_id(
        self,
        first_shapes: List[Dict],
        first_bordered: int,
        final_bordered: int
    ) -> float:
        """Rule-based: Check if unbordered shapes were identified and bordered."""
        if len(first_shapes) == 0:
            return 0.0
        
        # Calculate how many shapes needed borders
        unbordered_first = len(first_shapes) - first_bordered
        if unbordered_first == 0:
            return 1.0  # All were already bordered
        
        # How many got new borders
        new_borders = final_bordered - first_bordered
        
        # Score based on proportion of unbordered shapes that got borders
        if new_borders >= unbordered_first:
            return 1.0  # All unbordered shapes got borders
        elif new_borders > 0:
            return max(0.5, new_borders / unbordered_first)
        else:
            return 0.0
    
    def _evaluate_border_addition(
        self,
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        dark_increase: int
    ) -> float:
        """Rule-based: Check if black borders were added correctly."""
        # Borders should add significant dark pixels
        if dark_increase < 1000:
            return 0.0
        
        # Check if dark pixels form edges (borders should be lines, not blobs)
        edges = cv2.Canny(final_frame, 50, 150)
        first_edges = cv2.Canny(first_frame, 50, 150)
        
        edge_count = np.sum(edges > 0)
        first_edge_count = np.sum(first_edges > 0)
        edge_increase = edge_count - first_edge_count
        
        # Good borders should have both dark pixel increase AND edge increase
        if dark_increase > 3000 and edge_increase > 1000:
            return 1.0
        elif dark_increase > 2000 or edge_increase > 500:
            return 0.8
        elif dark_increase > 1000:
            return 0.6
        else:
            return 0.0
    
    def _evaluate_border_appearance(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate border appearance quality."""
        # Check edge structure
        edges = cv2.Canny(final_frame, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Also check for dark pixels (borders are typically dark)
        gray = cv2.cvtColor(final_frame, cv2.COLOR_BGR2GRAY)
        dark_ratio = np.sum(gray < 50) / gray.size
        
        # Reasonable edge ratio indicates clean borders
        # Lower threshold since some images have sparse borders
        if edge_ratio > 0.001 and dark_ratio > 0.001:
            return 1.0
        elif edge_ratio > 0.0005 or dark_ratio > 0.0005:
            return 0.8
        else:
            return 0.2
    
    def _evaluate_scene_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        first_shapes: List[Dict],
        final_shapes: List[Dict]
    ) -> float:
        """Rule-based: Check if shape colors and positions are preserved."""
        # Check if same number of shapes exist
        if len(first_shapes) == 0:
            return 0.5
        
        shape_count_diff = abs(len(first_shapes) - len(final_shapes))
        
        if shape_count_diff > 2:
            return 0.3  # Too many shapes changed
        
        # Compare color distributions
        first_colors = self._get_color_distribution(first_frame)
        final_colors = self._get_color_distribution(final_frame)
        
        # Similar color distributions indicate preservation
        color_diff = np.sum(np.abs(first_colors - final_colors))
        
        # Normalize by total histogram sum
        total = np.sum(first_colors) + np.sum(final_colors)
        if total > 0:
            normalized_diff = color_diff / total
        else:
            normalized_diff = 0
        
        # Use normalized difference for more consistent scoring
        if normalized_diff < 0.1 and shape_count_diff <= 1:
            return 1.0
        elif normalized_diff < 0.2:
            return 0.8
        elif normalized_diff < 0.4:
            return 0.6
        else:
            return 0.4
    
    def _count_bordered_shapes(self, frame: np.ndarray) -> int:
        """Count shapes with borders (dark edges)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count contours with substantial edges (bordered shapes)
        return sum(1 for cnt in contours if cv2.arcLength(cnt, True) > 100)
    
    def _count_dark_edge_pixels(self, frame: np.ndarray) -> int:
        """Count dark pixels (potential borders)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.sum(gray < 50)
    
    def _get_color_distribution(self, frame: np.ndarray) -> np.ndarray:
        """Get color histogram."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [30], [0, 180])
        return hist.flatten()


class IdentifyChineseCharacterEvaluator(BaseEvaluator):
    """
    G-247: Identify Chinese character evaluator.
    
    Rule-based evaluation:
    - Character recognition correctness (45%): Identify Chinese vs non-Chinese
    - Marking target accuracy (30%): Mark correct character
    - Marking position/range (15%): Circle contains character
    - Marking specification compliance (10%): Red circle, proper style
    """
    
    TASK_WEIGHTS = {
        'character_recognition': 0.45,
        'marking_target': 0.30,
        'marking_position': 0.15,
        'marking_specification': 0.10
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
        
        # 1. Character recognition (45%)
        scores['character_recognition'] = self._evaluate_character_recognition(
            first_frame, final_frame
        )
        
        # 2. Marking target (30%)
        scores['marking_target'] = self._evaluate_marking_target(
            first_frame, final_frame
        )
        
        # 3. Marking position (15%)
        scores['marking_position'] = self._evaluate_marking_position(final_frame)
        
        # 4. Marking specification (10%)
        scores['marking_specification'] = self._evaluate_marking_spec(final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_character_recognition(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if Chinese character is identified."""
        # Find Chinese character (more complex pattern)
        chinese_pos = self._find_chinese_character(first_frame)
        
        # Detect circle marking
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        if chinese_pos is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - chinese_pos[0])**2 + (circle[1] - chinese_pos[1])**2)
        
        if dist < 40:
            return 1.0
        elif dist < 80:
            return 0.7
        elif dist < 120:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_marking_target(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if correct character is marked."""
        circle = self._detect_red_circle(final_frame)
        chinese_pos = self._find_chinese_character(first_frame)
        
        if circle is None:
            return 0.0
        if chinese_pos is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - chinese_pos[0])**2 + (circle[1] - chinese_pos[1])**2)
        return max(0.0, 1.0 - dist / 60)
    
    def _evaluate_marking_position(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check if circle contains the character."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # Check if circle is in valid position
        h, w = final_frame.shape[:2]
        x, y, r = circle
        
        if 30 < x < w - 30 and 30 < y < h - 30:
            return 1.0
        else:
            return 0.5
    
    def _evaluate_marking_spec(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check marking specification."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        if 20 < r < 100:
            return 1.0
        else:
            return 0.5
    
    def _find_chinese_character(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find Chinese character (more complex than letters)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        characters = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 200 < area < 20000:
                # Check complexity (Chinese characters have more complexity)
                peri = cv2.arcLength(cnt, True)
                complexity = peri * peri / (area + 1)
                
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    characters.append((cx, cy, complexity, area))
        
        if len(characters) == 0:
            return None
        
        # Chinese character typically has highest complexity
        most_complex = max(characters, key=lambda c: c[2])
        return (most_complex[0], most_complex[1])
    
    def _detect_red_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect red circle."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                (x, y), r = cv2.minEnclosingCircle(largest)
                return (int(x), int(y), int(r))
        
        return None


class MarkAsymmetricalShapeEvaluator(BaseEvaluator):
    """
    G-248: Mark asymmetrical shape evaluator.
    
    Rule-based evaluation:
    - Symmetry identification correctness (45%): Find asymmetrical shape
    - Marking precision (30%): Circle accurately marks target
    - Marking quality (15%): Red circle quality
    - Scene preservation (10%): Original shapes unchanged
    """
    
    TASK_WEIGHTS = {
        'symmetry_identification': 0.45,
        'marking_precision': 0.30,
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
        
        # 1. Symmetry identification (45%)
        scores['symmetry_identification'] = self._evaluate_symmetry_id(
            first_frame, final_frame
        )
        
        # 2. Marking precision (30%)
        scores['marking_precision'] = self._evaluate_marking_precision(
            first_frame, final_frame
        )
        
        # 3. Marking quality (15%)
        scores['marking_quality'] = self._evaluate_marking_quality(final_frame)
        
        # 4. Scene preservation (10%)
        scores['scene_preservation'] = self._evaluate_scene_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_symmetry_id(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if asymmetrical shape is identified."""
        # Find asymmetrical shape
        asymmetric = self._find_asymmetrical_shape(first_frame)
        
        # Detect circle marking
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        if asymmetric is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - asymmetric[0])**2 + (circle[1] - asymmetric[1])**2)
        
        if dist < 50:
            return 1.0
        elif dist < 100:
            return 0.7
        elif dist < 150:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_marking_precision(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Evaluate marking precision."""
        circle = self._detect_red_circle(final_frame)
        asymmetric = self._find_asymmetrical_shape(first_frame)
        
        if circle is None:
            return 0.0
        if asymmetric is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - asymmetric[0])**2 + (circle[1] - asymmetric[1])**2)
        return max(0.0, 1.0 - dist / 80)
    
    def _evaluate_marking_quality(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate red circle quality."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        if 30 < r < 150:
            return 1.0
        else:
            return 0.5
    
    def _evaluate_scene_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if shapes are preserved."""
        first_count = self._count_shapes(first_frame)
        
        # Remove red marking
        hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        final_no_red = final_frame.copy()
        final_no_red[red_mask > 0] = [255, 255, 255]
        
        final_count = self._count_shapes(final_no_red)
        
        if abs(first_count - final_count) <= 1:
            return 1.0
        else:
            return 0.6
    
    def _find_asymmetrical_shape(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the asymmetrical shape (odd-sided polygon or lowest symmetry)."""
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
                    
                    # Get vertex count
                    perimeter = cv2.arcLength(cnt, True)
                    epsilon = 0.02 * perimeter
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    vertices = len(approx)
                    
                    # Calculate symmetry
                    symmetry = self._calculate_symmetry(cnt)
                    
                    shapes.append((cx, cy, symmetry, vertices))
        
        if len(shapes) == 0:
            return None
        
        # First, look for odd-sided polygons (they have no line of bilateral symmetry)
        odd_sided = [s for s in shapes if s[3] % 2 == 1 and s[3] >= 5]  # 5, 7, 9, etc.
        if odd_sided:
            # Return the odd-sided polygon (asymmetrical by definition)
            return (odd_sided[0][0], odd_sided[0][1])
        
        # Otherwise, find shape with lowest symmetry score
        most_asymmetric = min(shapes, key=lambda s: s[2])
        return (most_asymmetric[0], most_asymmetric[1])
    
    def _detect_shapes_with_symmetry(self, frame: np.ndarray) -> List[Tuple[int, int, float]]:
        """Detect shapes with their symmetry score."""
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
                    
                    # Calculate symmetry score
                    symmetry = self._calculate_symmetry(cnt)
                    shapes.append((cx, cy, symmetry))
        
        return shapes
    
    def _calculate_symmetry(self, contour: np.ndarray) -> float:
        """Calculate symmetry score for a contour using multiple methods."""
        # Get shape properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0 or area == 0:
            return 0.5
        
        # Circularity (how close to a circle)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Convex hull for solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Approximate polygon to count vertices
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # Symmetry score based on multiple factors:
        # - High circularity = more symmetric
        # - High solidity = more regular
        # - Even vertices = more symmetric
        # - Odd vertices (especially prime like 7) = less symmetric
        
        vertex_symmetry = 1.0 if vertices % 2 == 0 else 0.5
        if vertices in [3, 5, 7, 11, 13]:  # Odd-sided polygons are less symmetric
            vertex_symmetry = 0.3
        
        # Also do mirror symmetry check
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.drawContours(mask, [contour - [x-1, y-1]], -1, 255, -1)
        
        mid = w // 2
        left = mask[:, :mid]
        right = cv2.flip(mask[:, mid:], 1)
        
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]
        
        if left.size > 0 and right.size > 0:
            intersection = np.sum((left > 0) & (right > 0))
            union = np.sum((left > 0) | (right > 0))
            mirror_symmetry = intersection / union if union > 0 else 0.5
        else:
            mirror_symmetry = 0.5
        
        # Combine factors: lower score = more asymmetric
        symmetry = 0.3 * circularity + 0.2 * solidity + 0.2 * vertex_symmetry + 0.3 * mirror_symmetry
        
        return symmetry
    
    def _count_shapes(self, frame: np.ndarray) -> int:
        """Count shapes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return sum(1 for cnt in contours if cv2.contourArea(cnt) > 300)
    
    def _detect_red_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect red circle."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                (x, y), r = cv2.minEnclosingCircle(largest)
                return (int(x), int(y), int(r))
        
        return None

class ColorTripleIntersectionEvaluator(BaseEvaluator):
    """
    G-250: Color Triple Intersection Red
    
    Task: In a Venn diagram with 3 circles, identify and fill the triple 
    intersection region (where all 3 circles overlap) with red color.
    
    Rule-based evaluation:
    1. Triple intersection identification (40%) - Correct region identified
    2. Fill coverage (30%) - >=95% of triple intersection filled
    3. Fill precision (20%) - No overflow to other regions (>=95%)
    4. Visual quality (10%) - Pure red color, uniform fill
    """
    
    TASK_WEIGHTS = {
        'triple_intersection_identification': 0.40,
        'fill_coverage': 0.30,
        'fill_precision': 0.20,
        'visual_quality': 0.10
    }
    
    def _detect_red_region(self, frame: np.ndarray) -> np.ndarray:
        """Detect red-filled regions in frame."""
        if len(frame.shape) == 2:
            return np.zeros_like(frame, dtype=bool)

        # Red detection in HSV (frames are BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 80])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        return mask > 0
    
    def _check_red_purity(self, frame: np.ndarray, red_mask: np.ndarray) -> float:
        """Check if red color is pure (close to RGB 255,0,0)."""
        if red_mask.sum() < 10:
            return 0.0
        
        red_pixels = frame[red_mask]
        if len(red_pixels) == 0:
            return 0.0
        
        # BGR format - check blue channel is high (red in BGR)
        mean_b = np.mean(red_pixels[:, 0])
        mean_g = np.mean(red_pixels[:, 1])
        mean_r = np.mean(red_pixels[:, 2])
        
        # Check closeness to pure red (0, 0, 255) in BGR
        b_score = max(0, 1 - mean_b / 100)
        g_score = max(0, 1 - mean_g / 100)
        r_score = max(0, 1 - abs(mean_r - 255) / 100)
        
        return (r_score + g_score + b_score) / 3
    
    def _detect_circles(self, frame: np.ndarray) -> List[Dict]:
        """Detect circles in the Venn diagram."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                    param1=50, param2=30, minRadius=50, maxRadius=300)
        
        detected = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                detected.append({
                    'center': (i[0], i[1]),
                    'radius': i[2]
                })
        
        return detected
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate triple intersection filling accuracy."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        scores = {}
        
        # Detect red pixels in generated and GT
        red_mask_gen = self._detect_red_region(gen_final)
        red_mask_gt = self._detect_red_region(gt_final)
        
        # Calculate intersection and union for precision/recall
        intersection = np.logical_and(red_mask_gen, red_mask_gt).sum()
        gt_area = red_mask_gt.sum()
        gen_area = red_mask_gen.sum()
        
        # Fill coverage (recall) - how much of GT region is covered
        coverage = intersection / max(gt_area, 1)
        
        # Fill precision - how much of generated is correct
        precision = intersection / max(gen_area, 1)
        
        # Check if any red region was identified
        identification_score = 0.0
        if gen_area > 100:
            identification_score = 0.5
            if coverage > 0.5:
                identification_score = min(1.0, coverage + 0.3)
        
        # Visual quality - check red color purity
        visual_quality = self._check_red_purity(gen_final, red_mask_gen)
        
        scores['triple_intersection_identification'] = identification_score
        scores['fill_coverage'] = coverage
        scores['fill_precision'] = precision
        scores['visual_quality'] = visual_quality
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

class HighDensityLiquidEvaluator(BaseEvaluator):
    """
    G-273: High density liquid evaluator.
    
    Rule-based evaluation:
    - Physics reasoning accuracy (45%): Identify high-density liquid (object floats)
    - Marking object correctness (30%): Mark object in high-density liquid
    - Marking standardization (20%): Red rectangle marking
    - Element preservation (5%): Original elements unchanged
    """
    
    TASK_WEIGHTS = {
        'physics_reasoning': 0.45,
        'marking_correctness': 0.30,
        'marking_standardization': 0.20,
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
        
        # 1. Physics reasoning (45%)
        scores['physics_reasoning'] = self._evaluate_physics_reasoning(
            first_frame, final_frame
        )
        
        # 2. Marking correctness (30%)
        scores['marking_correctness'] = self._evaluate_marking_correctness(
            first_frame, final_frame
        )
        
        # 3. Marking standardization (20%)
        scores['marking_standardization'] = self._evaluate_marking_standard(final_frame)
        
        # 4. Element preservation (5%)
        scores['element_preservation'] = self._evaluate_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_physics_reasoning(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if high-density liquid (floating yellow square) is identified.
        
        CRITICAL RULES:
        1. There must be TWO yellow squares floating in liquid containers
        2. The squares should be in the middle portion of the frame (not at top)
        3. A red rectangle marking should be around the HIGHER floating square
        """
        h, w = final_frame.shape[:2]
        
        # Find yellow squares
        yellow_objects = self._find_yellow_squares(final_frame)
        
        # CRITICAL: Must have exactly 2 yellow squares
        if len(yellow_objects) != 2:
            return 0.1
        
        # CRITICAL: Yellow squares should be in the middle portion (not at top edge)
        # If squares are at y < 10% of height, they're not in liquid containers
        for obj in yellow_objects:
            if obj[2] < h * 0.1:  # top_y < 10% of height
                return 0.1  # Objects at top edge - wrong scene
        
        # Find the higher floating square (smaller y = higher on screen)
        higher_obj = min(yellow_objects, key=lambda o: o[2])
        lower_obj = max(yellow_objects, key=lambda o: o[2])
        
        # CRITICAL: There should be a noticeable height difference
        height_diff = lower_obj[2] - higher_obj[2]
        if height_diff < 30:
            return 0.3  # Not enough difference to determine which is higher
        
        # Detect red rectangle marking
        rect = self._detect_red_rectangle(final_frame)
        
        if rect is None:
            return 0.2  # No marking found
        
        rect_center = ((rect[0] + rect[2])//2, (rect[1] + rect[3])//2)
        
        # Check if rectangle marks the HIGHER floating square
        dist_to_higher = np.sqrt((rect_center[0] - higher_obj[0])**2 + (rect_center[1] - higher_obj[1])**2)
        dist_to_lower = np.sqrt((rect_center[0] - lower_obj[0])**2 + (rect_center[1] - lower_obj[1])**2)
        
        # Rectangle should be closer to the higher floating square
        if dist_to_higher < dist_to_lower:
            if dist_to_higher < 80:
                return 1.0
            elif dist_to_higher < 150:
                return 0.8
            else:
                return 0.5
        else:
            # Marked the wrong square!
            return 0.2
    
    def _find_yellow_squares(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find yellow squares with their positions.
        
        Returns: List of (center_x, center_y, top_y, area)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([45, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yellow_objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Yellow squares should be reasonably sized
            if 2000 < area < 50000:
                x, y, bw, bh = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    yellow_objects.append((cx, cy, y, area))
        
        return yellow_objects
    
    def _evaluate_marking_correctness(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if the correct yellow square is marked with red rectangle."""
        rect = self._detect_red_rectangle(final_frame)
        floating_obj = self._find_floating_object(final_frame)
        
        if rect is None:
            return 0.0
        
        h, w = final_frame.shape[:2]
        rect_center = ((rect[0] + rect[2])//2, (rect[1] + rect[3])//2)
        
        if floating_obj is None:
            # Fallback: check if rectangle is in the expected region
            if h * 0.3 < rect_center[1] < h * 0.7:
                return 0.8  # Rectangle is in reasonable position
            return 0.0
        
        dist = np.sqrt((rect_center[0] - floating_obj[0])**2 + (rect_center[1] - floating_obj[1])**2)
        
        # More lenient distance threshold for this task
        return max(0.3, 1.0 - dist / 150)
    
    def _evaluate_marking_standard(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check if red rectangle marking is used."""
        rect = self._detect_red_rectangle(final_frame)
        
        if rect is None:
            return 0.0
        
        x1, y1, x2, y2 = rect
        w = x2 - x1
        h = y2 - y1
        
        if 20 < w < 200 and 20 < h < 200:
            return 1.0
        else:
            return 0.5
    
    def _evaluate_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if elements are preserved."""
        # Count objects
        first_objects = self._count_objects(first_frame)
        
        # Remove red marking
        hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        final_no_red = final_frame.copy()
        final_no_red[red_mask > 0] = [255, 255, 255]
        
        final_objects = self._count_objects(final_no_red)
        
        if abs(first_objects - final_objects) <= 1:
            return 1.0
        else:
            return 0.6
    
    def _find_floating_object(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the yellow square (方块) that floats higher in the higher density liquid.
        
        This task has two containers with blue/green liquids. Yellow squares float
        in each. The one in higher density liquid floats HIGHER (top above liquid).
        We need to find the yellow square that floats higher and return its position.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        
        # Find yellow squares (方块)
        # Yellow hue range: 15-45 in OpenCV HSV
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([45, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yellow_objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Yellow squares are around 10000-15000 area
            if 5000 < area < 30000:
                x, y, bw, bh = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Store center and top y position
                    yellow_objects.append((cx, cy, y, area))
        
        if len(yellow_objects) == 0:
            return None
        
        # The yellow square in higher density liquid floats HIGHER (smaller y = top)
        # Find the one with smallest top y (highest on screen)
        topmost = min(yellow_objects, key=lambda o: o[2])
        return (topmost[0], topmost[1])
    
    def _count_objects(self, frame: np.ndarray) -> int:
        """Count objects in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return sum(1 for cnt in contours if 500 < cv2.contourArea(cnt) < 10000)
    
    def _detect_red_rectangle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect red rectangle marking in the frame.
        
        CRITICAL: The marking should be a small-to-medium sized rectangle,
        not a large container. Filter out areas that are too large.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_frame, w_frame = frame.shape[:2]
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size - marking should be small to medium
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Marking rectangle should be:
            # - Not too small (> 500 pixels)
            # - Not too large (< 10% of frame area)
            # - Roughly square-ish (aspect ratio between 0.3 and 3)
            frame_area = h_frame * w_frame
            if 500 < area < frame_area * 0.1:
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3:
                    valid_contours.append((cnt, area))
        
        if valid_contours:
            # Take the largest valid contour
            largest = max(valid_contours, key=lambda x: x[1])[0]
            x, y, w, h = cv2.boundingRect(largest)
            return (x, y, x + w, y + h)
        
        return None

class PigmentColorMixingEvaluator(BaseEvaluator):
    """
    O-2: Pigment color mixing (subtractive) evaluator.
    
    Rule-based evaluation:
    - Color mixing rule correctness (60%): CMY subtractive mixing result color
    - Mixing area fill accuracy (25%): Complete fill in marked zone
    - Scene preservation (10%): Original circles unchanged
    - Visual quality (5%): Clean edges, no artifacts
    
    CMY Subtractive Mixing Rules:
    - Cyan + Magenta = Blue
    - Cyan + Yellow = Green
    - Magenta + Yellow = Red
    - Cyan + Magenta + Yellow = Black
    """
    
    # CMY subtractive mixing expected results (in BGR format)
    CMY_MIXING_RULES = {
        ('cyan', 'magenta'): (255, 0, 0),      # Blue
        ('cyan', 'yellow'): (0, 255, 0),        # Green
        ('magenta', 'yellow'): (0, 0, 255),     # Red
        ('cyan', 'magenta', 'yellow'): (0, 0, 0),  # Black
    }
    
    TASK_WEIGHTS = {
        'mixing_correctness': 0.60,
        'fill_accuracy': 0.25,
        'scene_preservation': 0.10,
        'visual_quality': 0.05
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
        
        scores['mixing_correctness'] = self._evaluate_color_mixing(
            first_frame, final_frame, gt_final_frame
        )
        scores['fill_accuracy'] = self._evaluate_fill_region(
            first_frame, final_frame
        )
        scores['scene_preservation'] = self._evaluate_preservation(
            first_frame, final_frame
        )
        scores['visual_quality'] = self._evaluate_visual_quality(final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_color_mixing(
        self, 
        first_frame: np.ndarray, 
        final_frame: np.ndarray,
        gt_final_frame: Optional[np.ndarray] = None
    ) -> float:
        """Rule-based: Check if mixed color matches expected result."""
        # Get mixed color from final frame (center region)
        mixed_color = self._get_mixed_region_color(final_frame)
        
        if mixed_color is None:
            return 0.0
        
        # If GT final frame available, compare with GT center color
        if gt_final_frame is not None:
            gt_mixed_color = self._get_mixed_region_color(gt_final_frame)
            if gt_mixed_color is not None:
                color_diff = np.sqrt(np.sum((np.array(mixed_color) - np.array(gt_mixed_color)) ** 2))
                
                if color_diff < 30:
                    return 1.0
                elif color_diff < 60:
                    return 0.8
                elif color_diff < 100:
                    return 0.5
                else:
                    return 0.2
        
        # Fallback: Check if mixed color follows CMY rules
        input_colors = self._detect_input_colors(first_frame)
        
        if len(input_colors) < 2:
            return 0.2
        
        expected_color = self._calculate_expected_mix(input_colors)
        
        if expected_color is None:
            return 0.0
        
        color_diff = np.sqrt(np.sum((np.array(mixed_color) - np.array(expected_color)) ** 2))
        
        if color_diff < 50:
            return 1.0
        elif color_diff < 100:
            return 0.7
        elif color_diff < 150:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_fill_region(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if mixing region is properly filled."""
        h, w = first_frame.shape[:2]
        cx, cy = w // 2, h // 2
        size = 60
        
        # Get center region
        region = final_frame[max(0, cy-size):min(h, cy+size), max(0, cx-size):min(w, cx+size)]
        
        if region.size == 0:
            return 0.5
        
        # Check if region has uniform color (properly filled)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Low saturation variance indicates uniform fill
        sat_var = np.var(hsv[:, :, 1])
        
        if sat_var < 500:
            return 1.0
        elif sat_var < 1000:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_preservation(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if original pigment circles are preserved."""
        # Count colored regions in left/right thirds
        h, w = first_frame.shape[:2]
        
        first_left = self._count_colored_pixels(first_frame[:, :w//3])
        first_right = self._count_colored_pixels(first_frame[:, 2*w//3:])
        final_left = self._count_colored_pixels(final_frame[:, :w//3])
        final_right = self._count_colored_pixels(final_frame[:, 2*w//3:])
        
        # Circles should be preserved (similar pixel counts)
        left_ratio = min(first_left, final_left) / max(first_left, final_left, 1)
        right_ratio = min(first_right, final_right) / max(first_right, final_right, 1)
        
        avg_ratio = (left_ratio + right_ratio) / 2
        
        if avg_ratio > 0.8:
            return 1.0
        elif avg_ratio > 0.6:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_visual_quality(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate visual quality."""
        try:
            gray = cv2.cvtColor(final_frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size
            
            if 0.01 < edge_ratio < 0.15:
                return 1.0
            return 0.5
        except Exception:
            return 0.5
    
    def _detect_input_colors(self, frame: np.ndarray) -> List[str]:
        """Detect CMY colors present in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        colors_present = []
        
        # Cyan detection (hue ~90) - lower saturation threshold
        cyan_mask = cv2.inRange(hsv, np.array([80, 50, 50]), np.array([100, 255, 255]))
        if np.sum(cyan_mask > 0) > 500:
            colors_present.append('cyan')
        
        # Magenta detection (hue ~140-170) - lower saturation threshold
        magenta_mask = cv2.inRange(hsv, np.array([130, 50, 50]), np.array([170, 255, 255]))
        if np.sum(magenta_mask > 0) > 500:
            colors_present.append('magenta')
        
        # Yellow detection (hue ~20-40) - lower saturation threshold
        yellow_mask = cv2.inRange(hsv, np.array([15, 50, 50]), np.array([45, 255, 255]))
        if np.sum(yellow_mask > 0) > 500:
            colors_present.append('yellow')
        
        return colors_present
    
    def _get_mixed_region_color(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Get average color of the center mixing region."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        size = 40
        
        region = frame[max(0, cy-size):min(h, cy+size), max(0, cx-size):min(w, cx+size)]
        
        if region.size == 0:
            return None
        
        avg_color = np.mean(region, axis=(0, 1))
        return tuple(int(c) for c in avg_color)
    
    def _calculate_expected_mix(self, input_colors: List[str]) -> Optional[Tuple[int, int, int]]:
        """Calculate expected mixed color based on CMY rules."""
        key = tuple(sorted(input_colors))
        
        for rule_key, result in self.CMY_MIXING_RULES.items():
            if set(key) == set(rule_key):
                return result
        
        return None
    
    def _count_colored_pixels(self, region: np.ndarray) -> int:
        """Count non-white colored pixels."""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        return np.sum(hsv[:, :, 1] > 50)

# Export all evaluators
OUT_OF_DOMAIN_50_EVALUATORS_PART3 = {
    'G-219_select_leftmost_shape_data-generator': SelectLeftmostShapeEvaluator,
    'G-221_outline_innermost_square_data-generator': OutlineInnermostSquareEvaluator,
    'G-222_mark_tangent_point_of_circles_data-generator': MarkTangentPointEvaluator,
    'G-223_highlight_horizontal_lines_data-generator': HighlightHorizontalLinesEvaluator,
    'G-240_add_borders_to_unbordered_shapes_data-generator': AddBordersToUnborderedEvaluator,
    'G-247_identify_chinese_character_data-generator': IdentifyChineseCharacterEvaluator,
    'G-248_mark_asymmetrical_shape_data-generator': MarkAsymmetricalShapeEvaluator,
    'G-250_color_triple_intersection_red_data-generator': ColorTripleIntersectionEvaluator,
    'G-273_high_density_liquid_data-generator': HighDensityLiquidEvaluator,
    'O-2_pigment_color_mixing_subtractive_data-generator': PigmentColorMixingEvaluator
}
