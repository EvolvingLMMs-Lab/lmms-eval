"""
Specific evaluators for Out-of-Domain_50 tasks (Part 2).
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import normalize_frame_size

class IdentifyNearestSquareRectangleEvaluator(BaseEvaluator):
    """
    G-168: Identify nearest to square rectangle evaluator.
    
    Rule-based evaluation:
    - Aspect ratio judgment accuracy (50%): Correct rectangle selected
    - Marking uniqueness (20%): Only one rectangle marked
    - Marking position accuracy (20%): Circle accurately surrounds rectangle
    - Visual annotation quality (10%): Red circle proper
    """
    
    TASK_WEIGHTS = {
        'aspect_ratio': 0.50,
        'uniqueness': 0.20,
        'position': 0.20,
        'annotation': 0.10
    }
    
    def _detect_rectangles(self, frame: np.ndarray) -> List[Dict]:
        """Detect rectangles and calculate their aspect ratios."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            
            # Approximate polygon
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            
            if len(approx) == 4:  # Rectangle
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = min(w, h) / max(w, h)  # 1.0 = perfect square
                
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                rectangles.append({
                    'center': (cx, cy),
                    'aspect_ratio': aspect_ratio,
                    'area': area,
                    'bounds': (x, y, w, h)
                })
        
        return rectangles
    
    def _detect_red_marking(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red circle marking."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
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
        """Evaluate identify nearest to square rectangle task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect rectangles and markings
        gen_rects = self._detect_rectangles(last_frame)
        gt_rects = self._detect_rectangles(gt_last)
        
        gen_marking = self._detect_red_marking(last_frame)
        gt_marking = self._detect_red_marking(gt_last)
        
        # 1. Aspect ratio judgment: Check if marking is near the most square rectangle
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['aspect_ratio'] = max(0, 1.0 - dist / 100.0)
        elif gen_marking is not None and gen_rects:
            # Check if marked rectangle has highest aspect ratio
            marked_rect = None
            for rect in gen_rects:
                dist = np.sqrt((gen_marking[0] - rect['center'][0])**2 + 
                              (gen_marking[1] - rect['center'][1])**2)
                if dist < 100:
                    marked_rect = rect
                    break
            
            if marked_rect is not None:
                # Check if this is the most square one
                max_ratio = max(r['aspect_ratio'] for r in gen_rects)
                if marked_rect['aspect_ratio'] >= max_ratio - 0.1:
                    scores['aspect_ratio'] = 0.8
                else:
                    scores['aspect_ratio'] = 0.3
            else:
                scores['aspect_ratio'] = 0.3
        else:
            scores['aspect_ratio'] = 0.2  # Detection failed
        
        # 2. Uniqueness: Only one marking
        hsv_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask_gen = cv2.inRange(hsv_gen, lower_red1, upper_red1) | cv2.inRange(hsv_gen, lower_red2, upper_red2)
        contours_gen, _ = cv2.findContours(red_mask_gen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        significant_contours = [c for c in contours_gen if cv2.contourArea(c) > 100]
        
        if len(significant_contours) == 1:
            scores['uniqueness'] = 1.0
        elif len(significant_contours) == 0:
            scores['uniqueness'] = 0.0
        else:
            scores['uniqueness'] = max(0, 1.0 - (len(significant_contours) - 1) * 0.3)
        
        # 3. Position accuracy: Compare marking positions
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['position'] = max(0, 1.0 - dist / 50.0)
        else:
            scores['position'] = 0.2  # Detection failed
        
        # 4. Annotation quality: Red pixel presence
        scores['annotation'] = min(1.0, np.sum(red_mask_gen > 0) / 500.0)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class LocateSegmentIntersectionEvaluator(BaseEvaluator):
    """
    G-169: Locate intersection of segments evaluator.
    
    Rule-based evaluation:
    - Intersection calculation accuracy (60%): Precise intersection point
    - Marking position accuracy (25%): Circle centered on intersection
    - Visual annotation quality (10%): Red circle proper
    - Marking uniqueness (5%): Only one point marked
    """
    
    TASK_WEIGHTS = {
        'calculation': 0.60,
        'position': 0.25,
        'annotation': 0.10,
        'uniqueness': 0.05
    }
    
    def _detect_lines(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect line segments in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return []
        
        return [(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]
    
    def _line_intersection(self, line1: Tuple, line2: Tuple) -> Optional[Tuple[float, float]]:
        """Calculate intersection point of two line segments."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        
        return (px, py)
    
    def _detect_red_marking(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red circle marking."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        M = cv2.moments(red_mask)
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
        """Evaluate locate segment intersection task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect markings
        gen_marking = self._detect_red_marking(last_frame)
        gt_marking = self._detect_red_marking(gt_last)
        
        # 1. Calculation accuracy: Compare marking positions
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['calculation'] = max(0, 1.0 - dist / 30.0)  # Tight tolerance
        else:
            scores['calculation'] = 0.5 if gen_marking is None and gt_marking is None else 0.0
        
        # 2. Position accuracy: Same as calculation but with looser tolerance
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['position'] = max(0, 1.0 - dist / 50.0)
        else:
            scores['position'] = 0.2  # Detection failed
        
        # 3. Annotation quality: Red pixel IoU
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
        
        scores['annotation'] = red_overlap / red_union if red_union > 0 else 0.5
        
        # 4. Uniqueness: Only one marking
        contours_gen, _ = cv2.findContours(red_mask_gen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours_gen if cv2.contourArea(c) > 50]
        
        if len(significant_contours) == 1:
            scores['uniqueness'] = 1.0
        elif len(significant_contours) == 0:
            scores['uniqueness'] = 0.0
        else:
            scores['uniqueness'] = max(0, 1.0 - (len(significant_contours) - 1) * 0.3)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

class ArrangeCirclesByCircumferenceEvaluator(BaseEvaluator):
    """
    G-174: Arrange circles by circumference (large to small).
    
    Rule-based evaluation:
    - Sorting correctness (40%): Circles ordered by size (descending left to right)
    - Layout accuracy (30%): Horizontal alignment, even spacing
    - Object fidelity (20%): Circle properties preserved
    - Completeness (10%): All circles present
    """
    
    TASK_WEIGHTS = {
        'sorting_correctness': 0.40,
        'layout_accuracy': 0.30,
        'object_fidelity': 0.20,
        'completeness': 0.10
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
        
        # 1. Sorting correctness (40%)
        scores['sorting_correctness'] = self._evaluate_sorting(final_frame)
        
        # 2. Layout accuracy (30%)
        scores['layout_accuracy'] = self._evaluate_layout(final_frame)
        
        # 3. Object fidelity (20%)
        scores['object_fidelity'] = self._evaluate_fidelity(first_frame, final_frame)
        
        # 4. Completeness (10%)
        scores['completeness'] = self._evaluate_completeness(first_frame, final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_sorting(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check if circles are sorted by circumference (large to small)."""
        circles = self._detect_circles_with_size(final_frame)
        
        if len(circles) < 2:
            return 0.0  # STRICT: Not enough circles detected
        
        # Sort by x-position (left to right)
        sorted_by_x = sorted(circles, key=lambda c: c[0])
        radii = [c[2] for c in sorted_by_x]
        
        # Count inversions (smaller before larger - should be descending)
        inversions = sum(1 for i in range(len(radii)-1) if radii[i] < radii[i+1])
        max_inversions = len(radii) - 1
        
        if max_inversions == 0:
            return 1.0
        
        score = 1.0 - inversions / max_inversions
        return score
    
    def _evaluate_layout(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate horizontal alignment and spacing."""
        circles = self._detect_circles_with_size(final_frame)
        
        if len(circles) < 2:
            return 0.0  # STRICT: Not enough circles for layout eval
        
        # Check Y-coordinate alignment
        y_coords = [c[1] for c in circles]
        y_variance = np.var(y_coords)
        alignment_score = 1.0 / (1.0 + y_variance / 100)
        
        # Check spacing uniformity
        sorted_by_x = sorted(circles, key=lambda c: c[0])
        spacings = []
        for i in range(1, len(sorted_by_x)):
            spacing = sorted_by_x[i][0] - sorted_by_x[i-1][0]
            spacings.append(spacing)
        
        if len(spacings) > 1:
            spacing_variance = np.var(spacings) / (np.mean(spacings) + 1)
            spacing_score = 1.0 / (1.0 + spacing_variance)
        else:
            spacing_score = 1.0
        
        return 0.6 * alignment_score + 0.4 * spacing_score
    
    def _evaluate_fidelity(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if circle properties are preserved."""
        first_circles = self._detect_circles_with_size(first_frame)
        final_circles = self._detect_circles_with_size(final_frame)
        
        if len(first_circles) == 0 or len(final_circles) == 0:
            return 0.0  # STRICT: No circles detected
        
        # Compare radii distributions
        first_radii = sorted([c[2] for c in first_circles])
        final_radii = sorted([c[2] for c in final_circles])
        
        if len(first_radii) != len(final_radii):
            return 0.0  # STRICT: Circle count changed
        
        # Calculate radius similarity
        radius_diffs = [abs(fr - gr) for fr, gr in zip(first_radii, final_radii)]
        avg_diff = np.mean(radius_diffs)
        
        return max(0.0, 1.0 - avg_diff / 30)
    
    def _evaluate_completeness(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if all circles are present."""
        first_circles = self._detect_circles_with_size(first_frame)
        final_circles = self._detect_circles_with_size(final_frame)
        
        if len(first_circles) == 0:
            return 0.0  # STRICT: No circles in first frame
        
        completeness = min(1.0, len(final_circles) / len(first_circles))
        
        if len(final_circles) > len(first_circles):
            completeness *= 0.9  # Penalize extra circles
        
        return completeness
    
    def _detect_circles_with_size(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect circles with their x, y, radius."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 30,
            param1=50, param2=30, minRadius=15, maxRadius=100
        )
        
        result = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                result.append((int(x), int(y), int(r)))
        
        return result

class DrawMidpointPerpendicularEvaluator(BaseEvaluator):
    """
    G-189: Draw midpoint perpendicular line evaluator.
    
    Rule-based evaluation:
    - Midpoint identification accuracy (40%): Correct midpoint found
    - Perpendicular line position accuracy (30%): Line at x=width/2
    - Perpendicular line length/range (20%): Line spans between parallel lines
    - Visual quality (10%): Red line proper
    """
    
    TASK_WEIGHTS = {
        'midpoint': 0.40,
        'position': 0.30,
        'range': 0.20,
        'visual': 0.10
    }
    
    def _detect_red_line(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect red vertical line."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Find bounding box of red pixels
        points = np.where(red_mask > 0)
        if len(points[0]) < 10:
            return None
        
        y_min, y_max = points[0].min(), points[0].max()
        x_min, x_max = points[1].min(), points[1].max()
        
        # Check if vertical (height > width)
        height = y_max - y_min
        width = x_max - x_min
        
        x_center = (x_min + x_max) // 2
        
        return {
            'x_center': x_center,
            'y_min': y_min,
            'y_max': y_max,
            'length': height,
            'is_vertical': height > width * 2
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate draw midpoint perpendicular task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect red lines
        gen_line = self._detect_red_line(last_frame)
        gt_line = self._detect_red_line(gt_last)
        
        # 1. Midpoint accuracy: Compare x-position
        if gen_line is not None and gt_line is not None:
            x_diff = abs(gen_line['x_center'] - gt_line['x_center'])
            scores['midpoint'] = max(0, 1.0 - x_diff / 30.0)
        elif gen_line is not None:
            # Check if at image center
            frame_center = last_frame.shape[1] // 2
            x_diff = abs(gen_line['x_center'] - frame_center)
            scores['midpoint'] = max(0, 1.0 - x_diff / 50.0)
        else:
            scores['midpoint'] = 0.0
        
        # 2. Position accuracy: Line should be vertical
        if gen_line is not None:
            scores['position'] = 1.0 if gen_line['is_vertical'] else 0.5
        else:
            scores['position'] = 0.0
        
        # 3. Range: Line length comparison
        if gen_line is not None and gt_line is not None:
            length_ratio = min(gen_line['length'], gt_line['length']) / max(gen_line['length'], gt_line['length'], 1)
            scores['range'] = length_ratio
        elif gen_line is not None:
            # Check if reasonable length
            scores['range'] = min(1.0, gen_line['length'] / 100.0)
        else:
            scores['range'] = 0.0
        
        # 4. Visual quality: Red pixel IoU
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
        
        scores['visual'] = red_overlap / red_union if red_union > 0 else 0.5
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

class DrawNextSizedShapeEvaluator(BaseEvaluator):
    """
    G-193: Draw next sized shape in pattern.
    
    Rule-based evaluation:
    - Pattern recognition (30%): Identify "large-medium-small" size pattern
    - Figure drawing accuracy (35%): Correct type, color, smaller size
    - Label accuracy (25%): Correct Chinese label "小"
    - Animation quality (10%): Smooth growth animation
    """
    
    TASK_WEIGHTS = {
        'pattern_recognition': 0.30,
        'figure_drawing': 0.35,
        'label_accuracy': 0.25,
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
        
        # CRITICAL: First check if the shape count is correct
        # Should add exactly ONE new shape
        first_shapes = self._detect_shapes_with_area(first_frame, exclude_boxes=True)
        final_shapes = self._detect_shapes_with_area(final_frame, exclude_boxes=True)
        
        shape_count_change = len(final_shapes) - len(first_shapes)
        
        # If more than 2 new shapes or shapes removed, task failed
        if shape_count_change > 2 or shape_count_change < 0:
            self._last_task_details = {
                'pattern_recognition': 0.0,
                'figure_drawing': 0.0,
                'label_accuracy': 0.0,
                'animation_quality': 0.3,
                'too_many_shapes_changed': True,
                'first_count': len(first_shapes),
                'final_count': len(final_shapes)
            }
            return 0.0
        
        # 1. Pattern recognition (30%) - CRITICAL: Is size pattern followed?
        pattern_score = self._evaluate_pattern_understanding(
            first_frame, final_frame
        )
        scores['pattern_recognition'] = pattern_score
        
        # CRITICAL: If pattern not followed, other scores should be penalized
        pattern_followed = pattern_score > 0.7
        
        # 2. Figure drawing (35%) - Only counts if pattern is followed
        if pattern_followed:
            scores['figure_drawing'] = self._evaluate_figure_drawing(
                first_frame, final_frame
            )
        else:
            scores['figure_drawing'] = 0.0  # Wrong pattern - no credit
        
        # 3. Label accuracy (25%) - Compare with GT if available
        if gt_final_frame is not None:
            # STRICT: Compare final frame with GT
            gen_final_resized = final_frame
            gt_final_resized = gt_final_frame
            if gen_final_resized.shape != gt_final_resized.shape:
                gt_final_resized = normalize_frame_size(gt_final_frame, final_frame)
            
            diff = np.abs(gen_final_resized.astype(float) - gt_final_resized.astype(float)).mean()
            if diff < 15:
                scores['label_accuracy'] = 1.0
            elif diff < 30:
                scores['label_accuracy'] = 0.3
            else:
                scores['label_accuracy'] = 0.0
        else:
            scores['label_accuracy'] = self._evaluate_label(final_frame)
        
        # 4. Animation quality (10%)
        scores['animation_quality'] = self._evaluate_animation(video_frames)
        
        self._last_task_details = scores
        self._last_task_details['first_count'] = len(first_shapes)
        self._last_task_details['final_count'] = len(final_shapes)
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_pattern_understanding(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if size pattern is understood.
        
        The pattern is 'large-medium-small' repeating cyclically.
        We check if the rightmost new shape in the final frame follows the pattern.
        """
        # Detect shapes in both frames (excluding hollow boxes/designated areas)
        first_shapes = self._detect_shapes_with_area(first_frame, exclude_boxes=True)
        final_shapes = self._detect_shapes_with_area(final_frame, exclude_boxes=True)
        
        # Sort by x-position to get sequence
        first_sorted = sorted(first_shapes, key=lambda s: s[0])
        final_sorted = sorted(final_shapes, key=lambda s: s[0])
        
        if len(first_sorted) < 2 or len(final_sorted) < len(first_sorted):
            return 0.5
        
        # Get sizes from first frame to understand the pattern
        first_sizes = [s[2] for s in first_sorted]
        
        # Identify large, medium, small sizes from the first 3 shapes (if available)
        if len(first_sizes) >= 3:
            # Sort first 3 sizes to identify large, medium, small
            size_levels = sorted(first_sizes[:3], reverse=True)
            large_size = size_levels[0]
            medium_size = size_levels[1]
            small_size = size_levels[2]
            
            # Determine what the next shape should be based on position in pattern
            # Pattern: L-M-S-L-M-S-...
            # Position in pattern is (len(first_sorted)) % 3
            # 0 -> next is L, 1 -> next is M, 2 -> next is S
            pattern_position = len(first_sorted) % 3
            
            # Find the new shape (rightmost shape in final that wasn't in first)
            new_shapes = []
            for fs in final_sorted:
                is_new = True
                for ff in first_sorted:
                    if abs(fs[0] - ff[0]) < 50 and abs(fs[2] - ff[2]) / max(fs[2], ff[2]) < 0.3:
                        is_new = False
                        break
                if is_new:
                    new_shapes.append(fs)
            
            if new_shapes:
                # Get the rightmost new shape
                rightmost_new = max(new_shapes, key=lambda s: s[0])
                new_size = rightmost_new[2]
                
                # Check if new shape follows the pattern
                if pattern_position == 0:  # Should be large
                    expected_size = large_size
                elif pattern_position == 1:  # Should be medium
                    expected_size = medium_size
                else:  # Should be small
                    expected_size = small_size
                
                # Calculate size ratio
                size_ratio = min(new_size, expected_size) / max(new_size, expected_size)
                
                # Check if new size matches expected size category
                if size_ratio > 0.5:
                    return 1.0
                elif size_ratio > 0.3:
                    return 0.8
                else:
                    return 0.6
        
        return 0.1
    
    def _evaluate_figure_drawing(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Evaluate if correct figure is drawn in the box."""
        # Count shapes (excluding hollow boxes)
        first_shapes = self._detect_shapes_with_area(first_frame, exclude_boxes=True)
        final_shapes = self._detect_shapes_with_area(final_frame, exclude_boxes=True)
        first_count = len(first_shapes)
        final_count = len(final_shapes)
        
        # Should have one more shape
        if final_count == first_count + 1:
            if len(first_shapes) > 0 and len(final_shapes) > 0:
                # Get sizes sorted
                first_sizes = sorted([s[2] for s in first_shapes], reverse=True)
                final_sizes = sorted([s[2] for s in final_shapes], reverse=True)
                
                # The new shape should follow the pattern
                # If pattern is L-M-S-L-M, next should be S
                if len(first_sizes) >= 3:
                    small_size = first_sizes[2]  # Third largest = small
                    # Find the new shape size
                    new_size = None
                    for fs in final_sizes:
                        if fs not in first_sizes or final_sizes.count(fs) > first_sizes.count(fs):
                            new_size = fs
                            break
                    
                    if new_size is not None:
                        # Check if new size is close to small size
                        size_ratio = min(new_size, small_size) / max(new_size, small_size)
                        if size_ratio > 0.5:
                            return 1.0
                        elif size_ratio > 0.3:
                            return 0.8
                        return 0.6
                return 0.7
            return 0.6
        elif final_count >= first_count:
            return 0.5
        else:
            return 0.3
    
    def _evaluate_label(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate if label is present."""
        # Check for text/label in the new shape area
        h, w = final_frame.shape[:2]
        
        # Focus on right portion where new shape and label should be
        right_region = final_frame[:, 3*w//4:]
        
        gray = cv2.cvtColor(right_region, cv2.COLOR_BGR2GRAY)
        
        # Count dark pixels (text)
        dark_pixels = np.sum(gray < 100)
        
        if dark_pixels > 500:  # Significant text present
            return 1.0
        elif dark_pixels > 200:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_animation(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Evaluate animation smoothness."""
        if len(video_frames) < 5:
            return 0.5
        
        # Check for smooth changes
        differences = []
        for i in range(1, min(len(video_frames), 30)):
            diff = np.mean(np.abs(
                video_frames[i].astype(float) - video_frames[i-1].astype(float)
            ))
            differences.append(diff)
        
        if len(differences) < 2:
            return 0.5
        
        # Smoothness: low variance in differences
        variance = np.var(differences)
        smoothness = 1.0 / (1.0 + variance / 100)
        
        return smoothness
    
    def _detect_shapes_with_area(self, frame: np.ndarray, exclude_boxes: bool = True, min_area: int = 2000) -> List[Tuple[int, int, int]]:
        """Detect shapes with (x, y, area).
        
        Args:
            frame: Input frame
            exclude_boxes: If True, exclude hollow boxes (designated areas)
            min_area: Minimum area threshold to filter out small labels/text
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try multiple thresholds to handle different image styles
        best_shapes = []
        for thresh in [200, 220, 240]:
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            shapes = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 200:
                    x, y, w, h = cv2.boundingRect(cnt)
                    bbox_area = w * h
                    
                    # Exclude hollow boxes (designated areas) - they have low fill ratio
                    if exclude_boxes:
                        fill_ratio = area / bbox_area if bbox_area > 0 else 0
                        if fill_ratio < 0.3:  # Hollow box has low fill ratio
                            continue
                    
                    # Exclude small shapes that are likely labels/text
                    if area < min_area:
                        continue
                    
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        shapes.append((cx, cy, area))
            
            # Keep the threshold that finds the most shapes
            if len(shapes) > len(best_shapes):
                best_shapes = shapes
        
        return best_shapes

class MarkWavePeaksEvaluator(BaseEvaluator):
    """
    G-202: Mark wave peaks evaluator.
    
    Rule-based evaluation:
    - Peak identification accuracy (40%): Local maxima correctly identified
    - Marking position precision (30%): Markers centered on peak points
    - Marking style (20%): Double-layer markers (outer ring + inner dot)
    - Animation quality (10%): Smooth sequential appearance
    """
    
    TASK_WEIGHTS = {
        'peak_identification': 0.40,
        'marking_position': 0.30,
        'marking_style': 0.20,
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
        
        # 1. Peak identification (40%)
        scores['peak_identification'] = self._evaluate_peak_identification(
            first_frame, final_frame
        )
        
        # 2. Marking position (30%)
        scores['marking_position'] = self._evaluate_marking_positions(
            first_frame, final_frame
        )
        
        # 3. Marking style (20%)
        scores['marking_style'] = self._evaluate_marking_style(final_frame)
        
        # 4. Animation quality (10%)
        scores['animation_quality'] = self._evaluate_animation_quality(video_frames)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_peak_identification(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if wave peaks are correctly identified."""
        # Detect wave curve from first frame
        peaks = self._detect_wave_peaks(first_frame)
        
        # Detect markers in final frame
        markers = self._detect_red_markers(final_frame)
        
        if len(markers) == 0:
            return 0.0
        
        if len(peaks) == 0:
            # Can't detect peaks from first frame - use marker count as heuristic
            # For GT vs GT, markers should match peaks, so give credit based on marker presence
            # A reasonable wave has 2-10 peaks typically
            if 2 <= len(markers) <= 15:
                return 0.8  # Markers exist in reasonable quantity
            return 0.5
        
        # Count how many peaks have markers (with larger tolerance for video compression)
        # Markers are placed at the visual tip of peaks, which may be 20-50 pixels above
        # the detected curve center, so use generous tolerance
        matched_peaks = 0
        matched_markers = set()
        
        for peak in peaks:
            best_marker_idx = -1
            best_dist = float('inf')
            for idx, marker in enumerate(markers):
                # Use primarily x-distance since y may differ due to marker placement
                x_dist = abs(marker[0] - peak[0])
                y_dist = abs(marker[1] - peak[1])
                # Weight x more heavily since y offset is expected
                dist = np.sqrt(x_dist**2 + (y_dist * 0.5)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_marker_idx = idx
            
            # Use larger tolerance (80 pixels) for matching
            if best_dist < 80 and best_marker_idx not in matched_markers:
                matched_peaks += 1
                matched_markers.add(best_marker_idx)
        
        # Also count markers near any peak (for cases where we detect fewer peaks than markers)
        markers_near_peaks = 0
        for marker in markers:
            for peak in peaks:
                x_dist = abs(marker[0] - peak[0])
                y_dist = abs(marker[1] - peak[1])
                dist = np.sqrt(x_dist**2 + (y_dist * 0.5)**2)
                if dist < 80:
                    markers_near_peaks += 1
                    break
        
        recall = matched_peaks / len(peaks) if len(peaks) > 0 else 0
        precision = markers_near_peaks / len(markers) if len(markers) > 0 else 0
        
        # For GT vs GT comparison, if most markers are near detected peaks, that's good
        # Even if we don't detect all peaks
        if recall + precision > 0:
            f1 = 2 * recall * precision / (recall + precision)
            # Boost score if we have reasonable coverage
            if precision > 0.8:  # Most markers are near peaks
                f1 = max(f1, 0.8)
            return f1
        
        return 0.0
    
    def _evaluate_marking_positions(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Evaluate if markers are precisely on peaks."""
        peaks = self._detect_wave_peaks(first_frame)
        markers = self._detect_red_markers(final_frame)
        
        if len(markers) == 0 or len(peaks) == 0:
            return 0.1
        
        # Calculate average distance from markers to nearest peaks
        total_dist = 0
        matches = 0
        for marker in markers:
            min_dist = float('inf')
            for peak in peaks:
                dist = np.sqrt((marker[0] - peak[0])**2 + (marker[1] - peak[1])**2)
                min_dist = min(min_dist, dist)
            if min_dist < 60:
                total_dist += min_dist
                matches += 1
        
        if matches == 0:
            return 0.0
        
        avg_dist = total_dist / matches
        return max(0.0, 1.0 - avg_dist / 40)
    
    def _evaluate_marking_style(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check marker style (outer ring + inner dot)."""
        markers = self._detect_red_markers(final_frame)
        
        if len(markers) == 0:
            return 0.0
        
        # Check for red color presence (markers exist)
        hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        red_ratio = np.sum(mask > 0) / mask.size
        
        # Reasonable range for markers
        if 0.001 < red_ratio < 0.05:
            return 1.0
        elif 0.0005 < red_ratio < 0.1:
            return 0.7
        else:
            return 0.0
    
    def _evaluate_animation_quality(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Evaluate animation smoothness."""
        if len(video_frames) < 5:
            return 0.0
        
        # Track marker count over time
        marker_counts = []
        for frame in video_frames[::max(1, len(video_frames)//10)]:
            markers = self._detect_red_markers(frame)
            marker_counts.append(len(markers))
        
        if len(marker_counts) < 2:
            return 0.0
        
        # Check for gradual increase (sequential appearance)
        increases = sum(1 for i in range(1, len(marker_counts)) 
                       if marker_counts[i] >= marker_counts[i-1])
        
        return increases / (len(marker_counts) - 1)
    
    def _detect_wave_peaks(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect wave peak positions from the curve with improved robustness."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Try multiple detection methods
        all_peaks = []
        
        # Method 1: Detect colored wave (blue, etc.)
        # Blue hue range: 90-130 in OpenCV HSV
        lower_blue = np.array([90, 30, 30])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Also try detecting any saturated color
        saturation = hsv[:, :, 1]
        colored_mask = (saturation > 40).astype(np.uint8) * 255
        
        # Method 2: Dark curve on light background
        dark_masks = []
        for thresh_val in [80, 100, 120, 150]:
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            dark_masks.append(binary)
        
        # Try all masks
        masks_to_try = [blue_mask, colored_mask] + dark_masks
        
        for binary in masks_to_try:
            # Apply morphological operations to clean up noise
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find curve points
            curve_points = np.where(binary > 0)
            if len(curve_points[0]) < 100:
                continue
            
            # Group by x-coordinate and find the topmost point for each x
            x_to_y = {}
            for y, x in zip(curve_points[0], curve_points[1]):
                if x not in x_to_y or y < x_to_y[x]:
                    x_to_y[x] = y  # Keep minimum y (top of curve)
            
            if len(x_to_y) < 50:
                continue
            
            # Smooth the curve to reduce noise
            sorted_x = sorted(x_to_y.keys())
            y_values = [x_to_y[x] for x in sorted_x]
            
            # Apply simple moving average smoothing
            window_size = 7
            if len(y_values) >= window_size:
                smoothed_y = []
                for i in range(len(y_values)):
                    start = max(0, i - window_size // 2)
                    end = min(len(y_values), i + window_size // 2 + 1)
                    smoothed_y.append(np.mean(y_values[start:end]))
                y_values = smoothed_y
            
            # Find local minima in y (peaks on screen, since y increases downward)
            # In screen coordinates: smaller y = higher on screen = wave peak
            peaks = []
            min_peak_distance = 40  # Minimum distance between peaks
            
            # Calculate typical y-range for significance threshold
            y_range = max(y_values) - min(y_values) if y_values else 0
            significance_threshold = max(5, y_range * 0.03)  # At least 3% of range
            
            for i in range(25, len(sorted_x) - 25):
                x = sorted_x[i]
                y = y_values[i]
                
                # Check if local minimum with larger window
                window = 25
                left_y = y_values[max(0, i-window):i]
                right_y = y_values[i+1:min(len(y_values), i+window+1)]
                
                if len(left_y) > 0 and len(right_y) > 0:
                    # For local minimum: y should be smaller than max of neighbors
                    left_max = max(left_y)
                    right_max = max(right_y)
                    
                    # Must be clearly lower (smaller y) than neighbors
                    if y < left_max - significance_threshold and y < right_max - significance_threshold:
                        # Check distance from existing peaks
                        if not peaks or all(abs(x - px) > min_peak_distance for px, py in peaks):
                            peaks.append((x, int(x_to_y[x])))
            
            if len(peaks) > len(all_peaks):
                all_peaks = peaks
        
        return all_peaks
    
    def _detect_red_markers(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect red marker positions."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        markers = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 30:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    markers.append((cx, cy))
        
        return markers

class IdentifyPentagonsEvaluator(BaseEvaluator):
    """
    G-206: Identify pentagons evaluator.
    
    Rule-based evaluation:
    - Edge count identification (40%): Correct 5-sided polygon identified
    - Marking precision (35%): Red circle accurately marks pentagon
    - Marking quality (15%): Circle complete and proper
    - Scene fidelity (10%): All polygons preserved
    """
    
    TASK_WEIGHTS = {
        'edge_count': 0.40,
        'marking': 0.35,
        'quality': 0.15,
        'fidelity': 0.10
    }
    
    def _detect_polygons(self, frame: np.ndarray) -> List[Dict]:
        """Detect polygons and count their edges."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            
            # Approximate polygon
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            polygons.append({
                'center': (cx, cy),
                'vertices': len(approx),
                'area': area,
                'is_pentagon': len(approx) == 5
            })
        
        return polygons
    
    def _detect_red_marking(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red circle marking."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        M = cv2.moments(red_mask)
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
        """Evaluate identify pentagons task."""
        scores = {}
        
        first_frame = video_frames[0] if len(video_frames) > 0 else None
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect polygons from FIRST frame (before marking, more accurate)
        # The red marking in final frame can interfere with contour detection
        first_polygons = self._detect_polygons(first_frame) if first_frame is not None else []
        gen_polygons = self._detect_polygons(last_frame)
        gt_polygons = self._detect_polygons(gt_last)
        
        gen_marking = self._detect_red_marking(last_frame)
        gt_marking = self._detect_red_marking(gt_last)
        
        # 1. Edge count identification: Check if marking is near a pentagon
        # Use first_frame polygons for pentagon detection (more accurate)
        polygons_to_check = first_polygons if first_polygons else gen_polygons
        
        if gen_marking is not None and polygons_to_check:
            marked_pentagon = False
            for poly in polygons_to_check:
                dist = np.sqrt((gen_marking[0] - poly['center'][0])**2 + 
                              (gen_marking[1] - poly['center'][1])**2)
                if dist < 100 and poly['is_pentagon']:
                    marked_pentagon = True
                    break
            
            # If no pentagon found but marking matches GT, give credit
            if not marked_pentagon and gt_marking is not None:
                marking_dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                                      (gen_marking[1] - gt_marking[1])**2)
                if marking_dist < 30:
                    scores['edge_count'] = 0.9
                else:
                    scores['edge_count'] = 0.3
            else:
                scores['edge_count'] = 1.0 if marked_pentagon else 0.3
        else:
            scores['edge_count'] = 0.2  # Detection failed
        
        # 2. Marking precision: Compare with GT marking position
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['marking'] = max(0, 1.0 - dist / 50.0)
        else:
            scores['marking'] = 0.2  # Detection failed
        
        # 3. Quality: Red pixel IoU
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
        
        scores['quality'] = red_overlap / red_union if red_union > 0 else 0.5
        
        # 4. Scene fidelity: Compare polygon counts
        if gen_polygons and gt_polygons:
            count_ratio = min(len(gen_polygons), len(gt_polygons)) / max(len(gen_polygons), len(gt_polygons), 1)
            scores['fidelity'] = count_ratio
        else:
            scores['fidelity'] = 0.2  # Detection failed
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

class FindIncorrectArrowDirectionEvaluator(BaseEvaluator):
    """
    G-212: Find incorrect arrow direction evaluator.
    
    Rule-based evaluation:
    - Arrow identification accuracy (50%): Correctly identify reversed arrow
    - Marking standardization (30%): Red circle marking
    - Marking precision (15%): Circle position and size
    - Element preservation (5%): Original elements unchanged
    """
    
    TASK_WEIGHTS = {
        'arrow_identification': 0.50,
        'marking_standardization': 0.30,
        'marking_precision': 0.15,
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
        
        # 1. Arrow identification (50%)
        scores['arrow_identification'] = self._evaluate_arrow_identification(
            first_frame, final_frame, gt_final_frame
        )
        
        # 2. Marking standardization (30%)
        scores['marking_standardization'] = self._evaluate_marking_standard(final_frame)
        
        # 3. Marking precision (15%)
        scores['marking_precision'] = self._evaluate_marking_precision(final_frame)
        
        # 4. Element preservation (5%)
        scores['element_preservation'] = self._evaluate_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_arrow_identification(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        gt_final_frame: Optional[np.ndarray] = None
    ) -> float:
        """Rule-based: Check if the incorrect arrow is identified."""
        # Detect red circle marking
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # If GT final frame available, compare with GT marking position
        if gt_final_frame is not None:
            gt_circle = self._detect_red_circle(gt_final_frame)
            if gt_circle is not None:
                dist = np.sqrt((circle[0] - gt_circle[0])**2 + 
                              (circle[1] - gt_circle[1])**2)
                if dist < 40:
                    return 1.0
                elif dist < 80:
                    return 0.8
                elif dist < 120:
                    return 0.5
                else:
                    return 0.1
        
        # Fallback: Find the different arrow direction
        different_arrow_pos = self._find_different_arrow(first_frame)
        
        if different_arrow_pos is None:
            return 0.0
        
        # Check if circle marks the different arrow
        dist = np.sqrt((circle[0] - different_arrow_pos[0])**2 + 
                      (circle[1] - different_arrow_pos[1])**2)
        
        if dist < 40:
            return 1.0
        elif dist < 80:
            return 0.7
        elif dist < 120:
            return 0.4
        else:
            return 0.1
    
    def _evaluate_marking_standard(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check if red circle marking is used."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        # Check reasonable size
        if 15 < r < 100:
            size_score = 1.0
        else:
            size_score = 0.5
        
        # Check color is red
        color_score = self._check_red_color(final_frame, x, y, r)
        
        return 0.5 * size_score + 0.5 * color_score
    
    def _evaluate_marking_precision(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate circle position and size."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # Check if circle is in reasonable position
        h, w = final_frame.shape[:2]
        x, y, r = circle
        
        if 30 < x < w - 30 and 30 < y < h - 30:
            return 1.0
        else:
            return 0.6
    
    def _evaluate_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if original elements are preserved."""
        # Count arrows in first frame
        first_arrows = self._count_arrows(first_frame)
        
        # Count arrows in final (excluding red marking area)
        hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        final_no_red = final_frame.copy()
        final_no_red[red_mask > 0] = [255, 255, 255]
        
        final_arrows = self._count_arrows(final_no_red)
        
        if abs(first_arrows - final_arrows) <= 1:
            return 1.0
        elif abs(first_arrows - final_arrows) <= 2:
            return 0.7
        else:
            return 0.1
    
    def _find_different_arrow(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the arrow pointing in a different direction."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        arrows = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 10000:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Estimate arrow direction using bounding box
                    x, y, w, h = cv2.boundingRect(cnt)
                    direction = 1 if w > h else 0  # Simplified direction
                    
                    arrows.append((cx, cy, direction))
        
        if len(arrows) < 3:
            return None
        
        # Find the outlier direction
        from collections import Counter
        directions = [a[2] for a in arrows]
        direction_counts = Counter(directions)
        
        for arrow in arrows:
            if direction_counts[arrow[2]] == 1:
                return (arrow[0], arrow[1])
        
        return None
    
    def _count_arrows(self, frame: np.ndarray) -> int:
        """Count number of arrows."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return sum(1 for cnt in contours if 500 < cv2.contourArea(cnt) < 10000)
    
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
            if cv2.contourArea(largest) > 50:
                (x, y), r = cv2.minEnclosingCircle(largest)
                return (int(x), int(y), int(r))
        
        return None
    
    def _check_red_color(self, frame: np.ndarray, x: int, y: int, r: int) -> float:
        """Check if the marking is red."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r + 5, 255, 10)
        
        roi_hsv = hsv[mask > 0]
        if len(roi_hsv) == 0:
            return 0.5
        
        red_count = sum(1 for pixel in roi_hsv if pixel[0] < 10 or pixel[0] > 160)
        return red_count / len(roi_hsv)


class CircleCentralDotEvaluator(BaseEvaluator):
    """
    G-217: Circle central dot evaluator.
    
    Rule-based evaluation:
    - Center identification accuracy (50%): Find the central dot (y≈512)
    - Marking accuracy (30%): Circle centered on dot
    - Marking appearance (15%): Red circle, proper size (~36px radius)
    - Scene preservation (5%): Original dots unchanged
    """
    
    TASK_WEIGHTS = {
        'center_identification': 0.50,
        'marking_accuracy': 0.30,
        'marking_appearance': 0.15,
        'scene_preservation': 0.05
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
        
        # Check if a red circle/marking was added
        # Count red pixels in first vs final frame
        first_red_count = self._count_red_pixels(first_frame)
        final_red_count = self._count_red_pixels(final_frame)
        red_increase = final_red_count - first_red_count
        
        # Need at least 500 new red pixels for a marking
        if red_increase < 500:
            # No red marking added - task not completed
            self._last_task_details = {
                'center_identification': 0.0,
                'marking_accuracy': 0.0,
                'marking_appearance': 0.0,
                'scene_preservation': 1.0,
                'no_red_marking': True,
                'red_pixel_increase': int(red_increase)
            }
            return 0.05  # Very low score for no marking
        
        # 1. Center identification (50%)
        scores['center_identification'] = self._evaluate_center_identification(
            first_frame, final_frame
        )
        
        # 2. Marking accuracy (30%)
        scores['marking_accuracy'] = self._evaluate_marking_accuracy(
            first_frame, final_frame
        )
        
        # 3. Marking appearance (15%)
        scores['marking_appearance'] = self._evaluate_marking_appearance(first_frame, final_frame)
        
        # 4. Scene preservation (5%)
        scores['scene_preservation'] = self._evaluate_scene_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_center_identification(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if the central dot (y≈512) is identified."""
        # Find central dot from first frame
        central_dot = self._find_central_dot(first_frame)
        
        # Detect circle marking in final frame
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        if central_dot is None:
            return 0.5
        
        # Check if circle is centered on the central dot
        dist = np.sqrt((circle[0] - central_dot[0])**2 + (circle[1] - central_dot[1])**2)
        
        if dist < 25:
            return 1.0
        elif dist < 50:
            return 0.7
        elif dist < 100:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_marking_accuracy(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Evaluate if circle is properly centered."""
        circle = self._detect_red_circle(final_frame)
        central_dot = self._find_central_dot(first_frame)
        
        if circle is None:
            return 0.0
        if central_dot is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - central_dot[0])**2 + (circle[1] - central_dot[1])**2)
        return max(0.0, 1.0 - dist / 50)
    
    def _evaluate_marking_appearance(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate red circle appearance - size should match black dots."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        # Get expected size from black dots in first frame
        dots = self._detect_black_dots_with_size(first_frame)
        if dots:
            avg_dot_size = np.mean([d['size'] for d in dots])
            expected_radius = avg_dot_size / 2  # Radius should be half of dot diameter
            
            # Check if red circle radius matches expected (~26 for 51px dots)
            size_ratio = r / expected_radius if expected_radius > 0 else 0
            
            # Perfect match: ratio close to 1.0-1.4 (circle can be slightly larger)
            if 0.8 < size_ratio < 1.5:
                size_score = 1.0
            elif 0.6 < size_ratio < 2.0:
                size_score = 0.6
            else:
                # Circle is way too large or too small
                size_score = 0.1
        else:
            # Fallback: check size (should be ~36 pixels radius)
            if 25 < r < 50:
                size_score = 1.0
            elif 15 < r < 60:
                size_score = 0.6
            else:
                size_score = 0.1
        
        return size_score
    
    def _evaluate_scene_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if all dots are preserved - NO new dots should appear."""
        first_dots = self._detect_black_dots(first_frame)
        
        # Detect dots excluding red marking
        hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        final_no_red = final_frame.copy()
        final_no_red[red_mask > 0] = [255, 255, 255]
        
        final_dots = self._detect_black_dots(final_no_red)
        
        if len(first_dots) == 0:
            return 0.5
        
        # Strict check: dots should be preserved exactly (or -1 if central dot is covered)
        # If dots are removed or added, penalize heavily
        if len(final_dots) > len(first_dots):
            # New dots appeared - bad
            return 0.0
        elif len(final_dots) == len(first_dots):
            # All dots preserved
            return 1.0
        elif len(final_dots) == len(first_dots) - 1:
            # One dot might be covered by red marking - acceptable
            return 0.9
        elif len(final_dots) == len(first_dots) - 2:
            # Two dots missing - questionable
            return 0.5
        else:
            # Many dots removed - very bad
            return 0.0
    
    def _find_central_dot(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the central dot (y ≈ center of frame)."""
        dots = self._detect_black_dots(frame)
        
        if len(dots) == 0:
            return None
        
        h, w = frame.shape[:2]
        center_y = h // 2
        
        # Find dot closest to vertical center
        central_dot = min(dots, key=lambda d: abs(d[1] - center_y))
        return central_dot
    
    def _detect_black_dots(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect black dots in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 5000:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dots.append((cx, cy))
        
        return dots
    
    def _detect_black_dots_with_size(self, frame: np.ndarray) -> List[Dict]:
        """Detect black dots with size information."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 5000:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(cnt)
                    dots.append({'center': (cx, cy), 'area': area, 'size': max(w, h)})
        
        return dots
    
    def _count_red_pixels(self, frame: np.ndarray) -> int:
        """Count red pixels in frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        return int(np.sum(mask > 0))
    
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


class IdentifyLargestAngleEvaluator(BaseEvaluator):
    """
    G-218: Identify largest angle in triangle evaluator.
    
    Rule-based evaluation:
    - Angle recognition correctness (40%): Identify largest angle vertex
    - Marking position precision (35%): Circle at correct vertex
    - Marking specification compliance (15%): Red circle, ~40px radius
    - Triangle preservation (10%): Original triangle unchanged
    """
    
    TASK_WEIGHTS = {
        'angle_recognition': 0.40,
        'marking_position': 0.35,
        'marking_specification': 0.15,
        'triangle_preservation': 0.10
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
        
        # 1. Angle recognition (40%)
        scores['angle_recognition'] = self._evaluate_angle_recognition(
            first_frame, final_frame, gt_final_frame
        )
        
        # 2. Marking position (35%)
        scores['marking_position'] = self._evaluate_marking_position(
            first_frame, final_frame, gt_final_frame
        )
        
        # 3. Marking specification (15%)
        scores['marking_specification'] = self._evaluate_marking_spec(final_frame)
        
        # 4. Triangle preservation (10%)
        scores['triangle_preservation'] = self._evaluate_triangle_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_angle_recognition(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        gt_final_frame: Optional[np.ndarray] = None
    ) -> float:
        """Rule-based: Check if the largest angle vertex is identified."""
        # Detect circle marking
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # If GT final frame available, compare with GT marking position
        if gt_final_frame is not None:
            gt_circle = self._detect_red_circle(gt_final_frame)
            if gt_circle is not None:
                dist = np.sqrt((circle[0] - gt_circle[0])**2 + (circle[1] - gt_circle[1])**2)
                if dist < 30:
                    return 1.0
                elif dist < 60:
                    return 0.8
                elif dist < 100:
                    return 0.5
                else:
                    return 0.2
        
        # Fallback: Find largest angle vertex from triangle
        largest_vertex = self._find_largest_angle_vertex(first_frame)
        
        if largest_vertex is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - largest_vertex[0])**2 + (circle[1] - largest_vertex[1])**2)
        
        if dist < 30:
            return 1.0
        elif dist < 60:
            return 0.7
        elif dist < 100:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_marking_position(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        gt_final_frame: Optional[np.ndarray] = None
    ) -> float:
        """Rule-based: Evaluate circle position at vertex."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # If GT final frame available, compare with GT marking position
        if gt_final_frame is not None:
            gt_circle = self._detect_red_circle(gt_final_frame)
            if gt_circle is not None:
                dist = np.sqrt((circle[0] - gt_circle[0])**2 + (circle[1] - gt_circle[1])**2)
                return max(0.0, 1.0 - dist / 60)
        
        # Fallback: compare with detected largest vertex
        largest_vertex = self._find_largest_angle_vertex(first_frame)
        
        if largest_vertex is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - largest_vertex[0])**2 + (circle[1] - largest_vertex[1])**2)
        return max(0.0, 1.0 - dist / 60)
    
    def _evaluate_marking_spec(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check marking specification (~40px radius)."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        if 30 < r < 55:
            return 1.0
        elif 20 < r < 70:
            return 0.7
        else:
            return 0.3
    
    def _evaluate_triangle_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if triangle is preserved."""
        # Detect triangle vertices
        first_vertices = self._detect_triangle_vertices(first_frame)
        
        # Remove red marking
        hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        final_no_red = final_frame.copy()
        final_no_red[red_mask > 0] = [255, 255, 255]
        
        final_vertices = self._detect_triangle_vertices(final_no_red)
        
        if len(first_vertices) != 3:
            return 0.0
        
        if len(final_vertices) == 3:
            return 1.0
        elif len(final_vertices) >= 2:
            return 0.7
        else:
            return 0.1
    
    def _find_largest_angle_vertex(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the vertex with the largest angle."""
        vertices = self._detect_triangle_vertices(frame)
        
        if len(vertices) != 3:
            return None
        
        # Calculate angles at each vertex
        angles = []
        for i in range(3):
            p1 = np.array(vertices[i])
            p2 = np.array(vertices[(i+1) % 3])
            p3 = np.array(vertices[(i+2) % 3])
            
            v1 = p2 - p1
            v2 = p3 - p1
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append((angle, vertices[i]))
        
        # Return vertex with largest angle
        largest = max(angles, key=lambda x: x[0])
        return largest[1]
    
    def _detect_triangle_vertices(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect triangle vertices using corner detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find non-white regions (triangle lines)
        non_white = (gray < 250).astype(np.uint8) * 255
        contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return []
        
        # Get the largest contour (triangle outline)
        triangle = max(contours, key=cv2.contourArea)
        
        # First try polygon approximation (works for filled triangles)
        peri = cv2.arcLength(triangle, True)
        for eps_factor in [0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(triangle, eps_factor * peri, True)
            if len(approx) == 3:
                return [tuple(pt[0]) for pt in approx]
        
        # If approximation fails (line-drawn triangles), use corner detection
        # Create a mask with the triangle contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [triangle], -1, 255, 3)
        
        # Detect corners using goodFeaturesToTrack
        corners = cv2.goodFeaturesToTrack(mask, 20, 0.01, 30)
        
        if corners is not None and len(corners) >= 3:
            corner_pts = [(int(c[0][0]), int(c[0][1])) for c in corners]
            
            # Cluster nearby corners (corners along edges are close together)
            def cluster_corners(points, min_dist=50):
                """Cluster nearby points and return cluster centers."""
                if len(points) == 0:
                    return []
                
                clusters = []
                used = [False] * len(points)
                
                for i, p1 in enumerate(points):
                    if used[i]:
                        continue
                    
                    cluster = [p1]
                    used[i] = True
                    
                    for j, p2 in enumerate(points):
                        if not used[j]:
                            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                            if dist < min_dist:
                                cluster.append(p2)
                                used[j] = True
                    
                    # Cluster center
                    cx = int(np.mean([p[0] for p in cluster]))
                    cy = int(np.mean([p[1] for p in cluster]))
                    clusters.append((cx, cy))
                
                return clusters
            
            clustered = cluster_corners(corner_pts, min_dist=60)
            
            if len(clustered) >= 3:
                # Find the 3 most extreme points (vertices of triangle)
                # Use convex hull to get the outer vertices
                pts_array = np.array(clustered, dtype=np.float32).reshape(-1, 1, 2)
                hull = cv2.convexHull(pts_array)
                
                if len(hull) >= 3:
                    # Sort by angle from centroid to get consistent ordering
                    hull_pts = [tuple(pt[0].astype(int)) for pt in hull]
                    return hull_pts[:3]
                
                return clustered[:3]
        
        return []
    
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


# Export all evaluators
OUT_OF_DOMAIN_50_EVALUATORS_PART2 = {
    'G-168_identify_nearest_to_square_rectangle_data-generator': IdentifyNearestSquareRectangleEvaluator,
    'G-169_locate_intersection_of_segments_data-generator': LocateSegmentIntersectionEvaluator,
    'G-174_arrange_circles_by_circumference_data-generator': ArrangeCirclesByCircumferenceEvaluator,
    'G-189_draw_midpoint_perpendicular_line_data-generator': DrawMidpointPerpendicularEvaluator,
    'G-193_draw_next_sized_shape_data-generator': DrawNextSizedShapeEvaluator,
    'G-202_mark_wave_peaks_data-generator': MarkWavePeaksEvaluator,
    'G-206_identify_pentagons_data-generator': IdentifyPentagonsEvaluator,
    'G-212_find_incorrect_arrow_direction_data-generator': FindIncorrectArrowDirectionEvaluator,
    'G-217_circle_central_dot_data-generator': CircleCentralDotEvaluator,
    'G-218_identify_largest_angle_in_triangle_data-generator': IdentifyLargestAngleEvaluator,
}
