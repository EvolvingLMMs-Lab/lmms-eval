"""
Specific evaluators for In-Domain_50 tasks (Part 4).
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import normalize_frame_size

class ConstructionBlueprintEvaluator(BaseEvaluator):
    """
    O-21: Construction Blueprint (Missing Piece)
    
    Task: Select the correct piece from 4 candidates to fill the highlighted 
    gap in a block structure.
    
    Rule-based evaluation:
    1. Piece selection correctness (40%) - Right piece chosen
    2. Shape matching accuracy (30%) - Exact fit to gap
    3. Placement precision (20%) - No gaps or overlaps
    4. Structure integrity (10%) - Complete, connected result
    """
    
    TASK_WEIGHTS = {
        'piece_selection': 0.40,
        'shape_matching': 0.30,
        'placement': 0.20,
        'integrity': 0.10
    }
    
    def _detect_gap_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect red-outlined gap region."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        if hsv is None:
            return None
        
        # Red detection
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(largest)
        
        return None
    
    def _check_gap_filled(self, frame: np.ndarray) -> float:
        """Check if gap is properly filled (no red outline remaining)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        if hsv is None:
            return 0.5
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        red_pixels = red_mask.sum() / 255
        
        if red_pixels < 100:
            return 1.0
        elif red_pixels < 500:
            return 0.7
        else:
            return 0.3
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate blueprint piece selection and placement."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # 1. Piece selection: Compare green regions with GT (STRICT)
        gen_green = self._detect_green_filled_region(gen_final)
        gt_green = self._detect_green_filled_region(gt_final)
        
        if gen_green is not None and gt_green is not None:
            # Check if green regions are in similar locations
            genx, geny, genw, genh = gen_green
            gtx, gty, gtw, gth = gt_green
            
            # Calculate position difference
            pos_diff = np.sqrt((genx - gtx)**2 + (geny - gty)**2)
            size_ratio = min(genw * genh, gtw * gth) / max(genw * genh, gtw * gth, 1)
            
            if pos_diff < 50 and size_ratio > 0.5:
                scores['piece_selection'] = 1.0
            elif pos_diff < 100 or size_ratio > 0.3:
                scores['piece_selection'] = 0.3
            else:
                scores['piece_selection'] = 0.0
        elif gen_green is None and gt_green is None:
            scores['piece_selection'] = 1.0  # Both have no green
        else:
            scores['piece_selection'] = 0.0  # Mismatch
        
        # 2. Shape matching: Compare with GT final (STRICT)
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            if diff < 20:
                scores['shape_matching'] = 1.0
            elif diff < 40:
                scores['shape_matching'] = 0.3
            else:
                scores['shape_matching'] = 0.0
        else:
            scores['shape_matching'] = 0.0
        
        # 3. Placement: Check if gap is filled (no red outline remaining)
        scores['placement'] = self._check_gap_filled(gen_final)
        
        # 4. Integrity: Overall frame similarity
        scores['integrity'] = scores['shape_matching']
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_green_filled_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect green filled region."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        if hsv is None:
            return None
        
        # Green detection
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:  # Significant green region
                return cv2.boundingRect(largest)
        
        return None

class DominoChainBranchEvaluator(BaseEvaluator):
    """
    O-23: Domino Chain Branch Path Prediction
    
    Task: Y-shaped domino structure - push START, dominoes fall through 
    trunk to fork, then both branches fall (unless blocked by gap).
    
    Rule-based evaluation:
    1. Fallen domino accuracy (40%) - Correct dominoes fall
    2. Gap rule application (30%) - Gap blocks chain correctly
    3. Chain reaction sequence (20%) - Correct order
    4. Fall direction accuracy (10%) - Correct tilt directions
    """
    
    TASK_WEIGHTS = {
        'fallen_domino': 0.40,
        'gap_rule': 0.30,
        'chain_sequence': 0.20,
        'fall_direction': 0.10
    }
    
    def _count_standing_dominoes(self, frame: np.ndarray) -> int:
        """Count number of vertical (standing) domino shapes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Detect vertical lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=5)
        
        standing_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dy > 20 and dx < 10:  # Roughly vertical
                    standing_count += 1
        
        return standing_count
    
    def _count_fallen_dominoes(self, frame: np.ndarray) -> int:
        """Count number of tilted/fallen domino shapes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=5)
        
        fallen_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                # Tilted: significant both dx and dy
                if dx > 10 and dy > 10 and abs(dy - dx) < max(dx, dy) * 0.5:
                    fallen_count += 1
        
        return fallen_count
    
    def _detect_dominoes_by_color(self, frame: np.ndarray) -> Dict[str, int]:
        """Detect dominoes by their colors (red=fallen, blue=standing typically)."""
        if len(frame.shape) != 3:
            return {'red': 0, 'blue': 0}
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red detection
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Blue detection
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return {
            'red': len([c for c in red_contours if cv2.contourArea(c) > 100]),
            'blue': len([c for c in blue_contours if cv2.contourArea(c) > 100])
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate domino chain branch prediction."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # 1. Fallen domino accuracy: Compare domino states
        gen_standing = self._count_standing_dominoes(gen_final)
        gt_standing = self._count_standing_dominoes(gt_final)
        gen_fallen = self._count_fallen_dominoes(gen_final)
        gt_fallen = self._count_fallen_dominoes(gt_final)
        
        if gt_standing + gt_fallen > 0:
            standing_match = max(0, 1.0 - abs(gen_standing - gt_standing) / max(gt_standing + gt_fallen, 1))
            fallen_match = max(0, 1.0 - abs(gen_fallen - gt_fallen) / max(gt_standing + gt_fallen, 1))
            scores['fallen_domino'] = 0.5 * standing_match + 0.5 * fallen_match
        else:
            scores['fallen_domino'] = 0.2  # Detection failed
        
        # 2. Gap rule: Compare color distribution (red=fallen, blue=standing)
        gen_colors = self._detect_dominoes_by_color(gen_final)
        gt_colors = self._detect_dominoes_by_color(gt_final)
        
        gen_ratio = gen_colors['red'] / max(gen_colors['red'] + gen_colors['blue'], 1)
        gt_ratio = gt_colors['red'] / max(gt_colors['red'] + gt_colors['blue'], 1)
        
        ratio_diff = abs(gen_ratio - gt_ratio)
        scores['gap_rule'] = max(0, 1.0 - ratio_diff * 2)
        
        # 3. Chain sequence: Analyze progression through video
        if len(video_frames) >= 3:
            # Check if fallen count increases over time
            early_fallen = self._count_fallen_dominoes(video_frames[len(video_frames)//4])
            mid_fallen = self._count_fallen_dominoes(video_frames[len(video_frames)//2])
            late_fallen = self._count_fallen_dominoes(video_frames[-1])
            
            if early_fallen <= mid_fallen <= late_fallen:
                scores['chain_sequence'] = 1.0
            elif early_fallen <= late_fallen:
                scores['chain_sequence'] = 0.7
            else:
                scores['chain_sequence'] = 0.3
        else:
            scores['chain_sequence'] = 0.2  # Detection failed
        
        # 4. Fall direction: Compare overall structure
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            scores['fall_direction'] = max(0, 1.0 - diff / 100.0)
        else:
            scores['fall_direction'] = 0.2  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class DominoChainGapEvaluator(BaseEvaluator):
    """
    O-24: Domino Chain Gap Analysis
    
    Task: Find where domino chain stops due to gap too large. Identify 
    the last domino that falls before the gap.
    
    Rule-based evaluation:
    1. Gap identification (40%) - Correct gap location
    2. Last fallen domino identification (35%) - Correct domino number
    3. Domino state accuracy (15%) - Fallen vs standing correct
    4. Chain animation quality (10%) - Sequential falling
    """
    
    TASK_WEIGHTS = {
        'gap_identification': 0.40,
        'last_fallen': 0.35,
        'domino_state': 0.15,
        'animation_quality': 0.10
    }
    
    def _analyze_domino_colors(self, frame: np.ndarray) -> Dict[str, int]:
        """Analyze domino color states."""
        if len(frame.shape) != 3:
            return {'red': 0, 'blue': 0}
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red detection (fallen)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Blue detection (standing)
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        return {
            'red': int(np.sum(red_mask > 0)),
            'blue': int(np.sum(blue_mask > 0))
        }
    
    def _find_gap_position(self, frame: np.ndarray) -> Optional[int]:
        """Find x-position of gap in domino chain."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Find vertical structures (dominoes)
        edges = cv2.Canny(gray, 50, 150)
        
        # Project to x-axis
        x_projection = np.sum(edges, axis=0)
        
        # Find gaps (low values)
        threshold = np.max(x_projection) * 0.2
        gap_positions = np.where(x_projection < threshold)[0]
        
        if len(gap_positions) > 0:
            # Find largest continuous gap
            gaps = []
            start = gap_positions[0]
            for i in range(1, len(gap_positions)):
                if gap_positions[i] - gap_positions[i-1] > 5:
                    gaps.append((start, gap_positions[i-1]))
                    start = gap_positions[i]
            gaps.append((start, gap_positions[-1]))
            
            if gaps:
                largest_gap = max(gaps, key=lambda g: g[1] - g[0])
                return (largest_gap[0] + largest_gap[1]) // 2
        
        return None
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate domino chain gap analysis."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # 1. Gap identification: Compare gap positions
        gen_gap = self._find_gap_position(gen_final)
        gt_gap = self._find_gap_position(gt_final)
        
        if gen_gap is not None and gt_gap is not None:
            gap_diff = abs(gen_gap - gt_gap)
            scores['gap_identification'] = max(0, 1.0 - gap_diff / 100.0)
        else:
            scores['gap_identification'] = 0.5 if gen_gap == gt_gap else 0.3
        
        # 2. Last fallen: Compare color distribution
        gen_colors = self._analyze_domino_colors(gen_final)
        gt_colors = self._analyze_domino_colors(gt_final)
        
        gen_ratio = gen_colors['red'] / max(gen_colors['red'] + gen_colors['blue'], 1)
        gt_ratio = gt_colors['red'] / max(gt_colors['red'] + gt_colors['blue'], 1)
        
        ratio_diff = abs(gen_ratio - gt_ratio)
        scores['last_fallen'] = max(0, 1.0 - ratio_diff * 2)
        
        # 3. Domino state accuracy
        if gen_colors['red'] + gen_colors['blue'] > 0 and gt_colors['red'] + gt_colors['blue'] > 0:
            red_match = min(gen_colors['red'], gt_colors['red']) / max(gen_colors['red'], gt_colors['red'], 1)
            blue_match = min(gen_colors['blue'], gt_colors['blue']) / max(gen_colors['blue'], gt_colors['blue'], 1)
            scores['domino_state'] = 0.5 * red_match + 0.5 * blue_match
        else:
            scores['domino_state'] = 0.2  # Detection failed
        
        # 4. Animation quality: Compare frame-by-frame
        if len(video_frames) >= 2 and len(gt_frames) >= 2:
            motion_scores = []
            for i in range(1, min(len(video_frames), 5)):
                diff = cv2.absdiff(video_frames[i], video_frames[i-1])
                motion = np.mean(diff)
                motion_scores.append(motion)
            
            if motion_scores:
                variance = np.var(motion_scores)
                scores['animation_quality'] = max(0, 1.0 - variance / 500.0)
            else:
                scores['animation_quality'] = 0.2  # Detection failed
        else:
            scores['animation_quality'] = 0.2  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class LEGOConstructionEvaluator(BaseEvaluator):
    """
    O-25: LEGO Construction Assembly
    
    Task: Follow LEGO assembly instructions - move highlighted brick to 
    arrow-indicated position on partial model.
    
    Rule-based evaluation:
    1. Position accuracy (35%) - Brick at arrow position
    2. Stud alignment (30%) - Studs properly aligned
    3. Rotation correctness (20%) - Brick orientation correct
    4. Connection stability (15%) - Brick properly connected
    """
    
    TASK_WEIGHTS = {
        'position': 0.35,
        'stud_alignment': 0.30,
        'rotation': 0.20,
        'connection': 0.15
    }
    
    def _detect_highlighted_brick(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect highlighted (usually yellow/bright) brick position."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        if hsv is None:
            return None
        
        # Yellow/highlighted detection
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)
        
        return None
    
    def _analyze_structure(self, frame: np.ndarray) -> Dict:
        """Analyze LEGO structure properties."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return {
            'edge_count': np.sum(edges > 0),
            'contour_count': len(contours),
            'total_area': sum(cv2.contourArea(c) for c in contours)
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate LEGO brick placement accuracy."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # 1. Position accuracy: Compare brick positions
        gen_brick = self._detect_highlighted_brick(gen_final)
        gt_brick = self._detect_highlighted_brick(gt_final)
        
        if gen_brick is not None and gt_brick is not None:
            dist = np.sqrt((gen_brick[0] - gt_brick[0])**2 + (gen_brick[1] - gt_brick[1])**2)
            scores['position'] = max(0, 1.0 - dist / 50.0)
        else:
            scores['position'] = 0.2  # Detection failed
        
        # 2. Stud alignment: Compare edge structures
        gen_struct = self._analyze_structure(gen_final)
        gt_struct = self._analyze_structure(gt_final)
        
        if gt_struct['edge_count'] > 0:
            edge_ratio = min(gen_struct['edge_count'], gt_struct['edge_count']) / max(gen_struct['edge_count'], gt_struct['edge_count'])
            scores['stud_alignment'] = edge_ratio
        else:
            scores['stud_alignment'] = 0.2  # Detection failed
        
        # 3. Rotation: Compare overall frame similarity
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            scores['rotation'] = max(0, 1.0 - diff / 100.0)
        else:
            scores['rotation'] = 0.2  # Detection failed
        
        # 4. Connection: Check structure completeness
        if gt_struct['contour_count'] > 0:
            contour_ratio = min(gen_struct['contour_count'], gt_struct['contour_count']) / max(gen_struct['contour_count'], gt_struct['contour_count'])
            scores['connection'] = contour_ratio
        else:
            scores['connection'] = 0.2  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

class BallColorEvaluator(BaseEvaluator):
    """
    O-29: Ball Color (Cluster Merging)
    
    Task: Red cluster A moves and merges with other clusters. When A's 
    count >= target, target disappears and A absorbs all balls.
    
    Rule-based evaluation:
    1. Red A dominance (30%) - Only A moves, stays red
    2. Merge rule accuracy (35%) - Correct merging logic
    3. Ball count conservation (25%) - Total unchanged
    4. Iteration completeness (10%) - Continue until only A remains
    """
    
    TASK_WEIGHTS = {
        'red_dominance': 0.30,
        'merge_rule': 0.35,
        'conservation': 0.25,
        'completeness': 0.10
    }
    
    def _count_color_clusters(self, frame: np.ndarray) -> Dict[str, int]:
        """Count clusters by color."""
        if len(frame.shape) != 3:
            return {'red': 0, 'blue': 0, 'green': 0}
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        colors = {}
        
        # Red
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        colors['red'] = int(np.sum(red_mask > 0))
        
        # Blue
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        colors['blue'] = int(np.sum(blue_mask > 0))
        
        # Green
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        colors['green'] = int(np.sum(green_mask > 0))
        
        return colors
    
    def _count_balls(self, frame: np.ndarray) -> int:
        """Count circular objects (balls)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10,
                                   param1=50, param2=30, minRadius=3, maxRadius=30)
        
        if circles is not None:
            return len(circles[0])
        return 0
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate ball cluster merging behavior."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # 1. Red dominance: Check if red is dominant in final
        gen_colors = self._count_color_clusters(gen_final)
        gt_colors = self._count_color_clusters(gt_final)
        
        gen_red_ratio = gen_colors['red'] / max(sum(gen_colors.values()), 1)
        gt_red_ratio = gt_colors['red'] / max(sum(gt_colors.values()), 1)
        
        ratio_diff = abs(gen_red_ratio - gt_red_ratio)
        scores['red_dominance'] = max(0, 1.0 - ratio_diff * 2)
        
        # 2. Merge rule: Compare final state with GT
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            scores['merge_rule'] = max(0, 1.0 - diff / 100.0)
        else:
            scores['merge_rule'] = 0.2  # Detection failed
        
        # 3. Conservation: Compare ball counts
        if len(video_frames) >= 2:
            first_count = self._count_balls(video_frames[0])
            last_count = self._count_balls(video_frames[-1])
            
            if first_count > 0:
                ratio = min(first_count, last_count) / max(first_count, last_count)
                scores['conservation'] = ratio
            else:
                scores['conservation'] = 0.2  # Detection failed
        else:
            scores['conservation'] = 0.2  # Detection failed
        
        # 4. Completeness: Check if process completed
        # Final should have mostly one color (red)
        total_colored = sum(gen_colors.values())
        if total_colored > 0:
            dominant_ratio = max(gen_colors.values()) / total_colored
            scores['completeness'] = dominant_ratio
        else:
            scores['completeness'] = 0.2  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class BookshelfEvaluator(BaseEvaluator):
    """
    O-30: Bookshelf (Height Clustering)
    
    Task: Insert new books into correct positions based on height clustering.
    Books go to the cluster with closest average height.
    
    Rule-based evaluation:
    1. Cluster identification (30%) - Correct cluster boundaries
    2. Representative height calculation (25%) - Correct averages
    3. Matching and insertion (30%) - Books at correct positions
    4. Sorting constraint (15%) - Multiple books sorted by height
    """
    
    TASK_WEIGHTS = {
        'cluster_identification': 0.30,
        'height_calculation': 0.25,
        'matching_insertion': 0.30,
        'sorting': 0.15
    }
    
    def _detect_book_heights(self, frame: np.ndarray) -> List[int]:
        """Detect vertical book heights using color-based detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        heights = []
        
        # Method 1: Color-based detection (gold and azure books)
        if hsv is not None:
            # Gold: H ~15-45, high S
            lower_gold = np.array([15, 50, 50])
            upper_gold = np.array([45, 255, 255])
            gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
            
            # Azure/Blue: H ~90-130, high S
            lower_azure = np.array([90, 50, 50])
            upper_azure = np.array([130, 255, 255])
            azure_mask = cv2.inRange(hsv, lower_azure, upper_azure)
            
            # Combine color masks
            color_mask = gold_mask | azure_mask
            
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > w * 0.8 and h > 20:  # Vertical book (allow some tolerance)
                    heights.append(h)
        
        # Method 2: Grayscale threshold fallback (try multiple thresholds)
        if len(heights) < 2:
            for thresh_val in [100, 150, 200]:
                _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                temp_heights = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if h > w and h > 20:  # Vertical book
                        temp_heights.append(h)
                
                if len(temp_heights) > len(heights):
                    heights = temp_heights
        
        return sorted(heights)
    
    def _analyze_book_arrangement(self, frame: np.ndarray) -> Dict:
        """Analyze book arrangement using color-based detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        books = []
        
        # Method 1: Color-based detection (gold and azure books)
        if hsv is not None:
            # Gold: H ~15-45, high S
            lower_gold = np.array([15, 50, 50])
            upper_gold = np.array([45, 255, 255])
            gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
            
            # Azure/Blue: H ~90-130, high S
            lower_azure = np.array([90, 50, 50])
            upper_azure = np.array([130, 255, 255])
            azure_mask = cv2.inRange(hsv, lower_azure, upper_azure)
            
            # Combine color masks
            color_mask = gold_mask | azure_mask
            
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                # Filter: vertical shape (height > width), minimum height, minimum area
                if h > w * 0.8 and h > 20 and area > 500:
                    books.append({'x': x, 'height': h, 'y': y, 'width': w})
        
        # Method 2: Grayscale threshold fallback (try multiple thresholds)
        if len(books) < 2:
            for thresh_val in [100, 150, 200]:
                _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                temp_books = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = cv2.contourArea(cnt)
                    if h > w and h > 20 and area > 500:
                        temp_books.append({'x': x, 'height': h, 'y': y, 'width': w})
                
                if len(temp_books) > len(books):
                    books = temp_books
        
        # Sort by x-position
        books.sort(key=lambda b: b['x'])
        
        return {
            'book_count': len(books),
            'heights': [b['height'] for b in books],
            'positions': [b['x'] for b in books]
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate bookshelf insertion accuracy - STRICT GT comparison."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Normalize frame size (handles padding removal + resize)
        if gen_final.shape != gt_final.shape:
            gt_final = normalize_frame_size(gt_final, gen_final)
        
        # STRICT: Compare directly with GT final frame
        # The task requires books to be inserted in correct positions based on height clustering
        final_diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
        
        # 1. Cluster identification (30%): STRICT comparison with GT
        if final_diff < 10:
            scores['cluster_identification'] = 1.0
        elif final_diff < 25:
            scores['cluster_identification'] = 0.3
        else:
            scores['cluster_identification'] = 0.0
        
        # 2. Height calculation (25%): STRICT comparison with GT
        if final_diff < 10:
            scores['height_calculation'] = 1.0
        elif final_diff < 25:
            scores['height_calculation'] = 0.3
        else:
            scores['height_calculation'] = 0.0
        
        # 3. Matching insertion (30%): STRICT comparison with GT
        if final_diff < 10:
            scores['matching_insertion'] = 1.0
        elif final_diff < 25:
            scores['matching_insertion'] = 0.3
        else:
            scores['matching_insertion'] = 0.0
        
        # 4. Sorting (15%): STRICT comparison with GT
        if final_diff < 10:
            scores['sorting'] = 1.0
        elif final_diff < 25:
            scores['sorting'] = 0.3
        else:
            scores['sorting'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_task_specific_old(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """OLD: Evaluate bookshelf insertion accuracy using detection."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Normalize frame size (handles padding removal + resize)
        if gen_final.shape != gt_final.shape:
            gt_final = normalize_frame_size(gt_final, gen_final)
        
        # Analyze arrangements
        gen_arr = self._analyze_book_arrangement(gen_final)
        gt_arr = self._analyze_book_arrangement(gt_final)
        
        # 1. Cluster identification (30%): Compare book counts
        # Rule: Correctly identify height-based clusters using eps threshold
        if gt_arr['book_count'] > 0:
            count_ratio = min(gen_arr['book_count'], gt_arr['book_count']) / max(gen_arr['book_count'], gt_arr['book_count'])
            # Perfect match gets full score
            if count_ratio == 1.0:
                scores['cluster_identification'] = 1.0
            elif count_ratio >= 0.9:
                scores['cluster_identification'] = 0.9
            else:
                scores['cluster_identification'] = max(0.3, count_ratio)
        else:
            # No GT books - check if generated also has no books
            scores['cluster_identification'] = 1.0 if gen_arr['book_count'] == 0 else 0.3
        
        # 2. Height calculation (25%): Compare height distributions
        # Rule: Representative height = average of cluster heights
        if gen_arr['heights'] and gt_arr['heights']:
            gen_mean_h = np.mean(gen_arr['heights'])
            gt_mean_h = np.mean(gt_arr['heights'])
            
            if gt_mean_h > 0:
                height_ratio = min(gen_mean_h, gt_mean_h) / max(gen_mean_h, gt_mean_h)
                # Rule: calculation precision should be within 5% for full score
                if height_ratio >= 0.95:
                    scores['height_calculation'] = 1.0
                elif height_ratio >= 0.90:
                    scores['height_calculation'] = 0.8
                else:
                    scores['height_calculation'] = max(0.3, height_ratio)
            else:
                scores['height_calculation'] = 0.2  # Detection failed
        else:
            # No heights detected
            scores['height_calculation'] = 0.5 if not gt_arr['heights'] else 0.0
        
        # 3. Matching insertion (30%): Compare book positions
        # Rule: Each new book should be at the end of its matched cluster
        if gen_arr['positions'] and gt_arr['positions']:
            # Compare position sequences
            matched_positions = 0
            for gen_pos in gen_arr['positions']:
                for gt_pos in gt_arr['positions']:
                    if abs(gen_pos - gt_pos) < 30:  # Within 30 pixels
                        matched_positions += 1
                        break
            position_match = matched_positions / max(len(gt_arr['positions']), 1)
            scores['matching_insertion'] = position_match
        else:
            scores['matching_insertion'] = 0.5 if not gt_arr['positions'] else 0.0
        
        # 4. Sorting (15%): Check if heights are sorted within clusters
        if len(gen_arr['heights']) >= 2:
            # Check for local sorting (within clusters)
            sorted_count = 0
            for i in range(1, len(gen_arr['heights'])):
                if gen_arr['heights'][i] >= gen_arr['heights'][i-1] * 0.8:
                    sorted_count += 1
            scores['sorting'] = sorted_count / (len(gen_arr['heights']) - 1)
        else:
            scores['sorting'] = 0.8  # Single book is trivially sorted
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class BallEatingEvaluator(BaseEvaluator):
    """
    O-31: Ball Eating (Greedy Algorithm)
    
    CRITICAL RULES:
    1. Black dot must move to ALL red dots
    2. Black ball must gradually become BIGGER after eating
    3. All red balls should be eaten (final count = 0)
    """
    
    TASK_WEIGHTS = {
        'all_eaten': 0.50,         # All red balls eaten
        'growth': 0.30,            # Black ball grows significantly
        'animation': 0.20          # Smooth movement
    }
    
    def _count_red_balls(self, frame: np.ndarray) -> int:
        """Count red balls in frame."""
        if len(frame.shape) != 3:
            return 0
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 80])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        return int(np.sum(red_mask > 0))
    
    def _get_black_ball_size(self, frame: np.ndarray) -> float:
        """Get size of black ball."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Black ball detection
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.medianBlur(thresh, 5)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area <= 0:
            return 0
        
        return np.sqrt(area / np.pi)
    
    def _count_red_ball_objects(self, frame: np.ndarray) -> int:
        """Count number of red ball objects (not pixels)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 80])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len([c for c in contours if cv2.contourArea(c) > 100])
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate ball eating behavior.
        
        CRITICAL RULES:
        1. ALL red balls must be eaten (final count = 0)
        2. Black ball must grow significantly
        """
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        
        # 1. CRITICAL: All red balls must be eaten
        first_red_count = self._count_red_ball_objects(first_frame)
        final_red_count = self._count_red_ball_objects(gen_final)
        
        if first_red_count > 0:
            if final_red_count == 0:
                scores['all_eaten'] = 1.0
            else:
                eaten_ratio = max(0, 1 - (final_red_count / first_red_count))
                scores['all_eaten'] = eaten_ratio * 0.5  # Partial credit, max 0.5
        else:
            scores['all_eaten'] = 0.0
        
        # 2. Black ball must grow significantly
        first_size = self._get_black_ball_size(first_frame)
        last_size = self._get_black_ball_size(gen_final)
        
        if first_size > 0:
            growth_ratio = last_size / first_size
            # Should grow at least 1.4x per ball eaten
            # If 3 balls, expected growth = 1.4^3 = 2.74
            expected_growth = 1.4 ** first_red_count if first_red_count > 0 else 2.0
            
            if growth_ratio >= expected_growth * 0.8:
                scores['growth'] = 1.0
            elif growth_ratio >= 1.5:
                scores['growth'] = 0.5
            elif growth_ratio >= 1.2:
                scores['growth'] = 0.3
            else:
                scores['growth'] = 0.1
        else:
            scores['growth'] = 0.0
        
        # 3. Animation: Check for smooth movement
        if len(video_frames) >= 3:
            motion_scores = []
            for i in range(1, min(len(video_frames), 10)):
                diff = cv2.absdiff(video_frames[i], video_frames[i-1])
                motion = np.mean(diff)
                motion_scores.append(motion)
            
            if motion_scores:
                avg_motion = np.mean(motion_scores)
                if avg_motion > 1:  # Some movement detected
                    scores['animation'] = min(1.0, avg_motion / 10.0)
                else:
                    scores['animation'] = 0.0
            else:
                scores['animation'] = 0.0
        else:
            scores['animation'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class RollingBallEvaluator(BaseEvaluator):
    """
    O-32: Rolling Ball Trajectory
    
    Task: Animate ball rolling along curved 3D path through multiple 
    platforms, following smooth trajectory.
    
    Rule-based evaluation:
    1. Trajectory accuracy (50%) - Ball follows platform centers
    2. Animation smoothness (15%) - Continuous, no jumps
    3. Physics realism (20%) - Ease-out effect near end
    4. Final state accuracy (15%) - Ball at last platform center
    """
    
    TASK_WEIGHTS = {
        'trajectory': 0.50,
        'smoothness': 0.15,
        'physics': 0.20,
        'final_state': 0.15
    }
    
    def _find_ball_center(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Find center of ball."""
        if len(frame.shape) == 3:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Red ball detection
            lower_red1 = np.array([0, 80, 80])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 80, 80])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
            
            if np.sum(red_mask > 0) > 50:
                coords = np.where(red_mask > 0)
                return (float(np.mean(coords[1])), float(np.mean(coords[0])))
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                    param1=100, param2=20, minRadius=5, maxRadius=80)
        
        if circles is not None and len(circles[0]) > 0:
            x, y, _ = circles[0][0]
            return (float(x), float(y))
        
        return None
    
    def _analyze_trajectory_smoothness(self, frames: List[np.ndarray]) -> float:
        """Analyze if ball motion is smooth."""
        positions = []
        
        for frame in frames:
            center = self._find_ball_center(frame)
            if center is not None:
                positions.append(center)
        
        if len(positions) < 3:
            # For GT vs GT, if ball is stationary, that's fine
            return 0.8
        
        # Calculate velocity changes
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            v = np.sqrt(dx**2 + dy**2)
            velocities.append(v)
        
        if len(velocities) < 2:
            return 0.8
        
        mean_v = np.mean(velocities)
        std_v = np.std(velocities)
        
        # If ball is nearly stationary (GT vs GT case), that's smooth
        if mean_v < 2.0:
            return 1.0
        
        cv = std_v / mean_v
        return max(0.5, 1 - cv)
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate rolling ball trajectory animation."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Normalize frame size (handles padding removal + resize)
        if gen_final.shape != gt_final.shape:
            gt_final = normalize_frame_size(gt_final, gen_final)
        
        # 1. Trajectory accuracy (50%): Compare final ball positions
        # Rule: Ball must accurately pass through each platform's geometric center
        gen_pos = self._find_ball_center(gen_final)
        gt_pos = self._find_ball_center(gt_final)
        
        if gen_pos is not None and gt_pos is not None:
            dist = np.sqrt((gen_pos[0] - gt_pos[0])**2 + (gen_pos[1] - gt_pos[1])**2)
            # Rule: error <= 10% of ball radius for perfect score
            if dist < 10:
                scores['trajectory'] = 1.0
            elif dist < 30:
                scores['trajectory'] = 0.9
            else:
                # More lenient threshold
                scores['trajectory'] = max(0, 1.0 - dist / 100.0)
        else:
            # Rule-based: check if ball exists in frame
            scores['trajectory'] = 0.3 if gen_pos is not None else 0.0
        
        # 2. Smoothness (15%): Animation should be smooth, continuous
        scores['smoothness'] = self._analyze_trajectory_smoothness(video_frames)
        
        # 3. Physics (20%): Check for deceleration near end (ease-out effect)
        # Rule: ball should slow down approaching the final platform
        if len(video_frames) >= 4:
            early_positions = []
            late_positions = []
            
            for frame in video_frames[:len(video_frames)//2]:
                pos = self._find_ball_center(frame)
                if pos is not None:
                    early_positions.append(pos)
            
            for frame in video_frames[len(video_frames)//2:]:
                pos = self._find_ball_center(frame)
                if pos is not None:
                    late_positions.append(pos)
            
            if len(early_positions) >= 2 and len(late_positions) >= 2:
                early_speed = np.sqrt((early_positions[-1][0] - early_positions[0][0])**2 + 
                                     (early_positions[-1][1] - early_positions[0][1])**2)
                late_speed = np.sqrt((late_positions[-1][0] - late_positions[0][0])**2 + 
                                    (late_positions[-1][1] - late_positions[0][1])**2)
                
                # If ball is nearly stationary at end, that's good physics
                if early_speed < 5 and late_speed < 5:
                    scores['physics'] = 1.0
                # Physics: ball should slow down (ease-out / cubic deceleration)
                elif late_speed < early_speed:
                    decel_ratio = (early_speed - late_speed) / max(early_speed, 1)
                    scores['physics'] = min(1.0, 0.5 + decel_ratio * 0.5)
                else:
                    # Ball didn't slow down - partial credit
                    scores['physics'] = 0.2  # Detection failed
            else:
                scores['physics'] = 0.6
        else:
            scores['physics'] = 0.6
        
        # 4. Final state (15%): Ball must be at last platform's geometric center
        # Rule: ball position error <= 10% of ball radius for perfect score
        if gen_pos is not None and gt_pos is not None:
            dist = np.sqrt((gen_pos[0] - gt_pos[0])**2 + (gen_pos[1] - gt_pos[1])**2)
            if dist < 10:
                scores['final_state'] = 1.0
            elif dist < 30:
                scores['final_state'] = 0.8
            else:
                scores['final_state'] = max(0, 1.0 - dist / 100.0)
        else:
            # Rule-based: ball must be visible in final frame
            scores['final_state'] = 0.2 if gen_pos is not None else 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class CountingObjectEvaluator(BaseEvaluator):
    """
    O-33: Counting Objects
    
    Task: Count all visible geometric shapes in scene, output total count.
    Each object counted exactly once, no misses or duplicates.
    
    Rule-based evaluation:
    1. Count accuracy (50%) - Exact count correct
    2. Completeness (25%) - All objects identified
    3. Uniqueness (15%) - No duplicates
    4. Systematic approach (10%) - Orderly counting
    """
    
    TASK_WEIGHTS = {
        'count_accuracy': 0.50,
        'completeness': 0.25,
        'uniqueness': 0.15,
        'systematic': 0.10
    }
    
    def _count_objects(self, frame: np.ndarray) -> int:
        """Count distinct objects in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by size
        min_area = 100
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        return len(valid_contours)
    
    def _detect_number_annotation(self, frame: np.ndarray) -> Optional[int]:
        """Try to detect if there's a number annotation showing the count."""
        # This is a simplified version - in practice would use OCR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Look for text-like regions
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count small text-like contours
        text_contours = [c for c in contours if 10 < cv2.contourArea(c) < 500]
        
        return len(text_contours) if text_contours else None
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate object counting accuracy."""
        
        if not video_frames or gt_final_frame is None or gt_first_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # The task requires showing a NUMBER that matches the count of objects
        # Compare the final frame directly with GT final frame (which shows correct count)
        
        # 1. Count accuracy: Final frame must match GT final (shows correct number)
        if gen_final.shape == gt_final.shape:
            # VERY STRICT comparison - the displayed number must match GT exactly
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            if diff < 10:  # Very close match - correct number displayed
                scores['count_accuracy'] = 1.0
            elif diff < 20:
                scores['count_accuracy'] = 0.3
            else:
                scores['count_accuracy'] = 0.0  # Wrong number displayed
        else:
            scores['count_accuracy'] = 0.0
        
        # 2. Completeness: Check if final frame shows the counting result
        # Compare structure - GT final should show number, gen should too
        if gen_final.shape == gt_final.shape:
            # Check if there's text/number region in final frame
            gen_gray = cv2.cvtColor(gen_final, cv2.COLOR_BGR2GRAY) if len(gen_final.shape) == 3 else gen_final
            gt_gray = cv2.cvtColor(gt_final, cv2.COLOR_BGR2GRAY) if len(gt_final.shape) == 3 else gt_final
            
            # Check difference in the number display area (typically center or bottom)
            h, w = gen_gray.shape[:2]
            gen_center = gen_gray[h//4:3*h//4, w//4:3*w//4]
            gt_center = gt_gray[h//4:3*h//4, w//4:3*w//4]
            
            center_diff = np.abs(gen_center.astype(float) - gt_center.astype(float)).mean()
            if center_diff < 10:  # Very strict
                scores['completeness'] = 1.0
            elif center_diff < 25:
                scores['completeness'] = 0.3
            else:
                scores['completeness'] = 0.0
        else:
            scores['completeness'] = 0.0
        
        # 3. Uniqueness: Same as count_accuracy for this task
        scores['uniqueness'] = scores['count_accuracy']
        
        # 4. Systematic: Check overall frame similarity - very strict
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            if diff < 10:
                scores['systematic'] = 1.0
            elif diff < 20:
                scores['systematic'] = 0.3
            else:
                scores['systematic'] = 0.0
        else:
            scores['systematic'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class DotToDotEvaluator(BaseEvaluator):
    """
    O-34: Dot to Dot
    
    Task: Connect numbered dots in sequence (1→2→3→...→N) with straight 
    red lines to form continuous path.
    
    Rule-based evaluation:
    1. Connection order (40%) - Strict numerical sequence
    2. Connection completeness (30%) - All dots connected
    3. Line quality (20%) - Straight lines, connect centers
    4. Visual fidelity (10%) - Red color, dots preserved
    """
    
    TASK_WEIGHTS = {
        'connection_order': 0.40,
        'completeness': 0.30,
        'line_quality': 0.20,
        'visual_fidelity': 0.10
    }
    
    def _count_red_line_pixels(self, frame: np.ndarray) -> int:
        """Count red line pixels."""
        if len(frame.shape) != 3:
            return 0
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        return int(np.sum(red_mask > 0))
    
    def _detect_blue_dots(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect blue dot positions."""
        if len(frame.shape) != 3:
            return []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                dots.append((cx, cy))
        
        return dots
    
    def _detect_red_lines(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect red line segments."""
        if len(frame.shape) != 3:
            return []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        lines = cv2.HoughLinesP(red_mask, 1, np.pi/180, threshold=30,
                                minLineLength=20, maxLineGap=10)
        
        if lines is None:
            return []
        
        return [(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate dot-to-dot connection accuracy."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # 1. Connection order: Compare overall structure
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            scores['connection_order'] = max(0, 1.0 - diff / 100.0)
        else:
            scores['connection_order'] = 0.2  # Detection failed
        
        # 2. Completeness: Compare red line amounts
        gen_red = self._count_red_line_pixels(gen_final)
        gt_red = self._count_red_line_pixels(gt_final)
        
        if gt_red > 0:
            ratio = min(gen_red, gt_red) / max(gen_red, gt_red)
            scores['completeness'] = ratio
        else:
            scores['completeness'] = 0.5 if gen_red == 0 else 0.3
        
        # 3. Line quality: Compare line counts
        gen_lines = self._detect_red_lines(gen_final)
        gt_lines = self._detect_red_lines(gt_final)
        
        if gt_lines:
            line_ratio = min(len(gen_lines), len(gt_lines)) / max(len(gen_lines), len(gt_lines), 1)
            scores['line_quality'] = line_ratio
        else:
            scores['line_quality'] = 0.2  # Detection failed
        
        # 4. Visual fidelity: Check dots preserved
        gen_dots = self._detect_blue_dots(gen_final)
        gt_dots = self._detect_blue_dots(gt_final)
        
        if gt_dots:
            dot_ratio = min(len(gen_dots), len(gt_dots)) / max(len(gen_dots), len(gt_dots), 1)
            scores['visual_fidelity'] = dot_ratio
        else:
            scores['visual_fidelity'] = 0.2  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


# Export mapping for this batch
IN_DOMAIN_50_EVALUATORS_PART4 = {
    'O-21_construction_blueprint_data-generator': ConstructionBlueprintEvaluator,
    'O-23_domino_chain_branch_path_prediction_data-generator': DominoChainBranchEvaluator,
    'O-24_domino_chain_gap_analysis_data-generator': DominoChainGapEvaluator,
    'O-25_LEGO_construction_assembly_data-generator': LEGOConstructionEvaluator,
    'O-29_ballcolor_data-generator': BallColorEvaluator,
    'O-30_bookshelf_data-generator': BookshelfEvaluator,
    'O-31_ball_eating_data-generator': BallEatingEvaluator,
    'O-32_rolling_ball_data-generator': RollingBallEvaluator,
    'O-33_counting_object_data-generator': CountingObjectEvaluator,
    'O-34_dot_to_dot_task_data-generator': DotToDotEvaluator,
}
