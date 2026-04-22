"""
Utility functions for VBVR-Bench evaluation.
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from PIL import Image
import colorsys


def safe_distance(p1: Tuple, p2: Tuple) -> float:
    """
    Calculate Euclidean distance between two points, avoiding integer overflow.
    
    Args:
        p1: First point as (x, y) tuple
        p2: Second point as (x, y) tuple
        
    Returns:
        Euclidean distance as float
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def normalize_frame_size(frame: np.ndarray, target_frame: np.ndarray, 
                         background_color: Tuple[int, int, int] = None) -> np.ndarray:
    """
    Normalize frame size to match target_frame dimensions.
    
    Strategy:
    - If dimensions already match, return as-is
    - If aspect ratios are similar (within 5%), just resize
    - If aspect ratios differ, try to detect and crop padding (gray, white, or black)
      then resize to target dimensions
    
    Args:
        frame: Frame to normalize (source)
        target_frame: Frame with target dimensions
        background_color: Background color to detect for cropping (auto-detect if None)
        
    Returns:
        Frame normalized to target dimensions
    """
    if frame.shape == target_frame.shape:
        return frame
    
    h_src, w_src = frame.shape[:2]
    h_tgt, w_tgt = target_frame.shape[:2]
    
    # Calculate aspect ratios
    ar_src = w_src / h_src if h_src > 0 else 1
    ar_tgt = w_tgt / h_tgt if h_tgt > 0 else 1
    
    # If aspect ratios are similar (within 5%), just resize
    if abs(ar_src - ar_tgt) / max(ar_src, ar_tgt) < 0.05:
        return cv2.resize(frame, (w_tgt, h_tgt))
    
    # Try multiple background colors if not specified
    if background_color is None:
        # Try gray (128), white (255), and black (0)
        background_colors_to_try = [
            (128, 128, 128),  # Gray
            (255, 255, 255),  # White
            (0, 0, 0),        # Black
        ]
        
        best_cropped = frame
        best_crop_ratio = 1.0  # 1.0 means no cropping
        
        for bg_color in background_colors_to_try:
            cropped = _crop_padded_content(frame, bg_color)
            crop_ratio = (cropped.shape[0] * cropped.shape[1]) / (frame.shape[0] * frame.shape[1])
            
            # We want the smallest crop that still has content
            if 0.2 < crop_ratio < best_crop_ratio:
                best_cropped = cropped
                best_crop_ratio = crop_ratio
        
        result_frame = best_cropped
    else:
        result_frame = _crop_padded_content(frame, background_color)
    
    # Resize the cropped content to target dimensions
    return cv2.resize(result_frame, (w_tgt, h_tgt))


def _crop_padded_content(frame: np.ndarray, background_color: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """
    Crop out padding from a frame to extract the original content.
    Detects rows/columns that are mostly the background color and removes them.
    
    Args:
        frame: Frame that may have padding
        background_color: Color of the padding (BGR format)
        
    Returns:
        Cropped frame with padding removed
    """
    h, w = frame.shape[:2]
    bg_color = np.array(background_color, dtype=np.uint8)
    
    # Tolerance for background detection
    tolerance = 20
    
    # Create a mask of pixels that are NOT background
    diff = np.abs(frame.astype(np.int16) - bg_color.astype(np.int16))
    not_bg_mask = np.any(diff > tolerance, axis=2)
    
    # Find rows and columns with content
    row_has_content = np.any(not_bg_mask, axis=1)
    col_has_content = np.any(not_bg_mask, axis=0)
    
    # Find content boundaries
    rows_with_content = np.where(row_has_content)[0]
    cols_with_content = np.where(col_has_content)[0]
    
    if len(rows_with_content) == 0 or len(cols_with_content) == 0:
        # No content detected, return original
        return frame
    
    top = rows_with_content[0]
    bottom = rows_with_content[-1] + 1
    left = cols_with_content[0]
    right = cols_with_content[-1] + 1
    
    # Add small margin to avoid cutting off content edges
    margin = 2
    top = max(0, top - margin)
    bottom = min(h, bottom + margin)
    left = max(0, left - margin)
    right = min(w, right + margin)
    
    # Crop
    cropped = frame[top:bottom, left:right]
    
    # Only return cropped if it's significantly smaller (padding was actually removed)
    if cropped.shape[0] < h * 0.95 or cropped.shape[1] < w * 0.95:
        return cropped
    
    # No significant padding detected, return original
    return frame


def load_json(path: str) -> Dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_json(data: Any, path: str):
    """Save data to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def get_video_frames(
    video_path: str,
    max_frames: Optional[int] = None,
    frame_indices: Optional[List[int]] = None
) -> List[np.ndarray]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (evenly sampled)
        frame_indices: Specific frame indices to extract
        
    Returns:
        List of frames as numpy arrays (BGR format)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_indices is not None:
        indices = frame_indices
    elif max_frames is not None and max_frames < total_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
    else:
        indices = list(range(total_frames))
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames


def get_frame_count(video_path: str) -> int:
    """Get total frame count of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def get_video_info(video_path: str) -> Dict:
    """Get video information (fps, width, height, frame_count)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return info


def load_image(path: str) -> np.ndarray:
    """Load an image file."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return img


def load_gt_metadata(gt_path: str) -> Dict:
    """
    Load ground truth metadata for a task instance.
    
    Args:
        gt_path: Path to GT folder containing ground_truth.mp4, first_frame.png, etc.
        
    Returns:
        Dictionary with GT metadata
    """
    metadata = {
        'path': gt_path,
        'has_video': os.path.exists(os.path.join(gt_path, 'ground_truth.mp4')),
        'has_first_frame': os.path.exists(os.path.join(gt_path, 'first_frame.png')),
        'has_final_frame': os.path.exists(os.path.join(gt_path, 'final_frame.png')),
        'has_prompt': os.path.exists(os.path.join(gt_path, 'prompt.txt')),
    }
    
    if metadata['has_video']:
        metadata['video_info'] = get_video_info(os.path.join(gt_path, 'ground_truth.mp4'))
    
    if metadata['has_prompt']:
        with open(os.path.join(gt_path, 'prompt.txt'), 'r') as f:
            metadata['prompt'] = f.read().strip()
    
    return metadata


def extract_task_info_from_path(path: str) -> Dict:
    """Extract task information from a video path."""
    parts = Path(path).parts
    
    info = {
        'full_path': path,
        'filename': parts[-1] if parts else '',
    }
    
    # Try to extract split and task name
    for i, part in enumerate(parts):
        if part in ['In-Domain_50', 'Out-of-Domain_50']:
            info['split'] = part
            if i + 1 < len(parts):
                info['task_name'] = parts[i + 1]
            break
    
    return info


# ============================================================================
# Image/Frame Comparison Utilities
# ============================================================================

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    Returns value between 0 and 1, where 1 means identical.
    """
    if img1.shape != img2.shape:
        img2 = normalize_frame_size(img2, img1)
    
    # Convert to grayscale if color
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(ssim_map.mean())


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Squared Error between two images."""
    if img1.shape != img2.shape:
        img2 = normalize_frame_size(img2, img1)
    
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(255 ** 2 / mse))


def compute_histogram_similarity(img1: np.ndarray, img2: np.ndarray, method: str = 'correlation') -> float:
    """
    Compute histogram similarity between two images.
    
    Args:
        img1, img2: Input images (BGR)
        method: One of 'correlation', 'chi-square', 'intersection', 'bhattacharyya'
        
    Returns:
        Similarity score (higher is more similar for correlation/intersection)
    """
    # Convert to HSV
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    # Compute histograms
    h_bins, s_bins = 50, 60
    hist_size = [h_bins, s_bins]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges
    channels = [0, 1]
    
    hist1 = cv2.calcHist([hsv1], channels, None, hist_size, ranges)
    hist2 = cv2.calcHist([hsv2], channels, None, hist_size, ranges)
    
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    methods = {
        'correlation': cv2.HISTCMP_CORREL,
        'chi-square': cv2.HISTCMP_CHISQR,
        'intersection': cv2.HISTCMP_INTERSECT,
        'bhattacharyya': cv2.HISTCMP_BHATTACHARYYA
    }
    
    return float(cv2.compareHist(hist1, hist2, methods.get(method, cv2.HISTCMP_CORREL)))


# ============================================================================
# Color Analysis Utilities
# ============================================================================

def get_dominant_colors(img: np.ndarray, n_colors: int = 5) -> List[Tuple[int, int, int]]:
    """
    Extract dominant colors from an image using k-means clustering.
    
    Returns:
        List of (B, G, R) color tuples
    """
    # Reshape image
    pixels = img.reshape(-1, 3).astype(np.float32)
    
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Get colors sorted by frequency
    unique, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    
    colors = []
    for idx in sorted_indices:
        color = tuple(int(c) for c in centers[idx])
        colors.append(color)
    
    return colors


def color_distance(c1: Tuple, c2: Tuple, method: str = 'euclidean') -> float:
    """
    Compute distance between two colors.
    
    Args:
        c1, c2: Color tuples (B, G, R) or (H, S, V)
        method: 'euclidean' or 'deltaE' (perceptual)
        
    Returns:
        Distance value (lower = more similar)
    """
    if method == 'euclidean':
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
    else:
        # Simple euclidean as fallback
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to HSV (H: 0-360, S: 0-100, V: 0-100)."""
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h * 360, s * 100, v * 100


def color_name_match(color_bgr: Tuple[int, int, int], expected_name: str) -> float:
    """
    Check if a BGR color matches an expected color name.
    
    Args:
        color_bgr: (B, G, R) tuple
        expected_name: Color name like 'red', 'green', 'blue', etc.
        
    Returns:
        Match score between 0 and 1
    """
    b, g, r = color_bgr
    h, s, v = rgb_to_hsv(r, g, b)
    
    # Define color ranges (H: 0-360, S: 0-100, V: 0-100)
    color_ranges = {
        'red': [(0, 15), (345, 360)],  # Red wraps around
        'orange': [(15, 45)],
        'yellow': [(45, 70)],
        'green': [(70, 170)],
        'cyan': [(170, 200)],
        'blue': [(200, 260)],
        'purple': [(260, 290)],
        'magenta': [(290, 345)],
        'white': None,  # High V, low S
        'black': None,  # Low V
        'gray': None,   # Low S
    }
    
    expected_name = expected_name.lower()
    
    if expected_name == 'white':
        return 1.0 if v > 80 and s < 20 else max(0, (v - 50) / 50 * (1 - s / 100))
    elif expected_name == 'black':
        return 1.0 if v < 20 else max(0, (50 - v) / 50)
    elif expected_name == 'gray':
        return 1.0 if s < 20 and 20 < v < 80 else max(0, 1 - s / 50)
    elif expected_name in color_ranges:
        ranges = color_ranges[expected_name]
        for h_min, h_max in ranges:
            if h_min <= h <= h_max and s > 30 and v > 30:
                return 1.0
            elif h_min <= h <= h_max:
                return max(0, min(s / 50, v / 50))
    
    return 0.0


# ============================================================================
# Shape Detection Utilities
# ============================================================================

def detect_shapes(img: np.ndarray, min_area: int = 100) -> List[Dict]:
    """
    Detect shapes in an image.
    
    Returns:
        List of shape dictionaries with 'type', 'contour', 'center', 'area', 'bbox'
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Approximate contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Classify shape
        vertices = len(approx)
        shape_type = classify_shape(vertices, contour)
        
        # Get center and bounding box
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0
        
        x, y, w, h = cv2.boundingRect(contour)
        
        shapes.append({
            'type': shape_type,
            'contour': contour,
            'vertices': vertices,
            'center': (cx, cy),
            'area': area,
            'bbox': (x, y, w, h),
            'approx': approx
        })
    
    return shapes


def classify_shape(vertices: int, contour: np.ndarray) -> str:
    """Classify a shape based on number of vertices and contour properties."""
    if vertices == 3:
        return 'triangle'
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.9 <= aspect_ratio <= 1.1:
            return 'square'
        else:
            return 'rectangle'
    elif vertices == 5:
        return 'pentagon'
    elif vertices == 6:
        return 'hexagon'
    elif vertices > 6:
        # Check circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.7:
                return 'circle'
    
    return 'polygon'


def count_objects_by_color(img: np.ndarray, target_color_bgr: Tuple[int, int, int], tolerance: int = 30) -> int:
    """
    Count distinct objects of a specific color in an image.
    
    Args:
        img: Input image (BGR)
        target_color_bgr: Target color as (B, G, R)
        tolerance: Color matching tolerance
        
    Returns:
        Number of detected objects
    """
    # Create mask for target color
    lower = np.array([max(0, c - tolerance) for c in target_color_bgr])
    upper = np.array([min(255, c + tolerance) for c in target_color_bgr])
    mask = cv2.inRange(img, lower, upper)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by minimum area
    min_area = 50
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    return len(valid_contours)


# ============================================================================
# Motion and Flow Utilities
# ============================================================================

def compute_optical_flow(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute optical flow between two frames.
    
    Returns:
        Tuple of (flow array, average magnitude)
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    
    return flow, float(np.mean(magnitude))


def compute_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute normalized difference between two frames."""
    if frame1.shape != frame2.shape:
        frame2 = normalize_frame_size(frame2, frame1)
    
    diff = cv2.absdiff(frame1, frame2)
    return float(np.mean(diff) / 255.0)


def detect_motion_regions(frame1: np.ndarray, frame2: np.ndarray, threshold: int = 30) -> List[Tuple[int, int, int, int]]:
    """
    Detect regions with motion between two frames.
    
    Returns:
        List of bounding boxes (x, y, w, h) for motion regions
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Dilate to fill gaps
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, w, h))
    
    return bboxes


# ============================================================================
# Score Calculation Utilities
# ============================================================================

def linear_score(value: float, min_val: float, max_val: float, invert: bool = False) -> float:
    """
    Calculate a linear score between 0 and 1.
    
    Args:
        value: The value to score
        min_val: Value that maps to 0 (or 1 if inverted)
        max_val: Value that maps to 1 (or 0 if inverted)
        invert: If True, higher values give lower scores
        
    Returns:
        Score between 0 and 1
    """
    if max_val == min_val:
        return 0.5
    
    score = (value - min_val) / (max_val - min_val)
    score = max(0, min(1, score))
    
    if invert:
        score = 1 - score
    
    return score


def threshold_score(value: float, thresholds: List[Tuple[float, float]]) -> float:
    """
    Calculate score based on threshold ranges.
    
    Args:
        value: The value to score
        thresholds: List of (threshold, score) pairs, sorted by threshold ascending
                   Score is interpolated between thresholds
                   
    Returns:
        Score between 0 and 1
    """
    if not thresholds:
        return 0.5
    
    # Sort by threshold
    thresholds = sorted(thresholds, key=lambda x: x[0])
    
    if value <= thresholds[0][0]:
        return thresholds[0][1]
    if value >= thresholds[-1][0]:
        return thresholds[-1][1]
    
    # Interpolate
    for i in range(len(thresholds) - 1):
        t1, s1 = thresholds[i]
        t2, s2 = thresholds[i + 1]
        if t1 <= value <= t2:
            ratio = (value - t1) / (t2 - t1)
            return s1 + ratio * (s2 - s1)
    
    return 0.5


def weighted_average(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculate weighted average of scores.
    
    Args:
        scores: Dictionary of dimension -> score
        weights: Dictionary of dimension -> weight (should sum to 1 or will be normalized)
        
    Returns:
        Weighted average score
    """
    total_weight = sum(weights.get(dim, 0) for dim in scores)
    if total_weight == 0:
        return sum(scores.values()) / len(scores) if scores else 0
    
    weighted_sum = sum(scores[dim] * weights.get(dim, 0) for dim in scores)
    return weighted_sum / total_weight
