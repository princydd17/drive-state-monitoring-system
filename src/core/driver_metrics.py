#!/usr/bin/env python3
"""
Driver Metrics - Core detection algorithms for driver monitoring
Provides EAR, MAR, head pose, and distraction detection functions
"""

import cv2
import numpy as np
import math
from typing import Tuple, List, Dict, Optional

# Facial landmark indices for dlib's 68-point model
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
MOUTH_INDICES = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) for drowsiness detection.
    
    Args:
        eye_points: Array of 6 eye landmark points (x, y) coordinates
        
    Returns:
        float: Eye aspect ratio value
    """
    if len(eye_points) != 6:
        return 0.0
    
    # Calculate vertical distances
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    
    # Calculate horizontal distance
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    
    # Avoid division by zero
    if C == 0:
        return 0.0
    
    # Calculate EAR
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth_points: np.ndarray) -> float:
    """
    Calculate the Mouth Aspect Ratio (MAR) for yawn detection.
    
    Args:
        mouth_points: Array of mouth landmark points (x, y) coordinates
        
    Returns:
        float: Mouth aspect ratio value
    """
    if len(mouth_points) < 12:
        return 0.0
    
    # Calculate vertical distances
    A = np.linalg.norm(mouth_points[2] - mouth_points[6])
    B = np.linalg.norm(mouth_points[3] - mouth_points[5])
    
    # Calculate horizontal distance
    C = np.linalg.norm(mouth_points[0] - mouth_points[4])
    
    # Avoid division by zero
    if C == 0:
        return 0.0
    
    # Calculate MAR
    mar = (A + B) / (2.0 * C)
    return mar

def detect_drowsiness(left_ear: float, right_ear: float, threshold: float = 0.25) -> Tuple[bool, str, float]:
    """
    Detect drowsiness based on eye aspect ratios.
    
    Args:
        left_ear: Left eye aspect ratio
        right_ear: Right eye aspect ratio
        threshold: EAR threshold for drowsiness detection
        
    Returns:
        Tuple[bool, str, float]: (is_drowsy, severity, confidence)
    """
    avg_ear = (left_ear + right_ear) / 2.0
    
    if avg_ear < threshold:
        # Determine severity based on EAR value
        if avg_ear < 0.15:
            severity = "high"
            confidence = 0.95
        elif avg_ear < 0.20:
            severity = "medium"
            confidence = 0.85
        else:
            severity = "low"
            confidence = 0.75
        
        return True, severity, confidence
    
    return False, "none", 0.0

def detect_yawn(mar: float, threshold: float = 0.5) -> Tuple[bool, str, float]:
    """
    Detect yawning based on mouth aspect ratio.
    
    Args:
        mar: Mouth aspect ratio
        threshold: MAR threshold for yawn detection
        
    Returns:
        Tuple[bool, str, float]: (is_yawning, severity, confidence)
    """
    if mar > threshold:
        # Determine severity based on MAR value
        if mar > 0.8:
            severity = "severe"
            confidence = 0.95
        elif mar > 0.65:
            severity = "moderate"
            confidence = 0.85
        else:
            severity = "mild"
            confidence = 0.75
        
        return True, severity, confidence
    
    return False, "none", 0.0

def calculate_head_pose(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> Optional[Dict]:
    """
    Calculate head pose angles from facial landmarks.
    
    Args:
        landmarks: Array of facial landmarks
        image_shape: Image dimensions (height, width)
        
    Returns:
        Optional[Dict]: Head pose information or None
    """
    if len(landmarks) < 68:
        return None
    
    # 3D model points for head pose estimation
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    
    # 2D image points
    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],   # Chin
        landmarks[36],  # Left eye left corner
        landmarks[45],  # Right eye right corner
        landmarks[48],  # Left mouth corner
        landmarks[54]   # Right mouth corner
    ], dtype="double")
    
    # Camera matrix
    size = image_shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    
    # Distortion coefficients
    dist_coeffs = np.zeros((4,1))
    
    # Solve PnP
    try:
        (success, rotation_vec, translation_vec) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # Convert rotation vector to rotation matrix
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            
            # Get Euler angles
            pose_mat = cv2.hconcat((rotation_mat, translation_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            
            pitch = euler_angles[0]
            yaw = euler_angles[1]
            roll = euler_angles[2]
            
            # Determine head direction
            direction = "forward"
            if abs(yaw) > 15:
                direction = "left" if yaw > 0 else "right"
            elif abs(pitch) > 15:
                direction = "up" if pitch > 0 else "down"
            
            return {
                "pitch": float(pitch),
                "yaw": float(yaw),
                "roll": float(roll),
                "direction": direction,
                "confidence": 0.8
            }
    
    except Exception as e:
        print(f"Head pose calculation error: {e}")
    
    return None

def detect_distraction(head_pose: Dict, threshold: float = 15.0) -> Tuple[bool, str, float]:
    """
    Detect distraction based on head pose.
    
    Args:
        head_pose: Head pose information
        threshold: Angle threshold for distraction detection
        
    Returns:
        Tuple[bool, str, float]: (is_distracted, direction, confidence)
    """
    if not head_pose:
        return False, "none", 0.0
    
    yaw = abs(head_pose["yaw"])
    pitch = abs(head_pose["pitch"])
    
    if yaw > threshold or pitch > threshold:
        direction = head_pose["direction"]
        confidence = min(0.9, (max(yaw, pitch) - threshold) / 30.0 + 0.7)
        return True, direction, confidence
    
    return False, "none", 0.0

def calculate_perclos(ear_history: List[float], threshold: float = 0.25, 
                     time_window: float = 60.0, fps: float = 30.0) -> float:
    """
    Calculate PERCLOS (Percentage of Eye Closure) metric.
    
    Args:
        ear_history: List of EAR values over time
        threshold: EAR threshold for eye closure
        time_window: Time window in seconds
        fps: Frames per second
        
    Returns:
        float: PERCLOS percentage
    """
    if not ear_history:
        return 0.0
    
    # Calculate number of frames in time window
    window_frames = int(time_window * fps)
    
    # Get recent EAR values
    recent_ear = ear_history[-window_frames:] if len(ear_history) > window_frames else ear_history
    
    # Count frames with closed eyes
    closed_eye_frames = sum(1 for ear in recent_ear if ear < threshold)
    
    # Calculate PERCLOS
    perclos = (closed_eye_frames / len(recent_ear)) * 100.0
    
    return perclos

def calculate_eye_closure_duration(ear_history: List[float], threshold: float = 0.25, 
                                 fps: float = 30.0) -> float:
    """
    Calculate the duration of eye closure.
    
    Args:
        ear_history: List of EAR values over time
        threshold: EAR threshold for eye closure
        fps: Frames per second
        
    Returns:
        float: Eye closure duration in seconds
    """
    if not ear_history:
        return 0.0
    
    # Count consecutive frames with closed eyes
    consecutive_closed = 0
    for ear in reversed(ear_history):
        if ear < threshold:
            consecutive_closed += 1
        else:
            break
    
    # Convert to seconds
    duration = consecutive_closed / fps
    
    return duration

def calculate_blink_frequency(ear_history: List[float], threshold: float = 0.25,
                            time_window: float = 60.0, fps: float = 30.0) -> float:
    """
    Calculate blink frequency.
    
    Args:
        ear_history: List of EAR values over time
        threshold: EAR threshold for eye closure
        time_window: Time window in seconds
        fps: Frames per second
        
    Returns:
        float: Blink frequency (blinks per minute)
    """
    if len(ear_history) < 2:
        return 0.0
    
    # Calculate number of frames in time window
    window_frames = int(time_window * fps)
    
    # Get recent EAR values
    recent_ear = ear_history[-window_frames:] if len(ear_history) > window_frames else ear_history
    
    # Count blinks (transitions from open to closed to open)
    blinks = 0
    was_closed = recent_ear[0] < threshold
    
    for ear in recent_ear[1:]:
        is_closed = ear < threshold
        
        if not was_closed and is_closed:
            # Eye just closed
            pass
        elif was_closed and not is_closed:
            # Eye just opened - count as blink
            blinks += 1
        
        was_closed = is_closed
    
    # Convert to blinks per minute
    frequency = (blinks / time_window) * 60.0
    
    return frequency

def analyze_eye_movement(eye_points: np.ndarray, prev_eye_points: Optional[np.ndarray] = None) -> Dict:
    """
    Analyze eye movement patterns.
    
    Args:
        eye_points: Current eye landmark points
        prev_eye_points: Previous eye landmark points
        
    Returns:
        Dict: Eye movement analysis
    """
    analysis = {
        "movement_detected": False,
        "movement_direction": "none",
        "movement_magnitude": 0.0,
        "stability_score": 1.0
    }
    
    if prev_eye_points is None or len(eye_points) != len(prev_eye_points):
        return analysis
    
    # Calculate center of current and previous eye
    current_center = np.mean(eye_points, axis=0)
    prev_center = np.mean(prev_eye_points, axis=0)
    
    # Calculate movement vector
    movement_vector = current_center - prev_center
    movement_magnitude = np.linalg.norm(movement_vector)
    
    if movement_magnitude > 2.0:  # Threshold for significant movement
        analysis["movement_detected"] = True
        analysis["movement_magnitude"] = movement_magnitude
        
        # Determine movement direction
        if abs(movement_vector[0]) > abs(movement_vector[1]):
            analysis["movement_direction"] = "horizontal"
        else:
            analysis["movement_direction"] = "vertical"
        
        # Calculate stability score (inverse of movement)
        analysis["stability_score"] = max(0.0, 1.0 - (movement_magnitude / 10.0))
    
    return analysis

def validate_landmarks(landmarks: np.ndarray) -> bool:
    """
    Validate facial landmarks quality.
    
    Args:
        landmarks: Array of facial landmarks
        
    Returns:
        bool: True if landmarks are valid
    """
    if landmarks is None or len(landmarks) < 68:
        return False
    
    # Check for NaN or infinite values
    if np.any(np.isnan(landmarks)) or np.any(np.isinf(landmarks)):
        return False
    
    # Check for reasonable coordinate ranges
    if np.any(landmarks < -1000) or np.any(landmarks > 1000):
        return False
    
    return True

def smooth_ear_values(ear_history: List[float], window_size: int = 5) -> List[float]:
    """
    Apply smoothing to EAR values using moving average.
    
    Args:
        ear_history: List of EAR values
        window_size: Size of smoothing window
        
    Returns:
        List[float]: Smoothed EAR values
    """
    if len(ear_history) < window_size:
        return ear_history
    
    smoothed = []
    for i in range(len(ear_history)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(ear_history), i + window_size // 2 + 1)
        window = ear_history[start_idx:end_idx]
        smoothed.append(np.mean(window))
    
    return smoothed 