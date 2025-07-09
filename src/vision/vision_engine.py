#!/usr/bin/env python3
"""
Vision Engine - Advanced computer vision processing for driver monitoring
Provides facial landmark detection, eye tracking, and head pose estimation
"""

import cv2
import numpy as np
import dlib
import time
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class VisionProcessor:
    """
    Advanced computer vision processor for driver monitoring.
    """
    
    def __init__(self, predictor_path: str = None):
        """
        Initialize vision processor.
        
        Args:
            predictor_path: Path to dlib facial landmark predictor
        """
        if predictor_path is None:
            # Look for predictor in AI models directory
            ai_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ai')
            predictor_path = os.path.join(ai_dir, "shape_predictor_68_face_landmarks.dat")
        
        self.predictor_path = predictor_path
        self.detector = None
        self.predictor = None
        self.face_cascade = None
        
        # Initialize components
        self._initialize_components()
        
        # State variables
        self.landmark_history = []
        self.ear_history = []
        self.mar_history = []
        self.face_detection_history = []
        
        # Configuration
        self.config = {
            'min_face_size': 80,
            'confidence_threshold': 0.5,
            'smoothing_window': 5,
            'max_history_length': 100
        }
    
    def _initialize_components(self):
        """Initialize vision components."""
        try:
            # Initialize dlib face detector
            self.detector = dlib.get_frontal_face_detector()
            print("Dlib face detector initialized")
            
            # Initialize dlib facial landmark predictor
            if os.path.exists(self.predictor_path):
                self.predictor = dlib.shape_predictor(self.predictor_path)
                print("Dlib facial landmark predictor initialized")
            else:
                print(f"Warning: Predictor file not found at {self.predictor_path}")
                self.predictor = None
            
            # Initialize OpenCV face cascade as backup
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                print("Warning: OpenCV face cascade not loaded")
                self.face_cascade = None
            else:
                print("OpenCV face cascade initialized")
                
        except Exception as e:
            print(f"Error initializing vision components: {e}")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame for facial analysis.
        
        Args:
            frame: Input frame
            
        Returns:
            Dict: Processing results
        """
        results = {
            'face_detected': False,
            'landmarks': None,
            'eye_analysis': {
                'left_ear': 0.0,
                'right_ear': 0.0,
                'avg_ear': 0.0,
                'blink_confidence': 0.0,
                'eye_closure_duration': 0.0
            },
            'mouth_analysis': {
                'mar': 0.0,
                'is_yawning': False,
                'yawn_severity': 'none',
                'mouth_open_ratio': 0.0
            },
            'head_pose': None,
            'facial_expression': {
                'expression': 'neutral',
                'confidence': 0.0,
                'valence': 0.0,
                'arousal': 0.0
            },
            'processing_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Detect face
            face_rect = self._detect_face(frame)
            
            if face_rect is not None:
                results['face_detected'] = True
                
                # Extract landmarks
                landmarks = self._extract_landmarks(frame, face_rect)
                
                if landmarks is not None:
                    results['landmarks'] = landmarks
                    
                    # Analyze eyes
                    eye_analysis = self._analyze_eyes(landmarks)
                    results['eye_analysis'] = eye_analysis
                    
                    # Analyze mouth
                    mouth_analysis = self._analyze_mouth(landmarks)
                    results['mouth_analysis'] = mouth_analysis
                    
                    # Calculate head pose
                    head_pose = self._calculate_head_pose(landmarks, frame.shape)
                    results['head_pose'] = head_pose
                    
                    # Analyze facial expression
                    expression_analysis = self._analyze_facial_expression(landmarks)
                    results['facial_expression'] = expression_analysis
                    
                    # Update history
                    self._update_history(landmarks, eye_analysis, mouth_analysis)
            
            results['processing_time'] = time.time() - start_time
            
        except Exception as e:
            print(f"Frame processing error: {e}")
        
        return results
    
    def _detect_face(self, frame: np.ndarray) -> Optional[dlib.rectangle]:
        """
        Detect face in frame using multiple methods.
        
        Args:
            frame: Input frame
            
        Returns:
            Optional[dlib.rectangle]: Detected face rectangle
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try dlib detector first
        if self.detector is not None:
            try:
                faces = self.detector(gray, 1)
                if len(faces) > 0:
                    return faces[0]
            except Exception as e:
                print(f"Dlib face detection error: {e}")
        
        # Fallback to OpenCV cascade
        if self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5,
                    minSize=(self.config['min_face_size'], self.config['min_face_size'])
                )
                
                if len(faces) > 0:
                    # Convert OpenCV rectangle to dlib rectangle
                    x, y, w, h = faces[0]
                    return dlib.rectangle(x, y, x + w, y + h)
                    
            except Exception as e:
                print(f"OpenCV face detection error: {e}")
        
        return None
    
    def _extract_landmarks(self, frame: np.ndarray, face_rect: dlib.rectangle) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from detected face.
        
        Args:
            frame: Input frame
            face_rect: Face rectangle
            
        Returns:
            Optional[np.ndarray]: Array of landmark points
        """
        if self.predictor is None:
            return None
        
        try:
            # Predict landmarks
            shape = self.predictor(frame, face_rect)
            
            # Convert to numpy array
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            
            return landmarks
            
        except Exception as e:
            print(f"Landmark extraction error: {e}")
            return None
    
    def _analyze_eyes(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Analyze eye regions for drowsiness detection.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Dict: Eye analysis results
        """
        analysis = {
            'left_ear': 0.0,
            'right_ear': 0.0,
            'avg_ear': 0.0,
            'blink_confidence': 0.0,
            'eye_closure_duration': 0.0,
            'eye_movement': 'stable',
            'gaze_direction': 'center'
        }
        
        try:
            # Extract eye landmarks
            left_eye = landmarks[36:42]  # Left eye points
            right_eye = landmarks[42:48]  # Right eye points
            
            # Calculate EAR for each eye
            left_ear = self._calculate_ear(left_eye)
            right_ear = self._calculate_ear(right_eye)
            
            analysis['left_ear'] = left_ear
            analysis['right_ear'] = right_ear
            analysis['avg_ear'] = (left_ear + right_ear) / 2.0
            
            # Calculate blink confidence
            analysis['blink_confidence'] = self._calculate_blink_confidence(analysis['avg_ear'])
            
            # Calculate eye closure duration
            analysis['eye_closure_duration'] = self._calculate_eye_closure_duration()
            
            # Analyze eye movement
            analysis['eye_movement'] = self._analyze_eye_movement(left_eye, right_eye)
            
            # Determine gaze direction
            analysis['gaze_direction'] = self._determine_gaze_direction(left_eye, right_eye)
            
        except Exception as e:
            print(f"Eye analysis error: {e}")
        
        return analysis
    
    def _calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).
        
        Args:
            eye_points: Array of 6 eye landmark points
            
        Returns:
            float: EAR value
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
        return float(ear)
    
    def _calculate_blink_confidence(self, ear: float) -> float:
        """
        Calculate confidence in blink detection.
        
        Args:
            ear: Eye aspect ratio
            
        Returns:
            float: Blink confidence (0.0 to 1.0)
        """
        # Higher confidence for lower EAR values
        if ear < 0.15:
            return 0.95
        elif ear < 0.20:
            return 0.85
        elif ear < 0.25:
            return 0.75
        else:
            return 0.0
    
    def _calculate_eye_closure_duration(self) -> float:
        """
        Calculate current eye closure duration.
        
        Returns:
            float: Duration in seconds
        """
        if len(self.ear_history) < 2:
            return 0.0
        
        # Count consecutive frames with closed eyes
        consecutive_closed = 0
        for ear in reversed(self.ear_history):
            if ear < 0.25:  # Threshold for closed eyes
                consecutive_closed += 1
            else:
                break
        
        # Convert to seconds (assuming 30 FPS)
        duration = consecutive_closed / 30.0
        return duration
    
    def _analyze_eye_movement(self, left_eye: np.ndarray, right_eye: np.ndarray) -> str:
        """
        Analyze eye movement patterns.
        
        Args:
            left_eye: Left eye landmarks
            right_eye: Right eye landmarks
            
        Returns:
            str: Movement description
        """
        if len(self.landmark_history) < 2:
            return "stable"
        
        # Get previous eye positions
        prev_left_eye = self.landmark_history[-1][36:42]
        prev_right_eye = self.landmark_history[-1][42:48]
        
        # Calculate movement
        left_movement = np.linalg.norm(np.mean(left_eye, axis=0) - np.mean(prev_left_eye, axis=0))
        right_movement = np.linalg.norm(np.mean(right_eye, axis=0) - np.mean(prev_right_eye, axis=0))
        
        avg_movement = (left_movement + right_movement) / 2.0
        
        if avg_movement > 5.0:
            return "rapid"
        elif avg_movement > 2.0:
            return "moderate"
        else:
            return "stable"
    
    def _determine_gaze_direction(self, left_eye: np.ndarray, right_eye: np.ndarray) -> str:
        """
        Determine gaze direction.
        
        Args:
            left_eye: Left eye landmarks
            right_eye: Right eye landmarks
            
        Returns:
            str: Gaze direction
        """
        # Calculate eye centers
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        
        # Calculate relative positions
        left_iris = left_eye[3]  # Inner corner
        right_iris = right_eye[0]  # Outer corner
        
        # Determine direction based on iris position
        left_ratio = np.linalg.norm(left_iris - left_center) / np.linalg.norm(left_eye[0] - left_eye[3])
        right_ratio = np.linalg.norm(right_iris - right_center) / np.linalg.norm(right_eye[0] - right_eye[3])
        
        if left_ratio < 0.3 and right_ratio < 0.3:
            return "left"
        elif left_ratio > 0.7 and right_ratio > 0.7:
            return "right"
        else:
            return "center"
    
    def _analyze_mouth(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Analyze mouth region for yawn detection.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Dict: Mouth analysis results
        """
        analysis = {
            'mar': 0.0,
            'is_yawning': False,
            'yawn_severity': 'none',
            'mouth_open_ratio': 0.0,
            'lip_tension': 'normal'
        }
        
        try:
            # Extract mouth landmarks
            mouth_points = landmarks[48:60]  # Outer mouth points
            
            # Calculate MAR
            mar = self._calculate_mar(mouth_points)
            analysis['mar'] = mar
            
            # Detect yawning
            if mar > 29.0:
                analysis['is_yawning'] = True
                if mar > 50.0:
                    analysis['yawn_severity'] = 'severe'
                elif mar > 40.0:
                    analysis['yawn_severity'] = 'moderate'
                else:
                    analysis['yawn_severity'] = 'mild'
            
            # Calculate mouth open ratio
            analysis['mouth_open_ratio'] = self._calculate_mouth_open_ratio(mouth_points)
            
            # Analyze lip tension
            analysis['lip_tension'] = self._analyze_lip_tension(mouth_points)
            
        except Exception as e:
            print(f"Mouth analysis error: {e}")
        
        return analysis
    
    def _calculate_mar(self, mouth_points: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR).
        
        Args:
            mouth_points: Array of mouth landmark points
            
        Returns:
            float: MAR value
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
        return float(mar)
    
    def _calculate_mouth_open_ratio(self, mouth_points: np.ndarray) -> float:
        """
        Calculate mouth opening ratio.
        
        Args:
            mouth_points: Array of mouth landmark points
            
        Returns:
            float: Mouth open ratio
        """
        if len(mouth_points) < 12:
            return 0.0
        
        # Calculate vertical opening
        vertical_opening = np.linalg.norm(mouth_points[2] - mouth_points[6])
        
        # Calculate horizontal width
        horizontal_width = np.linalg.norm(mouth_points[0] - mouth_points[4])
        
        if horizontal_width == 0:
            return 0.0
        
        ratio = vertical_opening / horizontal_width
        return float(ratio)
    
    def _analyze_lip_tension(self, mouth_points: np.ndarray) -> str:
        """
        Analyze lip tension.
        
        Args:
            mouth_points: Array of mouth landmark points
            
        Returns:
            str: Lip tension description
        """
        if len(mouth_points) < 12:
            return "normal"
        
        # Calculate lip curvature
        top_lip_curve = np.linalg.norm(mouth_points[1] - mouth_points[5])
        bottom_lip_curve = np.linalg.norm(mouth_points[7] - mouth_points[11])
        
        # Determine tension based on curvature
        if top_lip_curve > 10.0 or bottom_lip_curve > 10.0:
            return "tense"
        elif top_lip_curve < 3.0 and bottom_lip_curve < 3.0:
            return "relaxed"
        else:
            return "normal"
    
    def _calculate_head_pose(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """
        Calculate head pose angles.
        
        Args:
            landmarks: Facial landmarks
            image_shape: Image dimensions
            
        Returns:
            Optional[Dict]: Head pose information
        """
        if len(landmarks) < 68:
            return None
        
        try:
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
                
                pitch = float(euler_angles[0])
                yaw = float(euler_angles[1])
                roll = float(euler_angles[2])
                
                # Determine head direction
                direction = "forward"
                if abs(yaw) > 15:
                    direction = "left" if yaw > 0 else "right"
                elif abs(pitch) > 15:
                    direction = "up" if pitch > 0 else "down"
                
                return {
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll,
                    "direction": direction,
                    "confidence": 0.8
                }
        
        except Exception as e:
            print(f"Head pose calculation error: {e}")
        
        return None
    
    def _analyze_facial_expression(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Analyze facial expression.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Dict: Expression analysis results
        """
        analysis = {
            'expression': 'neutral',
            'confidence': 0.0,
            'valence': 0.0,
            'arousal': 0.0
        }
        
        try:
            # Extract key facial features
            eyebrows = landmarks[17:27]  # Eyebrow points
            eyes = landmarks[36:48]      # Eye points
            mouth = landmarks[48:60]     # Mouth points
            
            # Analyze expression based on feature positions
            expression_score = self._calculate_expression_score(eyebrows, eyes, mouth)
            
            # Determine expression
            if expression_score > 0.7:
                analysis['expression'] = 'happy'
                analysis['valence'] = 0.8
                analysis['arousal'] = 0.6
            elif expression_score > 0.3:
                analysis['expression'] = 'slight_smile'
                analysis['valence'] = 0.5
                analysis['arousal'] = 0.3
            elif expression_score < -0.3:
                analysis['expression'] = 'frown'
                analysis['valence'] = -0.4
                analysis['arousal'] = 0.2
            else:
                analysis['expression'] = 'neutral'
                analysis['valence'] = 0.0
                analysis['arousal'] = 0.1
            
            analysis['confidence'] = abs(expression_score)
            
        except Exception as e:
            print(f"Expression analysis error: {e}")
        
        return analysis
    
    def _calculate_expression_score(self, eyebrows: np.ndarray, eyes: np.ndarray, mouth: np.ndarray) -> float:
        """
        Calculate facial expression score.
        
        Args:
            eyebrows: Eyebrow landmarks
            eyes: Eye landmarks
            mouth: Mouth landmarks
            
        Returns:
            float: Expression score (-1 to 1)
        """
        score = 0.0
        
        try:
            # Analyze mouth curvature (smile/frown)
            mouth_curve = self._calculate_mouth_curvature(mouth)
            score += mouth_curve * 0.6
            
            # Analyze eyebrow position
            eyebrow_position = self._calculate_eyebrow_position(eyebrows)
            score += eyebrow_position * 0.3
            
            # Analyze eye openness
            eye_openness = self._calculate_eye_openness(eyes)
            score += eye_openness * 0.1
            
        except Exception as e:
            print(f"Expression score calculation error: {e}")
        
        return float(np.clip(score, -1.0, 1.0))
    
    def _calculate_mouth_curvature(self, mouth: np.ndarray) -> float:
        """Calculate mouth curvature for smile detection."""
        if len(mouth) < 12:
            return 0.0
        
        # Calculate vertical distances at corners vs center
        left_corner = np.linalg.norm(mouth[2] - mouth[6])
        right_corner = np.linalg.norm(mouth[3] - mouth[5])
        center = np.linalg.norm(mouth[1] - mouth[7])
        
        if center == 0:
            return 0.0
        
        # Curvature is higher when corners are closer than center
        curvature = ((left_corner + right_corner) / 2.0) / center - 1.0
        return float(np.clip(curvature, -1.0, 1.0))
    
    def _calculate_eyebrow_position(self, eyebrows: np.ndarray) -> float:
        """Calculate eyebrow position."""
        if len(eyebrows) < 10:
            return 0.0
        
        # Calculate average eyebrow height
        avg_height = np.mean(eyebrows[:, 1])
        
        # Normalize to -1 to 1 range
        normalized_height = (avg_height - 100) / 50.0  # Adjust based on typical values
        return float(np.clip(normalized_height, -1.0, 1.0))
    
    def _calculate_eye_openness(self, eyes: np.ndarray) -> float:
        """Calculate eye openness."""
        if len(eyes) < 12:
            return 0.0
        
        # Calculate EAR for both eyes
        left_eye = eyes[0:6]
        right_eye = eyes[6:12]
        
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Normalize to -1 to 1 range
        normalized_ear = (avg_ear - 0.25) / 0.1  # Adjust based on typical EAR values
        return float(np.clip(normalized_ear, -1.0, 1.0))
    
    def _update_history(self, landmarks: np.ndarray, eye_analysis: Dict, mouth_analysis: Dict):
        """Update processing history."""
        # Update landmark history
        self.landmark_history.append(landmarks.copy())
        if len(self.landmark_history) > self.config['max_history_length']:
            self.landmark_history = self.landmark_history[-self.config['max_history_length']:]
        
        # Update EAR history
        self.ear_history.append(eye_analysis['avg_ear'])
        if len(self.ear_history) > self.config['max_history_length']:
            self.ear_history = self.ear_history[-self.config['max_history_length']:]
        
        # Update MAR history
        self.mar_history.append(mouth_analysis['mar'])
        if len(self.mar_history) > self.config['max_history_length']:
            self.mar_history = self.mar_history[-self.config['max_history_length']:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'total_frames_processed': len(self.landmark_history),
            'face_detection_rate': len([h for h in self.face_detection_history if h]) / max(1, len(self.face_detection_history)),
            'avg_processing_time': np.mean(self.face_detection_history) if self.face_detection_history else 0.0,
            'ear_statistics': {
                'mean': np.mean(self.ear_history) if self.ear_history else 0.0,
                'std': np.std(self.ear_history) if self.ear_history else 0.0,
                'min': np.min(self.ear_history) if self.ear_history else 0.0,
                'max': np.max(self.ear_history) if self.ear_history else 0.0
            },
            'mar_statistics': {
                'mean': np.mean(self.mar_history) if self.mar_history else 0.0,
                'std': np.std(self.mar_history) if self.mar_history else 0.0,
                'min': np.min(self.mar_history) if self.mar_history else 0.0,
                'max': np.max(self.mar_history) if self.mar_history else 0.0
            }
        }

def create_vision_processor(predictor_path: str = None) -> VisionProcessor:
    """
    Create a vision processor instance.
    
    Args:
        predictor_path: Path to facial landmark predictor
        
    Returns:
        VisionProcessor: Initialized vision processor
    """
    return VisionProcessor(predictor_path)

def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw facial landmarks on frame.
    
    Args:
        frame: Input frame
        landmarks: Facial landmarks
        color: Landmark color (B, G, R)
        
    Returns:
        np.ndarray: Frame with landmarks drawn
    """
    if landmarks is None:
        return frame
    
    result = frame.copy()
    
    for point in landmarks:
        x, y = int(point[0]), int(point[1])
        cv2.circle(result, (x, y), 2, color, -1)
    
    return result

def draw_eye_region(frame: np.ndarray, landmarks: np.ndarray, eye_indices: List[int], 
                   color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """
    Draw eye region on frame.
    
    Args:
        frame: Input frame
        landmarks: Facial landmarks
        eye_indices: Indices of eye landmarks
        color: Region color (B, G, R)
        
    Returns:
        np.ndarray: Frame with eye region drawn
    """
    if landmarks is None or len(landmarks) < max(eye_indices):
        return frame
    
    result = frame.copy()
    
    # Extract eye points
    eye_points = landmarks[eye_indices]
    
    # Draw eye region
    for i in range(len(eye_points)):
        pt1 = tuple(map(int, eye_points[i]))
        pt2 = tuple(map(int, eye_points[(i + 1) % len(eye_points)]))
        cv2.line(result, pt1, pt2, color, 2)
    
    return result 