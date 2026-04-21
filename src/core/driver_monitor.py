#!/usr/bin/env python3
"""
Driver Monitor System - Advanced Drowsiness Detection
Enhanced monitoring system with multiple detection algorithms
"""

import cv2
import numpy as np
import time
import threading
import json
import os
import sys
from collections import deque
from datetime import datetime
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from core.driver_metrics import (
    eye_aspect_ratio, mouth_aspect_ratio, detect_drowsiness,
    detect_yawn, calculate_head_pose, detect_distraction,
    calculate_perclos, calculate_eye_closure_duration,
    LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES
)
from vision.vision_engine import create_vision_processor, draw_landmarks, draw_eye_region
from audio.sound_system import create_sound_processor, create_smart_alert_system
from ai.ai_detector import DriverAIDetector, create_multi_detector
from ai.hybrid_scorer import HybridWeights, combine_scores
from utils.api_client import APIClient

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class DriverMonitorSystem:
    """
    Advanced driver monitoring system with multiple detection algorithms.
    """
    
    def __init__(self, config=None):
        """
        Initialize driver monitor system.
        
        Args:
            config: Configuration dictionary
        """
        # Default configuration
        self.config = {
            'camera_index': 0,
            'ear_threshold': 0.25,
            'mar_threshold': 0.5,
            'consecutive_frames_threshold': 4,
            'distraction_threshold': 15,
            'enable_ai': True,
            'enable_multi_detection': True,
            'enable_voice_commands': True,
            'enable_api_logging': True,
            'alert_cooldown': 3.0,
            'minimum_drowsy_duration': 1.5,
            'minimum_yawn_duration': 0.8,
            'calibration_duration': 45.0,
            'smoothing_alpha': 0.25,
            'hysteresis_margin': 0.02,
            'event_window_seconds': 60.0,
            'hybrid_ml_weight': 0.6,
            'hybrid_rule_weight': 0.4,
            'ml_confidence_threshold': 0.6,
            'adaptive_weight_floor': 0.35,
            'fatigue_decay': 0.9,
            'display_mode': 'full',  # full, minimal, none
            'save_video': False,
            'video_path': 'driver_monitor.mp4',
            'hard_case_logging': True
        }
        
        if config:
            self.config.update(config)
        
        # Initialize components
        self._initialize_components()
        
        # State variables
        self.is_running = False
        self.frame_count = 0
        self.detection_history = []
        self.alert_history = []
        self.last_alert_time = 0.0
        self.fps_estimate = 30.0
        self._last_frame_ts = None
        self._runtime_features = {}

        # Session calibration and temporal tracking
        self.dynamic_thresholds = {
            'ear': self.config['ear_threshold'],
            'mar': self.config['mar_threshold']
        }
        self.is_calibrated = False
        self.calibration_started_at = None
        self.calibration_data = {'ear': [], 'mar': []}
        self.ear_history = deque(maxlen=900)
        self.mar_history = deque(maxlen=900)
        self.eye_closed_history = deque(maxlen=900)
        self.blink_timestamps = deque()
        self.yawn_timestamps = deque()
        self.closed_eye_started_at = None
        self.yawn_started_at = None
        self.yawn_recorded = False
        self.drowsy_state = False
        self.drowsy_started_at = None
        self.fatigue_score = 0.0
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'drowsiness_detections': 0,
            'distraction_detections': 0,
            'yawn_detections': 0,
            'alerts_triggered': 0,
            'start_time': None
        }
    
    def _initialize_components(self):
        """Initialize all system components."""
        print("Initializing Driver Monitor System...")
        
        # Initialize computer vision
        try:
            self.vision_processor = create_vision_processor()
            print("Vision processor initialized")
        except Exception as e:
            print(f"Error initializing vision processor: {e}")
            self.vision_processor = None
        
        # Initialize audio system
        try:
            self.sound_processor = create_sound_processor()
            self.alert_system = create_smart_alert_system(self.sound_processor)
            print("Sound system initialized")
        except Exception as e:
            print(f"Error initializing sound system: {e}")
            self.sound_processor = None
            self.alert_system = None
        
        # Initialize AI detector
        if self.config['enable_ai']:
            try:
                self.ai_detector = DriverAIDetector()
                print("AI detector initialized")
            except Exception as e:
                print(f"Error initializing AI detector: {e}")
                self.ai_detector = None
        
        # Initialize multi detector
        if self.config['enable_multi_detection']:
            try:
                self.multi_detector = create_multi_detector()
                print("Multi detector initialized")
            except Exception as e:
                print(f"Error initializing multi detector: {e}")
                self.multi_detector = None
        
        # Initialize API client
        if self.config['enable_api_logging']:
            try:
                self.api_client = APIClient()
                print("API client initialized")
            except Exception as e:
                print(f"Error initializing API client: {e}")
                self.api_client = None
        
        # Initialize video writer
        if self.config['save_video']:
            self.video_writer = None
        
        print("System initialization complete")
    
    def start_monitoring(self):
        """Start the driver monitoring system."""
        print("Starting Driver Monitor System...")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.config['camera_index'])
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize video writer
        if self.config['save_video']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.config['video_path'], fourcc, 30.0, (640, 480)
            )
        
        # Start voice command listening
        if self.config['enable_voice_commands'] and self.sound_processor:
            self.sound_processor.start_voice_listening()
        
        # Initialize statistics
        self.stats['start_time'] = time.time()
        self.is_running = True
        
        print("Monitoring started. Press 'q' to quit, 's' for status, 'h' for help")
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            self.stop_monitoring()
    
    def _main_loop(self):
        """Main monitoring loop."""
        consecutive_frames = 0
        last_alert_time = 0
        
        while self.is_running:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            self.frame_count += 1
            self.stats['total_frames'] += 1
            
            # Process frame
            detection_result = self._process_frame(frame)
            
            # Handle voice commands
            if self.config['enable_voice_commands'] and self.sound_processor:
                command = self.sound_processor.get_next_command(timeout=0.1)
                if command:
                    self._handle_voice_command(command)
            
            # Display results
            if self.config['display_mode'] != 'none':
                self._display_results(frame, detection_result)
            
            # Save video
            if self.config['save_video'] and self.video_writer:
                self.video_writer.write(frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._show_status()
            elif key == ord('h'):
                self._show_help()
            elif key == ord('a'):
                self._adjust_alert_volume()
        
        self.cap.release()
        if self.config['save_video'] and self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
    
    def _process_frame(self, frame):
        """
        Process a single frame for drowsiness detection.
        
        Args:
            frame: Input frame
            
        Returns:
            dict: Detection results
        """
        now = time.time()
        self._update_fps(now)
        if self.calibration_started_at is None:
            self.calibration_started_at = now

        results = {
            'frame_number': self.frame_count,
            'timestamp': now,
            'face_detected': False,
            'system_state': 'alert',
            'risk_score': 0.0,
            'rule_score': 0.0,
            'ml_score': 0.0,
            'hybrid_score': 0.0,
            'fatigue_score': 0.0,
            'ml_confidence': 0.0,
            'model_disagreement': False,
            'policy_action': 'none',
            'fallback_triggered': False,
            'drowsiness': {
                'detected': False,
                'severity': 'none',
                'confidence': 0.0
            },
            'distraction': {
                'detected': False,
                'direction': 'none',
                'confidence': 0.0
            },
            'yawn': {
                'detected': False,
                'severity': 'none',
                'confidence': 0.0
            },
            'metrics': {
                'ear': 0.0,
                'left_ear': 0.0,
                'right_ear': 0.0,
                'mar': 0.0,
                'blink_frequency': 0.0,
                'eye_closure_duration': 0.0,
                'yawn_frequency': 0.0,
                'head_pose': (0.0, 0.0, 0.0)
            }
        }
        
        # Use vision processor if available
        if self.vision_processor:
            try:
                vision_results = self.vision_processor.process_frame(frame)
                if vision_results['face_detected']:
                    results['face_detected'] = True
                    
                    # Extract metrics
                    eye_analysis = vision_results['eye_analysis']
                    left_ear = float(eye_analysis.get('left_ear', 0.0))
                    right_ear = float(eye_analysis.get('right_ear', 0.0))
                    
                    # Analyze head pose
                    if vision_results['head_pose']:
                        head_pose = vision_results['head_pose']
                        results['metrics']['head_pose'] = (head_pose['pitch'], head_pose['yaw'], head_pose['roll'])
                        
                        if head_pose['direction'] != 'forward':
                            results['distraction']['detected'] = True
                            results['distraction']['direction'] = head_pose['direction']
                            results['distraction']['confidence'] = 0.8
                    
                    # Analyze mouth
                    mouth_analysis = vision_results['mouth_analysis']
                    mar = float(mouth_analysis.get('mar', 0.0))

                    runtime = self._update_temporal_state(left_ear, right_ear, mar, now, True)
                    self._runtime_features = runtime
                    results['metrics']['left_ear'] = runtime['left_ear']
                    results['metrics']['right_ear'] = runtime['right_ear']
                    results['metrics']['ear'] = runtime['ear']
                    results['metrics']['mar'] = runtime['mar']
                    results['metrics']['blink_frequency'] = runtime['blink_frequency']
                    results['metrics']['eye_closure_duration'] = runtime['eye_closure_duration']
                    results['metrics']['yawn_frequency'] = runtime['yawn_frequency']

                    # Detect drowsiness using smoothed value + hysteresis + duration
                    drowsy_detected, drowsy_severity, drowsy_confidence = self._detect_drowsy_state(runtime, now)
                    results['drowsiness']['detected'] = drowsy_detected
                    results['drowsiness']['severity'] = drowsy_severity
                    results['drowsiness']['confidence'] = drowsy_confidence

                    # Detect yawn using duration/frequency windows
                    yawn_detected, yawn_severity, yawn_confidence = self._detect_yawn_state(runtime)
                    results['yawn']['detected'] = yawn_detected
                    results['yawn']['severity'] = yawn_severity
                    results['yawn']['confidence'] = yawn_confidence

                    # Apply session calibration while face is visible
                    self._update_calibration(runtime['ear'], runtime['mar'], now)
                
            except Exception as e:
                print(f"Vision processing error: {e}")
        else:
            self._runtime_features = self._update_temporal_state(0.0, 0.0, 0.0, now, False)
            self.drowsy_state = False
            self.drowsy_started_at = None
        
        # Use AI detector if available
        model_signals = []
        if self.ai_detector and results['face_detected']:
            try:
                # Extract features for AI
                features = self._extract_ai_features(results)
                ai_prediction, ai_confidence, ai_level = self.ai_detector.predict(features)
                model_signals.append((bool(ai_prediction), float(ai_confidence)))
                
                # Combine with vision results
                if ai_prediction:
                    results['drowsiness']['detected'] = True
                    results['drowsiness']['confidence'] = max(
                        results['drowsiness']['confidence'], ai_confidence
                    )
                    if ai_level != 'none':
                        results['drowsiness']['severity'] = ai_level
                
            except Exception as e:
                print(f"AI processing error: {e}")
        
        # Use multi detector if available
        if self.multi_detector and results['face_detected']:
            try:
                features = self._extract_ai_features(results)
                multi_prediction, multi_confidence, multi_level = self.multi_detector.predict(features)
                model_signals.append((bool(multi_prediction), float(multi_confidence)))
                
                # Combine results
                if multi_prediction:
                    results['drowsiness']['detected'] = True
                    results['drowsiness']['confidence'] = max(
                        results['drowsiness']['confidence'], multi_confidence
                    )
                    if multi_level != 'none':
                        results['drowsiness']['severity'] = multi_level
                
            except Exception as e:
                print(f"Multi detector processing error: {e}")
        
        # Log detection
        self.detection_history.append(results)
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
        
        # Update statistics
        if results['drowsiness']['detected']:
            self.stats['drowsiness_detections'] += 1
        if results['distraction']['detected']:
            self.stats['distraction_detections'] += 1
        if results['yawn']['detected']:
            self.stats['yawn_detections'] += 1

        rule_score = self._compute_rule_score(results)
        ml_score, ml_confidence, model_disagreement = self._compute_ml_score(results, model_signals)
        weights = HybridWeights(
            ml_weight=float(self.config.get('hybrid_ml_weight', 0.6)),
            rule_weight=float(self.config.get('hybrid_rule_weight', 0.4)),
        )
        hybrid_score = self._compute_hybrid_score(rule_score, ml_score, ml_confidence, weights)
        fatigue_score = self._update_fatigue_score(hybrid_score)

        system_state, risk_score = self._derive_state_and_risk(results, hybrid_score, fatigue_score)
        results['system_state'] = system_state
        results['risk_score'] = risk_score
        results['rule_score'] = rule_score
        results['ml_score'] = ml_score
        results['hybrid_score'] = hybrid_score
        results['fatigue_score'] = fatigue_score
        results['ml_confidence'] = ml_confidence
        results['model_disagreement'] = model_disagreement
        results['fallback_triggered'] = ml_confidence < float(self.config.get('ml_confidence_threshold', 0.6))
        results['policy_action'] = self._maybe_trigger_alert(results, now)
        self._log_hard_case(results)
        
        # Send to API
        if self.api_client:
            try:
                self.api_client.log_event('driver_state', self._build_backend_event_payload(results))
            except Exception as e:
                print(f"API logging error: {e}")
        
        return results
    
    def _extract_ai_features(self, detection_result):
        """Extract features for AI prediction."""
        runtime = self._runtime_features if self._runtime_features else {}
        left_ear = detection_result['metrics'].get('left_ear', runtime.get('left_ear', 0.0))
        right_ear = detection_result['metrics'].get('right_ear', runtime.get('right_ear', 0.0))
        avg_ear = detection_result['metrics'].get('ear', runtime.get('ear', 0.0))
        ear_variance = runtime.get('ear_variance', 0.0)
        blink_frequency = detection_result['metrics'].get('blink_frequency', runtime.get('blink_frequency', 0.0))
        eye_closure_duration = detection_result['metrics'].get('eye_closure_duration', runtime.get('eye_closure_duration', 0.0))
        yawn_frequency = detection_result['metrics'].get('yawn_frequency', runtime.get('yawn_frequency', 0.0))

        features = [
            left_ear,
            right_ear,
            avg_ear,
            ear_variance,
            blink_frequency,
            eye_closure_duration,
            detection_result['metrics']['head_pose'][0],  # Pitch
            detection_result['metrics']['head_pose'][1],  # Yaw
            detection_result['metrics']['head_pose'][2],  # Roll
            detection_result['metrics']['mar'],
            yawn_frequency
        ]
        
        return np.array(features)

    def _update_fps(self, now):
        """Estimate FPS from frame timestamps."""
        if self._last_frame_ts is not None:
            dt = max(now - self._last_frame_ts, 1e-3)
            instant_fps = 1.0 / dt
            self.fps_estimate = (0.9 * self.fps_estimate) + (0.1 * instant_fps)
        self._last_frame_ts = now

    def _update_temporal_state(self, left_ear, right_ear, mar, now, face_detected):
        """Update smoothed streams, blink/yawn timers, and event frequencies."""
        if face_detected:
            raw_ear = (left_ear + right_ear) / 2.0
            prev_ear = self.ear_history[-1] if self.ear_history else raw_ear
            prev_mar = self.mar_history[-1] if self.mar_history else mar
            alpha = float(self.config.get('smoothing_alpha', 0.25))
            smoothed_ear = (alpha * raw_ear) + ((1 - alpha) * prev_ear)
            smoothed_mar = (alpha * mar) + ((1 - alpha) * prev_mar)
            self.ear_history.append(smoothed_ear)
            self.mar_history.append(smoothed_mar)
        else:
            smoothed_ear = self.ear_history[-1] if self.ear_history else 0.0
            smoothed_mar = self.mar_history[-1] if self.mar_history else 0.0

        ear_thr = self.dynamic_thresholds['ear']
        mar_thr = self.dynamic_thresholds['mar']

        eye_closed = face_detected and smoothed_ear < ear_thr
        self.eye_closed_history.append(1 if eye_closed else 0)
        if eye_closed and self.closed_eye_started_at is None:
            self.closed_eye_started_at = now
        if not eye_closed and self.closed_eye_started_at is not None:
            closure_dur = now - self.closed_eye_started_at
            if 0.08 <= closure_dur <= 0.8:
                self.blink_timestamps.append(now)
            self.closed_eye_started_at = None

        yawn_active = face_detected and smoothed_mar > mar_thr
        if yawn_active and self.yawn_started_at is None:
            self.yawn_started_at = now
            self.yawn_recorded = False
        if yawn_active and self.yawn_started_at is not None and not self.yawn_recorded:
            if (now - self.yawn_started_at) >= float(self.config.get('minimum_yawn_duration', 0.8)):
                self.yawn_timestamps.append(now)
                self.yawn_recorded = True
        if not yawn_active:
            self.yawn_started_at = None
            self.yawn_recorded = False

        window = float(self.config.get('event_window_seconds', 60.0))
        while self.blink_timestamps and (now - self.blink_timestamps[0]) > window:
            self.blink_timestamps.popleft()
        while self.yawn_timestamps and (now - self.yawn_timestamps[0]) > window:
            self.yawn_timestamps.popleft()

        eye_closure_duration = 0.0
        if self.closed_eye_started_at is not None:
            eye_closure_duration = now - self.closed_eye_started_at

        ear_variance = float(np.var(self.ear_history)) if len(self.ear_history) > 1 else 0.0
        return {
            'left_ear': float(left_ear),
            'right_ear': float(right_ear),
            'ear': float(smoothed_ear),
            'mar': float(smoothed_mar),
            'ear_variance': ear_variance,
            'eye_closure_duration': float(eye_closure_duration),
            'blink_frequency': float(len(self.blink_timestamps) * (60.0 / window)),
            'yawn_frequency': float(len(self.yawn_timestamps) * (60.0 / window)),
            'yawn_active': yawn_active
        }

    def _update_calibration(self, ear, mar, now):
        """Collect baseline and adapt thresholds once per session."""
        if self.is_calibrated:
            return
        if (now - self.calibration_started_at) > float(self.config.get('calibration_duration', 45.0)):
            if len(self.calibration_data['ear']) > 30:
                base_ear = float(np.median(self.calibration_data['ear']))
                base_mar = float(np.median(self.calibration_data['mar'])) if self.calibration_data['mar'] else self.config['mar_threshold']
                self.dynamic_thresholds['ear'] = min(self.config['ear_threshold'], max(0.12, base_ear * 0.78))
                self.dynamic_thresholds['mar'] = max(self.config['mar_threshold'], base_mar * 1.45)
                self.is_calibrated = True
                print(
                    f"Calibration complete: EAR threshold={self.dynamic_thresholds['ear']:.3f}, "
                    f"MAR threshold={self.dynamic_thresholds['mar']:.3f}"
                )
            return

        if ear > 0 and mar > 0:
            self.calibration_data['ear'].append(ear)
            self.calibration_data['mar'].append(mar)

    def _detect_drowsy_state(self, runtime, now):
        """Drowsiness detection with smoothing, hysteresis and minimum duration."""
        ear = runtime['ear']
        enter_thr = self.dynamic_thresholds['ear']
        exit_thr = enter_thr + float(self.config.get('hysteresis_margin', 0.02))
        min_duration = float(self.config.get('minimum_drowsy_duration', 1.5))
        eye_closure_duration = runtime['eye_closure_duration']

        if not self.drowsy_state and ear < enter_thr and eye_closure_duration >= min_duration:
            self.drowsy_state = True
            self.drowsy_started_at = now
        elif self.drowsy_state and ear > exit_thr:
            self.drowsy_state = False
            self.drowsy_started_at = None

        if not self.drowsy_state:
            return False, 'none', 0.0

        if ear < max(0.12, enter_thr - 0.08):
            return True, 'high', 0.95
        if ear < max(0.15, enter_thr - 0.04):
            return True, 'medium', 0.85
        return True, 'low', 0.75

    def _detect_yawn_state(self, runtime):
        """Yawn detection using duration and frequency window."""
        if not runtime['yawn_active']:
            return False, 'none', 0.0
        yawn_freq = runtime['yawn_frequency']
        if yawn_freq >= 4:
            return True, 'high', 0.95
        if yawn_freq >= 2:
            return True, 'medium', 0.85
        return True, 'low', 0.75

    def _compute_rule_score(self, results):
        """Compute rule-driven risk score from deterministic cues."""
        if not results['face_detected']:
            return 0.35
        drowsy = 1.0 if results['drowsiness']['detected'] else 0.0
        yawn = 1.0 if results['yawn']['detected'] else 0.0
        distraction = 1.0 if results['distraction']['detected'] else 0.0
        closure = min(1.0, float(results['metrics']['eye_closure_duration']) / 2.0)
        return min(1.0, (0.5 * drowsy) + (0.2 * yawn) + (0.2 * distraction) + (0.1 * closure))

    def _compute_ml_score(self, results, model_signals):
        """Compute ML-driven score and confidence from detector votes."""
        if not results['face_detected'] or not model_signals:
            return 0.2, 0.2, False
        confidences = [c for _, c in model_signals]
        weighted_positive = sum((1.0 if p else 0.0) * c for p, c in model_signals)
        total_conf = max(sum(confidences), 1e-6)
        ml_score = float(weighted_positive / total_conf)
        ml_confidence = float(max(confidences))
        model_disagreement = len(set(p for p, _ in model_signals)) > 1
        return ml_score, ml_confidence, model_disagreement

    def _compute_hybrid_score(self, rule_score, ml_score, ml_confidence, weights):
        """Confidence-aware hybrid scoring with safe fallback to rule score."""
        conf_threshold = float(self.config.get('ml_confidence_threshold', 0.6))
        if ml_confidence < conf_threshold:
            return float(rule_score)
        adaptive_floor = float(self.config.get('adaptive_weight_floor', 0.35))
        adaptive_ml_weight = max(adaptive_floor, min(ml_confidence, 0.95))
        adaptive_rule_weight = 1.0 - adaptive_ml_weight
        adaptive_weights = HybridWeights(ml_weight=adaptive_ml_weight, rule_weight=adaptive_rule_weight)
        return combine_scores(rule_score, ml_score, adaptive_weights)

    def _update_fatigue_score(self, hybrid_score):
        """Update fatigue accumulation with exponential decay."""
        decay = float(self.config.get('fatigue_decay', 0.9))
        self.fatigue_score = min(1.0, max(0.0, (decay * self.fatigue_score) + ((1.0 - decay) * hybrid_score)))
        return self.fatigue_score

    def _derive_state_and_risk(self, results, hybrid_score, fatigue_score):
        """Derive unified driver state and a normalized risk score."""
        if not results['face_detected']:
            return 'no_face', 0.35

        risk_score = min(1.0, (0.7 * hybrid_score) + (0.3 * fatigue_score))

        if results['drowsiness']['detected']:
            if results['drowsiness']['severity'] == 'high':
                return 'fatigue_risk_high', max(risk_score, 0.85)
            return 'fatigue_risk_low', max(risk_score, 0.6)
        if results['distraction']['detected']:
            return 'distraction_transient', max(risk_score, 0.5)
        if results['yawn']['detected']:
            return 'fatigue_risk_low', max(risk_score, 0.45)
        return 'alert', risk_score

    def _build_backend_event_payload(self, results):
        """Build a standardized backend payload for driver-state events."""
        return {
            'schema_version': '1.0',
            'state': results['system_state'],
            'confidence': float(max(
                results['drowsiness']['confidence'],
                results['yawn']['confidence'],
                results['distraction']['confidence'],
                0.2 if not results['face_detected'] else 0.0
            )),
            'risk_score': float(results['risk_score']),
            'rule_score': float(results.get('rule_score', 0.0)),
            'ml_score': float(results.get('ml_score', 0.0)),
            'hybrid_score': float(results.get('hybrid_score', 0.0)),
            'fatigue_score': float(results.get('fatigue_score', 0.0)),
            'ml_confidence': float(results.get('ml_confidence', 0.0)),
            'model_disagreement': bool(results.get('model_disagreement', False)),
            'fallback_triggered': bool(results.get('fallback_triggered', False)),
            'model_version': getattr(self.ai_detector, 'model_version', 'legacy') if self.ai_detector else 'none',
            'policy_action': results.get('policy_action', 'none'),
            'signal_bundle': {
                'face_detected': results['face_detected'],
                'ear': float(results['metrics']['ear']),
                'mar': float(results['metrics']['mar']),
                'blink_frequency': float(results['metrics']['blink_frequency']),
                'yawn_frequency': float(results['metrics']['yawn_frequency']),
                'eye_closure_duration': float(results['metrics']['eye_closure_duration']),
                'head_pose': tuple(results['metrics']['head_pose'])
            },
            'detection': {
                'drowsiness': results['drowsiness'],
                'yawn': results['yawn'],
                'distraction': results['distraction']
            },
            'frame_number': results['frame_number'],
            'timestamp': results['timestamp'],
            'latency_ms': int((1.0 / max(self.fps_estimate, 1e-6)) * 1000)
        }

    def _maybe_trigger_alert(self, results, now):
        """Trigger alerts with global cooldown and minimum event persistence."""
        if (now - self.last_alert_time) < float(self.config.get('alert_cooldown', 3.0)):
            return 'cooldown'
        if not self.alert_system:
            return 'none'

        triggered = False
        action = 'none'
        if results['drowsiness']['detected'] and self._runtime_features.get('eye_closure_duration', 0.0) >= float(self.config.get('minimum_drowsy_duration', 1.5)):
            self.alert_system.trigger_escalating_alert('drowsiness', results['drowsiness']['severity'])
            triggered = True
            action = 'escalating_audio_alert'
        elif results['yawn']['detected']:
            self.alert_system.trigger_alert('yawn', results['yawn']['severity'])
            triggered = True
            action = 'audio_alert'
        elif results['distraction']['detected']:
            self.alert_system.trigger_alert('distraction', 'medium')
            triggered = True
            action = 'audio_alert'

        if triggered:
            self.last_alert_time = now
            self.stats['alerts_triggered'] += 1
            return action
        return 'none'
    
    def _display_results(self, frame, results):
        """Display detection results on frame."""
        # Draw face detection status
        if results['face_detected']:
            cv2.putText(frame, "Face: Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Face: Not Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw drowsiness status
        if results['drowsiness']['detected']:
            color = (0, 0, 255) if results['drowsiness']['severity'] == 'high' else (0, 165, 255)
            cv2.putText(frame, f"Drowsy: {results['drowsiness']['severity']}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw distraction status
        if results['distraction']['detected']:
            cv2.putText(frame, f"Distracted: {results['distraction']['direction']}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Draw yawn status
        if results['yawn']['detected']:
            cv2.putText(frame, f"Yawning: {results['yawn']['severity']}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw metrics
        if self.config['display_mode'] == 'full':
            cv2.putText(frame, f"EAR: {results['metrics']['ear']:.3f}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {results['metrics']['mar']:.3f}", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Blinks/min: {results['metrics']['blink_frequency']:.1f}", (10, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Yawns/min: {results['metrics']['yawn_frequency']:.1f}", (10, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Risk: {results.get('risk_score', 0.0):.2f} Fatigue: {results.get('fatigue_score', 0.0):.2f}", (10, 270),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f"ML conf: {results.get('ml_confidence', 0.0):.2f}", (10, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 330), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw controls
        cv2.putText(frame, "Q: Quit | S: Status | H: Help | A: Volume", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Driver Monitor System', frame)
    
    def _handle_voice_command(self, command):
        """Handle voice commands."""
        command_type = command['type']
        print(f"Voice command: {command_type}")
        
        if command_type == 'stop':
            self.stop_monitoring()
        elif command_type == 'pause':
            # Implement pause functionality
            pass
        elif command_type == 'resume':
            # Implement resume functionality
            pass
        elif command_type == 'status':
            self._show_status()
        elif command_type == 'volume':
            self._adjust_alert_volume()
        elif command_type == 'help':
            self._show_help()
    
    def _show_status(self):
        """Show system status."""
        runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        print("\n" + "="*50)
        print("SYSTEM STATUS")
        print("="*50)
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Total frames: {self.stats['total_frames']}")
        print(f"Drowsiness detections: {self.stats['drowsiness_detections']}")
        print(f"Distraction detections: {self.stats['distraction_detections']}")
        print(f"Yawn detections: {self.stats['yawn_detections']}")
        print(f"Alerts triggered: {self.stats['alerts_triggered']}")
        print(f"Calibrated: {self.is_calibrated}")
        print(f"Active EAR threshold: {self.dynamic_thresholds['ear']:.3f}")
        print(f"Active MAR threshold: {self.dynamic_thresholds['mar']:.3f}")
        
        if self.alert_system:
            status = self.alert_system.get_system_status()
            print(f"Alert frequency: {status['alert_statistics']['alert_frequency']:.2f} alerts/hour")
        
        print("="*50)
    
    def _show_help(self):
        """Show help information."""
        print("\n" + "="*50)
        print("HELP - Driver Monitor System")
        print("="*50)
        print("Controls:")
        print("  Q - Quit monitoring")
        print("  S - Show status")
        print("  H - Show this help")
        print("  A - Adjust alert volume")
        print("\nVoice Commands:")
        print("  'Stop' - Stop monitoring")
        print("  'Status' - Show status")
        print("  'Volume' - Adjust volume")
        print("  'Help' - Show help")
        print("\nDetection Features:")
        print("  - Eye aspect ratio analysis")
        print("  - Head pose estimation")
        print("  - Yawn detection")
        print("  - AI-based prediction")
        print("  - Multi-detection system")
        print("  - Smart alert system")
        print("="*50)
    
    def _adjust_alert_volume(self):
        """Adjust alert volume."""
        if self.sound_processor:
            current_volume = self.sound_processor.volume
            new_volume = (current_volume + 0.2) % 1.0
            self.sound_processor.set_volume(new_volume)
            print(f"Alert volume set to: {new_volume:.1f}")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        print("Stopping monitoring system...")
        self.is_running = False
        
        # Stop voice listening
        if self.sound_processor:
            self.sound_processor.stop_voice_listening()
        
        # Save final statistics
        self._save_statistics()
        
        print("Monitoring system stopped")
    
    def _save_statistics(self):
        """Save monitoring statistics."""
        stats_file = f"monitor_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        final_stats = {
            'session_info': {
                'start_time': self.stats['start_time'],
                'end_time': time.time(),
                'duration': time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
            },
            'detection_stats': self.stats,
            'config': self.config,
            'timeline': self._build_timeline()
        }
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(final_stats, f, indent=2)
            print(f"Statistics saved to: {stats_file}")
        except Exception as e:
            print(f"Error saving statistics: {e}")

    def _build_timeline(self):
        """Create compact timeline for plotting/reporting."""
        timeline = []
        for d in self.detection_history:
            timeline.append(
                {
                    'timestamp': d.get('timestamp'),
                    'frame_number': d.get('frame_number'),
                    'state': d.get('system_state'),
                    'risk_score': d.get('risk_score', 0.0),
                    'fatigue_score': d.get('fatigue_score', 0.0),
                    'ml_confidence': d.get('ml_confidence', 0.0),
                    'fallback_triggered': d.get('fallback_triggered', False),
                    'policy_action': d.get('policy_action', 'none'),
                }
            )
        return timeline

    def _log_hard_case(self, results):
        """Store low-confidence/disagreement cases for future retraining."""
        if not self.config.get('hard_case_logging', True):
            return
        low_conf = results.get('ml_confidence', 1.0) < float(self.config.get('ml_confidence_threshold', 0.6))
        disagreement = bool(results.get('model_disagreement', False))
        if not (low_conf or disagreement):
            return
        try:
            hard_case_dir = os.path.join("data", "hard_cases")
            os.makedirs(hard_case_dir, exist_ok=True)
            out_file = os.path.join(hard_case_dir, "hard_cases.jsonl")
            payload = self._build_backend_event_payload(results)
            payload['hard_case_reason'] = 'low_confidence' if low_conf else 'model_disagreement'
            with open(out_file, "a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            print(f"Hard-case logging error: {e}")

def main():
    """Main function to run driver monitor system."""
    print("Driver Monitor System")
    print("Advanced drowsiness and distraction detection")
    print("="*60)
    
    # Configuration
    config = {
        'camera_index': 0,
        'ear_threshold': 0.25,
        'mar_threshold': 0.5,
        'enable_ai': True,
        'enable_multi_detection': True,
        'enable_voice_commands': True,
        'enable_api_logging': True,
        'display_mode': 'full',
        'save_video': False
    }
    
    # Create and start monitor
    monitor = DriverMonitorSystem(config)
    monitor.start_monitoring()

if __name__ == "__main__":
    main() 