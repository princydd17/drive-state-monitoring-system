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
            'mar_threshold': 29,
            'consecutive_frames_threshold': 4,
            'distraction_threshold': 15,
            'enable_ai': True,
            'enable_multi_detection': True,
            'enable_voice_commands': True,
            'enable_api_logging': True,
            'alert_cooldown': 3.0,
            'display_mode': 'full',  # full, minimal, none
            'save_video': False,
            'video_path': 'driver_monitor.mp4'
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
        results = {
            'frame_number': self.frame_count,
            'timestamp': time.time(),
            'face_detected': False,
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
                'mar': 0.0,
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
                    results['metrics']['ear'] = eye_analysis['avg_ear']
                    
                    # Detect drowsiness
                    if eye_analysis['avg_ear'] < self.config['ear_threshold']:
                        results['drowsiness']['detected'] = True
                        results['drowsiness']['confidence'] = eye_analysis['blink_confidence']
                        
                        # Determine severity
                        if eye_analysis['avg_ear'] < 0.15:
                            results['drowsiness']['severity'] = 'high'
                        elif eye_analysis['avg_ear'] < 0.20:
                            results['drowsiness']['severity'] = 'medium'
                        else:
                            results['drowsiness']['severity'] = 'low'
                    
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
                    results['metrics']['mar'] = mouth_analysis['mar']
                    
                    if mouth_analysis['is_yawning']:
                        results['yawn']['detected'] = True
                        results['yawn']['severity'] = mouth_analysis['yawn_severity']
                        results['yawn']['confidence'] = 0.9
                
            except Exception as e:
                print(f"Vision processing error: {e}")
        
        # Use AI detector if available
        if self.ai_detector and results['face_detected']:
            try:
                # Extract features for AI
                features = self._extract_ai_features(results)
                ai_prediction, ai_confidence, ai_level = self.ai_detector.predict(features)
                
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
        
        # Send to API
        if self.api_client:
            try:
                self.api_client.log_event('detection', results)
            except Exception as e:
                print(f"API logging error: {e}")
        
        return results
    
    def _extract_ai_features(self, detection_result):
        """Extract features for AI prediction."""
        features = [
            detection_result['metrics']['ear'],
            0.0,  # Placeholder for right EAR
            detection_result['metrics']['ear'],
            0.0,  # EAR variance
            0.0,  # Blink frequency
            0.0,  # Eye closure duration
            detection_result['metrics']['head_pose'][0],  # Pitch
            detection_result['metrics']['head_pose'][1],  # Yaw
            detection_result['metrics']['head_pose'][2],  # Roll
            detection_result['metrics']['mar'],
            0.0   # Yawn frequency
        ]
        
        return np.array(features)
    
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
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 210), 
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
            'config': self.config
        }
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(final_stats, f, indent=2)
            print(f"Statistics saved to: {stats_file}")
        except Exception as e:
            print(f"Error saving statistics: {e}")

def main():
    """Main function to run driver monitor system."""
    print("Driver Monitor System")
    print("Advanced drowsiness and distraction detection")
    print("="*60)
    
    # Configuration
    config = {
        'camera_index': 0,
        'ear_threshold': 0.25,
        'mar_threshold': 29,
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