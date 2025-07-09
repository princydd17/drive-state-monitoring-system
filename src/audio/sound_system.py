#!/usr/bin/env python3
"""
Sound System - Advanced audio processing for driver monitoring
Provides voice commands, intelligent alerts, and audio feedback
"""

import pygame
import numpy as np
import threading
import time
import queue
import speech_recognition as sr
from gtts import gTTS
import os
import sys
import tempfile
from typing import Dict, List, Optional, Callable, Any
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class SoundProcessor:
    """
    Advanced audio processor for driver monitoring system.
    """
    
    def __init__(self, volume: float = 0.7):
        """
        Initialize sound processor.
        
        Args:
            volume: Initial volume level (0.0 to 1.0)
        """
        self.volume = np.clip(volume, 0.0, 1.0)
        self.recognizer = None
        self.microphone = None
        self.voice_thread = None
        self.is_listening = False
        self.command_queue = queue.Queue()
        self.alert_sounds = {}
        self.voice_callbacks = {}
        
        # Initialize components
        self._initialize_components()
        
        # Configuration
        self.config = {
            'voice_timeout': 1.0,
            'phrase_time_limit': 3.0,
            'ambient_noise_duration': 1.0,
            'energy_threshold': 4000,
            'dynamic_energy_threshold': True,
            'pause_threshold': 0.8
        }
    
    def _initialize_components(self):
        """Initialize audio components."""
        try:
            # Initialize pygame mixer
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            print("Pygame mixer initialized")
            
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = self.config['energy_threshold']
            self.recognizer.dynamic_energy_threshold = self.config['dynamic_energy_threshold']
            self.recognizer.pause_threshold = self.config['pause_threshold']
            
            # Initialize microphone
            self.microphone = sr.Microphone()
            print("Speech recognition initialized")
            
            # Load default alert sounds
            self._load_default_sounds()
            
        except Exception as e:
            print(f"Error initializing sound components: {e}")
    
    def _load_default_sounds(self):
        """Load default alert sounds."""
        try:
            # Get audio directory path
            audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
            
            # Try to load existing sound files
            sound_files = {
                'alert': 'beep.mp3',
                'warning': 'hello.mp3',
                'critical': 'hello1.mp3'
            }
            
            for sound_name, filename in sound_files.items():
                file_path = os.path.join(audio_dir, filename)
                if os.path.exists(file_path):
                    try:
                        sound = pygame.mixer.Sound(file_path)
                        self.alert_sounds[sound_name] = sound
                        print(f"Loaded sound: {filename}")
                    except Exception as e:
                        print(f"Error loading sound {filename}: {e}")
                        self._create_beep_sound(sound_name)
                else:
                    # Create a simple beep sound
                    self._create_beep_sound(sound_name)
            
        except Exception as e:
            print(f"Error loading default sounds: {e}")
    
    def _create_beep_sound(self, sound_name: str):
        """Create a simple beep sound."""
        try:
            # Generate a simple beep
            sample_rate = 22050
            duration = 0.5
            frequency = 800 if sound_name == 'alert' else 600 if sound_name == 'warning' else 400
            
            # Generate sine wave
            samples = int(sample_rate * duration)
            wave = np.sin(2 * np.pi * frequency * np.arange(samples) / sample_rate)
            
            # Convert to 16-bit integer
            wave = (wave * 32767).astype(np.int16)
            
            # Create pygame sound
            sound = pygame.sndarray.make_sound(wave)
            self.alert_sounds[sound_name] = sound
            
        except Exception as e:
            print(f"Error creating beep sound: {e}")
    
    def play_sound(self, sound_name: str, volume: Optional[float] = None):
        """
        Play a sound.
        
        Args:
            sound_name: Name of the sound to play
            volume: Volume level (0.0 to 1.0), uses default if None
        """
        if sound_name not in self.alert_sounds:
            print(f"Sound '{sound_name}' not found")
            return
        
        try:
            sound = self.alert_sounds[sound_name]
            play_volume = volume if volume is not None else self.volume
            sound.set_volume(play_volume)
            sound.play()
            
        except Exception as e:
            print(f"Error playing sound: {e}")
    
    def play_alert(self, alert_type: str = 'alert', severity: str = 'medium'):
        """
        Play an alert based on type and severity.
        
        Args:
            alert_type: Type of alert ('alert', 'warning', 'critical')
            severity: Severity level ('low', 'medium', 'high')
        """
        # Map severity to volume
        volume_map = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        volume = volume_map.get(severity, 0.6)
        
        # Map alert type to sound
        sound_map = {
            'alert': 'alert',
            'warning': 'warning',
            'critical': 'critical'
        }
        sound_name = sound_map.get(alert_type, 'alert')
        
        self.play_sound(sound_name, volume)
    
    def speak_text(self, text: str, language: str = 'en'):
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
            language: Language code
        """
        try:
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_filename = temp_file.name
            
            # Generate speech
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(temp_filename)
            
            # Play the audio
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up
            os.unlink(temp_filename)
            
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
    
    def start_voice_listening(self):
        """Start listening for voice commands."""
        if self.is_listening:
            return
        
        self.is_listening = True
        self.voice_thread = threading.Thread(target=self._voice_listening_loop, daemon=True)
        self.voice_thread.start()
        print("Voice listening started")
    
    def stop_voice_listening(self):
        """Stop listening for voice commands."""
        self.is_listening = False
        if self.voice_thread:
            self.voice_thread.join(timeout=1.0)
        print("Voice listening stopped")
    
    def _voice_listening_loop(self):
        """Main voice listening loop."""
        # Adjust for ambient noise
        if self.microphone and self.recognizer:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=self.config['ambient_noise_duration'])
        
        while self.is_listening:
            try:
                if self.microphone and self.recognizer:
                    with self.microphone as source:
                        audio = self.recognizer.listen(
                            source,
                            timeout=self.config['voice_timeout'],
                            phrase_time_limit=self.config['phrase_time_limit']
                        )
                    
                    # Recognize speech
                    try:
                        command = self.recognizer.recognize_google(audio).lower()
                        print(f"Voice command detected: {command}")
                        
                        # Process command
                        self._process_voice_command(command)
                        
                    except sr.UnknownValueError:
                        pass  # Speech was unintelligible
                    except sr.RequestError as e:
                        print(f"Speech recognition error: {e}")
                
            except sr.WaitTimeoutError:
                pass  # No speech detected within timeout
            except Exception as e:
                print(f"Voice listening error: {e}")
                time.sleep(0.1)
    
    def _process_voice_command(self, command: str):
        """
        Process voice command.
        
        Args:
            command: Recognized voice command
        """
        # Define command mappings
        command_mappings = {
            'stop': {'type': 'stop', 'confidence': 0.9},
            'quit': {'type': 'stop', 'confidence': 0.9},
            'exit': {'type': 'stop', 'confidence': 0.9},
            'pause': {'type': 'pause', 'confidence': 0.8},
            'resume': {'type': 'resume', 'confidence': 0.8},
            'status': {'type': 'status', 'confidence': 0.8},
            'volume': {'type': 'volume', 'confidence': 0.7},
            'help': {'type': 'help', 'confidence': 0.7},
            'louder': {'type': 'volume_up', 'confidence': 0.8},
            'quieter': {'type': 'volume_down', 'confidence': 0.8},
            'mute': {'type': 'mute', 'confidence': 0.8},
            'unmute': {'type': 'unmute', 'confidence': 0.8}
        }
        
        # Find best matching command
        best_match = None
        best_confidence = 0.0
        
        for cmd, info in command_mappings.items():
            if cmd in command:
                if info['confidence'] > best_confidence:
                    best_match = info
                    best_confidence = info['confidence']
        
        if best_match:
            command_info = {
                'type': best_match['type'],
                'confidence': best_match['confidence'],
                'raw_command': command,
                'timestamp': time.time()
            }
            
            # Add to command queue
            self.command_queue.put(command_info)
    
    def get_next_command(self, timeout: float = 0.1) -> Optional[Dict]:
        """
        Get next voice command from queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Optional[Dict]: Command information or None
        """
        try:
            return self.command_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def set_volume(self, volume: float):
        """
        Set system volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.volume = np.clip(volume, 0.0, 1.0)
        print(f"Volume set to: {self.volume:.2f}")
    
    def register_voice_callback(self, command_type: str, callback: Callable):
        """
        Register a callback for voice commands.
        
        Args:
            command_type: Type of command to listen for
            callback: Function to call when command is received
        """
        self.voice_callbacks[command_type] = callback
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get sound system status."""
        return {
            'volume': self.volume,
            'is_listening': self.is_listening,
            'available_sounds': list(self.alert_sounds.keys()),
            'pending_commands': self.command_queue.qsize(),
            'voice_callbacks': list(self.voice_callbacks.keys())
        }

class SmartAlertSystem:
    """
    Intelligent alert system for driver monitoring.
    """
    
    def __init__(self, sound_processor: SoundProcessor):
        """
        Initialize smart alert system.
        
        Args:
            sound_processor: Sound processor instance
        """
        self.sound_processor = sound_processor
        self.alert_history = []
        self.alert_cooldowns = {}
        self.alert_patterns = {}
        
        # Configuration
        self.config = {
            'cooldown_duration': 3.0,
            'escalation_threshold': 3,
            'pattern_detection_window': 60.0,
            'adaptive_volume': True
        }
        
        # Initialize alert patterns
        self._initialize_alert_patterns()
    
    def _initialize_alert_patterns(self):
        """Initialize alert patterns and cooldowns."""
        self.alert_cooldowns = {
            'drowsiness': {'last_alert': 0, 'count': 0},
            'distraction': {'last_alert': 0, 'count': 0},
            'yawn': {'last_alert': 0, 'count': 0}
        }
        
        self.alert_patterns = {
            'drowsiness': {
                'low': {'sound': 'alert', 'volume': 0.4, 'message': 'Stay alert'},
                'medium': {'sound': 'warning', 'volume': 0.6, 'message': 'You seem tired'},
                'high': {'sound': 'critical', 'volume': 0.8, 'message': 'Pull over immediately'}
            },
            'distraction': {
                'low': {'sound': 'alert', 'volume': 0.3, 'message': 'Focus on the road'},
                'medium': {'sound': 'warning', 'volume': 0.5, 'message': 'Keep your eyes on the road'},
                'high': {'sound': 'critical', 'volume': 0.7, 'message': 'Stop and refocus'}
            },
            'yawn': {
                'low': {'sound': 'alert', 'volume': 0.3, 'message': 'Take a break'},
                'medium': {'sound': 'warning', 'volume': 0.5, 'message': 'Consider stopping'},
                'high': {'sound': 'critical', 'volume': 0.7, 'message': 'Pull over and rest'}
            }
        }
    
    def trigger_alert(self, alert_type: str, severity: str = 'medium', 
                     custom_message: Optional[str] = None):
        """
        Trigger an intelligent alert.
        
        Args:
            alert_type: Type of alert ('drowsiness', 'distraction', 'yawn')
            severity: Severity level ('low', 'medium', 'high')
            custom_message: Custom message to speak
        """
        current_time = time.time()
        
        # Check cooldown
        if alert_type in self.alert_cooldowns:
            cooldown_info = self.alert_cooldowns[alert_type]
            time_since_last = current_time - cooldown_info['last_alert']
            
            if time_since_last < self.config['cooldown_duration']:
                return  # Still in cooldown
        
        # Get alert pattern
        if alert_type in self.alert_patterns and severity in self.alert_patterns[alert_type]:
            pattern = self.alert_patterns[alert_type][severity]
            
            # Play sound
            self.sound_processor.play_sound(pattern['sound'], pattern['volume'])
            
            # Speak message
            message = custom_message if custom_message else pattern['message']
            self.sound_processor.speak_text(message)
            
            # Update cooldown
            if alert_type in self.alert_cooldowns:
                self.alert_cooldowns[alert_type]['last_alert'] = current_time
                self.alert_cooldowns[alert_type]['count'] += 1
            
            # Log alert
            alert_info = {
                'type': alert_type,
                'severity': severity,
                'message': message,
                'timestamp': current_time
            }
            self.alert_history.append(alert_info)
            
            # Keep history manageable
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
            
            print(f"Alert triggered: {alert_type} - {severity} - {message}")
    
    def trigger_escalating_alert(self, alert_type: str, base_severity: str = 'medium'):
        """
        Trigger an escalating alert based on frequency.
        
        Args:
            alert_type: Type of alert
            base_severity: Base severity level
        """
        if alert_type not in self.alert_cooldowns:
            return
        
        cooldown_info = self.alert_cooldowns[alert_type]
        count = cooldown_info['count']
        
        # Determine escalation level
        if count >= self.config['escalation_threshold'] * 3:
            severity = 'high'
        elif count >= self.config['escalation_threshold'] * 2:
            severity = 'medium'
        else:
            severity = base_severity
        
        self.trigger_alert(alert_type, severity)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics."""
        current_time = time.time()
        
        # Calculate recent alerts
        recent_window = 3600  # 1 hour
        recent_alerts = [
            alert for alert in self.alert_history
            if current_time - alert['timestamp'] < recent_window
        ]
        
        # Group by type
        alerts_by_type = {}
        for alert in recent_alerts:
            alert_type = alert['type']
            if alert_type not in alerts_by_type:
                alerts_by_type[alert_type] = []
            alerts_by_type[alert_type].append(alert)
        
        # Calculate statistics
        stats = {
            'total_alerts': len(self.alert_history),
            'recent_alerts': len(recent_alerts),
            'alert_frequency': len(recent_alerts) / (recent_window / 3600),  # alerts per hour
            'alerts_by_type': {alert_type: len(alerts) for alert_type, alerts in alerts_by_type.items()},
            'cooldown_status': {
                alert_type: {
                    'last_alert': cooldown_info['last_alert'],
                    'count': cooldown_info['count'],
                    'time_since_last': current_time - cooldown_info['last_alert']
                }
                for alert_type, cooldown_info in self.alert_cooldowns.items()
            }
        }
        
        return stats
    
    def reset_alert_counters(self):
        """Reset alert counters and cooldowns."""
        for cooldown_info in self.alert_cooldowns.values():
            cooldown_info['count'] = 0
            cooldown_info['last_alert'] = 0
        
        print("Alert counters reset")

def create_sound_processor(volume: float = 0.7) -> SoundProcessor:
    """
    Create a sound processor instance.
    
    Args:
        volume: Initial volume level
        
    Returns:
        SoundProcessor: Initialized sound processor
    """
    return SoundProcessor(volume)

def create_smart_alert_system(sound_processor: SoundProcessor) -> SmartAlertSystem:
    """
    Create a smart alert system instance.
    
    Args:
        sound_processor: Sound processor instance
        
    Returns:
        SmartAlertSystem: Initialized smart alert system
    """
    return SmartAlertSystem(sound_processor) 