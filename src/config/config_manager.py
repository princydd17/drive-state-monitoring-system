#!/usr/bin/env python3
"""
Configuration Management for Drowsiness Detection System
Handles system configuration, validation, and preset management
"""

import json
import os
import yaml
from typing import Dict, Any, Optional
import logging

class ConfigManager:
    """
    Configuration manager for drowsiness detection system.
    """
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self._get_default_config()
        self.load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            dict: Default configuration
        """
        return {
            # Camera settings
            "camera": {
                "index": 0,
                "width": 640,
                "height": 480,
                "fps": 30
            },
            
            # Detection thresholds
            "thresholds": {
                "ear": 0.25,
                "mar": 0.5,
                "consecutive_frames": 4,
                "distraction": 15,
                "blink_duration": 0.3
            },
            
            # Algorithm settings
            "algorithms": {
                "enable_ear": True,
                "enable_mar": True,
                "enable_head_pose": True,
                "enable_ml": True,
                "enable_ensemble": True,
                "enable_perclos": True
            },
            
            # Audio settings
            "audio": {
                "enable_alerts": True,
                "enable_voice_commands": True,
                "volume": 0.7,
                "alert_cooldown": 3.0,
                "alert_sounds": {
                    "low": "beep.mp3",
                    "medium": "hello.mp3",
                    "high": "hello1.mp3"
                }
            },
            
            # Display settings
            "display": {
                "mode": "full",  # full, minimal, none
                "show_landmarks": True,
                "show_metrics": True,
                "show_alerts": True,
                "window_title": "Drowsiness Detection"
            },
            
            # Data logging
            "logging": {
                "enable_csv_logging": True,
                "enable_backend_logging": True,
                "save_video": False,
                "video_path": "drowsiness_detection.mp4",
                "log_level": "INFO"
            },
            
            # ML settings
            "ml": {
                "model_path": None,
                "feature_scaling": True,
                "confidence_threshold": 0.6,
                "ensemble_weights": {
                    "ear": 0.5,
                    "blink": 0.3,
                    "head_pose": 0.2
                }
            },
            
            # Performance settings
            "performance": {
                "max_fps": 30,
                "buffer_size": 100,
                "processing_threads": 1,
                "memory_limit": 512  # MB
            },
            
            # User preferences
            "user": {
                "alert_sensitivity": 0.5,
                "preferred_alert_type": "sound",  # sound, voice, both
                "language": "en",
                "timezone": "UTC"
            }
        }
    
    def load_config(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Merge with default config
                self._merge_config(self.config, file_config)
                print(f"Configuration loaded from: {self.config_file}")
                return True
            else:
                print(f"Configuration file not found: {self.config_file}")
                print("Using default configuration")
                return False
                
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration")
            return False
    
    def save_config(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to: {self.config_file}")
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def _merge_config(self, default: Dict, override: Dict) -> None:
        """
        Recursively merge configuration dictionaries.
        
        Args:
            default: Default configuration
            override: Override configuration
        """
        for key, value in override.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., "camera.index")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., "camera.index")
            value: Value to set
            
        Returns:
            bool: True if successful, False otherwise
        """
        keys = key.split('.')
        config = self.config
        
        try:
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            return True
        except Exception as e:
            print(f"Error setting configuration: {e}")
            return False
    
    def validate_config(self) -> Dict[str, list]:
        """
        Validate configuration values.
        
        Returns:
            dict: Validation results with errors and warnings
        """
        errors = []
        warnings = []
        
        # Validate camera settings
        camera_index = self.get("camera.index")
        if not isinstance(camera_index, int) or camera_index < 0:
            errors.append("camera.index must be a non-negative integer")
        
        # Validate thresholds
        ear_threshold = self.get("thresholds.ear")
        if not isinstance(ear_threshold, (int, float)) or ear_threshold <= 0:
            errors.append("thresholds.ear must be a positive number")
        
        mar_threshold = self.get("thresholds.mar")
        if not isinstance(mar_threshold, (int, float)) or mar_threshold <= 0:
            errors.append("thresholds.mar must be a positive number")
        
        # Validate audio settings
        volume = self.get("audio.volume")
        if not isinstance(volume, (int, float)) or volume < 0 or volume > 1:
            errors.append("audio.volume must be between 0 and 1")
        
        # Validate performance settings
        max_fps = self.get("performance.max_fps")
        if not isinstance(max_fps, int) or max_fps <= 0:
            errors.append("performance.max_fps must be a positive integer")
        
        # Check for missing audio files
        alert_sounds = self.get("audio.alert_sounds", {})
        for severity, sound_file in alert_sounds.items():
            if not os.path.exists(sound_file):
                warnings.append(f"Alert sound file not found: {sound_file}")
        
        return {
            "errors": errors,
            "warnings": warnings
        }
    
    def get_preset_config(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get preset configuration.
        
        Args:
            preset_name: Name of preset
            
        Returns:
            dict: Preset configuration or None
        """
        presets = {
            "high_sensitivity": {
                "thresholds": {
                    "ear": 0.20,
                    "mar": 0.4,
                    "consecutive_frames": 2
                },
                "user": {
                    "alert_sensitivity": 0.8
                }
            },
            "low_sensitivity": {
                "thresholds": {
                    "ear": 0.30,
                    "mar": 0.6,
                    "consecutive_frames": 8
                },
                "user": {
                    "alert_sensitivity": 0.2
                }
            },
            "performance": {
                "algorithms": {
                    "enable_ml": False,
                    "enable_ensemble": False
                },
                "display": {
                    "mode": "minimal"
                },
                "performance": {
                    "max_fps": 60
                }
            },
            "research": {
                "logging": {
                    "enable_csv_logging": True,
                    "save_video": True
                },
                "algorithms": {
                    "enable_perclos": True
                },
                "display": {
                    "show_metrics": True
                }
            },
            "quiet": {
                "audio": {
                    "enable_alerts": False,
                    "enable_voice_commands": False
                },
                "display": {
                    "mode": "none"
                }
            }
        }
        
        return presets.get(preset_name)
    
    def apply_preset(self, preset_name: str) -> bool:
        """
        Apply preset configuration.
        
        Args:
            preset_name: Name of preset to apply
            
        Returns:
            bool: True if successful, False otherwise
        """
        preset = self.get_preset_config(preset_name)
        if preset is None:
            print(f"Preset not found: {preset_name}")
            return False
        
        self._merge_config(self.config, preset)
        print(f"Preset applied: {preset_name}")
        return True
    
    def export_config(self, output_file: str) -> bool:
        """
        Export configuration to file.
        
        Args:
            output_file: Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration exported to: {output_file}")
            return True
        except Exception as e:
            print(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, input_file: str) -> bool:
        """
        Import configuration from file.
        
        Args:
            input_file: Input file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(input_file, 'r') as f:
                imported_config = json.load(f)
            
            self._merge_config(self.config, imported_config)
            print(f"Configuration imported from: {input_file}")
            return True
        except Exception as e:
            print(f"Error importing configuration: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.config = self._get_default_config()
        print("Configuration reset to defaults")
    
    def print_config(self, section: str = None) -> None:
        """
        Print configuration.
        
        Args:
            section: Specific section to print (optional)
        """
        if section:
            config_to_print = self.get(section, {})
            print(f"\nConfiguration - {section}:")
        else:
            config_to_print = self.config
            print("\nFull Configuration:")
        
        print(json.dumps(config_to_print, indent=2))

class ConfigValidator:
    """
    Configuration validator with detailed validation rules.
    """
    
    @staticmethod
    def validate_camera_config(config: Dict) -> list:
        """Validate camera configuration."""
        errors = []
        
        if not isinstance(config.get("index"), int):
            errors.append("Camera index must be an integer")
        
        if config.get("width", 0) <= 0:
            errors.append("Camera width must be positive")
        
        if config.get("height", 0) <= 0:
            errors.append("Camera height must be positive")
        
        if config.get("fps", 0) <= 0:
            errors.append("Camera FPS must be positive")
        
        return errors
    
    @staticmethod
    def validate_thresholds_config(config: Dict) -> list:
        """Validate thresholds configuration."""
        errors = []
        
        if not isinstance(config.get("ear"), (int, float)) or config.get("ear", 0) <= 0:
            errors.append("EAR threshold must be a positive number")
        
        if not isinstance(config.get("mar"), (int, float)) or config.get("mar", 0) <= 0:
            errors.append("MAR threshold must be a positive number")
        
        if not isinstance(config.get("consecutive_frames"), int) or config.get("consecutive_frames", 0) <= 0:
            errors.append("Consecutive frames threshold must be a positive integer")
        
        return errors
    
    @staticmethod
    def validate_audio_config(config: Dict) -> list:
        """Validate audio configuration."""
        errors = []
        
        volume = config.get("volume", 0)
        if not isinstance(volume, (int, float)) or volume < 0 or volume > 1:
            errors.append("Audio volume must be between 0 and 1")
        
        cooldown = config.get("alert_cooldown", 0)
        if not isinstance(cooldown, (int, float)) or cooldown < 0:
            errors.append("Alert cooldown must be non-negative")
        
        return errors

# Utility functions
def create_config_file(config_file: str = "config.json") -> bool:
    """
    Create a new configuration file with defaults.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        manager = ConfigManager(config_file)
        return manager.save_config()
    except Exception as e:
        print(f"Error creating configuration file: {e}")
        return False

def load_preset_config(preset_name: str, config_file: str = "config.json") -> bool:
    """
    Load and apply a preset configuration.
    
    Args:
        preset_name: Name of preset
        config_file: Path to configuration file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        manager = ConfigManager(config_file)
        if manager.apply_preset(preset_name):
            return manager.save_config()
        return False
    except Exception as e:
        print(f"Error loading preset configuration: {e}")
        return False

def validate_config_file(config_file: str) -> Dict[str, list]:
    """
    Validate a configuration file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        dict: Validation results
    """
    try:
        manager = ConfigManager(config_file)
        return manager.validate_config()
    except Exception as e:
        return {
            "errors": [f"Error validating configuration: {e}"],
            "warnings": []
        } 