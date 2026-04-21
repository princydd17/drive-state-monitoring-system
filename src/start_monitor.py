#!/usr/bin/env python3
"""
Start Monitor - Driver monitoring system startup script
Launches the enhanced driver monitoring system with all features
"""

import sys
import os
import argparse
import json
import yaml
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main startup function."""
    print("Driver Monitor System")
    print("Advanced drowsiness and distraction detection")
    print("="*60)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Driver Monitor System")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--camera", "-cam", type=int, default=0, help="Camera index")
    parser.add_argument("--sensitivity", "-s", choices=["low", "medium", "high"], 
                       default="medium", help="Detection sensitivity")
    parser.add_argument("--mode", "-m", choices=["performance", "accuracy", "quiet"], 
                       default="accuracy", help="Operation mode")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI detection")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice commands")
    parser.add_argument("--no-api", action="store_true", help="Disable API logging")
    parser.add_argument("--save-video", action="store_true", help="Save video recording")
    parser.add_argument("--display", choices=["full", "minimal", "none"], 
                       default="full", help="Display mode")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_configuration(args)
    
    # Validate configuration
    if not validate_config(config):
        print("Configuration validation failed")
        return 1
    
    # Start monitoring system
    try:
        start_monitoring_system(config)
        return 0
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        return 0
    except Exception as e:
        print(f"Error starting monitoring system: {e}")
        return 1

def load_configuration(args):
    """Load configuration from file and command line arguments."""
    config = {
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
        'display_mode': 'full',
        'save_video': False,
        'video_path': 'driver_monitor.mp4',
        'sensitivity': 'medium',
        'operation_mode': 'accuracy'
    }
    
    # Load from config file if specified
    if args.config and os.path.exists(args.config):
        try:
            file_config = load_config_file(args.config)
            apply_config_aliases(file_config)
            config.update(file_config)
            print(f"Loaded configuration from {args.config}")
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    # Override with command line arguments
    config['camera_index'] = args.camera
    config['sensitivity'] = args.sensitivity
    config['operation_mode'] = args.mode
    config['enable_ai'] = not args.no_ai
    config['enable_voice_commands'] = not args.no_voice
    config['enable_api_logging'] = not args.no_api
    config['save_video'] = args.save_video
    config['display_mode'] = args.display
    
    # Apply sensitivity settings
    apply_sensitivity_settings(config)
    
    # Apply operation mode settings
    apply_operation_mode_settings(config)
    
    return config

def load_config_file(config_path):
    """Load configuration from JSON or YAML file."""
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, "r") as f:
        if ext in [".yaml", ".yml"]:
            loaded = yaml.safe_load(f)
        else:
            loaded = json.load(f)
    return loaded or {}

def apply_config_aliases(config):
    """Map shorthand config keys to monitor runtime config keys."""
    # Keep compatibility with stability experiment presets.
    if "smoothing_alpha" in config and "hybrid_smoothing_alpha" not in config:
        config["hybrid_smoothing_alpha"] = config["smoothing_alpha"]

    if "min_state_duration" in config:
        config.setdefault("min_state_duration_default", config["min_state_duration"])
        config.setdefault("min_state_duration_drowsy", config["min_state_duration"])
        config.setdefault("min_state_duration_distraction", config["min_state_duration"])
        config.setdefault("min_state_duration_no_face", config["min_state_duration"])

    if "alert_duration" in config and "alert_min_state_duration" not in config:
        config["alert_min_state_duration"] = config["alert_duration"]

    if "cooldown" in config and "alert_cooldown" not in config:
        config["alert_cooldown"] = config["cooldown"]

def apply_sensitivity_settings(config):
    """Apply sensitivity-based configuration adjustments."""
    sensitivity = config['sensitivity']
    
    if sensitivity == 'low':
        config['ear_threshold'] = 0.20
        config['mar_threshold'] = 0.6
        config['consecutive_frames_threshold'] = 6
        config['distraction_threshold'] = 20
        print("Applied low sensitivity settings")
    elif sensitivity == 'high':
        config['ear_threshold'] = 0.30
        config['mar_threshold'] = 0.4
        config['consecutive_frames_threshold'] = 2
        config['distraction_threshold'] = 10
        print("Applied high sensitivity settings")
    else:  # medium
        print("Applied medium sensitivity settings")

def apply_operation_mode_settings(config):
    """Apply operation mode-based configuration adjustments."""
    mode = config['operation_mode']
    
    if mode == 'performance':
        config['enable_ai'] = False
        config['enable_multi_detection'] = False
        config['display_mode'] = 'minimal'
        config['enable_voice_commands'] = False
        print("Applied performance mode settings")
    elif mode == 'quiet':
        config['enable_voice_commands'] = False
        config['display_mode'] = 'none'
        config['alert_cooldown'] = 5.0
        print("Applied quiet mode settings")
    else:  # accuracy
        print("Applied accuracy mode settings")

def validate_config(config):
    """Validate configuration parameters."""
    try:
        # Check camera index
        if config['camera_index'] < 0:
            print("Error: Camera index must be non-negative")
            return False
        
        # Check thresholds
        if config['ear_threshold'] <= 0 or config['ear_threshold'] > 1:
            print("Error: EAR threshold must be between 0 and 1")
            return False
        
        if config['mar_threshold'] <= 0 or config['mar_threshold'] > 2:
            print("Error: MAR threshold must be in a ratio range (0, 2]")
            return False
        
        # Check cooldown
        if config['alert_cooldown'] < 0:
            print("Error: Alert cooldown must be non-negative")
            return False
        
        # Check video path
        if config['save_video']:
            video_dir = os.path.dirname(config['video_path'])
            if video_dir and not os.path.exists(video_dir):
                try:
                    os.makedirs(video_dir)
                except Exception as e:
                    print(f"Error creating video directory: {e}")
                    return False
        
        print("Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"Configuration validation error: {e}")
        return False

def start_monitoring_system(config):
    """Start the driver monitoring system."""
    print("Starting Driver Monitor System...")
    print(f"Camera: {config['camera_index']}")
    print(f"Sensitivity: {config['sensitivity']}")
    print(f"Mode: {config['operation_mode']}")
    print(f"AI Detection: {'Enabled' if config['enable_ai'] else 'Disabled'}")
    print(f"Voice Commands: {'Enabled' if config['enable_voice_commands'] else 'Disabled'}")
    print(f"API Logging: {'Enabled' if config['enable_api_logging'] else 'Disabled'}")
    print(f"Display Mode: {config['display_mode']}")
    print(f"Video Recording: {'Enabled' if config['save_video'] else 'Disabled'}")
    print("-" * 60)
    
    # Import and start monitor
    try:
        from core.driver_monitor import DriverMonitorSystem
        
        # Create monitor instance
        monitor = DriverMonitorSystem(config)
        
        # Start monitoring
        monitor.start_monitoring()
        
    except ImportError as e:
        print(f"Error importing monitoring modules: {e}")
        print("Please ensure all required dependencies are installed")
        raise
    except Exception as e:
        print(f"Error starting monitoring system: {e}")
        raise

def show_help():
    """Show detailed help information."""
    help_text = """
Driver Monitor System - Help

USAGE:
    python start_monitor.py [OPTIONS]

OPTIONS:
    -c, --config FILE       Load configuration from JSON file
    -cam, --camera INDEX    Camera index (default: 0)
    -s, --sensitivity LEVEL Detection sensitivity (low/medium/high)
    -m, --mode MODE         Operation mode (performance/accuracy/quiet)
    --no-ai                 Disable AI detection
    --no-voice              Disable voice commands
    --no-api                Disable API logging
    --save-video            Save video recording
    --display MODE          Display mode (full/minimal/none)

SENSITIVITY LEVELS:
    low     - Fewer alerts, higher thresholds
    medium  - Balanced detection (default)
    high    - More alerts, lower thresholds

OPERATION MODES:
    performance - Optimized for speed, minimal features
    accuracy    - Full feature set, maximum accuracy (default)
    quiet       - Minimal audio/visual output

DISPLAY MODES:
    full    - Show all information and controls
    minimal - Show essential information only
    none    - No visual display

VOICE COMMANDS:
    "Stop"      - Stop monitoring
    "Status"    - Show system status
    "Volume"    - Adjust alert volume
    "Help"      - Show help information

CONTROLS:
    Q - Quit monitoring
    S - Show status
    H - Show help
    A - Adjust alert volume

EXAMPLES:
    python start_monitor.py
    python start_monitor.py --sensitivity high --mode performance
    python start_monitor.py --config my_config.json --save-video
    python start_monitor.py --no-voice --display minimal

CONFIGURATION FILE FORMAT:
    {
        "camera_index": 0,
        "ear_threshold": 0.25,
        "mar_threshold": 0.5,
        "enable_ai": true,
        "enable_voice_commands": true,
        "display_mode": "full"
    }
"""
    print(help_text)

if __name__ == "__main__":
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
        sys.exit(0)
    
    # Run main function
    sys.exit(main()) 