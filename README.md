# Drowsiness Driver Monitor System

A comprehensive, real-time drowsiness detection system with advanced computer vision, machine learning algorithms, voice commands, and backend integration for driver safety monitoring.

## 🚀 Features

### Core Detection
- **Multi-Algorithm Fusion**: EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), head pose estimation
- **Machine Learning Integration**: Random Forest and SVM models for enhanced accuracy
- **Real-time Processing**: 25-30 FPS performance with optimized algorithms
- **Facial Landmark Detection**: 68-point facial landmark tracking
- **Eye Tracking**: Precise eye state monitoring with blink detection

### Advanced Monitoring
- **Voice Commands**: Hands-free control with speech recognition
- **Intelligent Alerts**: Context-aware audio and visual warnings
- **Video Recording**: Automatic recording of drowsiness episodes
- **Data Analytics**: Comprehensive metrics and visualization
- **Session Management**: Track multiple driving sessions

### Backend Integration
- **RESTful API**: Flask-based backend with SQLite database
- **Event Logging**: Real-time event storage and retrieval
- **Analytics Dashboard**: Web-based data visualization
- **Configuration Management**: Remote configuration updates
- **Session Tracking**: Historical data analysis

### Configuration & Customization
- **Preset Modes**: Sensitivity, Performance, and Quiet modes
- **Real-time Tuning**: Adjustable detection parameters
- **Cross-platform**: Windows, macOS, and Linux support
- **Modular Architecture**: Easy to extend and customize

## 📁 Project Structure

```
Drowsiness_Detection-main/
├── backend/                 # Flask API backend
│   ├── app.py              # Main backend application
│   ├── requirements.txt    # Backend dependencies
│   └── templates/          # Web templates
├── src/                    # Main application source
│   ├── core/              # Core detection modules
│   │   ├── driver_monitor.py      # Main monitoring system
│   │   ├── driver_metrics.py      # Metrics calculation
│   │   └── distraction_detection.py # Distraction detection
│   ├── vision/            # Computer vision modules
│   │   └── vision_engine.py       # Vision processing engine
│   ├── audio/             # Audio processing
│   │   └── sound_system.py        # Audio alerts and voice commands
│   ├── ai/                # AI and ML modules
│   │   ├── ai_detector.py         # AI-based detection
│   │   └── *.dat                 # Pre-trained models
│   ├── utils/             # Utility modules
│   │   ├── api_client.py          # Backend API client
│   │   └── data_analyzer.py       # Data analysis utilities
│   └── config/            # Configuration management
│       └── config_manager.py      # Configuration handling
├── start_monitor.py       # Main startup script
├── start_backend.py       # Backend startup script
└── requirements.txt       # Main dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam access
- Microphone access (for voice commands)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/princydd17/drowsiness-driver-monitor-system.git
   cd drowsiness-driver-monitor-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies (macOS)**
   ```bash
   brew install cmake
   brew install boost
   ```

4. **Verify installation**
   ```bash
   python -c "import cv2, dlib, numpy; print('All dependencies installed successfully!')"
   ```

## 🚀 Usage

### Quick Start

1. **Start the backend server** (optional but recommended)
   ```bash
   python start_backend.py
   ```

2. **Start the monitoring system**
   ```bash
   python start_monitor.py
   ```

### Configuration

The system supports multiple configuration presets:

- **Sensitivity Mode**: High detection sensitivity for maximum safety
- **Performance Mode**: Balanced performance and accuracy
- **Quiet Mode**: Minimal alerts for less intrusive monitoring

### Voice Commands

Available voice commands:
- "Start monitoring" - Begin detection
- "Stop monitoring" - Pause detection
- "Increase sensitivity" - Make detection more sensitive
- "Decrease sensitivity" - Make detection less sensitive
- "Take screenshot" - Capture current frame
- "Start recording" - Begin video recording
- "Stop recording" - End video recording

## 🔧 Configuration

### Main Configuration Parameters

```python
# Detection thresholds
EAR_THRESHOLD = 0.2          # Eye aspect ratio threshold
MAR_THRESHOLD = 0.5          # Mouth aspect ratio threshold
BLINK_CONSEC_FRAMES = 2      # Consecutive frames for blink detection
DROWSY_CONSEC_FRAMES = 10    # Consecutive frames for drowsiness detection

# Performance settings
FRAME_RATE = 30              # Target frame rate
PROCESSING_INTERVAL = 1      # Processing interval in seconds

# Audio settings
ALERT_VOLUME = 0.7           # Alert volume (0.0 to 1.0)
VOICE_COMMANDS_ENABLED = True # Enable voice command recognition

# Backend settings
BACKEND_URL = "http://localhost:8080"  # Backend API URL
LOG_EVENTS = True            # Enable event logging to backend
```

### Configuration Presets

```python
# Sensitivity Mode
SENSITIVITY_PRESET = {
    "ear_threshold": 0.25,
    "mar_threshold": 0.6,
    "drowsy_consec_frames": 8,
    "alert_volume": 0.8
}

# Performance Mode
PERFORMANCE_PRESET = {
    "ear_threshold": 0.2,
    "mar_threshold": 0.5,
    "drowsy_consec_frames": 10,
    "alert_volume": 0.7
}

# Quiet Mode
QUIET_PRESET = {
    "ear_threshold": 0.15,
    "mar_threshold": 0.4,
    "drowsy_consec_frames": 15,
    "alert_volume": 0.5
}
```

## 🔍 Detection Algorithms

### 1. Eye Aspect Ratio (EAR)
- Calculates the ratio of eye height to width
- Detects eye closure and blink patterns
- Threshold-based classification

### 2. Mouth Aspect Ratio (MAR)
- Monitors mouth opening for yawning detection
- Helps identify fatigue indicators
- Combined with EAR for comprehensive detection

### 3. Head Pose Estimation
- 3D head pose calculation using facial landmarks
- Detects head nodding and tilting
- Provides additional drowsiness indicators

### 4. Machine Learning Detection
- Random Forest classifier for pattern recognition
- SVM model for binary classification
- Ensemble methods for improved accuracy

### 5. Multi-Algorithm Fusion
- Combines all detection methods
- Weighted voting system
- Reduces false positives and negatives

## 📊 Backend API

### Endpoints

- `POST /api/events` - Log detection events
- `GET /api/events` - Retrieve event history
- `POST /api/sessions` - Create new monitoring session
- `GET /api/sessions` - Get session data
- `GET /api/analytics` - Get analytics data
- `POST /api/config` - Update configuration

### Event Types

- `drowsiness_detected` - Drowsiness event
- `distraction_detected` - Distraction event
- `session_started` - Session start
- `session_ended` - Session end
- `alert_triggered` - Alert activation

## 📈 Data Analytics

### Metrics Calculated

- **PERCLOS**: Percentage of eye closure over time
- **Drowsiness Episodes**: Number and duration of drowsy periods
- **Severity Distribution**: Distribution of drowsiness severity levels
- **Session Statistics**: Overall session metrics
- **Trend Analysis**: Historical data trends

### Report Generation

- CSV export of session data
- Visualization charts and graphs
- Summary statistics
- Recommendations for improvement

## 🐛 Troubleshooting

### Common Issues

1. **Camera not found**
   - Check camera permissions
   - Verify camera is not in use by another application
   - Try different camera index (0, 1, 2)

2. **Performance issues**
   - Reduce frame rate in configuration
   - Lower processing resolution
   - Close other resource-intensive applications

3. **Audio not working**
   - Check system audio settings
   - Verify microphone permissions
   - Test with different audio devices

4. **Backend connection errors**
   - Ensure backend is running on correct port
   - Check firewall settings
   - Verify network connectivity

### Performance Optimization

- Use GPU acceleration if available
- Optimize camera resolution
- Adjust processing intervals
- Enable hardware acceleration

## 🔧 Development

### Adding New Features

1. **Create new detection algorithm**
   ```python
   class NewDetector:
       def __init__(self, config):
           self.config = config
       
       def detect(self, frame):
           # Implementation
           return result
   ```

2. **Integrate with main system**
   ```python
   # Add to driver_monitor.py
   from .new_detector import NewDetector
   
   # Initialize in __init__
   self.new_detector = NewDetector(config)
   
   # Use in detection loop
   result = self.new_detector.detect(frame)
   ```

### Testing

```bash
# Run compatibility tests
python tests/test_compatibility.py

# Run specific module tests
python -m pytest tests/
```

## 📝 License

This project is based on the original work from [sagardhande2942/Drowsiness_Detection](https://github.com/sagardhande2942/Drowsiness_Detection) and has been significantly enhanced with additional features and improvements.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the configuration documentation
- Open an issue on GitHub

## 🔮 Future Enhancements

- **Mobile App**: Companion mobile application
- **Cloud Integration**: Cloud-based data storage and analysis
- **Advanced ML**: Deep learning models for improved accuracy
- **IoT Integration**: Integration with vehicle systems
- **Real-time Alerts**: SMS/email notifications
- **Multi-camera Support**: Support for multiple camera feeds