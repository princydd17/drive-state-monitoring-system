# Driver Monitor System

Advanced driver drowsiness and distraction detection system with multiple detection algorithms, AI integration, voice commands, and intelligent alerts.

## Features

- **Multi-Algorithm Detection**: EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), head pose estimation
- **AI-Powered Detection**: Machine learning models (Random Forest, SVM) with ensemble voting
- **Voice Commands**: Hands-free control with speech recognition
- **Intelligent Alerts**: Smart alert system with escalation and cooldown management
- **Real-time Processing**: 25-30 FPS processing with optimized performance
- **Data Analytics**: Comprehensive data collection and analysis
- **Backend Integration**: RESTful API for data logging and session management
- **Configurable**: Multiple sensitivity levels and operation modes

## System Architecture

```
Driver Monitor System
├── driver_monitor.py          # Main monitoring system
├── driver_metrics.py          # Core detection algorithms
├── vision_engine.py           # Computer vision processing
├── sound_system.py            # Audio processing and voice commands
├── ai_detector.py             # Machine learning detection
├── data_analyzer.py           # Data analysis and visualization
├── config_manager.py          # Configuration management
├── start_monitor.py           # Startup script
└── backend/                   # Backend API services
    ├── app.py                 # Flask API server
    ├── requirements.txt       # Backend dependencies
    └── templates/             # Web interface
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Drowsiness_Detection-main
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required model files**:
   - `shape_predictor_68_face_landmarks.dat` (dlib facial landmark predictor)
   - `shape_predictor_5_face_landmarks.dat` (alternative predictor)

## Quick Start

### Basic Usage
```bash
python start_monitor.py
```

### Advanced Usage
```bash
# High sensitivity mode
python start_monitor.py --sensitivity high

# Performance mode (faster, fewer features)
python start_monitor.py --mode performance

# Custom configuration
python start_monitor.py --config my_config.json --save-video

# Minimal display with voice commands disabled
python start_monitor.py --display minimal --no-voice
```

## Configuration

### Command Line Options
- `--camera INDEX`: Camera index (default: 0)
- `--sensitivity LEVEL`: Detection sensitivity (low/medium/high)
- `--mode MODE`: Operation mode (performance/accuracy/quiet)
- `--no-ai`: Disable AI detection
- `--no-voice`: Disable voice commands
- `--no-api`: Disable API logging
- `--save-video`: Save video recording
- `--display MODE`: Display mode (full/minimal/none)

### Configuration File
Create a JSON configuration file:
```json
{
    "camera_index": 0,
    "ear_threshold": 0.25,
    "mar_threshold": 29,
    "enable_ai": true,
    "enable_voice_commands": true,
    "display_mode": "full",
    "save_video": false
}
```

## Detection Algorithms

### 1. Eye Aspect Ratio (EAR)
- Measures eye openness using facial landmarks
- Detects drowsiness based on eye closure patterns
- Configurable thresholds for sensitivity

### 2. Mouth Aspect Ratio (MAR)
- Detects yawning and mouth movements
- Indicates fatigue and drowsiness
- Separate threshold configuration

### 3. Head Pose Estimation
- 3D head pose calculation using PnP
- Detects distraction and attention direction
- Pitch, yaw, and roll angle analysis

### 4. AI-Powered Detection
- Random Forest classifier for drowsiness prediction
- SVM model for additional validation
- Ensemble voting for improved accuracy
- Feature-based machine learning approach

## Voice Commands

The system supports hands-free voice control:
- **"Stop"**: Stop monitoring
- **"Status"**: Show system status
- **"Volume"**: Adjust alert volume
- **"Help"**: Show help information

## Alert System

### Smart Alerts
- **Cooldown Management**: Prevents alert spam
- **Escalation**: Increases alert intensity with frequency
- **Severity Levels**: Low, medium, high based on detection confidence
- **Custom Messages**: Context-aware alert messages

### Alert Types
- **Drowsiness**: Eye closure and fatigue detection
- **Distraction**: Head pose and attention monitoring
- **Yawning**: Mouth movement and fatigue indicators

## Data Analytics

### Metrics Collected
- Eye aspect ratios over time
- Head pose angles and movements
- Alert frequency and patterns
- Detection confidence scores
- System performance metrics

### Analysis Features
- PERCLOS calculation (Percentage of Eye Closure)
- Drowsiness episode detection
- Severity distribution analysis
- Performance trend analysis
- Report generation

## Backend Integration

### API Endpoints
- `POST /api/events`: Log detection events
- `GET /api/sessions`: Retrieve session data
- `POST /api/sessions`: Create new sessions
- `GET /api/analytics`: Get analytics data
- `GET /api/config`: Get configuration

### Data Storage
- SQLite database for event logging
- CSV export for data analysis
- Session management and tracking

## Performance Optimization

### Operation Modes
1. **Performance Mode**: Optimized for speed
   - Disabled AI detection
   - Minimal display
   - Reduced feature set

2. **Accuracy Mode**: Maximum detection accuracy
   - Full feature set
   - AI-powered detection
   - Comprehensive analysis

3. **Quiet Mode**: Minimal audio/visual output
   - Disabled voice commands
   - No visual display
   - Extended alert cooldowns

## Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Check camera index with `--camera` option
   - Ensure camera is not in use by other applications

2. **Model files missing**:
   - Download required dlib predictor files
   - Place in project root directory

3. **Performance issues**:
   - Use performance mode: `--mode performance`
   - Disable AI detection: `--no-ai`
   - Reduce display mode: `--display minimal`

4. **Audio issues**:
   - Check microphone permissions
   - Disable voice commands: `--no-voice`

### System Requirements
- Python 3.7+
- OpenCV 4.5+
- dlib
- Webcam or camera device
- Microphone (for voice commands)

## Development

### Adding New Features
1. Create new module in appropriate directory
2. Update imports in main system
3. Add configuration options
4. Update documentation

### Testing
- Test with different lighting conditions
- Validate detection accuracy
- Performance benchmarking
- User experience testing

## License

This project is based on the original drowsiness detection system and has been enhanced with additional features and improvements.

## Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes
4. Test thoroughly
5. Submit pull request

## Support

For issues and questions:
1. Check troubleshooting section
2. Review configuration options
3. Test with different settings
4. Report bugs with detailed information

## Future Enhancements

- **Deep Learning Models**: CNN-based detection
- **Mobile Integration**: Smartphone app support
- **Cloud Analytics**: Remote data analysis
- **Multi-Camera Support**: Multiple driver monitoring
- **Integration APIs**: Fleet management systems
- **Real-time Alerts**: SMS/email notifications