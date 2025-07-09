#!/usr/bin/env python3
"""
Flask API Backend for Drowsiness Detection System
Highly compatible with existing detection scripts
"""

import os
import sys
from pathlib import Path

# Safety check - ensure we're running from the backend directory
def check_running_directory():
    """Check if we're running from the correct directory"""
    current_dir = Path.cwd()
    app_file = current_dir / "app.py"
    
    if not app_file.exists():
        print("ERROR: app.py not found in current directory!")
        print(f"Current directory: {current_dir}")
        print("\nTo fix this error:")
        print("1. Make sure you're in the backend directory:")
        print("   cd backend")
        print("2. Then run:")
        print("   python app.py")
        print("\nOr from the project root, use:")
        print("   python start_backend.py")
        print("\nExpected directory structure:")
        print("  Drowsiness_Detection-main/")
        print("  ├── backend/")
        print("  │   └── app.py  <- Run from here")
        print("  └── start_backend.py")
        sys.exit(1)

# Run the check
check_running_directory()

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import json
import threading
import time
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'drowsiness_detection_secret_key_2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///drowsiness_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Enable CORS for cross-origin requests
CORS(app)

# Initialize database
db = SQLAlchemy(app)

# Database Models
class DetectionEvent(db.Model):
    """Model for storing detection events"""
    id = db.Column(db.Integer, primary_key=True)
    event_type = db.Column(db.String(50), nullable=False)  # drowsiness, distraction, yawn
    severity = db.Column(db.String(20), nullable=False)    # low, medium, high
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    duration = db.Column(db.Float, default=0.0)  # duration in seconds
    details = db.Column(db.Text)  # JSON string for additional data
    user_id = db.Column(db.String(50), default='default_user')
    
    def to_dict(self):
        return {
            'id': self.id,
            'event_type': self.event_type,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat(),
            'duration': self.duration,
            'details': json.loads(self.details) if self.details else {},
            'user_id': self.user_id
        }

class UserSession(db.Model):
    """Model for tracking user sessions"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    total_events = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='active')  # active, completed, paused
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_events': self.total_events,
            'status': self.status
        }

class SystemConfig(db.Model):
    """Model for system configuration"""
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.Text, nullable=False)
    description = db.Column(db.String(200))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'key': self.key,
            'value': self.value,
            'description': self.description,
            'updated_at': self.updated_at.isoformat()
        }

# Global variables for real-time monitoring
active_sessions = {}
alert_callbacks = []

# Create database tables
with app.app_context():
    db.create_all()
    
    # Initialize default configuration
    default_configs = [
        ('EYE_AR_THRESH', '0.25', 'Eye Aspect Ratio threshold for drowsiness detection'),
        ('EYE_AR_CONSEC_FRAMES', '4', 'Consecutive frames threshold for drowsiness'),
        ('DISC_COUNT_THRES', '7', 'Distraction detection threshold'),
        ('MAR_THRES', '29', 'Mouth Aspect Ratio threshold'),
        ('ALERT_ENABLED', 'true', 'Enable audio alerts'),
        ('LOG_ENABLED', 'true', 'Enable event logging'),
        ('API_ENABLED', 'true', 'Enable API endpoints')
    ]
    
    for config_key, config_value, config_description in default_configs:
        if not SystemConfig.query.filter_by(key=config_key).first():
            config = SystemConfig(key=config_key, value=config_value, description=config_description)
            db.session.add(config)
    
    db.session.commit()

# Utility functions
def get_config(key, default=None):
    """Get configuration value"""
    config = SystemConfig.query.filter_by(key=key).first()
    return config.value if config else default

def set_config(key, value, description=None):
    """Set configuration value"""
    config = SystemConfig.query.filter_by(key=key).first()
    if config:
        config.value = value
        if description:
            config.description = description
    else:
        config = SystemConfig(key=key, value=value, description=description)
        db.session.add(config)
    db.session.commit()

def log_event(event_type, severity='medium', duration=0.0, details=None, user_id='default_user'):
    """Log detection event to database"""
    if get_config('LOG_ENABLED', 'true').lower() == 'true':
        event = DetectionEvent(
            event_type=event_type,
            severity=severity,
            duration=duration,
            details=json.dumps(details) if details else None,
            user_id=user_id
        )
        db.session.add(event)
        
        # Update session
        session = UserSession.query.filter_by(user_id=user_id, status='active').first()
        if session:
            session.total_events += 1
        else:
            session = UserSession(user_id=user_id, total_events=1)
            db.session.add(session)
        
        db.session.commit()
        return event.to_dict()
    return None

def broadcast_alert(event_type, severity, details=None):
    """Broadcast alert to connected clients"""
    alert_data = {
        'type': event_type,
        'severity': severity,
        'timestamp': datetime.utcnow().isoformat(),
        'details': details or {}
    }
    
    for callback in alert_callbacks:
        try:
            callback(alert_data)
        except Exception as e:
            print(f"Alert callback error: {e}")

# API Routes
@app.route('/')
def index():
    """API documentation page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'database': 'connected' if db.engine.pool.checkedin() > 0 else 'disconnected'
    })

@app.route('/api/events', methods=['POST'])
def log_detection_event():
    """Log a detection event"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        event_type = data.get('event_type', 'unknown')
        severity = data.get('severity', 'medium')
        duration = data.get('duration', 0.0)
        details = data.get('details', {})
        user_id = data.get('user_id', 'default_user')
        
        event = log_event(event_type, severity, duration, details, user_id)
        
        if event:
            # Broadcast alert if enabled
            if get_config('ALERT_ENABLED', 'true').lower() == 'true':
                broadcast_alert(event_type, severity, details)
            
            return jsonify({
                'success': True,
                'event': event,
                'message': f'{event_type} event logged successfully'
            })
        else:
            return jsonify({'error': 'Logging disabled'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/events', methods=['GET'])
def get_events():
    """Get detection events with filtering"""
    try:
        # Query parameters
        event_type = request.args.get('event_type')
        user_id = request.args.get('user_id', 'default_user')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = int(request.args.get('limit', 100))
        
        # Build query
        query = DetectionEvent.query
        
        if event_type:
            query = query.filter_by(event_type=event_type)
        
        if user_id:
            query = query.filter_by(user_id=user_id)
        
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                query = query.filter(DetectionEvent.timestamp >= start_dt)
            except ValueError:
                pass
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                query = query.filter(DetectionEvent.timestamp <= end_dt)
            except ValueError:
                pass
        
        # Order by timestamp and limit
        events = query.order_by(DetectionEvent.timestamp.desc()).limit(limit).all()
        
        return jsonify({
            'success': True,
            'events': [event.to_dict() for event in events],
            'count': len(events)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get user sessions"""
    try:
        user_id = request.args.get('user_id', 'default_user')
        sessions = UserSession.query.filter_by(user_id=user_id).order_by(UserSession.start_time.desc()).all()
        
        return jsonify({
            'success': True,
            'sessions': [session.to_dict() for session in sessions]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions', methods=['POST'])
def start_session():
    """Start a new user session"""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default_user')
        
        # End any existing active session
        active_session = UserSession.query.filter_by(user_id=user_id, status='active').first()
        if active_session:
            active_session.end_time = datetime.utcnow()
            active_session.status = 'completed'
        
        # Start new session
        new_session = UserSession(user_id=user_id, status='active')
        db.session.add(new_session)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'session': new_session.to_dict(),
            'message': 'Session started successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<int:session_id>', methods=['PUT'])
def end_session(session_id):
    """End a user session"""
    try:
        session = UserSession.query.get_or_404(session_id)
        session.end_time = datetime.utcnow()
        session.status = 'completed'
        db.session.commit()
        
        return jsonify({
            'success': True,
            'session': session.to_dict(),
            'message': 'Session ended successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_configs():
    """Get all configuration values"""
    try:
        configs = SystemConfig.query.all()
        return jsonify({
            'success': True,
            'configs': {config.key: config.value for config in configs}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config/<key>', methods=['GET'])
def get_config_value(key):
    """Get specific configuration value"""
    try:
        value = get_config(key)
        return jsonify({
            'success': True,
            'key': key,
            'value': value
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config/<key>', methods=['PUT'])
def update_config(key):
    """Update configuration value"""
    try:
        data = request.get_json()
        if not data or 'value' not in data:
            return jsonify({'error': 'Value is required'}), 400
        
        value = data['value']
        description = data.get('description')
        
        set_config(key, value, description)
        
        return jsonify({
            'success': True,
            'key': key,
            'value': value,
            'message': 'Configuration updated successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get system statistics"""
    try:
        user_id = request.args.get('user_id', 'default_user')
        
        # Get time range
        days = int(request.args.get('days', 7))
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Query events
        events = DetectionEvent.query.filter(
            DetectionEvent.user_id == user_id,
            DetectionEvent.timestamp >= start_date,
            DetectionEvent.timestamp <= end_date
        ).all()
        
        # Calculate statistics
        stats = {
            'total_events': len(events),
            'drowsiness_events': len([e for e in events if e.event_type == 'drowsiness']),
            'distraction_events': len([e for e in events if e.event_type == 'distraction']),
            'yawn_events': len([e for e in events if e.event_type == 'yawn']),
            'high_severity': len([e for e in events if e.severity == 'high']),
            'medium_severity': len([e for e in events if e.severity == 'medium']),
            'low_severity': len([e for e in events if e.severity == 'low']),
            'total_duration': sum(e.duration for e in events),
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': days
            }
        }
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/register', methods=['POST'])
def register_alert_callback():
    """Register for real-time alerts (WebSocket simulation)"""
    try:
        data = request.get_json()
        callback_url = data.get('callback_url')
        
        if not callback_url:
            return jsonify({'error': 'Callback URL is required'}), 400
        
        # Simple callback registration (in production, use WebSockets)
        alert_callbacks.append(lambda alert_data: print(f"Alert to {callback_url}: {alert_data}"))
        
        return jsonify({
            'success': True,
            'message': 'Alert callback registered successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Drowsiness Detection Backend...")
    print("API Documentation: http://localhost:8080")
    print("Health Check: http://localhost:8080/api/health")
    print("Database: drowsiness_detection.db")
    
    app.run(host='0.0.0.0', port=8080, debug=True) 