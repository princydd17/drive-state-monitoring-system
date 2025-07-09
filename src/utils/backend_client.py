#!/usr/bin/env python3
"""
Backend Client for Drowsiness Detection System
Integrates existing detection scripts with the Flask API backend
"""

import requests
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class DrowsinessDetectionClient:
    """Client for interacting with the Drowsiness Detection Backend API"""
    
    def __init__(self, base_url: str = "http://localhost:8080", user_id: str = "default_user"):
        """
        Initialize the client
        
        Args:
            base_url: Backend API base URL
            user_id: User identifier for session tracking
        """
        self.base_url = base_url.rstrip('/')
        self.user_id = user_id
        self.session_id = None
        self.is_connected = False
        self.retry_count = 0
        self.max_retries = 3
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to the backend API"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            if response.status_code == 200:
                self.is_connected = True
                print(f"Connected to backend at {self.base_url}")
                return True
            else:
                self.is_connected = False
                print(f"Backend health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.is_connected = False
            print(f"Cannot connect to backend at {self.base_url}: {e}")
            return False
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, **kwargs) -> Optional[Dict]:
        """Make HTTP request to the backend API with retry logic"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                if method.upper() == 'GET':
                    response = requests.get(url, timeout=10, **kwargs)
                elif method.upper() == 'POST':
                    response = requests.post(url, json=data, timeout=10, **kwargs)
                elif method.upper() == 'PUT':
                    response = requests.put(url, json=data, timeout=10, **kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                if response.status_code in [200, 201]:
                    return response.json()
                else:
                    print(f"API request failed: {response.status_code} - {response.text}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                print(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    print(f"All {self.max_retries} attempts failed")
                    return None
        
        return None
    
    def start_session(self) -> bool:
        """Start a new monitoring session"""
        data = {"user_id": self.user_id}
        result = self._make_request('POST', '/api/sessions', data)
        
        if result and result.get('success'):
            self.session_id = result['session']['id']
            print(f"Session started: {self.session_id}")
            return True
        else:
            print("Failed to start session")
            return False
    
    def end_session(self) -> bool:
        """End the current monitoring session"""
        if not self.session_id:
            print("No active session to end")
            return False
        
        result = self._make_request('PUT', f'/api/sessions/{self.session_id}')
        
        if result and result.get('success'):
            print(f"Session ended: {self.session_id}")
            self.session_id = None
            return True
        else:
            print("Failed to end session")
            return False
    
    def log_event(self, event_type: str, severity: str = "medium", duration: float = 0.0, 
                  details: Optional[Dict] = None) -> bool:
        """
        Log a detection event to the backend
        
        Args:
            event_type: Type of event (drowsiness, distraction, yawn)
            severity: Event severity (low, medium, high)
            duration: Event duration in seconds
            details: Additional event details
        """
        if not self.is_connected:
            print("Backend not connected, event not logged")
            return False
        
        data = {
            "event_type": event_type,
            "severity": severity,
            "duration": duration,
            "user_id": self.user_id,
            "details": details or {}
        }
        
        result = self._make_request('POST', '/api/events', data)
        
        if result and result.get('success'):
            print(f"Event logged: {event_type} ({severity})")
            return True
        else:
            print(f"Failed to log event: {event_type}")
            return False
    
    def log_drowsiness(self, ear: float, frame_count: int, duration: float = 0.0) -> bool:
        """Log drowsiness detection event"""
        severity = "high" if ear < 0.2 else "medium" if ear < 0.25 else "low"
        details = {
            "ear": ear,
            "frame_count": frame_count,
            "threshold": 0.25
        }
        return self.log_event("drowsiness", severity, duration, details)
    
    def log_distraction(self, direction: str, duration: float = 0.0) -> bool:
        """Log distraction detection event"""
        details = {
            "direction": direction,
            "detection_method": "gaze_analysis"
        }
        return self.log_event("distraction", "medium", duration, details)
    
    def log_yawn(self, mar: float, duration: float = 0.0) -> bool:
        """Log yawn detection event"""
        severity = "high" if mar > 35 else "medium" if mar > 29 else "low"
        details = {
            "mar": mar,
            "threshold": 29
        }
        return self.log_event("yawn", severity, duration, details)
    
    def get_statistics(self, days: int = 7) -> Optional[Dict]:
        """Get system statistics"""
        params = {"user_id": self.user_id, "days": days}
        return self._make_request('GET', '/api/stats', params=params)
    
    def get_events(self, event_type: Optional[str] = None, limit: int = 50) -> Optional[Dict]:
        """Get detection events"""
        params = {"user_id": self.user_id, "limit": limit}
        if event_type:
            params["event_type"] = event_type
        return self._make_request('GET', '/api/events', params=params)
    
    def update_config(self, key: str, value: str, description: Optional[str] = None) -> bool:
        """Update system configuration"""
        data = {"value": value}
        if description:
            data["description"] = description
        
        result = self._make_request('PUT', f'/api/config/{key}', data)
        return bool(result and result.get('success', False))
    
    def get_config(self, key: Optional[str] = None) -> Optional[Dict]:
        """Get system configuration"""
        if key:
            return self._make_request('GET', f'/api/config/{key}')
        else:
            return self._make_request('GET', '/api/config')

# Global client instance
_client_instance = None

def get_client(base_url: str = "http://localhost:8080", user_id: str = "default_user") -> DrowsinessDetectionClient:
    """Get or create a global client instance"""
    global _client_instance
    if _client_instance is None:
        _client_instance = DrowsinessDetectionClient(base_url, user_id)
    return _client_instance

def log_detection_event(event_type: str, severity: str = "medium", duration: float = 0.0, 
                       details: Optional[Dict] = None, user_id: str = "default_user") -> bool:
    """
    Convenience function to log detection events
    
    Args:
        event_type: Type of event (drowsiness, distraction, yawn)
        severity: Event severity (low, medium, high)
        duration: Event duration in seconds
        details: Additional event details
        user_id: User identifier
    """
    client = get_client(user_id=user_id)
    return client.log_event(event_type, severity, duration, details)

def log_drowsiness_event(ear: float, frame_count: int, duration: float = 0.0, 
                        user_id: str = "default_user") -> bool:
    """Convenience function to log drowsiness events"""
    client = get_client(user_id=user_id)
    return client.log_drowsiness(ear, frame_count, duration)

def log_distraction_event(direction: str, duration: float = 0.0, 
                         user_id: str = "default_user") -> bool:
    """Convenience function to log distraction events"""
    client = get_client(user_id=user_id)
    return client.log_distraction(direction, duration)

def log_yawn_event(mar: float, duration: float = 0.0, 
                  user_id: str = "default_user") -> bool:
    """Convenience function to log yawn events"""
    client = get_client(user_id=user_id)
    return client.log_yawn(mar, duration)

# Integration decorators for existing functions
def with_backend_logging(func):
    """Decorator to add backend logging to detection functions"""
    def wrapper(*args, **kwargs):
        # Call original function
        result = func(*args, **kwargs)
        
        # Extract event information from function name and result
        func_name = func.__name__.lower()
        
        if 'drowsiness' in func_name or 'drowsy' in func_name:
            # Try to extract EAR and frame count from result or args
            ear = kwargs.get('ear', 0.0)
            frame_count = kwargs.get('frame_count', 0)
            log_drowsiness_event(ear, frame_count)
        
        elif 'distraction' in func_name:
            # Try to extract direction from result or args
            direction = kwargs.get('direction', 'unknown')
            log_distraction_event(direction)
        
        elif 'yawn' in func_name:
            # Try to extract MAR from result or args
            mar = kwargs.get('mar', 0.0)
            log_yawn_event(mar)
        
        return result
    
    return wrapper

# Example usage and testing
if __name__ == "__main__":
    print("Testing Backend Client...")
    
    # Create client
    client = DrowsinessDetectionClient()
    
    if client.is_connected:
        # Start session
        client.start_session()
        
        # Log some test events
        client.log_drowsiness(0.15, 20, 5.2)
        client.log_distraction("left", 2.1)
        client.log_yawn(35.5, 1.8)
        
        # Get statistics
        stats = client.get_statistics(days=1)
        if stats and stats.get('success'):
            print("Statistics:", stats['statistics'])
        
        # End session
        client.end_session()
    else:
        print("Backend not available. Make sure to start the backend server first:")
        print("   cd backend && python app.py") 