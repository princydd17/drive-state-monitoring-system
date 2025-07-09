#!/usr/bin/env python3
"""
API Client - Backend integration for driver monitoring system
Provides RESTful API communication for data logging and session management
"""

import requests
import json
import time
from typing import Dict, Any, Optional
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class APIClient:
    """
    API client for driver monitoring system backend integration.
    """
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize API client.
        
        Args:
            base_url: Backend API base URL
        """
        self.base_url = base_url.rstrip('/')
        self.session_id = None
        self.is_connected = False
        
        # Configuration
        self.config = {
            'timeout': 5.0,
            'retry_attempts': 3,
            'retry_delay': 1.0
        }
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to backend API."""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=self.config['timeout'])
            if response.status_code == 200:
                self.is_connected = True
                print("Backend API connection established")
            else:
                print(f"Backend API health check failed: {response.status_code}")
        except Exception as e:
            print(f"Backend API connection failed: {e}")
            self.is_connected = False
    
    def create_session(self, session_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new monitoring session.
        
        Args:
            session_data: Optional session metadata
            
        Returns:
            bool: True if session created successfully
        """
        if not self.is_connected:
            return False
        
        try:
            # Prepare session data
            if session_data is None:
                session_data = {}
            
            session_data.update({
                'start_time': time.time(),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Send request
            response = requests.post(
                f"{self.base_url}/api/sessions",
                json=session_data,
                timeout=self.config['timeout']
            )
            
            if response.status_code == 201:
                result = response.json()
                self.session_id = result.get('session_id')
                print(f"Session created: {self.session_id}")
                return True
            else:
                print(f"Failed to create session: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error creating session: {e}")
            return False
    
    def log_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """
        Log an event to the backend.
        
        Args:
            event_type: Type of event (detection, alert, etc.)
            event_data: Event data
            
        Returns:
            bool: True if event logged successfully
        """
        if not self.is_connected:
            return False
        
        try:
            # Prepare event data
            event_payload = {
                'session_id': self.session_id,
                'event_type': event_type,
                'timestamp': time.time(),
                'data': event_data
            }
            
            # Send request with retry logic
            for attempt in range(self.config['retry_attempts']):
                try:
                    response = requests.post(
                        f"{self.base_url}/api/events",
                        json=event_payload,
                        timeout=self.config['timeout']
                    )
                    
                    if response.status_code == 201:
                        return True
                    else:
                        print(f"Failed to log event: {response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    print(f"Request error (attempt {attempt + 1}): {e}")
                    if attempt < self.config['retry_attempts'] - 1:
                        time.sleep(self.config['retry_delay'])
                    continue
            
            return False
            
        except Exception as e:
            print(f"Error logging event: {e}")
            return False
    
    def get_session_data(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get session data from backend.
        
        Args:
            session_id: Session ID (uses current session if None)
            
        Returns:
            Optional[Dict]: Session data or None
        """
        if not self.is_connected:
            return None
        
        try:
            target_session = session_id or self.session_id
            if not target_session:
                print("No session ID available")
                return None
            
            response = requests.get(
                f"{self.base_url}/api/sessions/{target_session}",
                timeout=self.config['timeout']
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get session data: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error getting session data: {e}")
            return None
    
    def get_analytics(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get analytics data from backend.
        
        Args:
            session_id: Session ID (uses current session if None)
            
        Returns:
            Optional[Dict]: Analytics data or None
        """
        if not self.is_connected:
            return None
        
        try:
            target_session = session_id or self.session_id
            params = {}
            if target_session:
                params['session_id'] = target_session
            
            response = requests.get(
                f"{self.base_url}/api/analytics",
                params=params,
                timeout=self.config['timeout']
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get analytics: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error getting analytics: {e}")
            return None
    
    def update_config(self, config_data: Dict[str, Any]) -> bool:
        """
        Update backend configuration.
        
        Args:
            config_data: Configuration data
            
        Returns:
            bool: True if configuration updated successfully
        """
        if not self.is_connected:
            return False
        
        try:
            response = requests.put(
                f"{self.base_url}/api/config",
                json=config_data,
                timeout=self.config['timeout']
            )
            
            if response.status_code == 200:
                print("Configuration updated successfully")
                return True
            else:
                print(f"Failed to update configuration: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error updating configuration: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get backend system status."""
        if not self.is_connected:
            return {'connected': False, 'error': 'Not connected'}
        
        try:
            response = requests.get(
                f"{self.base_url}/api/status",
                timeout=self.config['timeout']
            )
            
            if response.status_code == 200:
                status = response.json()
                status['connected'] = True
                return status
            else:
                return {
                    'connected': False,
                    'error': f'HTTP {response.status_code}'
                }
                
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
    
    def close_session(self) -> bool:
        """
        Close the current session.
        
        Returns:
            bool: True if session closed successfully
        """
        if not self.is_connected or not self.session_id:
            return False
        
        try:
            response = requests.delete(
                f"{self.base_url}/api/sessions/{self.session_id}",
                timeout=self.config['timeout']
            )
            
            if response.status_code == 200:
                print(f"Session closed: {self.session_id}")
                self.session_id = None
                return True
            else:
                print(f"Failed to close session: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error closing session: {e}")
            return False
    
    def export_data(self, session_id: Optional[str] = None, format: str = 'json') -> Optional[str]:
        """
        Export session data.
        
        Args:
            session_id: Session ID (uses current session if None)
            format: Export format ('json', 'csv')
            
        Returns:
            Optional[str]: Exported data or None
        """
        if not self.is_connected:
            return None
        
        try:
            target_session = session_id or self.session_id
            if not target_session:
                print("No session ID available")
                return None
            
            params = {'format': format}
            response = requests.get(
                f"{self.base_url}/api/sessions/{target_session}/export",
                params=params,
                timeout=self.config['timeout']
            )
            
            if response.status_code == 200:
                return response.text
            else:
                print(f"Failed to export data: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error exporting data: {e}")
            return None 