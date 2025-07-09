#!/usr/bin/env python3
"""
Startup script for Drowsiness Detection Backend
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_directory():
    """Check if we're in the correct directory and provide helpful guidance"""
    current_dir = Path.cwd()
    backend_dir = current_dir / "backend"
    app_file = backend_dir / "app.py"
    
    # Check if we're in the project root
    if not backend_dir.exists():
        print("ERROR: Backend directory not found!")
        print(f"Current directory: {current_dir}")
        print("Please make sure you're in the project root directory.")
        print("Expected structure:")
        print("  Drowsiness_Detection-main/")
        print("  ├── backend/")
        print("  │   └── app.py")
        print("  └── start_backend.py")
        return False
    
    # Check if app.py exists
    if not app_file.exists():
        print("ERROR: Backend app.py not found!")
        print(f"Expected location: {app_file}")
        print("Please check if the backend files are properly installed.")
        return False
    
    return True

def check_dependencies():
    """Check if backend dependencies are installed"""
    try:
        import flask
        import flask_sqlalchemy
        import flask_cors
        print("Backend dependencies are installed")
        return True
    except ImportError as e:
        print(f"Missing backend dependency: {e}")
        print("Installing backend dependencies...")
        
        backend_requirements = Path("backend/requirements.txt")
        if backend_requirements.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(backend_requirements)])
            return True
        else:
            print("Backend requirements file not found")
            return False

def start_backend():
    """Start the backend server"""
    backend_dir = Path("backend")
    app_file = backend_dir / "app.py"
    
    if not app_file.exists():
        print("Backend app.py not found")
        return False
    
    print("Starting Drowsiness Detection Backend...")
    print("API will be available at: http://localhost:8080")
    print("Health check: http://localhost:8080/api/health")
    print("Database: drowsiness_detection.db")
    print("=" * 50)
    
    # Change to backend directory and start server
    os.chdir(backend_dir)
    subprocess.run([sys.executable, "app.py"])

def test_backend():
    """Test if backend is running"""
    try:
        response = requests.get("http://localhost:8080/api/health", timeout=5)
        if response.status_code == 200:
            print("Backend is running successfully")
            return True
        else:
            print(f"Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("Backend is not responding")
        return False

def show_help():
    """Show helpful usage information"""
    print("\n" + "=" * 60)
    print("DROWSINESS DETECTION BACKEND - USAGE GUIDE")
    print("=" * 60)
    print("\nCORRECT WAYS TO START THE BACKEND:")
    print("1. From project root (recommended):")
    print("   python start_backend.py")
    print("\n2. From backend directory:")
    print("   cd backend")
    print("   python app.py")
    print("\nINCORRECT (will cause errors):")
    print("   python app.py  # From project root - WRONG!")
    print("\nTROUBLESHOOTING:")
    print("- Make sure you're in the project root directory")
    print("- Check that backend/app.py exists")
    print("- Ensure all dependencies are installed")
    print("- Port 8080 should be available")
    print("\nFor more help, see: BACKEND_INTEGRATION.md")
    print("=" * 60)

if __name__ == "__main__":
    print("==================================================")
    print("DROWSINESS DETECTION BACKEND STARTUP")
    print("==================================================")
    
    # Check if we're in the right directory
    if not check_directory():
        show_help()
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        show_help()
        sys.exit(1)
    
    # Start backend
    start_backend() 