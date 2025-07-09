#!/usr/bin/env python3
"""
Test script to verify compatibility with latest versions.
"""

import sys
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import dlib
        print(f"✓ dlib version: {dlib.__version__}")
    except ImportError as e:
        print(f"✗ dlib import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pygame
        print(f"✓ Pygame version: {pygame.version.ver}")
    except ImportError as e:
        print(f"✗ Pygame import failed: {e}")
        return False
    
    try:
        from imutils import face_utils
        print("✓ imutils.face_utils imported successfully")
    except ImportError as e:
        print(f"✗ imutils import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas version: {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import speech_recognition as sr
        print(f"✓ SpeechRecognition imported successfully")
    except ImportError as e:
        print(f"✗ SpeechRecognition import failed: {e}")
        return False
    
    try:
        from gtts import gTTS
        print("✓ gTTS imported successfully")
    except ImportError as e:
        print(f"✗ gTTS import failed: {e}")
        return False
    
    return True

def test_camera_access():
    """Test camera access."""
    print("\nTesting camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera access successful")
            cap.release()
            return True
        else:
            print("✗ Camera access failed - camera not available")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_face_detection():
    """Test face detection components."""
    print("\nTesting face detection components...")
    
    try:
        import dlib
        import cv2
        from imutils import face_utils
        
        # Test detector
        detector = dlib.get_frontal_face_detector()  # type: ignore
        print("✓ Face detector loaded successfully")
        
        # Test predictor (if file exists)
        import os
        if os.path.exists("shape_predictor_68_face_landmarks.dat"):
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # type: ignore
            print("✓ Face landmark predictor loaded successfully")
        else:
            print("⚠ Face landmark predictor file not found")
        
        # Test facial landmarks indices
        landmarks = face_utils.FACIAL_LANDMARKS_IDXS
        required_keys = ["jaw", "nose", "left_eye", "right_eye"]
        for key in required_keys:
            if key in landmarks:
                print(f"✓ Facial landmark '{key}' available")
            else:
                print(f"✗ Facial landmark '{key}' missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Face detection test failed: {e}")
        return False

def test_audio_components():
    """Test audio components."""
    print("\nTesting audio components...")
    
    try:
        import pygame
        pygame.mixer.init()
        print("✓ Pygame mixer initialized successfully")
        
        # Test if audio files exist
        import os
        audio_files = ["beep.mp3", "distractAlert.mp3"]
        for file in audio_files:
            if os.path.exists(file):
                print(f"✓ Audio file '{file}' found")
            else:
                print(f"⚠ Audio file '{file}' not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Audio test failed: {e}")
        return False

def test_our_modules():
    """Test our custom modules."""
    print("\nTesting custom modules...")
    
    try:
        import distraction_detection
        print("✓ distraction_detection module imported")
        
        import pnp_utils
        print("✓ pnp_utils module imported")
        
        # Test pnp_utils functions
        model = pnp_utils.ref3DModel()
        if model.shape == (6, 3):
            print("✓ 3D model created successfully")
        else:
            print("✗ 3D model has wrong shape")
            return False
        
        camera_matrix = pnp_utils.CameraMatrix(1000, (320, 240))
        if camera_matrix.shape == (3, 3):
            print("✓ Camera matrix created successfully")
        else:
            print("✗ Camera matrix has wrong shape")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Custom modules test failed: {e}")
        return False

def main():
    """Run all compatibility tests."""
    print("=" * 50)
    print("DROWSINESS DETECTION - COMPATIBILITY TEST")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Camera Access Test", test_camera_access),
        ("Face Detection Test", test_face_detection),
        ("Audio Components Test", test_audio_components),
        ("Custom Modules Test", test_our_modules),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! The code is compatible with latest versions.")
        return 0
    else:
        print("⚠ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 