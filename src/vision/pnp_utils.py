import numpy as np
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def ref3DModel():
    """
    Returns the 3D model for head pose estimation.
    """
    modelPoints = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)
    
    return modelPoints

def ref2dImagePoints(shape):
    """
    Returns the 2D image points from facial landmarks.
    """
    imagePoints = np.array([
        shape[30],     # Nose tip
        shape[8],      # Chin
        shape[36],     # Left eye left corner
        shape[45],     # Right eye right corner
        shape[48],     # Left mouth corner
        shape[54]      # Right mouth corner
    ], dtype=np.float64)
    
    return imagePoints

def CameraMatrix(fl, center):
    """
    Returns the camera matrix for head pose estimation.
    """
    cameraMatrix = np.array([
        [fl, 0, center[0]],
        [0, fl, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return cameraMatrix

