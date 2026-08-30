"""
Configuration management for SIBI Sign Language Recognition Application
"""
from dataclasses import dataclass, field
from pathlib import Path
import cv2

# Project root directory (parent of src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """Configuration constants for the application"""
    
    # File paths
    MODEL_PATH: str = field(default_factory=lambda: str(PROJECT_ROOT / 'model' / 'mymodel.sav'))
    DATA_PATH: str = field(default_factory=lambda: str(PROJECT_ROOT / 'data'))
    
    # MediaPipe Hand detection parameters
    MODEL_COMPLEXITY: int = 0
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5
    
    # Timing parameters (in seconds)
    GESTURE_DURATION: float = 2.5  # How long to hold gesture before detection
    DISPLAY_DURATION: float = 999999.0  # Effectively disabled (manual reset via R key)
    PREDICTION_DELAY: float = 2.5  # Delay between consecutive OUTPUT commits
    RAW_PREDICTION_INTERVAL: float = 0.1  # Sampling interval during hold (seconds)
    MAX_TEXT_LENGTH: int = 10  # Max characters to accumulate before requiring reset

    # Motion pattern detection (for dynamic letters like J and Z)
    ENABLE_JZ_MOTION: bool = True
    JZ_WINDOW_SECONDS: float = 0.6
    JZ_MIN_MOTION_LEVEL: str = "medium"  # low|medium|high

    # Prediction smoothing (stability vs latency)
    # 1 = no smoothing. 3 = lebih responsif, 5 = lebih stabil.
    PREDICTION_SMOOTHING_WINDOW: int = 3
    
    # Preprocessing constants
    NUM_LANDMARKS: int = 21  # Number of hand landmarks from MediaPipe
    NUM_COORDINATES: int = 42  # 21 landmarks * 2 coordinates (x, y)
    NUM_FEATURES: int = 210  # Total features needed for prediction (5 frames)
    NORMALIZATION_SCALE: int = 320  # Scale factor for normalization
    TARGET_PER_CLASS: int = 300  # Default target samples per class for data collection
    
    # UI parameters
    WINDOW_NAME: str = 'Aplikasi Bahasa Isyarat (SIBI)'
    CAMERA_INDEX: int = 0
    CAMERA_FLIP_HORIZONTAL: bool = True
    RIGHT_HAND_ONLY: bool = True
    # MediaPipe handedness label to use when RIGHT_HAND_ONLY=True.
    # If your physical right hand is detected as "Left", set this to "Left".
    TARGET_HAND_LABEL: str = "Left"
    FONT: int = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE: float = 1.0
    FONT_THICKNESS: int = 3
    FONT_COLOR: tuple = (255, 255, 255)  # White
    CHAR_SPACING: int = 4  # Spacing between characters in pixels
    TEXT_Y_OFFSET: int = 50  # Offset from bottom of screen
    
    # Frame processing
    FRAME_WAIT_TIME: int = 5  # milliseconds
    ESC_KEY: int = 27  # Escape key code
    
    def __post_init__(self):
        """Validate configuration values"""
        if self.GESTURE_DURATION <= 0:
            raise ValueError("GESTURE_DURATION must be positive")
        if self.DISPLAY_DURATION <= 0:
            raise ValueError("DISPLAY_DURATION must be positive")
        if self.PREDICTION_DELAY <= 0:
            raise ValueError("PREDICTION_DELAY must be positive")
        if self.NUM_LANDMARKS != 21:
            raise ValueError("NUM_LANDMARKS must be 21 for MediaPipe Hands")
