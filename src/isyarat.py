"""
SIBI Sign Language Recognition Application

This is a modular, maintainable version of the sign language recognition system.
Uses MediaPipe for hand detection and SVM for classification.

Author: Ananta Raihan
Date: March 2026
"""

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import pickle
import time
import os
from dataclasses import dataclass, field
from collections import deque, Counter
from typing import Optional, List, Deque
from src.config import Config
from src.app_logging import get_logger
from src.sibi_core import (
    center_keypoints,
    extract_keypoint_features,
    normalize_keypoints,
    scale_keypoints,
    select_target_hand_index,
)

logger = get_logger(__name__)


@dataclass
class RuntimeState:
    """Small container for mutable runtime state."""

    keypoint_frames: Deque[np.ndarray] = field(default_factory=deque)
    index_tip_track: Deque[tuple] = field(default_factory=deque)  # (t, x, y)
    pending_motion_letter: Optional[str] = None


class MotionPatternDetector:
    """Detect dynamic letters (J/Z) from index fingertip motion."""

    INDEX_TIP_LANDMARK = 8

    def __init__(self, config: Config):
        self.config = config

    def _motion_threshold(self) -> float:
        level = (getattr(self.config, "JZ_MIN_MOTION_LEVEL", "medium") or "medium").strip().lower()
        # Threshold in normalized coordinate space (0..1).
        if level == "low":
            return 0.08
        if level == "high":
            return 0.16
        return 0.12

    def update_track(self, track: Deque[tuple], hand_landmarks) -> None:
        if not getattr(self.config, "ENABLE_JZ_MOTION", True):
            return
        window_s = float(getattr(self.config, "JZ_WINDOW_SECONDS", 0.6) or 0.6)
        now = time.time()
        lm = hand_landmarks.landmark[self.INDEX_TIP_LANDMARK]
        track.append((now, float(lm.x), float(lm.y)))
        # Trim old samples
        cutoff = now - window_s
        while track and track[0][0] < cutoff:
            track.popleft()

    def _track_arrays(self, track: Deque[tuple]) -> tuple[np.ndarray, np.ndarray]:
        xs = np.array([p[1] for p in track], dtype=float)
        ys = np.array([p[2] for p in track], dtype=float)
        return xs, ys

    def _detect_from_motion_stats(self, dx: float, dy: float, net_y: float, sign_changes_x: int) -> Optional[str]:
        if dx > 0.12 and sign_changes_x >= 2 and dy > 0.04:
            return "Z"
        if net_y > 0.08 and dx > 0.05 and sign_changes_x <= 1:
            return "J"
        return None
    def detect(self, track: Deque[tuple]) -> Optional[str]:
        """Return 'J' or 'Z' if detected, else None."""
        if not getattr(self.config, "ENABLE_JZ_MOTION", True) or len(track) < 6:
            return None

        xs, ys = self._track_arrays(track)
        dx = float(xs.max() - xs.min())
        dy = float(ys.max() - ys.min())
        if float(np.hypot(dx, dy)) < self._motion_threshold():
            return None

        x = xs - xs[0]
        y = ys - ys[0]
        vx = np.diff(x)
        if len(vx) < 4:
            return None

        sign_changes_x = int(np.sum(np.sign(vx[1:]) != np.sign(vx[:-1])))
        net_y = float(y[-1] - y[0])
        return self._detect_from_motion_stats(dx, dy, net_y, sign_changes_x)

class HandKeypointProcessor:
    """Handles preprocessing of hand keypoints from MediaPipe."""

    def __init__(self, config: Config):
        self.config = config

    def center_keypoints(self, landmarks) -> np.ndarray:
        """Center keypoints relative to wrist (landmark 0)."""
        return center_keypoints(landmarks)

    def scale_keypoints(self, centered: np.ndarray) -> np.ndarray:
        """Scale keypoints to fixed range for normalization."""
        return scale_keypoints(centered, self.config)

    def normalize_keypoints(self, scaled: np.ndarray) -> np.ndarray:
        """Normalize keypoints to positive range."""
        return normalize_keypoints(scaled, self.config)

    def process_landmarks(self, landmarks) -> np.ndarray:
        """Full preprocessing pipeline for hand landmarks."""
        return extract_keypoint_features(landmarks, self.config)


class GestureDetector:
    """
    Manages gesture detection state and timing.
    
    Ensures gestures are held for a minimum duration before processing.
    """
    
    def __init__(self, gesture_duration: float):
        """
        Initialize gesture detector.
        
        Args:
            gesture_duration: Minimum time (seconds) to hold gesture
        """
        self.gesture_duration = gesture_duration
        self.gesture_start_time: Optional[float] = None
    
    def start_gesture(self):
        """Start tracking a new gesture."""
        if self.gesture_start_time is None:
            self.gesture_start_time = time.time()
    
    def reset_gesture(self):
        """Reset gesture tracking."""
        self.gesture_start_time = None
    
    def is_gesture_ready(self) -> bool:
        """
        Check if gesture has been held long enough.
        
        Returns:
            bool: True if gesture duration exceeded
        """
        if self.gesture_start_time is None:
            return False
        return (time.time() - self.gesture_start_time) >= self.gesture_duration
    
    def get_gesture_progress(self) -> float:
        """
        Get current gesture hold progress (0.0 to 1.0).
        
        Returns:
            float: Progress percentage
        """
        if self.gesture_start_time is None:
            return 0.0
        elapsed = time.time() - self.gesture_start_time
        return min(elapsed / self.gesture_duration, 1.0)


class PredictionManager:
    """
    Manages predictions and display timing.
    
    Handles text accumulation and automatic reset after display duration.
    """
    
    def __init__(
        self,
        display_duration: float,
        prediction_delay: float,
        smoothing_window: int = 1,
        raw_prediction_interval: float = 0.0,
        max_text_length: int = 10,
    ):
        """
        Initialize prediction manager.
        
        Args:
            display_duration: How long to display text (seconds)
            prediction_delay: Delay between predictions (seconds)
            max_text_length: Maximum characters to accumulate
        """
        self.display_duration = display_duration
        self.prediction_delay = prediction_delay
        self.max_text_length = max_text_length
        self.last_prediction_time: Optional[float] = None
        self.last_display_time: Optional[float] = None
        self.last_sample_time: Optional[float] = None
        self.display_text: str = ""
        self.smoothing_window = max(1, int(smoothing_window))
        self.raw_prediction_interval = max(0.0, float(raw_prediction_interval))
        self._pending_predictions: Deque[str] = deque(maxlen=self.smoothing_window)
    
    def can_predict(self) -> bool:
        """
        Check if enough time has passed for new prediction.
        
        Returns:
            bool: True if ready for new prediction
        """
        if self.last_prediction_time is None:
            return True
        return (time.time() - self.last_prediction_time) > self.prediction_delay

    def can_sample(self) -> bool:
        """Check if we can sample another raw prediction (for smoothing)."""
        if self.raw_prediction_interval <= 0:
            return True
        if self.last_sample_time is None:
            return True
        return (time.time() - self.last_sample_time) >= self.raw_prediction_interval
    
    def add_prediction(self, prediction: str):
        """
        Add new prediction to display text.
        
        Args:
            prediction: Predicted character/label
        """
        now = time.time()
        if len(self.display_text) >= self.max_text_length:
            # Buffer full - update timing but don't add character
            self.last_prediction_time = now
            return
        self.display_text += prediction
        self.last_prediction_time = now
        self.last_display_time = now

    def reset_pending_predictions(self):
        """Clear only pending predictions (does not touch display_text)."""
        self._pending_predictions.clear()
        self.last_sample_time = None

    def add_sample(self, prediction: str):
        """Collect a raw model prediction sample for smoothing."""
        if not self.can_sample():
            return
        self._pending_predictions.append(prediction)
        self.last_sample_time = time.time()

    def _commit_prediction(self, candidate: str) -> str:
        self._pending_predictions.clear()
        self.add_prediction(candidate)
        return candidate

    def _majority_vote(self) -> str:
        counts = Counter(self._pending_predictions)
        max_count = max(counts.values())
        candidates = {label for label, count in counts.items() if count == max_count}
        for label in reversed(self._pending_predictions):
            if label in candidates:
                return label
        return self._pending_predictions[-1]

    def has_pending_prediction(self, prediction: str) -> bool:
        """Return True when a label is present in the current smoothing window."""
        return prediction in self._pending_predictions
    def commit_pending_if_ready(self) -> Optional[str]:
        """Commit a character from pending samples when the hold completes."""
        if not self._pending_predictions or not self.can_predict():
            return None

        if self.smoothing_window > 1 and len(self._pending_predictions) < self.smoothing_window:
            return None

        candidate = self._pending_predictions[-1] if self.smoothing_window <= 1 else self._majority_vote()
        return self._commit_prediction(candidate)
    def should_reset_text(self) -> bool:
        """
        Check if text should be cleared.
        
        Returns:
            bool: True if display duration exceeded
        """
        if self.last_display_time is None:
            return False
        return (time.time() - self.last_display_time) > self.display_duration
    
    def reset_text(self):
        """Clear display text."""
        self.display_text = ""
        self.last_prediction_time = None
        self.last_display_time = None
        self._pending_predictions.clear()
        self.last_sample_time = None
    
    def get_display_text(self) -> str:
        """
        Get current display text.
        
        Returns:
            str: Text to display
        """
        return self.display_text


class ModelManager:
    """
    Manages machine learning model and label loading.
    """
    
    def __init__(self, config: Config):
        """
        Initialize model manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.model = self._load_model()
        self.labels = self._load_labels()
    
    def _load_model(self):
        """
        Load trained SVM model from disk.
        
        Returns:
            Trained model object
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            with open(self.config.MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            logger.info("Model loaded successfully from %s", self.config.MODEL_PATH)
            return model
        except FileNotFoundError:
            raise RuntimeError(f"Model file not found: {self.config.MODEL_PATH}")
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError, ValueError) as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _load_labels_from_encoder(self, le_path: str) -> List[str]:
        with open(le_path, "rb") as f:
            le = pickle.load(f)
        classes = getattr(le, "classes_", None)
        if classes is None:
            raise RuntimeError("Label encoder tidak memiliki classes_")
        labels = [str(c) for c in classes]
        if not labels:
            raise RuntimeError("classes_ kosong")
        return labels

    def _load_labels_from_data_folder(self) -> List[str]:
        files = sorted(os.listdir(self.config.DATA_PATH))
        labels = [f[5:-4] for f in files if f.startswith('data_') and f.endswith('.csv')]
        if not labels:
            raise RuntimeError("No data files found in data directory")
        return labels
    def _load_labels(self) -> List[str]:
        """Load class labels from le.sav, falling back to data_*.csv names."""
        model_folder = os.path.dirname(self.config.MODEL_PATH) or "."
        le_path = os.path.join(model_folder, "le.sav")

        try:
            labels = self._load_labels_from_encoder(le_path)
            logger.info("Loaded %s labels from encoder: %s", len(labels), labels)
            return labels
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError, TypeError, ValueError) as le_err:
            logger.warning("Gagal membaca %s, fallback ke data folder (%s)", le_path, le_err)

        try:
            labels = self._load_labels_from_data_folder()
            logger.info("Loaded %s labels from data folder: %s", len(labels), labels)
            return labels
        except FileNotFoundError:
            raise RuntimeError(f"Data directory not found: {self.config.DATA_PATH}")
        except (OSError, ValueError) as e:
            raise RuntimeError(f"Failed to load labels: {e}")
    def predict(self, features: np.ndarray) -> Optional[str]:
        """
        Make prediction using the model.
        
        Args:
            features: Feature vector for prediction
            
        Returns:
            Optional[str]: Predicted label or None if prediction fails
        """
        try:
            prediction = self.model.predict([features])
            pred = prediction[0]

            if isinstance(pred, (str, np.str_)):
                return str(pred)

            idx = int(pred)
            if 0 <= idx < len(self.labels):
                return self.labels[idx]

            logger.warning("Prediction index %s out of range", idx)
            return None
        except (AttributeError, TypeError, ValueError, IndexError) as e:
            logger.error("Prediction error: %s", e)
            return None


class UIRenderer:
    """
    Handles UI rendering for the application.
    """
    
    def __init__(self, config: Config):
        """
        Initialize UI renderer.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
    
    def draw_landmarks(self, image: np.ndarray, hand_landmarks):
        """
        Draw hand landmarks on image.
        
        Args:
            image: Image to draw on
            hand_landmarks: MediaPipe hand landmarks
        """
        self.mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
    
    def _text_layout(self, image: np.ndarray, text: str) -> tuple[list[int], int, int]:
        char_widths = [
            cv2.getTextSize(c, self.config.FONT, self.config.FONT_SCALE, self.config.FONT_THICKNESS)[0][0]
            for c in text
        ]
        total_width = sum(char_widths) + self.config.CHAR_SPACING * (len(text) - 1)
        x = int((image.shape[1] - total_width) / 2)
        y = image.shape[0] - self.config.TEXT_Y_OFFSET
        return char_widths, x, y

    def _draw_character(self, image: np.ndarray, char: str, x: int, y: int) -> None:
        cv2.putText(
            image,
            char,
            (x, y),
            self.config.FONT,
            self.config.FONT_SCALE,
            self.config.FONT_COLOR,
            self.config.FONT_THICKNESS,
            cv2.LINE_AA,
        )
    def draw_text(self, image: np.ndarray, text: str):
        """Draw centered text at bottom of image."""
        if not text:
            return

        char_widths, x, y = self._text_layout(image, text)
        for char, width in zip(text, char_widths):
            self._draw_character(image, char, x, y)
            x += width + self.config.CHAR_SPACING
    def draw_gesture_progress(self, image: np.ndarray, progress: float):
        """
        Draw gesture hold progress bar.
        
        Args:
            image: Image to draw on
            progress: Progress value (0.0 to 1.0)
        """
        if progress <= 0:
            return
        
        # Progress bar dimensions
        bar_width = 200
        bar_height = 10
        bar_x = 20
        bar_y = 20
        
        # Draw background
        cv2.rectangle(image, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        
        # Draw progress
        progress_width = int(bar_width * progress)
        color = (0, 255, 0) if progress >= 1.0 else (0, 255, 255)
        cv2.rectangle(image, (bar_x, bar_y), 
                     (bar_x + progress_width, bar_y + bar_height),
                     color, -1)


class SignLanguageRecognizer:
    """
    Main application class for SIBI sign language recognition.
    
    Orchestrates all components: video capture, hand detection, 
    preprocessing, prediction, and UI rendering.
    """
    
    def _build_prediction_manager(self) -> PredictionManager:
        return PredictionManager(
            self.config.DISPLAY_DURATION,
            self.config.PREDICTION_DELAY,
            smoothing_window=getattr(self.config, "PREDICTION_SMOOTHING_WINDOW", 1),
            raw_prediction_interval=getattr(self.config, "RAW_PREDICTION_INTERVAL", 0.0),
            max_text_length=getattr(self.config, "MAX_TEXT_LENGTH", 10),
        )

    def _build_hands_detector(self):
        return mp.solutions.hands.Hands(
            model_complexity=self.config.MODEL_COMPLEXITY,
            min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE,
        )
    def __init__(self, config: Config = None):
        """Initialize the sign language recognizer."""
        self.config = config or Config()
        self.model_manager = ModelManager(self.config)
        self.processor = HandKeypointProcessor(self.config)
        self.gesture_detector = GestureDetector(self.config.GESTURE_DURATION)
        self.prediction_manager = self._build_prediction_manager()
        self.ui_renderer = UIRenderer(self.config)
        self.motion_detector = MotionPatternDetector(self.config)
        self.hands = self._build_hands_detector()
        self.state = RuntimeState()
        frames_needed = self.config.NUM_FEATURES // self.config.NUM_COORDINATES
        self.state.keypoint_frames = deque(maxlen=frames_needed)
        logger.info("Sign Language Recognizer initialized successfully")
    def _reset_detection_state(self, clear_pending_predictions: bool = True):
        """Reset transient gesture state without clearing displayed text."""
        self.gesture_detector.reset_gesture()
        self.state.keypoint_frames.clear()
        self.state.index_tip_track.clear()
        self.state.pending_motion_letter = None
        if clear_pending_predictions:
            self.prediction_manager.reset_pending_predictions()

    def _target_landmarks(self, results):
        if not results.multi_hand_landmarks:
            return None
        hand_index = self._get_target_hand_index(results)
        if hand_index is None:
            return None
        hand_landmarks = results.multi_hand_landmarks[hand_index]
        if len(hand_landmarks.landmark) < self.config.NUM_LANDMARKS:
            return None
        return hand_landmarks
    def _process_hand_detection(self, results) -> Optional[np.ndarray]:
        """Process the target hand and return normalized keypoints."""
        hand_landmarks = self._target_landmarks(results)
        if hand_landmarks is None:
            self._reset_detection_state()
            return None

        self.gesture_detector.start_gesture()
        self.motion_detector.update_track(self.state.index_tip_track, hand_landmarks)
        return self.processor.process_landmarks(hand_landmarks.landmark)
    def _flip_landmarks_for_display(self, hand_landmarks):
        """Flip landmark X coordinates for drawing on a horizontally flipped image."""
        def _as_float(value) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        flipped = landmark_pb2.NormalizedLandmarkList()
        for lm in hand_landmarks.landmark:
            x = _as_float(getattr(lm, "x", 0.0))
            y = _as_float(getattr(lm, "y", 0.0))
            z = _as_float(getattr(lm, "z", 0.0))

            x = max(0.0, min(1.0, 1.0 - x))
            y = max(0.0, min(1.0, y))

            flipped.landmark.append(
                landmark_pb2.NormalizedLandmark(
                    x=x,
                    y=y,
                    z=z,
                )
            )
        return flipped

    def _get_target_hand_index(self, results) -> Optional[int]:
        """Return the index of the hand to use.

        If RIGHT_HAND_ONLY is enabled, tries to pick the physical right hand.
        MediaPipe handedness labels can be inverted when the camera image is
        horizontally flipped before processing, so we compensate for that.
        """
        if not results.multi_hand_landmarks:
            return None

        if not self.config.RIGHT_HAND_ONLY:
            return 0

        desired_label = (getattr(self.config, "TARGET_HAND_LABEL", "Right") or "Right").strip().title()
        if desired_label not in {"Left", "Right"}:
            desired_label = "Right"

        multi_handedness = getattr(results, "multi_handedness", None)
        if not multi_handedness:
            # Fallback: some devices/frames return landmarks without handedness.
            # In that case, keep the app functional by selecting the first hand.
            return 0

        for idx, handedness in enumerate(multi_handedness):
            classification = getattr(handedness, "classification", None)
            if not classification:
                continue
            label = getattr(classification[0], "label", None)
            if not isinstance(label, str):
                continue
            if label == desired_label:
                return idx

        return None
    
    def _update_keypoint_buffer(self, keypoints: np.ndarray):
        """Append a single frame of keypoints to the temporal buffer."""
        # Keep this as frames to avoid repeated large numpy allocations.
        self.state.keypoint_frames.append(np.asarray(keypoints, dtype=float))
    
    def _make_prediction(self) -> Optional[str]:
        """
        Make prediction if buffer has enough data.
        
        Returns:
            Optional[str]: Predicted label or None
        """
        frames_needed = self.config.NUM_FEATURES // self.config.NUM_COORDINATES
        if len(self.state.keypoint_frames) < frames_needed:
            return None

        # Check sampling timing (smoothing)
        if not self.prediction_manager.can_sample():
            return None
        
        # Make prediction
        # Concatenate frames into [NUM_FEATURES]
        features = np.concatenate(list(self.state.keypoint_frames)[-frames_needed:])
        prediction = self.model_manager.predict(features)
        
        return prediction
    
    def _draw_detected_landmarks(self, image: np.ndarray, results) -> None:
        """Draw all detected hand landmarks on the display frame."""
        if not results.multi_hand_landmarks:
            return

        for hand_landmarks in results.multi_hand_landmarks:
            if self.config.CAMERA_FLIP_HORIZONTAL:
                hand_landmarks = self._flip_landmarks_for_display(hand_landmarks)
            self.ui_renderer.draw_landmarks(image, hand_landmarks)

    def _commit_ready_prediction(self) -> None:
        """Commit a stable static or motion prediction after the hold timer completes."""
        if not self.gesture_detector.is_gesture_ready():
            return

        if self.state.pending_motion_letter is None:
            detected_motion = self.motion_detector.detect(self.state.index_tip_track)
            if detected_motion and self.prediction_manager.has_pending_prediction(detected_motion):
                self.state.pending_motion_letter = detected_motion

        motion_letter = self.state.pending_motion_letter
        if motion_letter is not None:
            if not self.prediction_manager.can_predict():
                return
            self.prediction_manager.add_prediction(motion_letter)
            self.prediction_manager.reset_pending_predictions()
            self._reset_detection_state(clear_pending_predictions=False)
            return

        committed = self.prediction_manager.commit_pending_if_ready()
        if committed is not None:
            self._reset_detection_state(clear_pending_predictions=False)

    def _draw_overlays(self, image: np.ndarray) -> None:
        """Draw progress and accumulated prediction text."""
        self.ui_renderer.draw_gesture_progress(
            image,
            self.gesture_detector.get_gesture_progress()
        )
        self.ui_renderer.draw_text(
            image,
            self.prediction_manager.get_display_text()
        )

    def _process_frame(self, image: np.ndarray):
        """
        Process a single video frame.
        
        Args:
            image: Input image from camera
            
        Returns:
            np.ndarray: Processed image with overlays
        """
        mp_frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(mp_frame_rgb)
        display_image = cv2.flip(image, 1) if self.config.CAMERA_FLIP_HORIZONTAL else image

        self._draw_detected_landmarks(display_image, results)
        keypoints = self._process_hand_detection(results)
        if keypoints is not None:
            self._update_keypoint_buffer(keypoints)
            prediction = self._make_prediction()
            if prediction:
                self.prediction_manager.add_sample(prediction)
            self._commit_ready_prediction()

        if self.prediction_manager.should_reset_text():
            self.prediction_manager.reset_text()

        self._draw_overlays(display_image)
        return display_image
    def process_frame(self, image):
        """
        Public interface for external GUI integration.
        
        Args:
            image: Input frame from camera
            
        Returns:
            tuple: (processed_image, current_prediction_text)
        """
        processed_image = self._process_frame(image)
        current_text = self.prediction_manager.get_display_text()
        return processed_image, current_text
    
    def reset_text(self):
        """Clear accumulated prediction text (for manual reset)."""
        self.prediction_manager.reset_text()
        self.gesture_detector.reset_gesture()
        self.state.keypoint_frames.clear()
        self.state.index_tip_track.clear()
        self.state.pending_motion_letter = None

    def cleanup(self):
        """
        Release MediaPipe resources.
        
        Call this when done using the recognizer.
        """
        if hasattr(self, 'hands'):
            self.hands.close()
    
    def _open_camera(self):
        cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
        return cap

    def _process_camera_frame(self, cap) -> bool:
        success, image = cap.read()
        if not success:
            logger.warning("Failed to read frame from camera")
            return True

        processed_image = self._process_frame(image)
        cv2.imshow(self.config.WINDOW_NAME, processed_image)
        if cv2.waitKey(self.config.FRAME_WAIT_TIME) & 0xFF == self.config.ESC_KEY:
            logger.info("Exiting application")
            return False
        return True

    def _release_camera(self, cap) -> None:
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        logger.info("Resources released")
    def run(self):
        """Capture video frames, process them, and display results."""
        cap = self._open_camera()
        logger.info("Starting %s", self.config.WINDOW_NAME)
        logger.info("Press ESC to exit")

        try:
            while cap.isOpened() and self._process_camera_frame(cap):
                pass
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except (RuntimeError, ValueError, OSError, cv2.error) as e:
            logger.error("Error during execution: %s", e)
            raise
        finally:
            self._release_camera(cap)

def main():
    """Entry point for the application."""
    try:
        logger.info("%s", "=" * 50)
        logger.info("SIBI Sign Language Recognition System")
        logger.info("%s", "=" * 50)
        
        app = SignLanguageRecognizer()
        app.run()
        
    except (RuntimeError, ValueError, OSError, cv2.error) as e:
        logger.error("Application error: %s", e)
        raise


if __name__ == "__main__":
    main()
