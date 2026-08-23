"""
Unit tests for SIBI Sign Language Recognition Application

Run with: pytest test_isyarat.py -v
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from config import Config
from making_data import extract_scaling
from sibi_core import select_target_hand_index
from isyarat import (
    HandKeypointProcessor,
    GestureDetector,
    PredictionManager,
    ModelManager,
    UIRenderer
)


class TestConfig:
    """Test configuration class"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = Config()
        assert config.MODEL_PATH == './model/mymodel.sav'
        assert config.NUM_LANDMARKS == 21
        assert config.GESTURE_DURATION == 2.5
        assert config.DISPLAY_DURATION == 999999.0
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config should not raise
        config = Config()
        config.__post_init__()
        
        # Invalid gesture duration should raise
        with pytest.raises(ValueError):
            config = Config()
            config.GESTURE_DURATION = -1
            config.__post_init__()


class TestHandKeypointProcessor:
    """Test hand keypoint preprocessing"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        config = Config()
        return HandKeypointProcessor(config)
    
    @pytest.fixture
    def mock_landmarks(self):
        """Create mock landmarks"""
        landmarks = []
        for i in range(21):
            landmark = Mock()
            landmark.x = 0.5 + i * 0.01
            landmark.y = 0.5 + i * 0.01
            landmarks.append(landmark)
        return landmarks
    
    def test_center_keypoints(self, processor, mock_landmarks):
        """Test keypoint centering"""
        centered = processor.center_keypoints(mock_landmarks)
        
        # Check shape
        assert centered.shape == (2, 21)
        
        # First point should be (0, 0) as it's the reference
        assert centered[0, 0] == 0.0
        assert centered[1, 0] == 0.0
    
    def test_scale_keypoints(self, processor):
        """Test keypoint scaling"""
        # Create test data
        centered = np.random.randn(2, 21)
        scaled = processor.scale_keypoints(centered)
        
        # Check shape
        assert scaled.shape == (42,)
        
        # Check scaling range
        assert np.max(np.abs(scaled)) <= 320
    
    def test_normalize_keypoints(self, processor):
        """Test keypoint normalization"""
        scaled = np.array([-100, 0, 100, 200])
        normalized = processor.normalize_keypoints(scaled)
        
        # Check offset applied
        assert normalized[0] == 220  # -100 + 320
        assert normalized[1] == 320  # 0 + 320
        assert normalized[2] == 420  # 100 + 320
    
    def test_process_landmarks_pipeline(self, processor, mock_landmarks):
        """Test full preprocessing pipeline"""
        result = processor.process_landmarks(mock_landmarks)
        
        # Check output shape
        assert result.shape == (42,)
        
        # Check values are positive (after normalization)
        assert np.all(result >= 0)


class TestGestureDetector:
    """Test gesture detection timing"""
    
    def test_initial_state(self):
        """Test initial detector state"""
        detector = GestureDetector(gesture_duration=1.0)
        assert detector.gesture_start_time is None
        assert not detector.is_gesture_ready()
    
    def test_start_gesture(self):
        """Test gesture start"""
        detector = GestureDetector(gesture_duration=1.0)
        detector.start_gesture()
        
        assert detector.gesture_start_time is not None
        assert isinstance(detector.gesture_start_time, float)
    
    def test_reset_gesture(self):
        """Test gesture reset"""
        detector = GestureDetector(gesture_duration=1.0)
        detector.start_gesture()
        detector.reset_gesture()
        
        assert detector.gesture_start_time is None
    
    def test_gesture_ready_timing(self):
        """Test gesture ready after duration"""
        detector = GestureDetector(gesture_duration=0.1)
        
        # Should not be ready immediately
        detector.start_gesture()
        assert not detector.is_gesture_ready()
        
        # Should be ready after duration
        time.sleep(0.15)
        assert detector.is_gesture_ready()
    
    def test_gesture_progress(self):
        """Test gesture progress calculation"""
        detector = GestureDetector(gesture_duration=1.0)
        
        # No progress initially
        assert detector.get_gesture_progress() == 0.0
        
        # Some progress after starting
        detector.start_gesture()
        time.sleep(0.3)
        progress = detector.get_gesture_progress()
        assert 0.2 < progress < 0.5
        
        # Full progress after duration
        time.sleep(0.8)
        assert detector.get_gesture_progress() >= 1.0


class TestPredictionManager:
    """Test prediction management"""
    
    def test_initial_state(self):
        """Test initial manager state"""
        manager = PredictionManager(display_duration=3.0, prediction_delay=2.0)
        
        assert manager.display_text == ""
        assert manager.last_prediction_time is None
        assert manager.can_predict()
    
    def test_add_prediction(self):
        """Test adding predictions"""
        manager = PredictionManager(display_duration=3.0, prediction_delay=2.0)
        
        manager.add_prediction("A")
        assert manager.get_display_text() == "A"
        
        manager.add_prediction("B")
        assert manager.get_display_text() == "AB"
    
    def test_can_predict_timing(self):
        """Test prediction delay"""
        manager = PredictionManager(display_duration=3.0, prediction_delay=0.1)
        
        # Can predict initially
        assert manager.can_predict()
        
        # Cannot predict immediately after
        manager.add_prediction("A")
        assert not manager.can_predict()
        
        # Can predict after delay
        time.sleep(0.15)
        assert manager.can_predict()
    
    def test_should_reset_text(self):
        """Test text reset timing"""
        manager = PredictionManager(display_duration=0.1, prediction_delay=0.05)
        
        # Should not reset without prediction
        assert not manager.should_reset_text()
        
        # Should not reset immediately
        manager.add_prediction("A")
        assert not manager.should_reset_text()
        
        # Should reset after display duration
        time.sleep(0.15)
        assert manager.should_reset_text()
    
    def test_reset_text(self):
        """Test text reset"""
        manager = PredictionManager(display_duration=3.0, prediction_delay=2.0)
        
        manager.add_prediction("ABC")
        manager.reset_text()
        
        assert manager.get_display_text() == ""
        assert manager.last_prediction_time is None

    def test_prediction_smoothing_majority_vote(self):
        """Test majority-vote smoothing commits only after window is filled."""
        manager = PredictionManager(display_duration=3.0, prediction_delay=0.0, smoothing_window=5)

        # Not enough samples -> no commit
        manager.add_sample("A")
        manager.add_sample("A")
        manager.add_sample("B")
        manager.add_sample("A")
        assert manager.commit_pending_if_ready() is None

        # 5th sample -> commit majority (A)
        manager.add_sample("C")
        committed = manager.commit_pending_if_ready()
        assert committed == "A"
        assert manager.get_display_text() == "A"


class TestModelManager:
    """Test model management"""
    
    @patch('pickle.load')
    @patch('builtins.open')
    def test_model_loading_success(self, mock_open, mock_pickle_load):
        """Test successful model loading"""
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.classes_ = ['A', 'B']
        mock_pickle_load.side_effect = [mock_model, mock_encoder]
        
        config = Config()
        manager = ModelManager(config)
        
        assert manager.model == mock_model
        assert manager.labels == ['A', 'B']
    
    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_model_loading_failure(self, mock_open):
        """Test model loading failure"""
        config = Config()
        
        with pytest.raises(RuntimeError, match="Model file not found"):
            ModelManager(config)
    
    @patch('pickle.load')
    @patch('builtins.open')
    @patch('os.listdir', return_value=['data_A.csv', 'data_B.csv', 'data_C.csv'])
    def test_label_loading(self, mock_listdir, mock_open, mock_pickle_load):
        """Test fallback label loading from data directory"""
        mock_pickle_load.side_effect = [Mock(), TypeError("bad encoder")]
        config = Config()
        manager = ModelManager(config)
        
        assert manager.labels == ['A', 'B', 'C']
    
    @patch('pickle.load')
    @patch('builtins.open')
    @patch('os.listdir', return_value=['data_A.csv'])
    def test_predict_success(self, mock_listdir, mock_open, mock_pickle_load):
        """Test successful prediction"""
        mock_model = Mock()
        mock_model.predict.return_value = [0]
        mock_encoder = Mock()
        mock_encoder.classes_ = ['A']
        mock_pickle_load.side_effect = [mock_model, mock_encoder]
        
        config = Config()
        manager = ModelManager(config)
        
        features = np.random.randn(210)
        result = manager.predict(features)
        
        assert result == 'A'
        mock_model.predict.assert_called_once()
    
    @patch('pickle.load')
    @patch('builtins.open')
    @patch('os.listdir', return_value=['data_A.csv'])
    def test_predict_index_error(self, mock_listdir, mock_open, mock_pickle_load):
        """Test prediction with invalid index"""
        mock_model = Mock()
        mock_model.predict.return_value = [999]  # Invalid index
        mock_encoder = Mock()
        mock_encoder.classes_ = ['A']
        mock_pickle_load.side_effect = [mock_model, mock_encoder]
        
        config = Config()
        manager = ModelManager(config)
        
        features = np.random.randn(210)
        result = manager.predict(features)
        
        assert result is None


class TestUIRenderer:
    """Test UI rendering"""
    
    @pytest.fixture
    def renderer(self):
        """Create renderer instance"""
        config = Config()
        return UIRenderer(config)
    
    def test_renderer_initialization(self, renderer):
        """Test renderer initialization"""
        assert renderer.mp_drawing is not None
        assert renderer.mp_drawing_styles is not None
        assert renderer.mp_hands is not None
    
    @patch('cv2.putText')
    def test_draw_text_empty(self, mock_puttext, renderer):
        """Test drawing empty text"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        renderer.draw_text(image, "")
        
        # Should not call putText for empty string
        mock_puttext.assert_not_called()
    
    @patch('cv2.putText')
    @patch('cv2.getTextSize', return_value=((50, 30), 0))
    def test_draw_text_single_char(self, mock_getsize, mock_puttext, renderer):
        """Test drawing single character"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        renderer.draw_text(image, "A")
        
        # Should call putText once
        assert mock_puttext.call_count == 1
    
    @patch('cv2.rectangle')
    def test_draw_gesture_progress_zero(self, mock_rectangle, renderer):
        """Test drawing zero progress"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        renderer.draw_gesture_progress(image, 0.0)
        
        # Should not draw for zero progress
        mock_rectangle.assert_not_called()
    
    @patch('cv2.rectangle')
    def test_draw_gesture_progress_partial(self, mock_rectangle, renderer):
        """Test drawing partial progress"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        renderer.draw_gesture_progress(image, 0.5)
        
        # Should draw background and progress
        assert mock_rectangle.call_count == 2


class TestIntegration:
    """Integration tests"""
    
    def test_full_preprocessing_pipeline(self):
        """Test full preprocessing pipeline with realistic data"""
        config = Config()
        processor = HandKeypointProcessor(config)
        
        # Create realistic landmarks
        landmarks = []
        for i in range(21):
            landmark = Mock()
            landmark.x = 0.5 + np.random.randn() * 0.1
            landmark.y = 0.5 + np.random.randn() * 0.1
            landmarks.append(landmark)
        
        # Process
        result = processor.process_landmarks(landmarks)
        
        # Validate
        assert result.shape == (42,)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
    
    def test_gesture_and_prediction_timing(self):
        """Test gesture detection with prediction timing"""
        gesture_detector = GestureDetector(gesture_duration=0.1)
        prediction_manager = PredictionManager(display_duration=0.3, prediction_delay=0.1)
        
        # Start gesture
        gesture_detector.start_gesture()
        time.sleep(0.15)
        
        # Gesture should be ready
        assert gesture_detector.is_gesture_ready()
        
        # Make first prediction
        assert prediction_manager.can_predict()
        prediction_manager.add_prediction("A")
        
        # Cannot predict immediately
        assert not prediction_manager.can_predict()
        
        # Wait for delay
        time.sleep(0.15)
        assert prediction_manager.can_predict()
        prediction_manager.add_prediction("B")
        
        # Check accumulated text
        assert prediction_manager.get_display_text() == "AB"



class TestSharedCore:
    """Tests for shared preprocessing and hand-selection helpers."""

    def test_runtime_and_collection_preprocessing_match(self):
        config = Config()
        processor = HandKeypointProcessor(config)
        landmarks = []
        for i in range(21):
            landmark = Mock()
            landmark.x = 0.4 + i * 0.01
            landmark.y = 0.6 - i * 0.005
            landmarks.append(landmark)

        hand_landmarks = Mock()
        hand_landmarks.landmark = landmarks

        np.testing.assert_allclose(
            processor.process_landmarks(landmarks),
            extract_scaling(hand_landmarks, config),
        )

    def test_shared_hand_selection_prefers_configured_label(self):
        config = Config()
        config.RIGHT_HAND_ONLY = True
        config.TARGET_HAND_LABEL = "Left"

        results = Mock()
        results.multi_hand_landmarks = [Mock(), Mock()]
        right = Mock()
        right.classification = [Mock(label="Right")]
        left = Mock()
        left.classification = [Mock(label="Left")]
        results.multi_handedness = [right, left]

        assert select_target_hand_index(results, config) == 1

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])







