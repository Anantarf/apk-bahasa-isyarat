"""Shared core utilities for SIBI recognition and data collection."""
from typing import Optional

import numpy as np

from src.config import Config


def select_target_hand_index(results, cfg: Config) -> Optional[int]:
    """Select the hand index using the configured handedness rule."""
    if not getattr(results, "multi_hand_landmarks", None):
        return None

    if not cfg.RIGHT_HAND_ONLY:
        return 0

    desired_label = (getattr(cfg, "TARGET_HAND_LABEL", "Right") or "Right").strip().title()
    if desired_label not in {"Left", "Right"}:
        desired_label = "Right"

    multi_handedness = getattr(results, "multi_handedness", None)
    if not multi_handedness:
        return 0

    for idx, handedness in enumerate(multi_handedness):
        classification = getattr(handedness, "classification", None)
        if not classification:
            continue
        label = getattr(classification[0], "label", None)
        if label == desired_label:
            return idx

    return None


def center_keypoints(landmarks) -> np.ndarray:
    """Center landmark coordinates relative to wrist landmark 0."""
    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y
    centered_x = np.array([wrist_x - lm.x for lm in landmarks])
    centered_y = np.array([wrist_y - lm.y for lm in landmarks])
    return np.vstack([centered_x, centered_y])


def scale_keypoints(centered: np.ndarray, cfg: Config) -> np.ndarray:
    """Scale centered coordinates per axis to the configured normalization scale."""
    axis_max = np.max(np.abs(centered), axis=1, keepdims=True)
    axis_max[axis_max == 0] = 1.0
    return ((centered / axis_max) * cfg.NORMALIZATION_SCALE).reshape(-1)


def normalize_keypoints(scaled: np.ndarray, cfg: Config) -> np.ndarray:
    """Shift scaled coordinates into the positive feature range."""
    return scaled + cfg.NORMALIZATION_SCALE


def extract_keypoint_features(landmarks, cfg: Config) -> np.ndarray:
    """Extract normalized 42-dimensional hand keypoint features."""
    if len(landmarks) < cfg.NUM_LANDMARKS:
        raise ValueError(f"Landmark tidak lengkap: {len(landmarks)} < {cfg.NUM_LANDMARKS}")

    features = normalize_keypoints(scale_keypoints(center_keypoints(landmarks), cfg), cfg)
    if features.shape != (cfg.NUM_COORDINATES,):
        raise ValueError(f"Fitur tidak sesuai NUM_COORDINATES: {features.shape}")
    return features
