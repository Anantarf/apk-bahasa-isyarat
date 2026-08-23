import csv
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from config import Config
from app_logging import get_logger
from sibi_core import extract_keypoint_features, select_target_hand_index

logger = get_logger(__name__)

@dataclass
class CollectionContext:
    cfg: Config
    label_data: str
    file_path: str
    sequence_length: int
    target_per_class: int
    mp_hands: object
    mp_drawing: object
    mp_styles: object
    window_name: str = "Make Data SIBI"


def extract_scaling(hand_landmarks, cfg: Config) -> np.ndarray:
    """Extract features using the shared runtime preprocessing pipeline."""
    return extract_keypoint_features(hand_landmarks.landmark, cfg)
def _get_target_hand_index(results, cfg: Config) -> Optional[int]:
    """Select hand index using the shared runtime handedness rule."""
    return select_target_hand_index(results, cfg)
def _put_overlay_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def _draw_status_overlay(
    frame: np.ndarray,
    label_data: str,
    data_counter: int,
    start_time: float,
    recording: bool,
) -> None:
    status = "REC" if recording else "PREVIEW"
    status_color = (0, 0, 255) if recording else (0, 255, 0)
    _put_overlay_text(frame, status, (10, 30), 1, status_color, 2)
    _put_overlay_text(frame, f"{label_data}: {data_counter}", (10, 70), 0.7, (255, 255, 0), 2)
    _put_overlay_text(frame, f"Time: {int(time.time() - start_time)}s", (10, 100), 0.6, (200, 200, 200), 1)
def _handle_keypress(key: int, recording: bool, frame_buffer: list[np.ndarray]) -> tuple[bool, bool]:
    if key == ord(" "):
        frame_buffer.clear()
        return not recording, False
    if key == 27:
        return recording, True
    return recording, False


def _validate_sequence_config(cfg: Config) -> int:
    if cfg.NUM_COORDINATES <= 0:
        raise ValueError("NUM_COORDINATES harus > 0")
    if cfg.NUM_FEATURES % cfg.NUM_COORDINATES != 0:
        raise ValueError(
            f"NUM_FEATURES ({cfg.NUM_FEATURES}) harus kelipatan NUM_COORDINATES ({cfg.NUM_COORDINATES})"
        )
    return cfg.NUM_FEATURES // cfg.NUM_COORDINATES


def _create_camera_error_frame(width: int, height: int, read_fail_count: int) -> np.ndarray:
    fallback = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(fallback, "Camera read failed", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(fallback, f"retry: {read_fail_count}", (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(fallback, "Press ESC to exit", (20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return fallback


def _process_frame_with_hands(frame: np.ndarray, hands) -> tuple[Optional[object], Optional[str]]:
    try:
        frame.flags.writeable = False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return hands.process(rgb), None
    except cv2.error as e:
        return None, f"OpenCV process error: {e}"
    except KeyboardInterrupt:
        raise
    except (RuntimeError, ValueError, TypeError) as e:
        return None, f"MediaPipe/process error: {e}"
    finally:
        frame.flags.writeable = True


def _draw_hand_landmarks(frame: np.ndarray, results, mp_hands, mp_drawing, mp_styles) -> None:
    if not results.multi_hand_landmarks:
        return

    for hand_landmarks in results.multi_hand_landmarks:
        try:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
        except cv2.error as e:
            logger.warning("OpenCV draw error: %s", e)


def _write_sequence_if_ready(
    frame_buffer: list[np.ndarray],
    sequence_length: int,
    cfg: Config,
    file_path: str,
) -> bool:
    if len(frame_buffer) != sequence_length:
        return False

    try:
        sequence = np.concatenate(frame_buffer).flatten()
    except ValueError as e:
        logger.warning("Gagal menyusun sequence: %s", e)
        frame_buffer.clear()
        return False

    if sequence.shape[0] != cfg.NUM_FEATURES:
        logger.warning("Dilewati: dimensi sequence %s != %s", sequence.shape[0], cfg.NUM_FEATURES)
        frame_buffer.clear()
        return False

    try:
        with open(file_path, "a", newline="") as f:
            csv.writer(f).writerow(sequence)
    except OSError as e:
        logger.error("Gagal simpan CSV: %s", e)
        frame_buffer.clear()
        return False

    frame_buffer.clear()
    return True


def _show_frame_and_handle_keys(
    window_name: str,
    frame: np.ndarray,
    recording: bool,
    frame_buffer: list[np.ndarray],
) -> tuple[bool, bool]:
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(5) & 0xFF
    return _handle_keypress(key, recording, frame_buffer)


def _handle_frame_processing_error(
    bgr: np.ndarray,
    label_data: str,
    data_counter: int,
    start_time: float,
    recording: bool,
    window_name: str,
    frame_buffer: list[np.ndarray],
) -> tuple[bool, bool]:
    frame_buffer.clear()
    display_frame = cv2.flip(bgr, 1)
    cv2.putText(
        display_frame,
        "Frame processing error",
        (10, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    _draw_status_overlay(display_frame, label_data, data_counter, start_time, recording)
    return _show_frame_and_handle_keys(window_name, display_frame, recording, frame_buffer)


def _collect_target_sample(
    results,
    cfg: Config,
    recording: bool,
    frame_buffer: list[np.ndarray],
    sequence_length: int,
    file_path: str,
) -> bool:
    if not results.multi_hand_landmarks:
        frame_buffer.clear()
        return False

    target_idx = _get_target_hand_index(results, cfg)
    if target_idx is None:
        frame_buffer.clear()
        return False

    if not recording:
        return False

    try:
        feat = extract_scaling(results.multi_hand_landmarks[target_idx], cfg)
    except ValueError as e:
        logger.warning("Fitur tidak valid: %s", e)
        frame_buffer.clear()
        return False

    frame_buffer.append(feat)
    return _write_sequence_if_ready(frame_buffer, sequence_length, cfg, file_path)


def _handle_camera_read_failure(
    window_name: str,
    fallback_width: int,
    fallback_height: int,
    read_fail_count: int,
    recording: bool,
    frame_buffer: list[np.ndarray],
) -> tuple[bool, bool]:
    fallback = _create_camera_error_frame(fallback_width, fallback_height, read_fail_count)
    return _show_frame_and_handle_keys(window_name, fallback, recording, frame_buffer)


def _handle_successful_capture(
    frame: np.ndarray,
    hands,
    context: CollectionContext,
    data_counter: int,
    start_time: float,
    recording: bool,
    frame_buffer: list[np.ndarray],
) -> tuple[int, bool, bool]:
    bgr = frame.copy()
    results, process_error_message = _process_frame_with_hands(frame, hands)
    if process_error_message is not None:
        logger.error(process_error_message)
        recording, should_break = _handle_frame_processing_error(
            bgr, context.label_data, data_counter, start_time, recording, context.window_name, frame_buffer
        )
        return data_counter, recording, should_break

    _draw_hand_landmarks(bgr, results, context.mp_hands, context.mp_drawing, context.mp_styles)
    if _collect_target_sample(results, context.cfg, recording, frame_buffer, context.sequence_length, context.file_path):
        data_counter += 1
        logger.info("Data ke-%s disimpan", data_counter)
        if data_counter >= context.target_per_class:
            logger.info("Target per kelas tercapai")
            recording = False

    display_frame = cv2.flip(bgr, 1)
    _draw_status_overlay(display_frame, context.label_data, data_counter, start_time, recording)
    recording, should_break = _show_frame_and_handle_keys(context.window_name, display_frame, recording, frame_buffer)
    return data_counter, recording, should_break

def _run_collection_loop(cap, hands, context: CollectionContext) -> None:
    frame_buffer: list[np.ndarray] = []
    data_counter = 0
    recording = False
    read_fail_count = 0
    start_time = time.time()
    fallback_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    fallback_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            read_fail_count += 1
            recording, should_break = _handle_camera_read_failure(
                context.window_name, fallback_width, fallback_height, read_fail_count, recording, frame_buffer
            )
            if should_break:
                break
            continue

        read_fail_count = 0
        fallback_height, fallback_width = frame.shape[:2]
        data_counter, recording, should_break = _handle_successful_capture(
            frame, hands, context, data_counter, start_time, recording, frame_buffer
        )
        if should_break:
            break

def collect_data(label_data: str, cfg: Config) -> None:
    """Collect sequence data and append into data_[label].csv."""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    sequence_length = _validate_sequence_config(cfg)
    target_per_class = int(cfg.TARGET_PER_CLASS)
    file_path = os.path.join(cfg.DATA_PATH, f"data_{label_data}.csv")
    os.makedirs(cfg.DATA_PATH, exist_ok=True)

    cap = cv2.VideoCapture(cfg.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak bisa dibuka. Periksa CAMERA_INDEX di config.py")

    try:
        with mp_hands.Hands(
            model_complexity=cfg.MODEL_COMPLEXITY,
            min_detection_confidence=cfg.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=cfg.MIN_TRACKING_CONFIDENCE,
        ) as hands:
            context = CollectionContext(
                cfg, label_data, file_path, sequence_length, target_per_class, mp_hands, mp_drawing, mp_styles
            )
            try:
                _run_collection_loop(cap, hands, context)
            except KeyboardInterrupt:
                logger.info("Pengambilan data dihentikan oleh pengguna.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
def main() -> None:
    cfg = Config()
    label_data = input("Masukkan label gesture: ").strip().upper()
    if not label_data:
        logger.warning("Label kosong, batal.")
        return

    collect_data(label_data, cfg)


if __name__ == "__main__":
    main()















