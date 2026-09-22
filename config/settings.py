"""
Guardian Matrix - Centralized Configuration & Settings
Loads configuration from .env or environment variables with typed defaults.
Supports dynamic override without modifying source code.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Automatically find and load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
env_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def _get_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "t")


@dataclass
class Settings:
    # RTSP / CCTV Camera Settings
    cctv_rtsp_url: str = _get_str("CCTV_RTSP_URL", "rtsp://192.168.1.50:8554/stream")
    target_fps: int = _get_int("TARGET_FPS", 15)
    frame_skip: int = _get_int("FRAME_SKIP", 1)
    reconnect_max_backoff: float = _get_float("RECONNECT_MAX_BACKOFF", 15.0)
    reconnect_initial_backoff: float = _get_float("RECONNECT_INITIAL_BACKOFF", 1.0)
    frozen_stream_timeout: float = _get_float("FROZEN_STREAM_TIMEOUT", 3.0)

    # Models & Paths
    model_path: str = _get_str("MODEL_PATH", "yolov8n.pt")
    pose_model_path: str = _get_str("POSE_MODEL_PATH", "yolov8n-pose.pt")
    device: str = _get_str("DEVICE", "cpu")

    # Thresholds & Tracking
    detection_confidence: float = _get_float("DETECTION_CONFIDENCE", 0.40)
    pose_confidence: float = _get_float("POSE_CONFIDENCE", 0.40)
    max_tracks: int = _get_int("MAX_TRACKS", 20)

    # Risk Scoring & Alerts
    alert_confidence: float = _get_float("ALERT_CONFIDENCE", 0.80)
    alert_confirmation_frames: int = _get_int("ALERT_CONFIRMATION_FRAMES", 10)
    alert_cooldown_seconds: float = _get_float("ALERT_COOLDOWN_SECONDS", 30.0)

    # Incident Recording
    incident_output_dir: str = _get_str("INCIDENT_OUTPUT_DIR", "incidents")
    pre_event_buffer_seconds: float = _get_float("PRE_EVENT_BUFFER_SECONDS", 5.0)
    post_event_buffer_seconds: float = _get_float("POST_EVENT_BUFFER_SECONDS", 5.0)

    def reload(self):
        """Reload settings from .env."""
        load_dotenv(dotenv_path=env_path, override=True)
        self.cctv_rtsp_url = _get_str("CCTV_RTSP_URL", self.cctv_rtsp_url)
        self.target_fps = _get_int("TARGET_FPS", self.target_fps)
        self.frame_skip = _get_int("FRAME_SKIP", self.frame_skip)
        self.model_path = _get_str("MODEL_PATH", self.model_path)
        self.pose_model_path = _get_str("POSE_MODEL_PATH", self.pose_model_path)
        self.device = _get_str("DEVICE", self.device)
        self.detection_confidence = _get_float("DETECTION_CONFIDENCE", self.detection_confidence)
        self.pose_confidence = _get_float("POSE_CONFIDENCE", self.pose_confidence)
        self.max_tracks = _get_int("MAX_TRACKS", self.max_tracks)
        self.alert_confidence = _get_float("ALERT_CONFIDENCE", self.alert_confidence)
        self.alert_confirmation_frames = _get_int("ALERT_CONFIRMATION_FRAMES", self.alert_confirmation_frames)
        self.alert_cooldown_seconds = _get_float("ALERT_COOLDOWN_SECONDS", self.alert_cooldown_seconds)
        self.incident_output_dir = _get_str("INCIDENT_OUTPUT_DIR", self.incident_output_dir)
        self.pre_event_buffer_seconds = _get_float("PRE_EVENT_BUFFER_SECONDS", self.pre_event_buffer_seconds)
        self.post_event_buffer_seconds = _get_float("POST_EVENT_BUFFER_SECONDS", self.post_event_buffer_seconds)


settings = Settings()
