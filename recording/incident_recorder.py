"""
Guardian Matrix - Incident Video & Evidence Recorder
Maintains a rolling ring buffer (5s pre-event) and records 5s post-event frames
when an alert triggers, writing event.mp4 and metadata.json asynchronously.
"""

import os
import json
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
from config.settings import settings


class IncidentRecorder:
    """
    Asynchronous event clip recorder with pre-event rolling buffer and post-event capture.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        target_fps: Optional[int] = None,
        pre_event_seconds: Optional[float] = None,
        post_event_seconds: Optional[float] = None,
        camera_id: str = "camera_01",
    ):
        self.output_dir = Path(output_dir or settings.incident_output_dir)
        self.target_fps = target_fps or settings.target_fps
        self.pre_event_seconds = pre_event_seconds if pre_event_seconds is not None else settings.pre_event_buffer_seconds
        self.post_event_seconds = post_event_seconds if post_event_seconds is not None else settings.post_event_buffer_seconds
        self.camera_id = camera_id

        self.pre_buffer_size = max(1, int(self.target_fps * self.pre_event_seconds))
        self.post_buffer_size = max(1, int(self.target_fps * self.post_event_seconds))

        # Circular buffer for pre-event frames
        self._pre_buffer: deque = deque(maxlen=self.pre_buffer_size)
        self._is_recording_incident = False
        self._post_frames: list = []
        self._active_alert_meta: Optional[Dict[str, Any]] = None
        self._last_save_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def add_frame(self, frame: np.ndarray):
        """Adds incoming frame to rolling pre-event buffer or active incident recording."""
        if frame is None or frame.size == 0:
            return

        with self._lock:
            if not self._is_recording_incident:
                # Save a copy to prevent in-place mutation
                self._pre_buffer.append(frame.copy())
            else:
                self._post_frames.append(frame.copy())
                # Check if post-event duration completed
                if len(self._post_frames) >= self.post_buffer_size:
                    self._finalize_recording()

    def trigger_recording(self, alert_data: Dict[str, Any]):
        """Called when an alert is triggered to initiate incident clip capture."""
        with self._lock:
            if self._is_recording_incident:
                return  # Already recording an active incident

            self._is_recording_incident = True
            self._active_alert_meta = dict(alert_data)
            self._post_frames = []

    def _finalize_recording(self):
        """Asynchronously writes video file and metadata.json."""
        pre_frames = list(self._pre_buffer)
        post_frames = list(self._post_frames)
        all_frames = pre_frames + post_frames
        meta = self._active_alert_meta or {}

        # Reset states
        self._is_recording_incident = False
        self._active_alert_meta = None
        self._post_frames = []

        # Run writer in worker thread to avoid blocking AI pipeline
        t = threading.Thread(
            target=self._write_incident_to_disk,
            args=(all_frames, meta),
            daemon=True,
            name="IncidentDiskWriter",
        )
        self._last_save_thread = t
        t.start()

    def _write_incident_to_disk(self, frames: list, meta: Dict[str, Any]):
        if not frames:
            return

        try:
            timestamp_slug = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            incident_dir = self.output_dir / timestamp_slug
            incident_dir.mkdir(parents=True, exist_ok=True)

            video_path = incident_dir / "event.mp4"
            meta_path = incident_dir / "metadata.json"

            # Write Video
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, float(self.target_fps), (w, h))

            for f in frames:
                # Ensure frame matches resolution
                if f.shape[0] != h or f.shape[1] != w:
                    f = cv2.resize(f, (w, h))
                writer.write(f)
            writer.release()

            # Write Metadata
            meta_payload = {
                "timestamp": meta.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "camera": self.camera_id,
                "incident_id": meta.get("incident_id", f"INCIDENT_{int(time.time())}"),
                "track_ids": meta.get("track_ids", []),
                "risk_score": meta.get("risk_score", 0.0),
                "risk_level": meta.get("risk_level", "UNKNOWN"),
                "duration": meta.get("duration", 0.0),
                "evidence": meta.get("evidence", []),
                "frames_count": len(frames),
                "fps": self.target_fps,
                "video_file": "event.mp4",
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, indent=2)

            print(f"[INCIDENT SAVED] Video & metadata written to: {incident_dir}")
        except Exception as e:
            print(f"[WARNING] Error saving incident recording: {e}")
