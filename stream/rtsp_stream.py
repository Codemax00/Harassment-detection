"""
Guardian Matrix - Robust RTSP Stream Receiver
Implements background threaded frame grabbing, exponential backoff auto-reconnect,
frozen stream detection, latest-frame bounded buffer, and telemetry (FPS, latency, dropped frames).
"""

import os
import time
import queue
import threading
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

# Force TCP transport for FFmpeg RTSP to avoid UDP packet loss artifacts
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class RTSPStream:
    """
    Production-grade RTSP stream receiver with automatic reconnection,
    frozen stream watchdog, bounded buffer latest-frame strategy, and telemetry.
    """

    def __init__(
        self,
        url: str,
        target_fps: int = 15,
        reconnect_initial_backoff: float = 1.0,
        reconnect_max_backoff: float = 15.0,
        frozen_stream_timeout: float = 8.0,
    ):
        self.url = str(url).strip()
        self.target_fps = target_fps
        self.initial_backoff = reconnect_initial_backoff
        self.max_backoff = reconnect_max_backoff
        self.frozen_timeout = frozen_stream_timeout

        # Connection state: OFFLINE, CONNECTING, CONNECTED, RECONNECTING, STOPPED
        self.status = "OFFLINE"
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Latest frame buffer (capacity 1 to prevent latency buildup)
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()

        # Telemetry metrics
        self.frames_received = 0
        self.frames_dropped = 0
        self.fps = 0.0
        self.latency_ms = 0.0
        self.last_frame_time = 0.0
        self.reconnect_count = 0
        self.last_error = ""

    def connect(self) -> bool:
        """Starts the background capture thread."""
        if self._thread is not None and self._thread.is_alive():
            return True

        self._stop_event.clear()
        self.status = "CONNECTING"
        self._thread = threading.Thread(target=self._capture_loop, name="RTSPStreamWorker", daemon=True)
        self._thread.start()
        return True

    def _open_capture(self) -> bool:
        """Internal helper to initialize OpenCV VideoCapture with low buffer size."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        # Configure FFmpeg options dynamically based on stream protocol
        if ".m3u8" in self.url.lower() or "http" in self.url.lower():
            # Enable automatic HTTP/HLS segment reconnection without dropping the stream
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "reconnect;1|reconnect_at_eof;1|reconnect_streamed;1|reconnect_delay_max;2"
        else:
            # Force TCP for RTSP to prevent packet loss
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;500000"

        # Check if URL is numeric (webcam index)
        if self.url.isdigit():
            cap = cv2.VideoCapture(int(self.url))
        else:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            return False

        # Only set buffer size 1 for raw RTSP; HLS needs segment buffering
        if ".m3u8" not in self.url.lower():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        self._cap = cap
        return True

    def _capture_loop(self):
        """Worker loop running in background thread with paced frame ingestion and HLS resilience."""
        current_backoff = self.initial_backoff
        consecutive_read_failures = 0
        frame_interval = 1.0 / max(5, self.target_fps)

        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                self.status = "CONNECTING" if self.reconnect_count == 0 else "RECONNECTING"
                success = self._open_capture()
                if not success:
                    self.status = "OFFLINE"
                    self.reconnect_count += 1
                    time.sleep(current_backoff)
                    current_backoff = min(current_backoff * 2.0, self.max_backoff)
                    continue
                else:
                    self.status = "CONNECTED"
                    current_backoff = self.initial_backoff
                    self.last_frame_time = time.time()
                    consecutive_read_failures = 0

            # Read frame from stream
            t_start = time.time()
            ret, frame = self._cap.read()

            if not ret or frame is None:
                consecutive_read_failures += 1
                # HLS segment refresh tolerance: wait up to 4.0s (50 attempts * 0.08s)
                # without tearing down the connection!
                if consecutive_read_failures < 50:
                    time.sleep(0.08)
                    continue

                # Stream has genuinely dropped after sustained failure (> 4s)
                self.status = "OFFLINE"
                self.last_error = "Frame read failed (stream disconnected or ended)"
                if self._cap:
                    self._cap.release()
                    self._cap = None
                self.reconnect_count += 1
                consecutive_read_failures = 0
                time.sleep(current_backoff)
                current_backoff = min(current_backoff * 2.0, self.max_backoff)
                continue

            # Successfully received a frame: reset failure counter
            consecutive_read_failures = 0

            now = time.time()
            self.latency_ms = (now - t_start) * 1000.0
            time_since_last = now - self.last_frame_time if self.last_frame_time > 0 else 0.033
            self.last_frame_time = now
            self.frames_received += 1
            if time_since_last > 0:
                instant_fps = 1.0 / time_since_last
                self.fps = 0.8 * self.fps + 0.2 * instant_fps if self.fps > 0 else instant_fps

            # Pace frame ingestion to real camera speed to prevent buffer exhaustion
            read_elapsed = time.time() - t_start
            if read_elapsed < frame_interval:
                time.sleep(frame_interval - read_elapsed)

            # Frozen stream watchdog
            if self.last_frame_time > 0 and (now - self.last_frame_time) > self.frozen_timeout:
                self.status = "OFFLINE"
                self.last_error = f"Stream frozen: no frames received for > {self.frozen_timeout}s"
                if self._cap:
                    self._cap.release()
                    self._cap = None
                continue

            self.status = "CONNECTED"

            # Enqueue latest frame; if queue is full, drop older frame to ensure 0-latency
            meta = {
                "timestamp": now,
                "frame_id": self.frames_received,
                "fps": round(self.fps, 1),
                "latency_ms": round(self.latency_ms, 1),
            }
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass

            try:
                self._frame_queue.put_nowait((frame, meta))
            except queue.Full:
                self.frames_dropped += 1

    def read(self, timeout: float = 0.5) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Retrieves the latest available frame without lag.
        Returns: (success: bool, frame: Optional[np.ndarray], metadata: dict)
        """
        try:
            frame, meta = self._frame_queue.get(timeout=timeout)
            return True, frame, meta
        except queue.Empty:
            # Check if frozen
            if self.status == "CONNECTED" and self.last_frame_time > 0:
                if (time.time() - self.last_frame_time) > self.frozen_timeout:
                    self.status = "OFFLINE"
            return False, None, {
                "status": self.status,
                "fps": round(self.fps, 1),
                "latency_ms": round(self.latency_ms, 1),
                "dropped_frames": self.frames_dropped,
                "reconnect_count": self.reconnect_count,
            }

    def reconnect(self):
        """Force manual reconnection."""
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self.status = "RECONNECTING"

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns current stream diagnostics."""
        return {
            "status": self.status,
            "url": self.url,
            "fps": round(self.fps, 1),
            "latency_ms": round(self.latency_ms, 1),
            "frames_received": self.frames_received,
            "frames_dropped": self.frames_dropped,
            "reconnect_count": self.reconnect_count,
            "last_error": self.last_error,
        }

    def release(self):
        """Cleanly releases all resources and stops background thread."""
        self.status = "STOPPED"
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        # Drain queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break


class StreamProvider:
    """Factory and manager for video streams (RTSP, Webcam, Video file)."""

    @staticmethod
    def create(source_type: str = "rtsp", source_path: Optional[str] = None, **kwargs) -> RTSPStream:
        if source_type.lower() == "rtsp":
            from config.settings import settings
            url = source_path or settings.cctv_rtsp_url
            return RTSPStream(url=url, target_fps=settings.target_fps, **kwargs)
        elif source_type.lower() in ("webcam", "camera"):
            url = source_path or "0"
            return RTSPStream(url=url, **kwargs)
        elif source_type.lower() == "video":
            if not source_path:
                raise ValueError("Video source requires valid file path.")
            return RTSPStream(url=source_path, **kwargs)
        else:
            raise ValueError(f"Unknown stream source type: {source_type}")
