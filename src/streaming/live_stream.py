"""
Live Video Stream Ingestion & Processing Layer for Guardian Matrix.
Supports webcams, network RTSP CCTV feeds, HTTP streams, and continuous file streams.
Implements asynchronous threaded frame capture to ensure real-time latency without buffer lag.
"""

from dataclasses import dataclass, field
from enum import Enum
import os
import queue
import threading
import time
from typing import Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union, Any
import cv2
import numpy as np

from ..pipeline import GuardianMatrixPipeline, PipelineFrameResult


class StreamType(str, Enum):
    WEBCAM = "WEBCAM"
    RTSP = "RTSP"
    HTTP = "HTTP"
    FILE_STREAM = "FILE_STREAM"
    CUSTOM = "CUSTOM"


@dataclass
class LiveStreamConfig:
    """Configuration settings for live stream processing."""
    source: Union[int, str] = 0
    target_fps: float = 15.0
    width: int = 640
    height: int = 480
    buffer_size: int = 1
    enable_loop: bool = True  # For file streams, replay when reaching end
    reconnect_delay_seconds: float = 2.0
    max_reconnect_attempts: int = 5
    drop_frames_when_busy: bool = True


class LiveStreamSource:
    """Handles raw camera/network connection and asynchronous frame capture."""

    def __init__(self, config: Optional[LiveStreamConfig] = None):
        self.config = config or LiveStreamConfig()
        self.stream_type = self._detect_stream_type(self.config.source)
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep latest frame for zero latency
        self._thread: Optional[threading.Thread] = None
        self.actual_fps: float = 0.0
        self.frames_captured: int = 0
        self.dropped_frames: int = 0
        self._lock = threading.Lock()

    def _detect_stream_type(self, source: Union[int, str]) -> StreamType:
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            return StreamType.WEBCAM
        src_str = str(source).lower()
        if src_str.startswith("rtsp://"):
            return StreamType.RTSP
        elif src_str.startswith("http://") or src_str.startswith("https://"):
            return StreamType.HTTP
        elif os.path.exists(str(source)):
            return StreamType.FILE_STREAM
        return StreamType.CUSTOM

    def start(self) -> bool:
        """Opens capture stream and starts asynchronous capture thread."""
        with self._lock:
            if self.is_running:
                return True
            success = self._open_capture()
            if not success:
                return False
            self.is_running = True
            self._thread = threading.Thread(target=self._capture_worker, daemon=True)
            self._thread.start()
            return True

    def _open_capture(self) -> bool:
        src = self.config.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        try:
            if self.stream_type == StreamType.WEBCAM:
                if os.name == 'nt' and isinstance(src, int):
                    self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(src)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            elif self.stream_type in (StreamType.RTSP, StreamType.HTTP):
                # Set RTSP transport over TCP for stability
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                self.cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            else:
                # File stream simulation
                self.cap = cv2.VideoCapture(str(src))

            if not self.cap or not self.cap.isOpened():
                return False

            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.actual_fps = fps if fps > 0 else self.config.target_fps
            return True
        except Exception as e:
            print(f"Error opening stream {src}: {e}")
            return False

    def _capture_worker(self):
        """Worker thread grabbing frames at stream rate."""
        frame_interval = 1.0 / max(1.0, self.actual_fps)
        last_frame_time = time.time()

        while self.is_running:
            cap = self.cap
            if cap is None:
                break
            now = time.time()
            # For file streams, pace playback to mimic genuine live broadcast
            if self.stream_type == StreamType.FILE_STREAM:
                sleep_needed = frame_interval - (now - last_frame_time)
                if sleep_needed > 0:
                    time.sleep(sleep_needed)
                last_frame_time = time.time()

            ret, frame = cap.read()
            if not ret:
                if self.stream_type == StreamType.FILE_STREAM and self.config.enable_loop:
                    # Loop video file continuously for live stream simulation
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.01)
                    continue

            self.frames_captured += 1

            # Put into queue; discard old frame if full to ensure zero latency
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.dropped_frames += 1
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def read_latest_frame(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """Fetch the freshest frame from the live capture worker."""
        if not self.is_running:
            return None
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Stops capture thread and releases video device."""
        with self._lock:
            self.is_running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
                self._thread = None
            if self.cap:
                self.cap.release()
                self.cap = None
                self.cap = None
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)


class LiveStreamProcessor:
    """
    Consumes frames from LiveStreamSource, passes them through GuardianMatrixPipeline,
    and provides real-time event feeds and MJPEG outputs.
    """

    def __init__(
        self,
        source: Optional[LiveStreamSource] = None,
        pipeline: Optional[GuardianMatrixPipeline] = None
    ):
        self.source = source or LiveStreamSource()
        self.pipeline = pipeline or GuardianMatrixPipeline()
        self.is_processing = False
        self._proc_thread: Optional[threading.Thread] = None
        self.latest_result: Optional[PipelineFrameResult] = None
        self.listeners: List[Callable[[PipelineFrameResult], None]] = []
        self.fps_tracker: List[float] = []

    def add_event_listener(self, callback: Callable[[PipelineFrameResult], None]):
        self.listeners.append(callback)

    def start_processing(self):
        if self.is_processing:
            return
        self.source.start()
        self.is_processing = True
        self._proc_thread = threading.Thread(target=self._process_worker, daemon=True)
        self._proc_thread.start()

    def _process_worker(self):
        frame_idx = 0
        t_start = time.time()

        while self.is_processing:
            frame = self.source.read_latest_frame(timeout=0.1)
            if frame is None:
                continue

            frame_idx += 1
            now = time.time()
            t_stamp = now - t_start

            result = self.pipeline.process_frame(
                frame=frame,
                timestamp=t_stamp,
                frame_id=frame_idx,
                annotate=True
            )
            self.latest_result = result

            for listener in self.listeners:
                try:
                    listener(result)
                except Exception as e:
                    print(f"Error in live event listener: {e}")

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """
        Yields multipart MJPEG frame stream for Flask `/video_feed` endpoints.
        """
        if not self.is_processing:
            self.start_processing()

        while self.is_processing:
            if self.latest_result is not None and self.latest_result.annotated_frame is not None:
                frame = self.latest_result.annotated_frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                    )
            time.sleep(0.04)  # ~25 FPS stream delivery

    def stop_processing(self):
        self.is_processing = False
        self.source.stop()
        if self._proc_thread and self._proc_thread.is_alive():
            self._proc_thread.join(timeout=1.0)
