"""
Unit and Integration Tests for Live Stream Ingestion & Processing.
"""

import unittest
import os
import time
import numpy as np

from src.streaming.live_stream import (
    LiveStreamSource,
    LiveStreamProcessor,
    LiveStreamConfig,
    StreamType
)
from helpers.guardian_matrix_adapter import GuardianMatrixLiveStreamManager


class TestLiveStream(unittest.TestCase):

    def setUp(self):
        self.sample_video = os.path.join("test_videos", "WhatsApp Video 2025-09-12 at 10.26.50 AM.mp4")

    def test_stream_type_detection(self):
        cfg_cam = LiveStreamConfig(source=0)
        src_cam = LiveStreamSource(cfg_cam)
        self.assertEqual(src_cam.stream_type, StreamType.WEBCAM)

        cfg_cam_str = LiveStreamConfig(source="1")
        src_cam_str = LiveStreamSource(cfg_cam_str)
        self.assertEqual(src_cam_str.stream_type, StreamType.WEBCAM)

        cfg_rtsp = LiveStreamConfig(source="rtsp://192.168.1.100:554/live/ch0")
        src_rtsp = LiveStreamSource(cfg_rtsp)
        self.assertEqual(src_rtsp.stream_type, StreamType.RTSP)

        cfg_http = LiveStreamConfig(source="http://camera.local/stream.mjpg")
        src_http = LiveStreamSource(cfg_http)
        self.assertEqual(src_http.stream_type, StreamType.HTTP)

        if os.path.exists(self.sample_video):
            cfg_file = LiveStreamConfig(source=self.sample_video)
            src_file = LiveStreamSource(cfg_file)
            self.assertEqual(src_file.stream_type, StreamType.FILE_STREAM)

    def test_live_stream_capture_and_mjpeg(self):
        if not os.path.exists(self.sample_video):
            self.skipTest("Sample video file not present")

        cfg = LiveStreamConfig(source=self.sample_video, target_fps=30.0, enable_loop=True)
        source = LiveStreamSource(cfg)
        self.assertTrue(source.start())

        # Test asynchronous frame retrieval
        frame = source.read_latest_frame(timeout=2.0)
        self.assertIsNotNone(frame)
        self.assertTrue(isinstance(frame, np.ndarray))
        self.assertTrue(frame.shape[0] > 0 and frame.shape[1] > 0)

        # Test processor and event listener
        processor = LiveStreamProcessor(source=source)
        received_events = []

        def on_event(res):
            received_events.append(res.frame_id)

        processor.add_event_listener(on_event)
        processor.start_processing()

        # Wait for processor to consume at least one frame
        t0 = time.time()
        while len(received_events) == 0 and (time.time() - t0 < 5.0):
            time.sleep(0.1)
        self.assertTrue(len(received_events) > 0)
        self.assertIsNotNone(processor.latest_result)

        # Test MJPEG stream generation
        mjpeg_gen = processor.generate_mjpeg_stream()
        chunk = next(mjpeg_gen)
        self.assertTrue(chunk.startswith(b'--frame\r\n'))
        self.assertIn(b'Content-Type: image/jpeg', chunk)

        processor.stop_processing()

    def test_guardian_matrix_live_stream_manager(self):
        if not os.path.exists(self.sample_video):
            self.skipTest("Sample video file not present")

        manager = GuardianMatrixLiveStreamManager()
        started = manager.start_stream(source=self.sample_video, fps=30.0)
        self.assertTrue(started)

        t0 = time.time()
        while time.time() - t0 < 5.0:
            status = manager.get_latest_status()
            if status.get("frame_id", 0) > 0:
                break
            time.sleep(0.1)

        status = manager.get_latest_status()
        self.assertTrue(status.get("active"))
        self.assertIn("pattern", status)
        self.assertIn("policy_state", status)
        self.assertIn("confidence", status)

        # Retrieve MJPEG stream chunks
        stream_iter = manager.get_mjpeg_stream()
        first_chunk = next(stream_iter)
        self.assertTrue(first_chunk.startswith(b'--frame\r\n'))

        manager.stop_stream()
        self.assertIsNone(manager.active_source)


if __name__ == "__main__":
    unittest.main()
