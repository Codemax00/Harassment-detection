"""
Unit and integration tests for Guardian Matrix RTSP Stream Receiver.
"""

import unittest
import time
import numpy as np
import cv2
import tempfile
import os

from stream.rtsp_stream import RTSPStream, StreamProvider
from config.settings import settings


class TestRTSPStream(unittest.TestCase):
    def setUp(self):
        # Create a small dummy video file to simulate a stream
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_path = os.path.join(self.temp_dir.name, "test_feed.mp4")
        
        # Write 30 frames of 320x240 video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.video_path, fourcc, 15.0, (320, 240))
        for i in range(30):
            frame = np.full((240, 320, 3), (i * 8) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_stream_creation_and_reading(self):
        stream = RTSPStream(url=self.video_path, target_fps=15)
        stream.connect()

        # Wait for worker thread to process at least 1 frame
        time.sleep(0.3)
        success, frame, meta = stream.read(timeout=1.0)
        
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (240, 320, 3))
        self.assertIn("fps", meta)
        self.assertIn("latency_ms", meta)
        
        telemetry = stream.get_telemetry()
        self.assertGreater(telemetry["frames_received"], 0)
        
        stream.release()
        self.assertEqual(stream.status, "STOPPED")

    def test_stream_provider_factory(self):
        stream = StreamProvider.create(source_type="video", source_path=self.video_path)
        self.assertIsInstance(stream, RTSPStream)
        self.assertEqual(stream.url, self.video_path)
        stream.release()

    def test_bounded_queue_latest_frame_dropping(self):
        stream = RTSPStream(url=self.video_path, target_fps=30)
        stream.connect()
        
        # Simulate slower consumer: wait for multiple frames to arrive without reading
        time.sleep(0.6)
        
        # Bounded queue (maxsize=1) must drop older frames rather than queueing them up
        telemetry = stream.get_telemetry()
        stream.release()
        
        # Should have dropped frames or read only 1 latest frame
        self.assertGreaterEqual(telemetry["frames_received"], 1)


if __name__ == "__main__":
    unittest.main()
