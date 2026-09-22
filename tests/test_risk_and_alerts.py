"""
Unit and integration tests for RiskAnalyzer, AlertManager, and IncidentRecorder.
"""

import unittest
import tempfile
import time
import os
import json
import numpy as np

from analysis.risk_analyzer import RiskAnalyzer
from alerts.alert_manager import AlertManager
from recording.incident_recorder import IncidentRecorder


class TestRiskAndAlerts(unittest.TestCase):
    def setUp(self):
        self.risk_analyzer = RiskAnalyzer(alert_confidence=0.80)
        self.alert_manager = AlertManager(confirmation_frames=3, cooldown_seconds=2.0, confidence_threshold=0.80)

    def test_normal_proximity_does_not_trigger_high_risk(self):
        """Verify Requirement 11: Normal close proximity must not be flagged as harassment."""
        tracks = [
            {"track_id": 1, "bbox": [10, 10, 50, 100], "center": (30, 55), "confidence": 0.9},
            {"track_id": 2, "bbox": [60, 10, 100, 100], "center": (80, 55), "confidence": 0.9},
        ]
        # Normal peaceful proximity: close distance, zero relative velocity, zero contact, zero pursuit
        interactions = [
            {
                "person_a": 1,
                "person_b": 2,
                "distance": 45.0,
                "relative_velocity": 0.0,
                "duration": 1.0,
                "contact_probability": 0.05,
                "pursuit_score": 0.0,
                "repeated_contact_score": 0.0,
            }
        ]
        result = self.risk_analyzer.evaluate(tracks, interactions)
        self.assertEqual(result["risk_level"], "LOW")
        self.assertLess(result["risk_score"], 0.50)

    def test_aggressive_interaction_triggers_high_risk(self):
        """Verify high velocity closure + contact produces HIGH/CRITICAL risk."""
        tracks = [
            {"track_id": 1, "bbox": [10, 10, 50, 100], "center": (30, 55), "confidence": 0.9},
            {"track_id": 2, "bbox": [55, 10, 95, 100], "center": (75, 55), "confidence": 0.9},
        ]
        interactions = [
            {
                "person_a": 1,
                "person_b": 2,
                "distance": 20.0,
                "relative_velocity": -35.0,  # rapid approach
                "duration": 4.5,
                "contact_probability": 0.75,
                "pursuit_score": 0.60,
                "repeated_contact_score": 0.70,
            }
        ]
        result = self.risk_analyzer.evaluate(tracks, interactions)
        self.assertIn(result["risk_level"], ("HIGH", "CRITICAL"))
        self.assertGreaterEqual(result["risk_score"], 0.80)
        self.assertIn(1, result["track_ids"])
        self.assertIn(2, result["track_ids"])

    def test_alert_confirmation_frames_and_cooldown(self):
        """Verify single-frame does not alert, requires 3 consecutive frames, and cooldown suppresses repeats."""
        high_risk_result = {
            "risk_score": 0.88,
            "risk_level": "HIGH",
            "evidence": ["rapid approach", "persistent close interaction"],
            "track_ids": [1, 2],
            "duration": 3.5,
        }

        # Frame 1: qualifying, but not enough frames yet
        alert1 = self.alert_manager.evaluate(high_risk_result, timestamp=100.0)
        self.assertIsNone(alert1)

        # Frame 2: still waiting for 3rd confirmation frame
        alert2 = self.alert_manager.evaluate(high_risk_result, timestamp=100.1)
        self.assertIsNone(alert2)

        # Frame 3: confirmed!
        alert3 = self.alert_manager.evaluate(high_risk_result, timestamp=100.2)
        self.assertIsNotNone(alert3)
        self.assertEqual(alert3["risk_level"], "HIGH")

        # Frame 4 (immediately after at timestamp 100.5): should be suppressed by cooldown (2.0s)
        alert4 = self.alert_manager.evaluate(high_risk_result, timestamp=100.5)
        self.assertIsNone(alert4)

        # Frame 5 (after cooldown elapsed, e.g. timestamp 102.5): should trigger again
        alert5 = self.alert_manager.evaluate(high_risk_result, timestamp=102.5)
        self.assertIsNotNone(alert5)

    def test_incident_recorder_saves_clip_and_metadata(self):
        """Verify pre-event buffer + post-event writes video and metadata.json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = IncidentRecorder(
                output_dir=temp_dir,
                target_fps=10,
                pre_event_seconds=0.5,  # 5 frames
                post_event_seconds=0.5, # 5 frames
            )

            # Feed pre-event frames
            for _ in range(10):
                dummy_frame = np.zeros((120, 160, 3), dtype=np.uint8)
                recorder.add_frame(dummy_frame)

            # Trigger alert
            alert_meta = {
                "incident_id": "TEST_INCIDENT",
                "timestamp": "2026-09-22 12:00:00",
                "risk_score": 0.92,
                "risk_level": "CRITICAL",
                "evidence": ["pursuit pattern"],
                "track_ids": [1, 2],
                "duration": 4.0,
            }
            recorder.trigger_recording(alert_meta)

            # Feed post-event frames to trigger finalization
            for _ in range(6):
                dummy_frame = np.zeros((120, 160, 3), dtype=np.uint8)
                recorder.add_frame(dummy_frame)

            # Wait for background worker thread to finish writing
            if recorder._last_save_thread:
                recorder._last_save_thread.join(timeout=3.0)
            else:
                time.sleep(0.5)

            # Check directory
            subdirs = [os.path.join(temp_dir, d) for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
            self.assertEqual(len(subdirs), 1)
            inc_dir = subdirs[0]
            
            video_file = os.path.join(inc_dir, "event.mp4")
            meta_file = os.path.join(inc_dir, "metadata.json")

            self.assertTrue(os.path.exists(video_file))
            self.assertTrue(os.path.exists(meta_file))

            with open(meta_file, "r") as f:
                saved_meta = json.load(f)
            self.assertEqual(saved_meta["incident_id"], "TEST_INCIDENT")
            self.assertEqual(saved_meta["risk_score"], 0.92)


if __name__ == "__main__":
    unittest.main()
