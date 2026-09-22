"""
Unit tests for Pose Representation, Normalization, and Temporal Motion Engine.
"""

import unittest
import numpy as np

from src.pose.pose_representation import PoseFrame, normalize_pose, KeypointName
from src.pose.pose_estimator import YOLOPoseEstimator, SapiensPoseEstimator, GEMXPoseEstimator
from src.temporal.temporal_engine import TemporalMotionEngine, compute_angle_3points
from src.tracking.tracker import TrackedPerson


class TestPoseAndTemporal(unittest.TestCase):

    def test_pose_normalization(self):
        kps = np.zeros((17, 2), dtype=np.float32)
        kps[0] = [150.0, 100.0]  # Nose
        bbox = (100.0, 50.0, 200.0, 250.0)  # Center (150, 150), height 200
        raw_pose = PoseFrame(
            track_id=1,
            timestamp=0.0,
            keypoints_2d=kps,
            pose_confidence=0.85,
            bbox=bbox
        )

        norm = normalize_pose(raw_pose, bbox=bbox)
        self.assertTrue(norm.is_normalized)
        # Check normalized nose coordinate: (150 - 150)/200 = 0.0, (100 - 150)/200 = -0.25
        self.assertAlmostEqual(norm.keypoints_2d[0, 0], 0.0, places=4)
        self.assertAlmostEqual(norm.keypoints_2d[0, 1], -0.25, places=4)
        # 3D keypoints must be generated
        self.assertIsNotNone(norm.keypoints_3d)
        self.assertEqual(norm.keypoints_3d.shape, (17, 3))

    def test_angle_computation(self):
        # Right angle triangle: (0, 1) -> (0, 0) -> (1, 0) is 90 degrees
        p1 = (0.0, 1.0)
        p2 = (0.0, 0.0)
        p3 = (1.0, 0.0)
        angle = compute_angle_3points(p1, p2, p3)
        self.assertAlmostEqual(angle, 90.0, places=3)

    def test_temporal_motion_engine(self):
        engine = TemporalMotionEngine(window_seconds=2.0)
        # Simulate a person moving across frames
        poses = []
        for i in range(5):
            kps = np.zeros((17, 2), dtype=np.float32)
            # Torso moving rightwards by 0.1 normalized unit per second
            kps[11] = [0.1 * i, 0.0]  # Left hip
            kps[12] = [0.1 * i, 0.0]  # Right hip
            kps[5] = [0.1 * i, -0.5]  # Left shoulder
            kps[6] = [0.1 * i, -0.5]  # Right shoulder

            pf = PoseFrame(
                track_id=1,
                timestamp=i * 0.1,  # 100ms apart
                keypoints_2d=kps,
                pose_confidence=0.9,
                frame_id=i,
                is_normalized=True
            )
            features = engine.update([pf])

        self.assertIn(1, features)
        mf = features[1]
        # Velocity in x should be approximately +0.1 / 0.1 = 1.0
        self.assertAlmostEqual(mf.torso_velocity[0], 1.0, places=2)
        self.assertTrue(mf.torso_speed > 0.5)


if __name__ == "__main__":
    unittest.main()
