"""
Unit tests for Pairwise Interaction Engine and Structured Evidence Builder.
"""

import unittest
import numpy as np

from src.pose.pose_representation import PoseFrame, KeypointName
from src.interaction.interaction_engine import InteractionEngine, InteractionFeatures
from src.temporal.temporal_engine import TemporalMotionFeatures
from src.evidence.structured_evidence import build_structured_evidence, StructuredEvidence
from src.taxonomy.event_taxonomy import EventLabel
from src.tracking.tracker import TrackedPerson


class TestInteractionAndEvidence(unittest.TestCase):

    def test_pairwise_interaction_analysis(self):
        engine = InteractionEngine()

        # Person 1 at origin
        kps1 = np.zeros((17, 2), dtype=np.float32)
        kps1[9] = [0.1, 0.0]  # Left wrist
        pose1 = PoseFrame(track_id=1, timestamp=0.0, keypoints_2d=kps1, pose_confidence=0.9, is_normalized=True)

        # Person 2 close at distance 0.3
        kps2 = np.zeros((17, 2), dtype=np.float32)
        kps2[10] = [0.12, 0.0]  # Right wrist very close to Person 1's hand
        pose2 = PoseFrame(track_id=2, timestamp=0.0, keypoints_2d=kps2, pose_confidence=0.88, is_normalized=True)

        interactions = engine.update([pose1, pose2], motion_features={}, timestamp=0.0)
        self.assertEqual(len(interactions), 1)
        inter = interactions[0]
        self.assertEqual(inter.person_a, 1)
        self.assertEqual(inter.person_b, 2)
        # Hand to hand distance is very small (0.02)
        self.assertTrue(inter.hand_to_hand_distance < 0.1)
        self.assertTrue(inter.hand_contact_probability > 0.5)

    def test_structured_evidence_compilation(self):
        tracked = [
            TrackedPerson(track_id=1, bbox=(10, 10, 50, 100), detection_confidence=0.9, timestamp=1.0),
            TrackedPerson(track_id=2, bbox=(55, 10, 95, 100), detection_confidence=0.85, timestamp=1.0)
        ]
        poses = [
            PoseFrame(track_id=1, timestamp=1.0, keypoints_2d=np.zeros((17, 2)), pose_confidence=0.9),
            PoseFrame(track_id=2, timestamp=1.0, keypoints_2d=np.zeros((17, 2)), pose_confidence=0.85)
        ]
        inter = InteractionFeatures(
            person_a=1, person_b=2, distance=0.45, relative_velocity=-0.8,
            relative_direction=(1.0, 0.0), orientation_similarity=0.2, is_facing_each_other=True,
            hand_to_body_min_distance=0.1, hand_to_hand_distance=0.05, hand_contact_probability=0.8,
            movement_synchronization=0.6, proximity_duration_seconds=2.5, repeated_contact_score=0.75,
            pursuit_score=0.2, separation_score=0.0, restraint_likelihood=0.65
        )

        evidence = build_structured_evidence(
            timestamp=1.0,
            frame_id=30,
            tracked_people=tracked,
            poses=poses,
            interactions=[inter],
            motion_features={},
            classified_event=EventLabel.REPEATED_CONTACT_PATTERN,
            event_confidence=0.84,
            temporal_probs={EventLabel.REPEATED_CONTACT_PATTERN.value: 0.84}
        )

        self.assertEqual(evidence.scene["person_count"], 2)
        self.assertIsNotNone(evidence.interaction)
        self.assertEqual(evidence.interaction["pair"], [1, 2])
        self.assertEqual(evidence.temporal_model["interaction_pattern"], "REPEATED_CONTACT_PATTERN")
        self.assertIn("pose_quality", evidence.quality_metrics)

        # Ensure serialization to valid JSON works
        json_str = evidence.to_json()
        self.assertIn("REPEATED_CONTACT_PATTERN", json_str)


if __name__ == "__main__":
    unittest.main()
