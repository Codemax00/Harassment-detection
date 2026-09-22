"""
Unit tests for Deterministic Safety Policy and Event Logger.
"""

import unittest
import os
import shutil

from src.policy.safety_policy import SafetyPolicy, PolicyState
from src.logging.event_logger import EventLogger
from src.system1.laya_decision import LayaDecision, DecisionState
from src.system2.verifier import System2VerificationResult
from src.evidence.structured_evidence import StructuredEvidence
from src.taxonomy.event_taxonomy import EventLabel


class TestSafetyPolicyAndLogger(unittest.TestCase):

    def setUp(self):
        self.test_log_dir = "outputs/test_events_log"
        self.logger = EventLogger(log_dir=self.test_log_dir, random_seed=123)
        self.policy = SafetyPolicy()

    def tearDown(self):
        if os.path.exists(self.test_log_dir):
            shutil.rmtree(self.test_log_dir, ignore_errors=True)

    def test_safety_policy_transitions(self):
        evidence = StructuredEvidence(
            timestamp=1.0, frame_id=30, scene={}, interaction=None, motion={},
            pose={}, temporal_model={"interaction_pattern": EventLabel.RESTRAINT_PATTERN.value},
            quality_metrics={}
        )

        s1_review = LayaDecision(
            decision=DecisionState.REVIEW, calibrated_confidence=0.88, raw_confidence=0.85,
            passes_quality_gate=True, quality_gate_details={}, escalate_to_system2=True,
            rationale="Review trigger", timestamp=1.0, frame_id=30
        )

        # Verified by System-2 -> Triggers ALERT
        s2_verified = System2VerificationResult(
            is_verified=True, verified_decision="CONFIRMED_RISK",
            verification_confidence=0.9, analysis_text="Confirmed risk", inspected_elements=[]
        )
        action_alert = self.policy.evaluate_policy(evidence, s1_review, s2_verified)
        self.assertEqual(action_alert.state, PolicyState.ALERT)
        self.assertTrue(action_alert.trigger_alert)
        self.assertTrue(action_alert.request_human_review)

    def test_event_logger_provenance(self):
        evidence = StructuredEvidence(
            timestamp=1.0, frame_id=30, scene={"tracked_ids": [7, 12]}, interaction={"pair": [7, 12]},
            motion={}, pose={}, temporal_model={"interaction_pattern": "REPEATED_CONTACT_PATTERN"},
            quality_metrics={}
        )
        s1 = LayaDecision(
            decision=DecisionState.REVIEW, calibrated_confidence=0.85, raw_confidence=0.8,
            passes_quality_gate=True, quality_gate_details={}, escalate_to_system2=False,
            rationale="Review", timestamp=1.0, frame_id=30
        )
        action = self.policy.evaluate_policy(evidence, s1, None)

        record = self.logger.log_event(
            video_id="TEST_VID_001",
            evidence=evidence,
            policy_action=action,
            confidence=0.85,
            pose_model_name="YOLO_Pose",
            temporal_model_name="ST-GCN"
        )

        self.assertTrue(record.event_id.startswith("GM-"))
        self.assertEqual(record.track_ids, [7, 12])
        self.assertIn("python_version", record.reproducibility)
        self.assertEqual(record.reproducibility["random_seed"], 123)

        # Verify file written to disk
        expected_file = os.path.join(self.test_log_dir, f"{record.event_id}_{action.state.value}.json")
        self.assertTrue(os.path.exists(expected_file))


if __name__ == "__main__":
    unittest.main()
