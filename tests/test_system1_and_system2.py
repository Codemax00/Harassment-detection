"""
Unit tests for Laya System-1 and System-2 Verification Layers.
"""

import unittest

from src.system1.laya_decision import LayaSystem1Engine, DecisionState
from src.system2.verifier import System2Verifier
from src.evidence.structured_evidence import StructuredEvidence
from src.taxonomy.event_taxonomy import EventLabel


class TestSystem1AndSystem2(unittest.TestCase):

    def setUp(self):
        self.s1 = LayaSystem1Engine(
            min_pose_quality=0.6,
            min_tracking_quality=0.6,
            min_temporal_consistency=0.6,
            min_evidence_completeness=0.6
        )
        self.s2 = System2Verifier()

    def test_system1_normal_activity(self):
        evidence = StructuredEvidence(
            timestamp=0.5,
            frame_id=15,
            scene={"person_count": 2},
            interaction=None,
            motion={"rapid_approach": 0.1},
            pose={"confidence": 0.85},
            temporal_model={"interaction_pattern": EventLabel.NORMAL_ACTIVITY.value, "confidence": 0.9},
            quality_metrics={
                "pose_quality": 0.85,
                "tracking_quality": 0.90,
                "temporal_consistency": 0.80,
                "evidence_completeness": 0.85
            }
        )
        decision = self.s1.evaluate(evidence)
        self.assertEqual(decision.decision, DecisionState.NORMAL)
        self.assertFalse(decision.escalate_to_system2)
        self.assertTrue(decision.passes_quality_gate)

    def test_system1_quality_gate_fallback_uncertain(self):
        # Degraded pose quality below 0.6 threshold
        evidence = StructuredEvidence(
            timestamp=0.5,
            frame_id=15,
            scene={"person_count": 2},
            interaction=None,
            motion={},
            pose={"confidence": 0.3},
            temporal_model={"interaction_pattern": EventLabel.AGGRESSIVE_MOTION_PATTERN.value, "confidence": 0.9},
            quality_metrics={
                "pose_quality": 0.3,  # Fails gate!
                "tracking_quality": 0.9,
                "temporal_consistency": 0.8,
                "evidence_completeness": 0.8
            }
        )
        decision = self.s1.evaluate(evidence)
        # Cannot emit high confidence alert when quality gates fail
        self.assertEqual(decision.decision, DecisionState.UNCERTAIN)
        self.assertFalse(decision.passes_quality_gate)
        # Escalates to System-2 for verification
        self.assertTrue(decision.escalate_to_system2)

    def test_system2_verification_confirmed_risk(self):
        evidence = StructuredEvidence(
            timestamp=2.0,
            frame_id=60,
            scene={"person_count": 2},
            interaction={
                "pair": [1, 2],
                "repeated_contact_score": 0.75,
                "pursuit_score": 0.65,
                "duration_s": 2.5
            },
            motion={"repeated_contact": 0.75},
            pose={"confidence": 0.85},
            temporal_model={"interaction_pattern": EventLabel.REPEATED_CONTACT_PATTERN.value, "confidence": 0.88},
            quality_metrics={"pose_quality": 0.85, "tracking_quality": 0.85, "temporal_consistency": 0.85, "evidence_completeness": 0.9}
        )
        s1_decision = self.s1.evaluate(evidence)
        self.assertIn(s1_decision.decision, (DecisionState.REVIEW, DecisionState.SUSPICIOUS))
        self.assertTrue(s1_decision.escalate_to_system2)

        s2_res = self.s2.verify(s1_decision, evidence)
        self.assertTrue(s2_res.is_verified)
        self.assertEqual(s2_res.verified_decision, "CONFIRMED_RISK")


if __name__ == "__main__":
    unittest.main()
