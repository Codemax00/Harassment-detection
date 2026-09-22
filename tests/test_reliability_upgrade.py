"""
Guardian Matrix Final Reliability & Validation Test Suite.
Verifies:
1. Dual-threshold hysteresis (0.81 -> 0.65 -> 0.50 -> 0.44)
2. Single-frame spike rejection
3. Fall vs Aggression strict separation
4. Aggressive interaction confirmation (directional strike)
5. Normal walking arm swing non-aggression
6. Close proximity standing non-aggression
7. Duplicate person suppression (IoU > 0.40)
8. Forensic evidence bundle generation (metadata.json, timeline.json, keyframe.jpg, clip.mp4)
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from src.taxonomy.event_taxonomy import (
    EventCategory,
    ActionType,
    OperationalUrgency,
    EvidenceState,
    EventClassification,
    EventTaxonomy
)
from src.evidence.structured_evidence import EvidenceSnapshot, RollingEvidenceBuffer
from src.temporal.temporal_decision_engine import (
    TemporalDecisionEngine,
    TemporalDecisionConfig,
    TemporalDecisionOutput
)
from src.policy.incident_state_machine import (
    IncidentStateMachine,
    IncidentStateMachineConfig,
    StateMachineResult
)
from src.evidence.evidence_store import EvidenceStore, EvidenceStoreConfig
from src.pose.pose_representation import PoseFrame, KeypointName
from src.temporal.temporal_engine import TemporalMotionFeatures
from src.interaction.interaction_engine import InteractionEngine, InteractionFeatures
from src.detection.detector import PersonDetection
from src.pipeline import GuardianMatrixPipeline


class TestGuardianMatrixReliability(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_dummy_decision(
        self,
        timestamp: float,
        score: float,
        category: EventCategory = EventCategory.AGGRESSIVE_INTERACTION,
        action: ActionType = ActionType.STRIKE,
        is_uncertain: bool = False
    ) -> TemporalDecisionOutput:
        snap = EvidenceSnapshot(
            timestamp=timestamp,
            frame_id=int(timestamp * 15),
            aggressive_motion_score=score,
            directional_impact_score=score,
            rapid_approach_score=score,
            proximity_score=0.8,
            pursuit_score=0.0,
            fall_score=0.0,
            pose_quality=0.85,
            tracking_quality=0.90,
            event_probability=score,
            category=category,
            action=action,
            urgency=OperationalUrgency.CRITICAL_ALERT if score > 0.7 else OperationalUrgency.BENIGN,
            evidence_state=EvidenceState.NORMAL,
            track_ids=[1, 2]
        )
        return TemporalDecisionOutput(
            timestamp=timestamp,
            frame_id=int(timestamp * 15),
            category=category,
            action=action,
            aggression_risk=score if category == EventCategory.AGGRESSIVE_INTERACTION else 0.0,
            safety_risk=score if category == EventCategory.SAFETY_INCIDENT else 0.0,
            decision_score=score,
            urgency=OperationalUrgency.CRITICAL_ALERT if score > 0.7 else OperationalUrgency.BENIGN,
            evidence_snapshot=snap,
            persistence_score=0.8,
            temporal_features={"directional_impact_score": score},
            quality_metrics={"pose_quality": 0.85, "tracking_quality": 0.90},
            dominant_track_ids=[1, 2],
            is_uncertain=is_uncertain,
            rationale="Test decision"
        )

    # -------------------------------------------------------------------------
    # TEST 1: Dual-Threshold Hysteresis
    # -------------------------------------------------------------------------
    def test_hysteresis_state_transitions(self):
        """Verify 0.81 -> ACTIVE, 0.65 -> ACTIVE, 0.50 -> ACTIVE, 0.44 -> exits ACTIVE."""
        sm = IncidentStateMachine(IncidentStateMachineConfig(
            observing_threshold=0.40,
            suspected_threshold=0.60,
            active_threshold=0.80,
            exit_threshold=0.45,
            cooldown_seconds=1.0
        ))

        # Step 1: Score = 0.81 -> Immediately triggers ACTIVE_INCIDENT
        res1 = sm.update(self._make_dummy_decision(timestamp=1.0, score=0.81))
        self.assertEqual(res1.state, EvidenceState.ACTIVE_INCIDENT)
        self.assertTrue(res1.is_incident_active)
        self.assertIsNotNone(res1.incident_id)

        # Step 2: Score drops to 0.65 -> Must REMAIN in ACTIVE_INCIDENT due to hysteresis
        res2 = sm.update(self._make_dummy_decision(timestamp=1.2, score=0.65))
        self.assertEqual(res2.state, EvidenceState.ACTIVE_INCIDENT)
        self.assertTrue(res2.is_incident_active)

        # Step 3: Score drops to 0.50 -> Above exit_threshold (0.45), must REMAIN ACTIVE_INCIDENT
        res3 = sm.update(self._make_dummy_decision(timestamp=1.4, score=0.50))
        self.assertEqual(res3.state, EvidenceState.ACTIVE_INCIDENT)
        self.assertTrue(res3.is_incident_active)

        # Step 4: Score drops to 0.44 -> Below exit threshold (0.45), transitions to COOLDOWN
        res4 = sm.update(self._make_dummy_decision(timestamp=1.6, score=0.44))
        self.assertEqual(res4.state, EvidenceState.COOLDOWN)
        self.assertFalse(res4.is_incident_active)

        # Step 5: After cooldown expires with low score -> transitions to NORMAL
        res5 = sm.update(self._make_dummy_decision(timestamp=2.8, score=0.20))
        self.assertEqual(res5.state, EvidenceState.NORMAL)

    # -------------------------------------------------------------------------
    # TEST 2: Single-Frame Spike Rejection
    # -------------------------------------------------------------------------
    def test_single_frame_spike_rejection(self):
        """Single-frame anomaly (NORMAL -> NORMAL -> HIGH -> NORMAL -> NORMAL) must NOT alert."""
        tde = TemporalDecisionEngine(TemporalDecisionConfig(
            temporal_window_seconds=2.0,
            min_observation_seconds=0.30
        ))
        sm = IncidentStateMachine(IncidentStateMachineConfig(
            active_threshold=0.80,
            observing_threshold=0.40
        ))

        # Frame 1: Normal (t=0.0)
        out1 = tde.process_frame_evidence(timestamp=0.0, frame_id=1, poses=[], motion_features={}, interactions=[])
        res1 = sm.update(out1)
        self.assertEqual(res1.state, EvidenceState.NORMAL)

        # Frame 2: Normal (t=0.1)
        out2 = tde.process_frame_evidence(timestamp=0.1, frame_id=2, poses=[], motion_features={}, interactions=[])
        res2 = sm.update(out2)
        self.assertEqual(res2.state, EvidenceState.NORMAL)

        # Frame 3: Single isolated spike (t=0.2, high strike spike)
        mock_inter = [InteractionFeatures(
            person_a=1, person_b=2, distance=0.4, relative_velocity=-1.0,
            relative_direction=(1.0, 0.0), orientation_similarity=-0.8,
            is_facing_each_other=True, hand_to_body_min_distance=0.2,
            hand_to_hand_distance=0.2, hand_contact_probability=0.7,
            aggressive_strike_score=0.9, directional_impact_score=0.85
        )]
        out3 = tde.process_frame_evidence(timestamp=0.2, frame_id=3, poses=[], motion_features={}, interactions=mock_inter)
        res3 = sm.update(out3)
        # Because persistence duration is 0 (first frame of elevation), must not be ACTIVE_INCIDENT
        self.assertNotEqual(res3.state, EvidenceState.ACTIVE_INCIDENT)

        # Frame 4: Returns immediately to normal (t=0.3)
        out4 = tde.process_frame_evidence(timestamp=0.3, frame_id=4, poses=[], motion_features={}, interactions=[])
        res4 = sm.update(out4)
        self.assertNotEqual(res4.state, EvidenceState.ACTIVE_INCIDENT)

    # -------------------------------------------------------------------------
    # TEST 3: Strict Fall Separation
    # -------------------------------------------------------------------------
    def test_fall_separation(self):
        """Fall evidence must strictly classify as SAFETY_INCIDENT and NEVER AGGRESSIVE_INTERACTION."""
        tde = TemporalDecisionEngine()
        
        # Simulate severe downward collapse and horizontal prone posture
        fall_motion = {
            1: TemporalMotionFeatures(
                track_id=1, window_duration_seconds=1.0, num_frames=15,
                torso_velocity=(0.0, 1.2, 0.0), torso_speed=1.2,
                torso_acceleration=(0.0, 0.0, 0.0), peak_acceleration=0.0,
                left_hand_velocity=0.0, right_hand_velocity=0.0, peak_hand_speed=0.0,
                left_foot_velocity=0.0, right_foot_velocity=0.0, peak_foot_speed=0.0,
                left_elbow_angle=180.0, right_elbow_angle=180.0, left_knee_angle=180.0,
                right_knee_angle=180.0, torso_tilt_degrees=85.0, body_orientation_degrees=0.0,
                motion_jitter=0.1, is_rapid_motion=False, is_falling=True
            )
        }

        # Feed 5 consecutive fall frames across 0.4 seconds
        decision = None
        for i in range(5):
            t = 1.0 + (i * 0.08)
            decision = tde.process_frame_evidence(
                timestamp=t, frame_id=i+1, poses=[],
                motion_features=fall_motion, interactions=[]
            )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.category, EventCategory.SAFETY_INCIDENT)
        self.assertEqual(decision.action, ActionType.FALL)
        self.assertNotEqual(decision.category, EventCategory.AGGRESSIVE_INTERACTION)
        self.assertGreater(decision.safety_risk, 0.5)
        self.assertEqual(decision.aggression_risk, 0.0)

    # -------------------------------------------------------------------------
    # TEST 4: Aggressive Interaction Confirmation
    # -------------------------------------------------------------------------
    def test_aggressive_interaction_confirmation(self):
        """Directional strike with rapid approach and proximity confirms AGGRESSIVE_INTERACTION."""
        tde = TemporalDecisionEngine(TemporalDecisionConfig(min_observation_seconds=0.25))
        sm = IncidentStateMachine(IncidentStateMachineConfig(min_observation_dwell_seconds=0.15))

        inter = [InteractionFeatures(
            person_a=1, person_b=2, distance=0.45, relative_velocity=-0.8,
            relative_direction=(1.0, 0.0), orientation_similarity=-0.7,
            is_facing_each_other=True, hand_to_body_min_distance=0.15,
            hand_to_hand_distance=0.25, hand_contact_probability=0.85,
            directional_impact_score=0.92, aggressive_strike_score=0.90
        )]

        # Feed sequence over 0.5s with confident poses
        kps = np.zeros((17, 2), dtype=np.float32)
        mock_poses = [
            PoseFrame(track_id=1, timestamp=10.0, keypoints_2d=kps, pose_confidence=0.85, is_normalized=True),
            PoseFrame(track_id=2, timestamp=10.0, keypoints_2d=kps, pose_confidence=0.85, is_normalized=True)
        ]

        final_state = None
        for i in range(6):
            t = 10.0 + (i * 0.1)
            dec = tde.process_frame_evidence(timestamp=t, frame_id=i+1, poses=mock_poses, motion_features={}, interactions=inter)
            final_state = sm.update(dec)

        self.assertEqual(final_state.category, EventCategory.AGGRESSIVE_INTERACTION)
        self.assertEqual(final_state.state, EvidenceState.ACTIVE_INCIDENT)
        self.assertEqual(final_state.urgency, OperationalUrgency.CRITICAL_ALERT)

    # -------------------------------------------------------------------------
    # TEST 5: Normal Walking Arm Swing Non-Aggression
    # -------------------------------------------------------------------------
    def test_normal_walking_arm_swing(self):
        """Walking with hand movement but no directional impact must remain NORMAL_ACTIVITY."""
        tde = TemporalDecisionEngine()
        sm = IncidentStateMachine()

        # Arm is moving, but persons are separated (d=1.5) and directional impact is 0
        walk_inter = [InteractionFeatures(
            person_a=1, person_b=2, distance=1.5, relative_velocity=0.0,
            relative_direction=(1.0, 0.0), orientation_similarity=0.9,
            is_facing_each_other=False, hand_to_body_min_distance=1.4,
            hand_to_hand_distance=1.3, hand_contact_probability=0.0,
            directional_impact_score=0.0, aggressive_strike_score=0.0
        )]

        for i in range(5):
            t = 2.0 + (i * 0.1)
            dec = tde.process_frame_evidence(timestamp=t, frame_id=i+1, poses=[], motion_features={}, interactions=walk_inter)
            state_res = sm.update(dec)
            self.assertEqual(state_res.state, EvidenceState.NORMAL)
            self.assertEqual(dec.category, EventCategory.NORMAL_ACTIVITY)

    # -------------------------------------------------------------------------
    # TEST 6: Two People Standing Close
    # -------------------------------------------------------------------------
    def test_close_standing_non_aggression(self):
        """Two people standing in conversational proximity without strike or pursuit must not alert."""
        tde = TemporalDecisionEngine()
        sm = IncidentStateMachine()

        close_inter = [InteractionFeatures(
            person_a=1, person_b=2, distance=0.6, relative_velocity=0.0,
            relative_direction=(1.0, 0.0), orientation_similarity=-0.8,
            is_facing_each_other=True, hand_to_body_min_distance=0.55,
            hand_to_hand_distance=0.5, hand_contact_probability=0.0,
            directional_impact_score=0.0, aggressive_strike_score=0.0
        )]

        for i in range(10):
            t = 5.0 + (i * 0.1)
            dec = tde.process_frame_evidence(timestamp=t, frame_id=i+1, poses=[], motion_features={}, interactions=close_inter)
            state_res = sm.update(dec)
            self.assertNotEqual(state_res.state, EvidenceState.ACTIVE_INCIDENT)
            self.assertNotEqual(state_res.category, EventCategory.AGGRESSIVE_INTERACTION)

    # -------------------------------------------------------------------------
    # TEST 7: Duplicate Person Detection Suppression
    # -------------------------------------------------------------------------
    def test_duplicate_person_suppression(self):
        """Overlapping duplicate bounding boxes (IoU > 0.40) must NOT create pair interaction."""
        engine = InteractionEngine()

        # Two poses representing the same person (IoU ~ 0.85)
        kps = np.zeros((17, 2), dtype=np.float32)
        pose1 = PoseFrame(track_id=1, timestamp=1.0, keypoints_2d=kps, bbox=(100, 50, 180, 250), is_normalized=True)
        pose2 = PoseFrame(track_id=2, timestamp=1.0, keypoints_2d=kps, bbox=(105, 52, 182, 248), is_normalized=True)

        interactions = engine.update([pose1, pose2], motion_features={}, timestamp=1.0)
        # Must filter out duplicate boxes so no interaction pair is produced
        self.assertEqual(len(interactions), 0)

    # -------------------------------------------------------------------------
    # TEST 8: Forensic Evidence Store Bundle
    # -------------------------------------------------------------------------
    def test_evidence_store_bundling(self):
        """Trigger synthetic incident and verify metadata.json, timeline.json, keyframe.jpg, clip.mp4."""
        store_cfg = EvidenceStoreConfig(
            base_incidents_dir=self.temp_dir,
            pre_event_seconds=0.5,
            post_event_seconds=0.5,
            target_fps=10.0,
            save_video_clips=True
        )
        store = EvidenceStore(config=store_cfg)

        # Pre-event frames
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(5):
            t = i * 0.1
            dec = self._make_dummy_decision(timestamp=t, score=0.2)
            state_res = StateMachineResult(
                state=EvidenceState.NORMAL, category=EventCategory.NORMAL_ACTIVITY,
                action=ActionType.WALKING, urgency=OperationalUrgency.BENIGN,
                decision_score=0.2, is_incident_active=False, requires_review=False,
                incident_id=None, transition_occurred=False, state_dwell_seconds=t,
                active_track_ids=[], rationale=""
            )
            store.on_frame(dummy_frame, t, i+1, dec, state_res)

        # Active incident frame
        dec_active = self._make_dummy_decision(timestamp=0.6, score=0.88)
        state_active = StateMachineResult(
            state=EvidenceState.ACTIVE_INCIDENT, category=EventCategory.AGGRESSIVE_INTERACTION,
            action=ActionType.STRIKE, urgency=OperationalUrgency.CRITICAL_ALERT,
            decision_score=0.88, is_incident_active=True, requires_review=False,
            incident_id="INC-TEST-001_STRIKE", transition_occurred=True,
            state_dwell_seconds=0.1, active_track_ids=[1, 2], rationale="Strike confirmed"
        )
        store.on_frame(dummy_frame, 0.6, 6, dec_active, state_active)

        # Post-event frames to finalize
        saved_dir = None
        for i in range(7, 15):
            t = i * 0.1
            dec_norm = self._make_dummy_decision(timestamp=t, score=0.2)
            state_norm = StateMachineResult(
                state=EvidenceState.COOLDOWN if t < 1.0 else EvidenceState.NORMAL,
                category=EventCategory.NORMAL_ACTIVITY, action=ActionType.WALKING,
                urgency=OperationalUrgency.BENIGN, decision_score=0.2,
                is_incident_active=False, requires_review=False,
                incident_id=None, transition_occurred=False, state_dwell_seconds=t-0.6,
                active_track_ids=[], rationale=""
            )
            res = store.on_frame(dummy_frame, t, i, dec_norm, state_norm)
            if res:
                saved_dir = res
                break

        self.assertIsNotNone(saved_dir)
        self.assertTrue(os.path.exists(saved_dir))
        self.assertTrue(os.path.exists(os.path.join(saved_dir, "metadata.json")))
        self.assertTrue(os.path.exists(os.path.join(saved_dir, "timeline.json")))
        self.assertTrue(os.path.exists(os.path.join(saved_dir, "keyframe.jpg")))
        self.assertTrue(os.path.exists(os.path.join(saved_dir, "clip.mp4")))


if __name__ == "__main__":
    unittest.main()
