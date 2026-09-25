"""
Guardian Matrix Unified Pipeline Orchestrator.
Executes the research-grade surveillance safety architecture:
Detection -> Tracking -> Pose -> Interaction Features -> Directional Impact ->
Temporal Decision Engine (Time-weighted aggregation & spike rejection) ->
Incident State Machine (5-state hysteresis & uncertainty gate) ->
Evidence Store (auditable forensic bundling with rolling frame buffer).
"""

from dataclasses import dataclass, field
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from .detection.detector import BasePersonDetector, PersonDetection, load_detector
from .tracking.tracker import BaseTracker, TrackedPerson, load_tracker
from .pose.pose_representation import PoseFrame, KeypointName
from .pose.pose_estimator import BasePoseEstimator, load_pose_estimator
from .temporal.temporal_engine import TemporalMotionEngine, TemporalMotionFeatures
from .temporal.temporal_action_classifier import BaseTemporalClassifier, load_temporal_classifier
from .temporal.temporal_decision_engine import TemporalDecisionEngine, TemporalDecisionOutput
from .interaction.interaction_engine import InteractionEngine, InteractionFeatures
from .taxonomy.event_taxonomy import (
    EventLabel,
    EventCategory,
    ActionType,
    OperationalUrgency,
    EvidenceState,
    EventTaxonomy
)
from .evidence.structured_evidence import StructuredEvidence, build_structured_evidence
from .evidence.evidence_store import EvidenceStore
from .system1.laya_decision import LayaDecision, LayaSystem1Engine, DecisionState
from .system2.verifier import System2Verifier, System2VerificationResult
from .policy.safety_policy import SafetyPolicy, PolicyState, PolicyAction
from .policy.incident_state_machine import IncidentStateMachine, StateMachineResult
from .logging.event_logger import EventLogger, EventRecord

try:
    from config.settings import settings
except ImportError:
    import sys
    from pathlib import Path
    _proj_root = str(Path(__file__).resolve().parent.parent)
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    from config.settings import settings


@dataclass
class PipelineFrameResult:
    """End-to-end result for a single processed video frame."""
    frame_id: int
    timestamp: float
    detections: List[PersonDetection]
    tracked_people: List[TrackedPerson]
    poses: List[PoseFrame]
    motion_features: Dict[int, TemporalMotionFeatures]
    interactions: List[InteractionFeatures]
    classified_event: EventLabel
    event_confidence: float
    evidence: StructuredEvidence
    system1_decision: LayaDecision
    system2_result: Optional[System2VerificationResult]
    policy_action: PolicyAction
    event_record: Optional[EventRecord]
    annotated_frame: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0
    temporal_output: Optional[TemporalDecisionOutput] = None
    state_machine_result: Optional[StateMachineResult] = None
    saved_incident_bundle: Optional[str] = None

    def to_telemetry_dict(self) -> Dict[str, Any]:
        """JSON-serializable telemetry payload matching Section 17 & 34."""
        state_str = self.state_machine_result.state.value if self.state_machine_result else self.policy_action.state.value
        cat_str = self.temporal_output.category.value if self.temporal_output else "NORMAL_ACTIVITY"
        act_str = self.temporal_output.action.value if self.temporal_output else self.classified_event.value
        risk_val = self.state_machine_result.decision_score if self.state_machine_result else self.system1_decision.calibrated_confidence
        urgency_str = self.state_machine_result.urgency.value if self.state_machine_result else "BENIGN"

        return {
            "state": state_str,
            "category": cat_str,
            "action": act_str,
            "risk_score": round(float(risk_val), 4),
            "urgency": urgency_str,
            "track_ids": [p.track_id for p in self.tracked_people],
            "evidence": {
                "directional_impact": round(float(self.temporal_output.temporal_features.get("directional_impact_score", 0.0)), 3) if self.temporal_output else 0.0,
                "rapid_approach": round(float(self.temporal_output.temporal_features.get("rapid_approach_score", 0.0)), 3) if self.temporal_output else 0.0,
                "persistence": round(float(self.temporal_output.persistence_score), 3) if self.temporal_output else 0.0
            },
            "quality": {
                "pose": round(float(self.temporal_output.quality_metrics.get("pose_quality", 0.0)), 3) if self.temporal_output else 0.0,
                "tracking": round(float(self.temporal_output.quality_metrics.get("tracking_quality", 0.0)), 3) if self.temporal_output else 0.0
            },
            "saved_incident_bundle": self.saved_incident_bundle,
            "processing_time_ms": round(float(self.processing_time_ms), 2)
        }


class GuardianMatrixPipeline:
    """
    Guardian Matrix Research-Grade Multi-Stage Surveillance Safety Pipeline.
    """

    def __init__(
        self,
        detector: Optional[BasePersonDetector] = None,
        tracker: Optional[BaseTracker] = None,
        pose_estimator: Optional[BasePoseEstimator] = None,
        temporal_engine: Optional[TemporalMotionEngine] = None,
        temporal_classifier: Optional[BaseTemporalClassifier] = None,
        interaction_engine: Optional[InteractionEngine] = None,
        temporal_decision_engine: Optional[TemporalDecisionEngine] = None,
        incident_state_machine: Optional[IncidentStateMachine] = None,
        evidence_store: Optional[EvidenceStore] = None,
        system1_engine: Optional[LayaSystem1Engine] = None,
        system2_verifier: Optional[System2Verifier] = None,
        safety_policy: Optional[SafetyPolicy] = None,
        event_logger: Optional[EventLogger] = None,
        video_id: str = "GM_STREAM"
    ):
        self.video_id = video_id
        self.detector = detector or load_detector("yolo", config={
            "model_path": settings.model_path,
            "conf_threshold": settings.detection_confidence,
            "device": settings.device
        })
        self.tracker = tracker or load_tracker("bytetrack", config={
            "track_buffer": max(15, int(settings.target_fps * 2))
        })
        self.pose_estimator = pose_estimator or load_pose_estimator("yolo_pose", config={
            "model_path": settings.pose_model_path,
            "conf_threshold": settings.pose_confidence,
            "device": settings.device
        })
        self.temporal_engine = temporal_engine or TemporalMotionEngine(window_seconds=2.0)
        self.temporal_classifier = temporal_classifier or load_temporal_classifier("ensemble")
        self.interaction_engine = interaction_engine or InteractionEngine()
        self.temporal_decision_engine = temporal_decision_engine or TemporalDecisionEngine()
        self.incident_state_machine = incident_state_machine or IncidentStateMachine()
        self.evidence_store = evidence_store or EvidenceStore()
        self.system1_engine = system1_engine or LayaSystem1Engine()
        self.system2_verifier = system2_verifier or System2Verifier()
        self.safety_policy = safety_policy or SafetyPolicy()
        self.event_logger = event_logger or EventLogger()

        self.pose_history: Dict[int, List[PoseFrame]] = {}
        self.frame_counter = 0

    def reset(self):
        """Resets all internal states for a fresh video or stream session."""
        self.pose_history.clear()
        self.frame_counter = 0
        self.tracker = load_tracker("bytetrack", config={
            "track_buffer": max(15, int(settings.target_fps * 2))
        })
        self.temporal_engine = TemporalMotionEngine(window_seconds=2.0)
        self.interaction_engine = InteractionEngine()
        self.temporal_decision_engine = TemporalDecisionEngine()
        self.incident_state_machine.reset()
        self.evidence_store = EvidenceStore()

    def finalize(self) -> Optional[str]:
        """Finalizes any pending or active incident bundle and flushes evidence store."""
        return self.evidence_store.finalize_pending_incident()

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        frame_id: int = 0,
        annotate: bool = True
    ) -> PipelineFrameResult:
        t0 = time.perf_counter()
        self.frame_counter += 1
        current_fid = frame_id if frame_id > 0 else self.frame_counter

        # Stage 1: Person Detection
        detections = self.detector.detect(frame, frame_id=current_fid)

        # Stage 2: Multi-Object Persistent Tracking with Visual Appearance Re-ID
        tracked_people = self.tracker.update(detections, timestamp=timestamp, frame_id=current_fid, frame=frame)

        # Stage 3: High-Precision Pose Estimation & Normalization
        poses = self.pose_estimator.estimate(frame, tracked_people, timestamp=timestamp, frame_id=current_fid)

        # Maintain temporal pose sequences and prune dead tracks to prevent memory leak
        active_tids = {p.track_id for p in tracked_people}
        dead_tracks = [tid for tid in self.pose_history if tid not in active_tids]
        for tid in dead_tracks:
            del self.pose_history[tid]

        for p in poses:
            if p.track_id not in self.pose_history:
                self.pose_history[p.track_id] = []
            self.pose_history[p.track_id].append(p)
            if len(self.pose_history[p.track_id]) > 60:
                self.pose_history[p.track_id].pop(0)

        # Stage 4: Temporal Motion Feature Extraction
        motion_features = self.temporal_engine.update(poses)

        # Stage 5: Person-Person Interaction Engine & Directional Impact
        interactions = self.interaction_engine.update(poses, motion_features, timestamp=timestamp)

        # Stage 6: Temporal Action Classification (Baseline Model)
        classified_event, event_conf, temporal_probs = self.temporal_classifier.classify(
            self.pose_history, interactions, motion_features
        )

        # Stage 7: Structured Evidence Compilation
        evidence = build_structured_evidence(
            timestamp=timestamp,
            frame_id=current_fid,
            tracked_people=tracked_people,
            poses=poses,
            interactions=interactions,
            motion_features=motion_features,
            classified_event=classified_event,
            event_confidence=event_conf,
            temporal_probs=temporal_probs
        )

        # Stage 8: Temporal Decision Engine (Rolling Aggregation & Spike Protection)
        temporal_output: TemporalDecisionOutput = self.temporal_decision_engine.process_frame_evidence(
            timestamp=timestamp,
            frame_id=current_fid,
            poses=poses,
            motion_features=motion_features,
            interactions=interactions
        )

        # Stage 9: Incident State Machine (Hysteresis & Uncertainty Gating)
        state_result: StateMachineResult = self.incident_state_machine.update(temporal_output)

        # Stage 10: Evidence Store Incident Bundling (Forensic recording)
        saved_bundle = self.evidence_store.on_frame(
            frame=frame,
            timestamp=timestamp,
            frame_id=current_fid,
            decision=temporal_output,
            state_result=state_result
        )

        # Stage 11: System-1 & System-2 Decision Layer (Legacy Alignment)
        s1_decision = self.system1_engine.evaluate(evidence)
        s2_result: Optional[System2VerificationResult] = None
        if s1_decision.escalate_to_system2:
            s2_result = self.system2_verifier.verify(s1_decision, evidence, keyframes=[frame])

        # Stage 12: Deterministic Safety Policy (Synchronized with State Machine)
        policy_action = self.safety_policy.evaluate_policy(evidence, s1_decision, s2_result)
        
        # Override policy action with verified State Machine decision
        if state_result.state == EvidenceState.ACTIVE_INCIDENT:
            policy_action.state = PolicyState.ALERT
            policy_action.target_event_label = state_result.action.value
            policy_action.rationale = state_result.rationale
        elif state_result.requires_review or state_result.state == EvidenceState.SUSPECTED:
            policy_action.state = PolicyState.REVIEW
            policy_action.target_event_label = state_result.action.value
            policy_action.rationale = state_result.rationale
        elif state_result.state == EvidenceState.OBSERVING:
            policy_action.state = PolicyState.MONITOR
            policy_action.target_event_label = state_result.action.value
            policy_action.rationale = state_result.rationale
        else:
            policy_action.state = PolicyState.NORMAL
            policy_action.target_event_label = state_result.action.value
            policy_action.rationale = state_result.rationale

        # Event Logging for scientific provenance
        event_record: Optional[EventRecord] = None
        if policy_action.state in (PolicyState.REVIEW, PolicyState.ALERT):
            event_record = self.event_logger.log_event(
                video_id=self.video_id,
                evidence=evidence,
                policy_action=policy_action,
                confidence=float(state_result.decision_score),
                pose_model_name=self.pose_estimator.model_name
            )

        # Stage 13: Visualization Annotation
        annotated: Optional[np.ndarray] = None
        if annotate and frame is not None:
            annotated = self._annotate_frame(
                frame.copy(), tracked_people, poses, interactions, state_result, temporal_output
            )

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0

        return PipelineFrameResult(
            frame_id=current_fid,
            timestamp=timestamp,
            detections=detections,
            tracked_people=tracked_people,
            poses=poses,
            motion_features=motion_features,
            interactions=interactions,
            classified_event=classified_event,
            event_confidence=event_conf,
            evidence=evidence,
            system1_decision=s1_decision,
            system2_result=s2_result,
            policy_action=policy_action,
            event_record=event_record,
            annotated_frame=annotated,
            processing_time_ms=elapsed_ms,
            temporal_output=temporal_output,
            state_machine_result=state_result,
            saved_incident_bundle=saved_bundle
        )

    def _annotate_frame(
        self,
        img: np.ndarray,
        tracked_people: List[TrackedPerson],
        poses: List[PoseFrame],
        interactions: List[InteractionFeatures],
        state_result: StateMachineResult,
        temporal_output: TemporalDecisionOutput
    ) -> np.ndarray:
        """
        Renders research dashboard overlay on frame matching Section 34 & 39.
        """
        # Determine status colors
        if state_result.state == EvidenceState.ACTIVE_INCIDENT:
            box_color = (0, 0, 255)      # Red
            badge_text = f"ACTIVE INCIDENT [{state_result.action.value}]"
        elif state_result.requires_review:
            box_color = (0, 165, 255)    # Orange
            badge_text = f"REVIEW REQUIRED [{state_result.action.value}]"
        elif state_result.state == EvidenceState.SUSPECTED:
            box_color = (0, 200, 255)    # Amber
            badge_text = f"SUSPECTED [{state_result.action.value}]"
        elif state_result.state == EvidenceState.OBSERVING:
            box_color = (0, 255, 255)    # Yellow
            badge_text = "OBSERVING"
        elif state_result.state == EvidenceState.COOLDOWN:
            box_color = (255, 200, 0)    # Cyan-Orange
            badge_text = "COOLDOWN"
        else:
            box_color = (0, 255, 0)      # Green
            badge_text = "NORMAL"

        # 1. Draw bounding boxes and track IDs
        for person in tracked_people:
            x1, y1, x2, y2 = [int(v) for v in person.bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                img,
                f"ID {person.track_id} ({person.detection_confidence:.2f})",
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2
            )

        # 2. Draw skeletons
        for pose in poses:
            if not pose.is_normalized or not pose.bbox:
                continue
            bx1, by1, bx2, by2 = pose.bbox
            bh = max(1.0, by2 - by1)
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0

            kps_px = []
            for kp in pose.keypoints_2d[:17]:
                px = int(kp[0] * bh + cx)
                py = int(kp[1] * bh + cy)
                kps_px.append((px, py))
                cv2.circle(img, (px, py), 3, (255, 200, 0), -1)

            edges = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
            for i, j in edges:
                if i < len(kps_px) and j < len(kps_px):
                    cv2.line(img, kps_px[i], kps_px[j], (255, 255, 100), 1)

        # 3. Draw interaction connecting lines
        for inter in interactions:
            if inter.distance < 1.0:
                p_a = next((p for p in tracked_people if p.track_id == inter.person_a), None)
                p_b = next((p for p in tracked_people if p.track_id == inter.person_b), None)
                if p_a and p_b:
                    ca = (int(p_a.center[0]), int(p_a.center[1]))
                    cb = (int(p_b.center[0]), int(p_b.center[1]))
                    line_col = (0, 0, 255) if inter.directional_impact_score > 0.4 else (0, 200, 255)
                    cv2.line(img, ca, cb, line_col, 2)
                    mid = ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2)
                    cv2.putText(
                        img,
                        f"d={inter.distance:.2f}m",
                        mid,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1
                    )

        # 4. Status Banner overlay (Top left)
        banner_h = 78
        banner_w = 460
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (banner_w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)

        cv2.putText(
            img,
            f"GUARDIAN MATRIX: {badge_text}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            box_color,
            2
        )
        
        category_name = state_result.category.value
        score_val = state_result.decision_score
        cv2.putText(
            img,
            f"Category: {category_name} | Score: {score_val:.2f} | Urgency: {state_result.urgency.value}",
            (20, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (220, 220, 220),
            1
        )
        cv2.putText(
            img,
            f"People: {len(tracked_people)} | IDs: {state_result.active_track_ids} | Dwell: {state_result.state_dwell_seconds:.1f}s",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 180, 180),
            1
        )

        return img
