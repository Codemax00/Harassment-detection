"""
Guardian Matrix Unified Pipeline Orchestrator.
Executes the research-grade 10-stage human-interaction analysis architecture
as specified in Upgrade_Plan.md Section 36.
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
from .interaction.interaction_engine import InteractionEngine, InteractionFeatures
from .taxonomy.event_taxonomy import EventLabel
from .evidence.structured_evidence import StructuredEvidence, build_structured_evidence
from .system1.laya_decision import LayaDecision, LayaSystem1Engine, DecisionState
from .system2.verifier import System2Verifier, System2VerificationResult
from .policy.safety_policy import SafetyPolicy, PolicyState, PolicyAction
from .logging.event_logger import EventLogger, EventRecord


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


class GuardianMatrixPipeline:
    """
    Guardian Matrix Research-Grade Multi-Stage Pipeline.
    """

    def __init__(
        self,
        detector: Optional[BasePersonDetector] = None,
        tracker: Optional[BaseTracker] = None,
        pose_estimator: Optional[BasePoseEstimator] = None,
        temporal_engine: Optional[TemporalMotionEngine] = None,
        temporal_classifier: Optional[BaseTemporalClassifier] = None,
        interaction_engine: Optional[InteractionEngine] = None,
        system1_engine: Optional[LayaSystem1Engine] = None,
        system2_verifier: Optional[System2Verifier] = None,
        safety_policy: Optional[SafetyPolicy] = None,
        event_logger: Optional[EventLogger] = None,
        video_id: str = "GM_STREAM"
    ):
        self.video_id = video_id
        self.detector = detector or load_detector("yolo")
        self.tracker = tracker or load_tracker("bytetrack")
        self.pose_estimator = pose_estimator or load_pose_estimator("yolo_pose")
        self.temporal_engine = temporal_engine or TemporalMotionEngine(window_seconds=2.0)
        self.temporal_classifier = temporal_classifier or load_temporal_classifier("ensemble")
        self.interaction_engine = interaction_engine or InteractionEngine()
        self.system1_engine = system1_engine or LayaSystem1Engine()
        self.system2_verifier = system2_verifier or System2Verifier()
        self.safety_policy = safety_policy or SafetyPolicy()
        self.event_logger = event_logger or EventLogger()

        self.pose_history: Dict[int, List[PoseFrame]] = {}
        self.frame_counter = 0

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

        # Stage 2: Multi-Object Persistent Tracking
        tracked_people = self.tracker.update(detections, timestamp=timestamp, frame_id=current_fid)

        # Stage 3: High-Precision Pose Estimation & Normalization
        poses = self.pose_estimator.estimate(frame, tracked_people, timestamp=timestamp, frame_id=current_fid)

        # Maintain temporal pose sequences
        for p in poses:
            if p.track_id not in self.pose_history:
                self.pose_history[p.track_id] = []
            self.pose_history[p.track_id].append(p)
            if len(self.pose_history[p.track_id]) > 60:
                self.pose_history[p.track_id].pop(0)

        # Stage 4: Temporal Motion Feature Extraction
        motion_features = self.temporal_engine.update(poses)

        # Stage 5: Person-Person Interaction Engine
        interactions = self.interaction_engine.update(poses, motion_features, timestamp=timestamp)

        # Stage 6: Temporal Action Classification
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

        # Stage 8: Laya System-1 Decision Layer
        s1_decision = self.system1_engine.evaluate(evidence)

        # Stage 9: System-2 Deep Verification (if escalated)
        s2_result: Optional[System2VerificationResult] = None
        if s1_decision.escalate_to_system2:
            s2_result = self.system2_verifier.verify(s1_decision, evidence, keyframes=[frame])

        # Stage 10: Deterministic Safety Policy
        policy_action = self.safety_policy.evaluate_policy(evidence, s1_decision, s2_result)

        # Event Logging for scientific provenance
        event_record: Optional[EventRecord] = None
        if policy_action.state in (PolicyState.REVIEW, PolicyState.ALERT):
            event_record = self.event_logger.log_event(
                video_id=self.video_id,
                evidence=evidence,
                policy_action=policy_action,
                confidence=s1_decision.calibrated_confidence,
                pose_model_name=self.pose_estimator.model_name
            )

        # Visualization Annotation
        annotated: Optional[np.ndarray] = None
        if annotate and frame is not None:
            annotated = self._annotate_frame(
                frame.copy(), tracked_people, poses, interactions, s1_decision, policy_action
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
            processing_time_ms=elapsed_ms
        )

    def _annotate_frame(
        self,
        img: np.ndarray,
        tracked_people: List[TrackedPerson],
        poses: List[PoseFrame],
        interactions: List[InteractionFeatures],
        decision: LayaDecision,
        action: PolicyAction
    ) -> np.ndarray:
        """
        Renders research dashboard overlay on frame matching Section 30.
        """
        # 1. Draw bounding boxes and track IDs
        for person in tracked_people:
            x1, y1, x2, y2 = [int(v) for v in person.bbox]
            # Color based on policy state
            color = (0, 255, 0)  # Green
            if action.state == PolicyState.MONITOR:
                color = (0, 255, 255)  # Yellow
            elif action.state == PolicyState.REVIEW:
                color = (0, 165, 255)  # Orange
            elif action.state == PolicyState.ALERT:
                color = (0, 0, 255)  # Red

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                f"ID {person.track_id} ({person.detection_confidence:.2f})",
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # 2. Draw skeletons
        for pose in poses:
            if not pose.is_normalized:
                continue
            box = pose.bbox
            if not box:
                continue
            bx1, by1, bx2, by2 = box
            bh = max(1.0, by2 - by1)
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0

            # Convert normalized keypoints back to pixel coords for visualization
            kps_px = []
            for kp in pose.keypoints_2d[:17]:
                px = int(kp[0] * bh + cx)
                py = int(kp[1] * bh + cy)
                kps_px.append((px, py))
                cv2.circle(img, (px, py), 3, (255, 200, 0), -1)

            # Draw skeletal lines
            edges = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
            for i, j in edges:
                if i < len(kps_px) and j < len(kps_px):
                    cv2.line(img, kps_px[i], kps_px[j], (255, 255, 100), 1)

        # 3. Draw interaction connecting lines
        for inter in interactions:
            if inter.distance < 1.0:
                # Find positions of person A and person B
                p_a = next((p for p in tracked_people if p.track_id == inter.person_a), None)
                p_b = next((p for p in tracked_people if p.track_id == inter.person_b), None)
                if p_a and p_b:
                    ca = (int(p_a.center[0]), int(p_a.center[1]))
                    cb = (int(p_b.center[0]), int(p_b.center[1]))
                    line_col = (0, 0, 255) if inter.hand_contact_probability > 0.5 else (0, 200, 255)
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
        banner_h = 70
        banner_w = 420
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (banner_w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        state_color = (0, 255, 0)
        if action.state == PolicyState.MONITOR:
            state_color = (0, 255, 255)
        elif action.state == PolicyState.REVIEW:
            state_color = (0, 165, 255)
        elif action.state == PolicyState.ALERT:
            state_color = (0, 0, 255)

        cv2.putText(
            img,
            f"GUARDIAN MATRIX: {action.state.value}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            state_color,
            2
        )
        pattern = action.target_event_label
        cv2.putText(
            img,
            f"Pattern: {pattern} | Conf: {decision.calibrated_confidence:.2f} | People: {len(tracked_people)}",
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1
        )

        return img
