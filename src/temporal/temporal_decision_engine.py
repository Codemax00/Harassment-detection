"""
Temporal Decision Engine for Guardian Matrix.
Aggregates kinematic and interaction evidence over a rolling time window (default 3.0s).
Implements time-weighted evidence accumulation, event-type-aware persistence, and strict
isolation of the Safety Branch (Falls/Collapses) from the Aggression Branch (Strikes/Kicks/Pursuit).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..taxonomy.event_taxonomy import (
    EventCategory,
    ActionType,
    OperationalUrgency,
    EvidenceState,
    EventTaxonomy
)
from ..evidence.structured_evidence import EvidenceSnapshot, RollingEvidenceBuffer
from ..interaction.interaction_engine import InteractionFeatures
from ..temporal.temporal_engine import TemporalMotionFeatures
from ..pose.pose_representation import PoseFrame, KeypointName


@dataclass
class TemporalDecisionConfig:
    """Configurable temporal parameters matching Section 8."""
    temporal_window_seconds: float = 3.0
    min_observation_seconds: float = 0.30
    suspected_confirmation_seconds: float = 0.30
    active_confirmation_seconds: float = 0.30
    alert_exit_seconds: float = 1.0
    half_life_seconds: float = 1.2  # Time-decay weighting half-life
    min_pose_quality_threshold: float = 0.30
    min_tracking_quality_threshold: float = 0.35


@dataclass
class TemporalDecisionOutput:
    """Aggregated temporal assessment for downstream state machine."""
    timestamp: float
    frame_id: int
    category: EventCategory
    action: ActionType
    aggression_risk: float
    safety_risk: float
    decision_score: float  # Peak independent decision score (NOT sum)
    urgency: OperationalUrgency
    evidence_snapshot: EvidenceSnapshot
    persistence_score: float
    temporal_features: Dict[str, float]
    quality_metrics: Dict[str, float]
    dominant_track_ids: List[int]
    is_uncertain: bool
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "frame_id": self.frame_id,
            "category": self.category.value,
            "action": self.action.value,
            "aggression_risk": round(float(self.aggression_risk), 4),
            "safety_risk": round(float(self.safety_risk), 4),
            "decision_score": round(float(self.decision_score), 4),
            "urgency": self.urgency.value,
            "persistence_score": round(float(self.persistence_score), 4),
            "temporal_features": {k: round(float(v), 4) for k, v in self.temporal_features.items()},
            "quality_metrics": {k: round(float(v), 4) for k, v in self.quality_metrics.items()},
            "dominant_track_ids": self.dominant_track_ids,
            "is_uncertain": self.is_uncertain,
            "rationale": self.rationale
        }


class TemporalDecisionEngine:
    """
    Temporal Decision Engine.
    Aggregates physical evidence over rolling time windows, applies exponential time decay,
    and independently computes Safety Risk (falls) and Aggressive Risk (attacks).
    """

    def __init__(self, config: Optional[TemporalDecisionConfig] = None):
        self.config = config or TemporalDecisionConfig()
        # Rolling buffer of snapshots (approx 120 frames at 30 fps or 60 at 15 fps)
        self.buffer = RollingEvidenceBuffer(maxlen=150)

    def process_frame_evidence(
        self,
        timestamp: float,
        frame_id: int,
        poses: List[PoseFrame],
        motion_features: Dict[int, TemporalMotionFeatures],
        interactions: List[InteractionFeatures]
    ) -> TemporalDecisionOutput:
        """
        Processes instant frame observations into a rolling temporal decision.
        """
        # 1. Compute Instantaneous Feature Scores from this frame
        instant_snap = self._extract_instantaneous_snapshot(
            timestamp=timestamp,
            frame_id=frame_id,
            poses=poses,
            motion_features=motion_features,
            interactions=interactions
        )
        self.buffer.append(instant_snap)

        # 2. Retrieve Rolling History
        history = self.buffer.get_recent(self.config.temporal_window_seconds, current_timestamp=timestamp)
        if not history:
            history = [instant_snap]

        # 3. Compute Time-Weighted Temporal Aggregates
        # More recent snapshots receive higher weight: w = exp(-ln(2) * dt / half_life)
        weights = []
        for s in history:
            dt = max(0.0, timestamp - s.timestamp)
            w = float(np.exp(-0.693147 * dt / max(0.1, self.config.half_life_seconds)))
            weights.append(w)
        weight_sum = max(1e-4, sum(weights))
        norm_weights = [w / weight_sum for w in weights]

        # Aggregate kinematic & interaction evidence scores
        agg_motion = float(sum(s.aggressive_motion_score * w for s, w in zip(history, norm_weights)))
        dir_impact = float(sum(s.directional_impact_score * w for s, w in zip(history, norm_weights)))
        rapid_app = float(sum(s.rapid_approach_score * w for s, w in zip(history, norm_weights)))
        prox_score = float(sum(s.proximity_score * w for s, w in zip(history, norm_weights)))
        pursuit_sc = float(sum(s.pursuit_score * w for s, w in zip(history, norm_weights)))
        fall_sc = float(sum(s.fall_score * w for s, w in zip(history, norm_weights)))
        pose_qual = float(sum(s.pose_quality * w for s, w in zip(history, norm_weights)))
        track_qual = float(sum(s.tracking_quality * w for s, w in zip(history, norm_weights)))

        # Also retain peak values in the window to detect short rapid bursts (e.g. sudden strike or kick)
        peak_impact = float(max(s.directional_impact_score for s in history))
        peak_motion = float(max(s.aggressive_motion_score for s in history))
        peak_fall = float(max(s.fall_score for s in history))
        peak_pursuit = float(max(s.pursuit_score for s in history))

        # 4. Measure Temporal Persistence
        # Calculate how long elevated evidence has persisted
        elevated_motion_snaps = [s for s in history if (s.aggressive_motion_score > 0.25 or s.directional_impact_score > 0.25)]
        elevated_fall_snaps = [s for s in history if s.fall_score > 0.5]
        elevated_pursuit_snaps = [s for s in history if s.pursuit_score > 0.4]

        dt_motion = (elevated_motion_snaps[-1].timestamp - elevated_motion_snaps[0].timestamp) if len(elevated_motion_snaps) >= 2 else 0.0
        dt_fall = (elevated_fall_snaps[-1].timestamp - elevated_fall_snaps[0].timestamp) if len(elevated_fall_snaps) >= 2 else 0.0
        dt_pursuit = (elevated_pursuit_snaps[-1].timestamp - elevated_pursuit_snaps[0].timestamp) if len(elevated_pursuit_snaps) >= 2 else 0.0

        # Persistence score reflects confirmation duration
        # Strike needs burst + directional impact; Pursuit needs sustained time; Fall needs prone persistence
        strike_persistence = float(np.clip(dt_motion / self.config.min_observation_seconds, 0.0, 1.0))
        fall_persistence = float(np.clip(dt_fall / self.config.min_observation_seconds, 0.0, 1.0))
        pursuit_persistence = float(np.clip(dt_pursuit / 0.6, 0.0, 1.0))

        # 5. Independent Risk Evaluation: SAFETY BRANCH (Falls / Collapses)
        # Gated strictly on prone orientation, downward velocity, and persistence
        safety_risk = 0.0
        safety_action = ActionType.FALL
        if peak_fall > 0.5:
            # Fall risk combines peak fall evidence and persistence
            safety_risk = float(0.55 * peak_fall + 0.25 * fall_sc + 0.20 * fall_persistence)
            safety_risk = float(np.clip(safety_risk, 0.0, 1.0))

        # 6. Independent Risk Evaluation: AGGRESSION BRANCH (Strikes, Kicks, Pursuits)
        # Combines directional impact, proximity, rapid approach, and burst velocity
        aggression_risk = 0.0
        aggression_action = ActionType.WALKING

        # Aggression strictly requires real pairwise interactions between distinct people
        if interactions:
            peak_strike_val = max(peak_impact, peak_motion)
            strike_component = float(0.55 * peak_strike_val + 0.25 * agg_motion + 0.20 * strike_persistence)
            if prox_score < 0.15:
                # If persons are far apart (> 1.5 body heights), strike component is heavily attenuated
                strike_component *= 0.35

            # Branch B: Pursuit / Chasing
            pursuit_component = float(0.60 * peak_pursuit + 0.40 * pursuit_persistence) if pursuit_persistence > 0.35 else 0.0

            # Branch C: Inter-person rapid closing collision / push
            push_component = float(rapid_app * prox_score) if (rapid_app > 0.5 and prox_score > 0.5) else 0.0

            agg_candidates = [
                (strike_component, ActionType.STRIKE if peak_impact > 0.3 else ActionType.PUSH),
                (pursuit_component, ActionType.PURSUIT),
                (push_component, ActionType.PUSH)
            ]
            agg_candidates.sort(key=lambda x: x[0], reverse=True)
            aggression_risk = float(np.clip(agg_candidates[0][0], 0.0, 1.0))
            aggression_action = agg_candidates[0][1] if aggression_risk > 0.30 else ActionType.WALKING

        # 7. Disambiguation: Strict Separation of Safety vs Aggression
        # Never add safety_risk to aggression_risk!
        if safety_risk >= aggression_risk and safety_risk > 0.40:
            final_category = EventCategory.SAFETY_INCIDENT
            final_action = safety_action
            decision_score = safety_risk
            rationale = f"Safety incident confirmed: fall/collapse pattern with score {safety_risk:.2f}"
        elif aggression_risk > 0.40:
            final_category = EventCategory.AGGRESSIVE_INTERACTION
            final_action = aggression_action
            decision_score = aggression_risk
            rationale = f"Aggressive interaction confirmed: {final_action.value} pattern with score {aggression_risk:.2f}"
        else:
            final_category = EventCategory.NORMAL_ACTIVITY
            final_action = ActionType.WALKING
            decision_score = max(safety_risk, aggression_risk)
            rationale = "Normal ambient activity"

        # 8. Uncertainty Gating
        # If observation quality is low (small figures, occluded keypoints, tracking jitter), flag uncertainty
        is_uncertain = bool(
            pose_qual < self.config.min_pose_quality_threshold or
            track_qual < self.config.min_tracking_quality_threshold
        )

        # Operational Urgency assignment
        if is_uncertain and decision_score >= 0.50:
            urgency = OperationalUrgency.REVIEW_REQUIRED
        elif decision_score >= 0.75:
            urgency = OperationalUrgency.CRITICAL_ALERT if final_category == EventCategory.AGGRESSIVE_INTERACTION else OperationalUrgency.SAFETY_ALERT
        elif decision_score >= 0.45:
            urgency = OperationalUrgency.MONITOR
        else:
            urgency = OperationalUrgency.BENIGN

        # Dominant Track IDs
        dominant_tids = instant_snap.track_ids

        temporal_feats = {
            "aggressive_motion_score": agg_motion,
            "directional_impact_score": dir_impact,
            "rapid_approach_score": rapid_app,
            "proximity_score": prox_score,
            "pursuit_score": pursuit_sc,
            "fall_score": fall_sc,
            "peak_directional_impact": peak_impact,
            "peak_fall_score": peak_fall,
            "strike_persistence": strike_persistence,
            "fall_persistence": fall_persistence
        }

        quality_dict = {
            "pose_quality": pose_qual,
            "tracking_quality": track_qual
        }

        return TemporalDecisionOutput(
            timestamp=timestamp,
            frame_id=frame_id,
            category=final_category,
            action=final_action,
            aggression_risk=aggression_risk,
            safety_risk=safety_risk,
            decision_score=decision_score,
            urgency=urgency,
            evidence_snapshot=instant_snap,
            persistence_score=max(strike_persistence, fall_persistence, pursuit_persistence),
            temporal_features=temporal_feats,
            quality_metrics=quality_dict,
            dominant_track_ids=dominant_tids,
            is_uncertain=is_uncertain,
            rationale=rationale
        )

    def _extract_instantaneous_snapshot(
        self,
        timestamp: float,
        frame_id: int,
        poses: List[PoseFrame],
        motion_features: Dict[int, TemporalMotionFeatures],
        interactions: List[InteractionFeatures]
    ) -> EvidenceSnapshot:
        """Extracts physical scores from current frame extractions."""
        # Quality
        pose_conf = float(np.mean([p.pose_confidence for p in poses])) if poses else 0.0
        track_conf = 0.85 if poses else 0.0

        # Motion & Fall analysis
        fall_scores = []
        for tid, m in motion_features.items():
            if m.is_falling:
                # Tilt and plunge severity
                score = 0.70 + (0.30 * min(1.0, m.torso_tilt_degrees / 90.0))
                fall_scores.append(score)
            else:
                fall_scores.append(0.0)

        # Inspect current poses for horizontal collapse / prone posture
        for p in poses:
            if p.bbox:
                bw = max(1.0, p.bbox[2] - p.bbox[0])
                bh = max(1.0, p.bbox[3] - p.bbox[1])
                # Lying down aspect ratio (prominently wider than tall)
                if bw > bh * 1.30:
                    fall_scores.append(0.85)
            ls = p.get_scene_keypoint_2d(KeypointName.LEFT_SHOULDER)
            rs = p.get_scene_keypoint_2d(KeypointName.RIGHT_SHOULDER)
            lh = p.get_scene_keypoint_2d(KeypointName.LEFT_HIP)
            rh = p.get_scene_keypoint_2d(KeypointName.RIGHT_HIP)
            if (ls or rs) and (lh or rh):
                s_pt = ls or rs
                h_pt = lh or rh
                dx = abs(h_pt[0] - s_pt[0])
                dy = abs(h_pt[1] - s_pt[1])
                if dx > dy * 1.2:  # Horizontal torso
                    fall_scores.append(0.80)

        instant_fall_score = float(max(fall_scores, default=0.0))

        # Pairwise interaction features
        instant_dir_impact = 0.0
        instant_strike_score = 0.0
        instant_rapid_app = 0.0
        instant_prox = 0.0
        instant_pursuit = 0.0
        tracked_ids: List[int] = [p.track_id for p in poses]

        if interactions:
            # Sort by highest aggressive strike or contact
            sorted_inter = sorted(interactions, key=lambda x: (x.directional_impact_score + x.aggressive_strike_score + x.pursuit_score), reverse=True)
            top_i = sorted_inter[0]
            instant_dir_impact = float(top_i.directional_impact_score)
            instant_strike_score = float(top_i.aggressive_strike_score)
            instant_rapid_app = float(max(0.0, -top_i.relative_velocity))
            instant_prox = float(np.clip(1.0 - (top_i.distance / 1.5), 0.0, 1.0))
            instant_pursuit = float(top_i.pursuit_score)
            tracked_ids = [top_i.person_a, top_i.person_b]

        return EvidenceSnapshot(
            timestamp=timestamp,
            frame_id=frame_id,
            aggressive_motion_score=instant_strike_score,
            directional_impact_score=instant_dir_impact,
            rapid_approach_score=instant_rapid_app,
            proximity_score=instant_prox,
            pursuit_score=instant_pursuit,
            fall_score=instant_fall_score,
            pose_quality=pose_conf,
            tracking_quality=track_conf,
            event_probability=0.0,
            category=EventCategory.NORMAL_ACTIVITY,
            action=ActionType.WALKING,
            urgency=OperationalUrgency.BENIGN,
            evidence_state=EvidenceState.NORMAL,
            track_ids=tracked_ids,
            metadata={}
        )
