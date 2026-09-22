"""
Structured Event Evidence Layer for Guardian Matrix.
Formats multi-model computer vision extractions into standardized, typed JSON evidence
as specified in Upgrade_Plan.md Section 15.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np

from collections import deque
from enum import Enum
from ..taxonomy.event_taxonomy import (
    EventLabel,
    EventCategory,
    ActionType,
    OperationalUrgency,
    EvidenceState
)
from ..tracking.tracker import TrackedPerson
from ..pose.pose_representation import PoseFrame
from ..temporal.temporal_engine import TemporalMotionFeatures
from ..interaction.interaction_engine import InteractionFeatures


@dataclass
class StructuredEvidence:
    """
    Structured Evidence Data Object matching Section 15.
    Contains strictly grounded physical observables.
    """
    timestamp: float
    frame_id: int
    scene: Dict[str, Any]
    interaction: Optional[Dict[str, Any]]
    motion: Dict[str, Any]
    pose: Dict[str, Any]
    temporal_model: Dict[str, Any]
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def build_structured_evidence(
    timestamp: float,
    frame_id: int,
    tracked_people: List[TrackedPerson],
    poses: List[PoseFrame],
    interactions: List[InteractionFeatures],
    motion_features: Dict[int, TemporalMotionFeatures],
    classified_event: EventLabel,
    event_confidence: float,
    temporal_probs: Dict[str, float]
) -> StructuredEvidence:
    """
    Compiles raw module extractions into clean structured evidence.
    """
    # 1. Scene information
    scene_data = {
        "person_count": len(tracked_people),
        "tracked_ids": [p.track_id for p in tracked_people]
    }

    # 2. Pairwise interaction evidence (select highest risk pair if multiple)
    interaction_data = None
    if interactions:
        # Prioritize pair with smallest distance or highest contact/pursuit
        sorted_inter = sorted(
            interactions,
            key=lambda x: (x.repeated_contact_score + x.pursuit_score + x.restraint_likelihood - x.distance),
            reverse=True
        )
        primary_inter = sorted_inter[0]
        interaction_data = {
            "pair": [primary_inter.person_a, primary_inter.person_b],
            "distance_m": round(float(primary_inter.distance), 3),
            "duration_s": round(float(primary_inter.proximity_duration_seconds), 2),
            "relative_velocity": round(float(primary_inter.relative_velocity), 3),
            "is_facing_each_other": primary_inter.is_facing_each_other,
            "hand_to_body_min_distance": round(float(primary_inter.hand_to_body_min_distance), 3),
            "hand_contact_probability": round(float(primary_inter.hand_contact_probability), 3),
            "repeated_contact_score": round(float(primary_inter.repeated_contact_score), 3),
            "pursuit_score": round(float(primary_inter.pursuit_score), 3),
            "separation_score": round(float(primary_inter.separation_score), 3),
            "restraint_likelihood": round(float(primary_inter.restraint_likelihood), 3)
        }

    # 3. Motion summaries
    rapid_approach_val = 0.0
    rapid_separation_val = 0.0
    repeated_contact_val = 0.0
    if interactions:
        rapid_approach_val = float(max(0.0, -interactions[0].relative_velocity))
        rapid_separation_val = float(interactions[0].separation_score)
        repeated_contact_val = float(interactions[0].repeated_contact_score)

    motion_data = {
        "rapid_approach": round(min(1.0, rapid_approach_val), 3),
        "rapid_separation": round(min(1.0, rapid_separation_val), 3),
        "repeated_contact": round(min(1.0, repeated_contact_val), 3)
    }

    # 4. Pose quality
    avg_pose_conf = float(np.mean([p.pose_confidence for p in poses])) if poses else 0.0
    pose_data = {
        "confidence": round(avg_pose_conf, 3),
        "num_poses": len(poses),
        "has_3d_keypoints": any(p.keypoints_3d is not None for p in poses)
    }

    # 5. Temporal model
    temporal_data = {
        "interaction_pattern": classified_event.value,
        "confidence": round(event_confidence, 3),
        "top_probabilities": {k: round(v, 3) for k, v in temporal_probs.items() if v >= 0.05}
    }

    # 6. Quality gating metrics
    tracking_qual = float(np.mean([p.detection_confidence for p in tracked_people])) if tracked_people else 0.0
    quality_metrics = {
        "pose_quality": avg_pose_conf,
        "tracking_quality": tracking_qual,
        "temporal_consistency": 0.85 if motion_features else 0.50,
        "evidence_completeness": 1.0 if (poses and interactions and motion_features) else 0.6
    }

    return StructuredEvidence(
        timestamp=timestamp,
        frame_id=frame_id,
        scene=scene_data,
        interaction=interaction_data,
        motion=motion_data,
        pose=pose_data,
        temporal_model=temporal_data,
        quality_metrics=quality_metrics
    )


@dataclass
class EvidenceSnapshot:
    """
    Standardized, self-contained snapshot of physical evidence at a given timestamp.
    Used for rolling temporal reasoning, state transitions, and auditable forensic incident export.
    """
    timestamp: float
    frame_id: int
    aggressive_motion_score: float = 0.0
    directional_impact_score: float = 0.0
    rapid_approach_score: float = 0.0
    proximity_score: float = 0.0
    pursuit_score: float = 0.0
    fall_score: float = 0.0
    pose_quality: float = 0.0
    tracking_quality: float = 0.0
    event_probability: float = 0.0
    category: EventCategory = EventCategory.NORMAL_ACTIVITY
    action: ActionType = ActionType.WALKING
    urgency: OperationalUrgency = OperationalUrgency.BENIGN
    evidence_state: EvidenceState = EvidenceState.NORMAL
    track_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(float(self.timestamp), 3),
            "frame_id": int(self.frame_id),
            "aggressive_motion_score": round(float(self.aggressive_motion_score), 4),
            "directional_impact_score": round(float(self.directional_impact_score), 4),
            "rapid_approach_score": round(float(self.rapid_approach_score), 4),
            "proximity_score": round(float(self.proximity_score), 4),
            "pursuit_score": round(float(self.pursuit_score), 4),
            "fall_score": round(float(self.fall_score), 4),
            "pose_quality": round(float(self.pose_quality), 4),
            "tracking_quality": round(float(self.tracking_quality), 4),
            "event_probability": round(float(self.event_probability), 4),
            "category": self.category.value if isinstance(self.category, EventCategory) else str(self.category),
            "action": self.action.value if isinstance(self.action, ActionType) else str(self.action),
            "urgency": self.urgency.value if isinstance(self.urgency, OperationalUrgency) else str(self.urgency),
            "evidence_state": self.evidence_state.value if isinstance(self.evidence_state, EvidenceState) else str(self.evidence_state),
            "track_ids": self.track_ids,
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class RollingEvidenceBuffer:
    """
    Bounded rolling queue for EvidenceSnapshot objects to prevent unbounded memory growth.
    """
    def __init__(self, maxlen: int = 120):
        self.maxlen = maxlen
        self.snapshots: deque[EvidenceSnapshot] = deque(maxlen=maxlen)

    def append(self, snapshot: EvidenceSnapshot) -> None:
        self.snapshots.append(snapshot)

    def clear(self) -> None:
        self.snapshots.clear()

    def get_recent(self, seconds: float, current_timestamp: Optional[float] = None) -> List[EvidenceSnapshot]:
        """Returns snapshots within the last `seconds` elapsed time."""
        if not self.snapshots:
            return []
        now = current_timestamp if current_timestamp is not None else self.snapshots[-1].timestamp
        cutoff = now - seconds
        return [s for s in self.snapshots if s.timestamp >= cutoff]

    def __len__(self) -> int:
        return len(self.snapshots)

    def __iter__(self):
        return iter(self.snapshots)
