"""
Structured Event Evidence Layer for Guardian Matrix.
Formats multi-model computer vision extractions into standardized, typed JSON evidence
as specified in Upgrade_Plan.md Section 15.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np

from ..taxonomy.event_taxonomy import EventLabel
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
