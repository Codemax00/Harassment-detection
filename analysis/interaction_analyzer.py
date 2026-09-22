"""
Guardian Matrix - Interaction Analyzer
Computes pairwise interaction dynamics: distances, approach velocity, orientation,
contact probability, pursuit metrics, and persistence across a temporal window.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import numpy as np
from src.interaction.interaction_engine import InteractionEngine, InteractionFeatures
from src.temporal.temporal_engine import TemporalMotionEngine, TemporalMotionFeatures
from src.pose.pose_representation import PoseFrame


class InteractionAnalyzerInterface(ABC):
    @abstractmethod
    def analyze(self, tracks: List[Dict[str, Any]], poses: List[Dict[str, Any]], timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        pass


class InteractionAnalyzer(InteractionAnalyzerInterface):
    """
    Analyzes person-to-person interaction metrics across temporal frames.
    """

    def __init__(self, proximity_threshold: float = 0.8, contact_threshold: float = 0.25):
        self.engine = InteractionEngine(
            proximity_threshold=proximity_threshold,
            contact_threshold=contact_threshold,
        )
        self.motion_engine = TemporalMotionEngine(window_seconds=2.0)
        self.frame_id = 0

    def analyze(
        self,
        tracks: List[Dict[str, Any]],
        poses: List[Dict[str, Any]],
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Calculates pairwise interaction features for all person pairs in the frame.
        Outputs:
        [
            {
                "person_a": int,
                "person_b": int,
                "distance": float,
                "relative_velocity": float,
                "duration": float,
                "is_facing": bool,
                "contact_probability": float,
                "pursuit_score": float,
                "repeated_contact_score": float,
                "_raw": InteractionFeatures
            }
        ]
        """
        self.frame_id += 1
        now = timestamp if timestamp is not None else time.time()

        if len(poses) < 2:
            return []

        # Extract PoseFrame objects and update timestamp/frame_id
        pose_frames: List[PoseFrame] = []
        for p in poses:
            pf = p.get("_pose_frame")
            if pf is not None and isinstance(pf, PoseFrame):
                pf.timestamp = now
                pf.frame_id = self.frame_id
                pose_frames.append(pf)

        motion_features = self.motion_engine.update(pose_frames)

        raw_interactions: List[InteractionFeatures] = self.engine.update(
            poses=pose_frames,
            motion_features=motion_features,
            timestamp=now,
        )

        results: List[Dict[str, Any]] = []
        for item in raw_interactions:
            results.append({
                "person_a": item.person_a,
                "person_b": item.person_b,
                "distance": round(item.distance, 2),
                "relative_velocity": round(item.relative_velocity, 2),
                "duration": round(item.proximity_duration_seconds, 2),
                "is_facing": item.is_facing_each_other,
                "contact_probability": round(item.hand_contact_probability, 3),
                "pursuit_score": round(item.pursuit_score, 3),
                "repeated_contact_score": round(item.repeated_contact_score, 3),
                "_raw": item,
            })

        return results
