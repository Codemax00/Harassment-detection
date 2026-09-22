"""
Guardian Matrix - Multi-Person Tracking Module
Wraps ByteTracker to provide stable track IDs, trajectory state estimation,
and maintains temporal person histories (positions, velocity, poses, timestamps).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import math
import time
import numpy as np
from src.tracking.tracker import ByteTracker, TrackedPerson
from src.detection.detector import PersonDetection
from config.settings import settings


class Tracker(ABC):
    """Abstract Tracker interface."""

    @abstractmethod
    def update(self, detections: List[Dict[str, Any]], timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        pass


class MultiPersonTracker(Tracker):
    """
    Multi-person tracker with Kalman filtering, stable IDs, and movement history tracking.
    """

    def __init__(self, max_tracks: Optional[int] = None, max_time_lost: int = 30):
        self.max_tracks = max_tracks or settings.max_tracks
        self.underlying_tracker = ByteTracker(
            track_thresh=settings.detection_confidence,
            match_thresh=0.55,
            max_time_lost=max_time_lost,
            min_hits=2,  # Require confirmation across frames to prevent flickering IDs
        )
        self.frame_id = 0
        self.person_history: Dict[int, Dict[str, Any]] = {}
        self.smoothed_bboxes: Dict[int, List[float]] = {}

    def update(self, detections: List[Dict[str, Any]], timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Takes raw detections list and updates tracking states.
        Returns list of active, actively-visible tracked persons with smoothed bboxes and stable track_ids.
        """
        self.frame_id += 1
        now = timestamp if timestamp is not None else time.time()

        # Convert input dict detections to PersonDetection dataclasses
        person_detections: List[PersonDetection] = []
        for det in detections:
            if "_raw" in det and isinstance(det["_raw"], PersonDetection):
                person_detections.append(det["_raw"])
            else:
                bbox = det.get("bbox", [0, 0, 0, 0])
                conf = det.get("confidence", 0.5)
                person_detections.append(
                    PersonDetection(
                        bounding_box=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        confidence=conf,
                        frame_id=self.frame_id,
                    )
                )

        tracked_people: List[TrackedPerson] = self.underlying_tracker.update(
            person_detections, timestamp=now, frame_id=self.frame_id
        )

        results: List[Dict[str, Any]] = []
        active_ids = set()

        for tp in tracked_people[:self.max_tracks]:
            # CRITICAL: Do NOT display ghost tracks that were not updated in the current frame
            if tp.time_since_update > 0:
                continue

            tid = tp.track_id
            active_ids.add(tid)

            # Bounding box smoothing (EMA) to eliminate visual jitter
            raw_box = [tp.x1, tp.y1, tp.x2, tp.y2]
            if tid in self.smoothed_bboxes:
                prev = self.smoothed_bboxes[tid]
                alpha = 0.65  # Weight for previous frame (stability)
                smooth_box = [
                    alpha * prev[0] + (1 - alpha) * raw_box[0],
                    alpha * prev[1] + (1 - alpha) * raw_box[1],
                    alpha * prev[2] + (1 - alpha) * raw_box[2],
                    alpha * prev[3] + (1 - alpha) * raw_box[3],
                ]
            else:
                smooth_box = raw_box
            self.smoothed_bboxes[tid] = smooth_box

            cx = (smooth_box[0] + smooth_box[2]) / 2.0
            cy = (smooth_box[1] + smooth_box[3]) / 2.0

            if tid not in self.person_history:
                self.person_history[tid] = {
                    "positions": [],
                    "poses": [],
                    "timestamps": [],
                    "velocity": [],
                    "interaction_history": [],
                }

            history = self.person_history[tid]
            history["positions"].append((cx, cy))
            history["timestamps"].append(now)

            # Compute velocity based on smoothed centers
            vel = 0.0
            if len(history["positions"]) >= 2:
                prev_x, prev_y = history["positions"][-2]
                dt = history["timestamps"][-1] - history["timestamps"][-2]
                if dt > 0.01:
                    dist = math.hypot(cx - prev_x, cy - prev_y)
                    vel = dist / dt
            history["velocity"].append(round(vel, 1))

            # Trim history
            if len(history["positions"]) > 60:
                history["positions"].pop(0)
                history["timestamps"].pop(0)
                history["velocity"].pop(0)
                if len(history["poses"]) > 60:
                    history["poses"].pop(0)
                if len(history["interaction_history"]) > 60:
                    history["interaction_history"].pop(0)

            results.append({
                "track_id": tid,
                "bbox": [round(smooth_box[0], 1), round(smooth_box[1], 1), round(smooth_box[2], 1), round(smooth_box[3], 1)],
                "center": (round(cx, 1), round(cy, 1)),
                "confidence": round(tp.detection_confidence, 3),
                "class": "person",
                "velocity": round(vel, 1),
                "_tracked_person": tp,
            })

        # Purge inactive tracks
        expired_ids = [
            tid for tid in self.person_history
            if tid not in active_ids and (now - self.person_history[tid]["timestamps"][-1]) > 5.0
        ]
        for tid in expired_ids:
            del self.person_history[tid]
            if tid in self.smoothed_bboxes:
                del self.smoothed_bboxes[tid]

        return results

    def attach_pose(self, track_id: int, pose_data: Dict[str, Any]):
        """Attaches latest pose estimation to a track's history."""
        if track_id in self.person_history:
            self.person_history[track_id]["poses"].append(pose_data)
            if len(self.person_history[track_id]["poses"]) > 60:
                self.person_history[track_id]["poses"].pop(0)

    def get_history(self, track_id: int) -> Optional[Dict[str, Any]]:
        return self.person_history.get(track_id)
