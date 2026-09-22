"""
Multi-Object Tracking Layer for Guardian Matrix.
Implements persistent identity tracking based on ByteTrack / Kalman-IoU matching.
Produces TrackedPerson representations enabling downstream temporal reasoning.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from ..detection.detector import PersonDetection


class TrackState(Enum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class TrackedPerson:
    """
    Tracked Person representation as specified in Upgrade_Plan.md Section 6.
    """
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    detection_confidence: float
    timestamp: float  # video time or frame timestamp
    frame_id: int = 0
    state: TrackState = TrackState.TRACKED
    age: int = 1
    time_since_update: int = 0
    history: List[Tuple[float, float]] = field(default_factory=list)  # [(x_center, y_center), ...]

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union_area = areaA + areaB - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class KalmanBoxTracker:
    """
    2D bounding box Kalman filter state estimator.
    State vector: [x, y, s, r, dx, dy, ds]
    where (x, y) is box center, s is scale (area), r is aspect ratio (w/h).
    """
    count = 0

    def __init__(self, bbox: Tuple[float, float, float, float]):
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        
        # [x, y, s, r, dx, dy, ds]
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)

        self.state = np.array([x, y, s, r, 0.0, 0.0, 0.0], dtype=np.float32)
        # Covariance estimate error
        self.P = np.eye(7, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # High uncertainty in initial velocities

        # State transition model
        self.F = np.eye(7, dtype=np.float32)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        # Measurement model: observes [x, y, s, r]
        self.H = np.zeros((4, 7), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Process and measurement noises
        self.Q = np.eye(7, dtype=np.float32) * 0.01
        self.Q[4:, 4:] *= 0.1
        self.R = np.eye(4, dtype=np.float32) * 1.0

        self.time_since_update = 0
        self.history_centers: List[Tuple[float, float]] = [(x, y)]
        self.hits = 1
        self.hit_streak = 1
        self.age = 1

    def predict(self) -> Tuple[float, float, float, float]:
        """Advance the state vector and return the predicted bounding box."""
        if (self.state[6] + self.state[2]) <= 0:
            self.state[6] = 0.0
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_bbox()

    def update(self, bbox: Tuple[float, float, float, float]):
        """Update the state estimate with observed measurement."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        z = np.array([x, y, s, r], dtype=np.float32)

        # Innovation
        y_innov = z - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.state = self.state + np.dot(K, y_innov)
        self.P = np.dot((np.eye(7) - np.dot(K, self.H)), self.P)
        self.history_centers.append((self.state[0], self.state[1]))
        if len(self.history_centers) > 60:
            self.history_centers.pop(0)

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """Convert state [x, y, s, r] to [x1, y1, x2, y2]."""
        x, y, s, r = self.state[0], self.state[1], max(1.0, self.state[2]), max(0.1, self.state[3])
        w = np.sqrt(s * r)
        h = s / w
        return (float(x - w / 2.0), float(y - h / 2.0), float(x + w / 2.0), float(y + h / 2.0))


class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: List[PersonDetection], timestamp: float = 0.0, frame_id: int = 0) -> List[TrackedPerson]:
        pass

    @abstractmethod
    def reset(self):
        pass


class ByteTracker(BaseTracker):
    """
    ByteTrack-style Multi-Object Tracker.
    Associates high-confidence detections first, followed by remaining unmatched tracks
    associated with low-confidence detections, preserving track consistency across occlusions.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.6,
        max_time_lost: int = 30,
        min_hits: int = 2
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        self.min_hits = min_hits
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def reset(self):
        self.trackers.clear()
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(
        self,
        detections: List[PersonDetection],
        timestamp: float = 0.0,
        frame_id: int = 0
    ) -> List[TrackedPerson]:
        self.frame_count += 1
        current_frame = frame_id if frame_id > 0 else self.frame_count

        # Step 1: Predict positions for existing trackers
        for trk in self.trackers:
            trk.predict()

        # Step 2: Split detections into high confidence and low confidence
        high_dets = [d for d in detections if d.confidence >= self.track_thresh]
        low_dets = [d for d in detections if 0.1 <= d.confidence < self.track_thresh]

        # Step 3: Associate high confidence detections with existing tracks
        matched_trks, unmatched_dets, unmatched_trks = self._associate(
            [d.bounding_box for d in high_dets],
            [trk.get_bbox() for trk in self.trackers],
            iou_threshold=self.match_thresh
        )

        for det_idx, trk_idx in matched_trks:
            self.trackers[trk_idx].update(high_dets[det_idx].bounding_box)

        # Step 4: Associate remaining unmatched tracks with low confidence detections
        if len(unmatched_trks) > 0 and len(low_dets) > 0:
            remaining_trks = [self.trackers[i] for i in unmatched_trks]
            low_matched_trks, _, remaining_unmatched = self._associate(
                [d.bounding_box for d in low_dets],
                [trk.get_bbox() for trk in remaining_trks],
                iou_threshold=0.4
            )
            for det_idx, trk_idx_local in low_matched_trks:
                trk_actual_idx = unmatched_trks[trk_idx_local]
                self.trackers[trk_actual_idx].update(low_dets[det_idx].bounding_box)

        # Step 5: Initialize new tracks for unmatched high-confidence detections
        for det_idx in unmatched_dets:
            new_trk = KalmanBoxTracker(high_dets[det_idx].bounding_box)
            self.trackers.append(new_trk)

        # Step 6: Prune dead tracks and build output TrackedPerson list
        active_tracked_persons: List[TrackedPerson] = []
        alive_trackers: List[KalmanBoxTracker] = []

        for trk in self.trackers:
            if trk.time_since_update <= self.max_time_lost:
                alive_trackers.append(trk)
                # Ensure track has enough hits to be reliable
                if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                    # Find matching detection confidence if recently updated
                    conf = 0.85
                    if trk.time_since_update == 0:
                        # Find closest detection
                        c = trk.get_bbox()
                        for d in detections:
                            if compute_iou(c, d.bounding_box) > 0.3:
                                conf = d.confidence
                                break

                    tracked_person = TrackedPerson(
                        track_id=trk.id,
                        bbox=trk.get_bbox(),
                        detection_confidence=conf,
                        timestamp=timestamp,
                        frame_id=current_frame,
                        state=TrackState.TRACKED if trk.time_since_update == 0 else TrackState.LOST,
                        age=trk.age,
                        time_since_update=trk.time_since_update,
                        history=list(trk.history_centers)
                    )
                    active_tracked_persons.append(tracked_person)

        self.trackers = alive_trackers
        return active_tracked_persons

    def _associate(
        self,
        det_boxes: List[Tuple[float, float, float, float]],
        trk_boxes: List[Tuple[float, float, float, float]],
        iou_threshold: float = 0.5
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy IoU bipartite matching."""
        if len(det_boxes) == 0 or len(trk_boxes) == 0:
            return [], list(range(len(det_boxes))), list(range(len(trk_boxes)))

        iou_matrix = np.zeros((len(det_boxes), len(trk_boxes)), dtype=np.float32)
        for d, det in enumerate(det_boxes):
            for t, trk in enumerate(trk_boxes):
                iou_matrix[d, t] = compute_iou(det, trk)

        # Greedy match from highest IoU
        matched_pairs: List[Tuple[int, int]] = []
        matched_dets = set()
        matched_trks = set()

        indices = np.argsort(-iou_matrix, axis=None)
        for idx in indices:
            d = idx // len(trk_boxes)
            t = idx % len(trk_boxes)
            if d in matched_dets or t in matched_trks:
                continue
            if iou_matrix[d, t] < iou_threshold:
                break
            matched_dets.add(d)
            matched_trks.add(t)
            matched_pairs.append((int(d), int(t)))

        unmatched_dets = [d for d in range(len(det_boxes)) if d not in matched_dets]
        unmatched_trks = [t for t in range(len(trk_boxes)) if t not in matched_trks]
        return matched_pairs, unmatched_dets, unmatched_trks


def load_tracker(tracker_type: str = "bytetrack", config: Optional[Dict[str, Any]] = None) -> BaseTracker:
    cfg = config or {}
    track_thresh = cfg.get("track_thresh", 0.5)
    match_thresh = cfg.get("match_thresh", 0.6)
    track_buffer = cfg.get("track_buffer", 30)
    return ByteTracker(track_thresh=track_thresh, match_thresh=match_thresh, max_time_lost=track_buffer)
