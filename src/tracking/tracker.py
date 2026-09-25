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


import cv2


def extract_appearance_feature(image: Optional[np.ndarray], bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """
    Extracts a 64-dimensional spatial-color appearance descriptor:
    - Upper half (torso/clothing): 16 Hue, 8 Saturation, 8 Value bins (32 bins)
    - Lower half (pants/skirt): 16 Hue, 8 Saturation, 8 Value bins (32 bins)
    Returns L2-normalized vector. Extremely fast (<0.1ms) and robust to scale/lighting.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None

    h_img, w_img = image.shape[:2]
    x1 = max(0, min(int(round(bbox[0])), w_img - 1))
    y1 = max(0, min(int(round(bbox[1])), h_img - 1))
    x2 = max(0, min(int(round(bbox[2])), w_img))
    y2 = max(0, min(int(round(bbox[3])), h_img))

    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w < 4 or crop_h < 8:
        return None

    crop = image[y1:y2, x1:x2]
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    except Exception:
        return None

    mid_y = max(1, crop_h // 2)
    upper = hsv[:mid_y, :]
    lower = hsv[mid_y:, :]

    # Upper body histogram
    h_up = cv2.calcHist([upper], [0], None, [16], [0, 180]).flatten()
    s_up = cv2.calcHist([upper], [1], None, [8], [0, 256]).flatten()
    v_up = cv2.calcHist([upper], [2], None, [8], [0, 256]).flatten()
    up_feat = np.concatenate([h_up, s_up, v_up])

    # Lower body histogram
    h_low = cv2.calcHist([lower], [0], None, [16], [0, 180]).flatten()
    s_low = cv2.calcHist([lower], [1], None, [8], [0, 256]).flatten()
    v_low = cv2.calcHist([lower], [2], None, [8], [0, 256]).flatten()
    low_feat = np.concatenate([h_low, s_low, v_low])

    feat = np.concatenate([up_feat, low_feat]).astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm > 1e-6:
        feat /= norm
    return feat


def cosine_similarity(featA: Optional[np.ndarray], featB: Optional[np.ndarray]) -> float:
    """Compute cosine similarity between two unit-normalized appearance vectors."""
    if featA is None or featB is None:
        return 0.0
    dot = float(np.dot(featA, featB))
    return max(0.0, min(1.0, dot))


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
    2D bounding box Kalman filter state estimator with Visual Appearance Memory.
    State vector: [x, y, s, r, dx, dy, ds]
    where (x, y) is box center, s is scale (area), r is aspect ratio (w/h).
    """
    count = 0  # Fallback counter for standalone tracker usage

    def __init__(self, bbox: Tuple[float, float, float, float], track_id: Optional[int] = None):
        if track_id is not None:
            self.id = track_id
        else:
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
        self.last_detection_box: Optional[Tuple[float, float, float, float]] = bbox

        # Visual Appearance Re-ID state
        self.appearance: Optional[np.ndarray] = None
        self.feature_history: List[np.ndarray] = []

    def update_appearance(self, feat: Optional[np.ndarray]):
        """Update visual appearance descriptor with exponential moving average and history bank."""
        if feat is None:
            return
        if self.appearance is None:
            self.appearance = feat.copy()
        else:
            self.appearance = 0.85 * self.appearance + 0.15 * feat
            norm = np.linalg.norm(self.appearance)
            if norm > 1e-6:
                self.appearance /= norm
        self.feature_history.append(feat.copy())
        if len(self.feature_history) > 10:
            self.feature_history.pop(0)

    def match_appearance(self, feat: Optional[np.ndarray]) -> float:
        """Compute maximum appearance similarity across EMA model and feature history bank."""
        if feat is None or self.appearance is None:
            return 0.0
        ema_sim = cosine_similarity(self.appearance, feat)
        if self.feature_history:
            bank_max = max(cosine_similarity(f, feat) for f in self.feature_history)
            return max(ema_sim, bank_max)
        return ema_sim

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

    def update(self, bbox: Tuple[float, float, float, float], feat: Optional[np.ndarray] = None):
        """Update the state estimate with observed measurement and visual feature."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_detection_box = bbox

        if feat is not None:
            self.update_appearance(feat)

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
        """Convert state [x, y, s, r] to [x1, y1, x2, y2] with jitter smoothing."""
        x, y, s, r = self.state[0], self.state[1], max(1.0, self.state[2]), max(0.1, self.state[3])
        w = np.sqrt(s * r)
        h = s / w
        pred_box = (float(x - w / 2.0), float(y - h / 2.0), float(x + w / 2.0), float(y + h / 2.0))

        # When freshly updated, blend measurement (70%) and Kalman prediction (30%) to eliminate box jitter
        if self.time_since_update == 0 and self.last_detection_box is not None:
            m = self.last_detection_box
            return (
                0.70 * m[0] + 0.30 * pred_box[0],
                0.70 * m[1] + 0.30 * pred_box[1],
                0.70 * m[2] + 0.30 * pred_box[2],
                0.70 * m[3] + 0.30 * pred_box[3]
            )
        return pred_box


class BaseTracker(ABC):
    @abstractmethod
    def update(
        self,
        detections: List[PersonDetection],
        timestamp: float = 0.0,
        frame_id: int = 0,
        frame: Optional[np.ndarray] = None
    ) -> List[TrackedPerson]:
        pass

    @abstractmethod
    def reset(self):
        pass


class ByteTracker(BaseTracker):
    """
    ByteTrack Multi-Object Tracker with Visual Appearance Re-Identification.
    Associates high-confidence detections with active tracks using combined IoU and
    HSV Appearance Cosine Similarity, performs low-confidence association, and
    executes lost-track recovery to prevent ID explosion and duplicate counting.
    """

    def __init__(
        self,
        track_thresh: float = 0.4,
        match_thresh: float = 0.35,
        max_time_lost: int = 45,
        min_hits: int = 2
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        self.min_hits = min_hits
        self.trackers: List[KalmanBoxTracker] = []
        self.lost_trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self._next_id = 1
        self.total_unique_seen = 0

    def reset(self):
        self.trackers.clear()
        self.lost_trackers.clear()
        self.frame_count = 0
        self._next_id = 1
        self.total_unique_seen = 0
        KalmanBoxTracker.count = 0

    def update(
        self,
        detections: List[PersonDetection],
        timestamp: float = 0.0,
        frame_id: int = 0,
        frame: Optional[np.ndarray] = None
    ) -> List[TrackedPerson]:
        self.frame_count += 1
        current_frame = frame_id if frame_id > 0 else self.frame_count

        # Step 1: Predict positions for all active and lost trackers
        for trk in self.trackers:
            trk.predict()
        for trk in self.lost_trackers:
            trk.predict()

        # Step 2: Extract visual appearance features for detections
        det_features: List[Optional[np.ndarray]] = []
        for det in detections:
            feat = extract_appearance_feature(frame, det.bounding_box) if frame is not None else None
            det_features.append(feat)

        # Step 3: Split detections into high confidence and low confidence
        high_indices = [i for i, d in enumerate(detections) if d.confidence >= self.track_thresh]
        low_indices = [i for i, d in enumerate(detections) if 0.10 <= d.confidence < self.track_thresh]

        high_dets = [detections[i] for i in high_indices]
        high_feats = [det_features[i] for i in high_indices]

        low_dets = [detections[i] for i in low_indices]
        low_feats = [det_features[i] for i in low_indices]

        # Step 4: Associate high confidence detections with active tracks
        matched_trks, unmatched_high_dets, unmatched_trks = self._associate_hybrid(
            high_dets,
            high_feats,
            self.trackers,
            affinity_threshold=self.match_thresh
        )

        for det_idx, trk_idx in matched_trks:
            self.trackers[trk_idx].update(
                high_dets[det_idx].bounding_box,
                feat=high_feats[det_idx]
            )

        # Step 5: Associate remaining unmatched active tracks with low confidence detections
        unmatched_active_trackers = [self.trackers[i] for i in unmatched_trks]
        if len(unmatched_active_trackers) > 0 and len(low_dets) > 0:
            low_matched_trks, _, remaining_unmatched_idx = self._associate_hybrid(
                low_dets,
                low_feats,
                unmatched_active_trackers,
                affinity_threshold=0.25
            )
            for det_idx, trk_local_idx in low_matched_trks:
                actual_trk_idx = unmatched_trks[trk_local_idx]
                self.trackers[actual_trk_idx].update(
                    low_dets[det_idx].bounding_box,
                    feat=low_feats[det_idx]
                )
            # Update remaining unmatched active trackers
            unmatched_trks = [unmatched_trks[i] for i in remaining_unmatched_idx]

        # Step 6: Visual Re-ID Recovery of Lost Tracks before spawning NEW IDs
        # This directly eliminates ID explosion when persons cross or briefly occlude!
        unmatched_after_recovery: List[int] = []
        for det_idx in unmatched_high_dets:
            det_box = high_dets[det_idx].bounding_box
            det_feat = high_feats[det_idx]
            det_center = ((det_box[0] + det_box[2]) / 2.0, (det_box[1] + det_box[3]) / 2.0)
            det_h = max(1.0, det_box[3] - det_box[1])

            recovered_trk_idx = -1
            best_recovery_score = 0.0

            for l_idx, lost_trk in enumerate(self.lost_trackers):
                lost_box = lost_trk.get_bbox()
                iou = compute_iou(det_box, lost_box)
                app_sim = lost_trk.match_appearance(det_feat) if det_feat is not None else 0.0

                lost_center = lost_trk.history_centers[-1] if lost_trk.history_centers else (0.0, 0.0)
                dist = np.hypot(det_center[0] - lost_center[0], det_center[1] - lost_center[1])
                spatial_plausible = dist < (det_h * 3.5 + lost_trk.time_since_update * 15.0)

                # Recovery criteria:
                # 1. High visual appearance match with spatial plausibility
                # 2. Moderate IoU with predicted box
                if (app_sim >= 0.58 and spatial_plausible) or (iou >= 0.20):
                    score = 0.5 * app_sim + 0.5 * max(iou, 0.2)
                    if score > best_recovery_score:
                        best_recovery_score = score
                        recovered_trk_idx = l_idx

            if recovered_trk_idx >= 0:
                # Re-activate lost track with original ID!
                recovered_trk = self.lost_trackers.pop(recovered_trk_idx)
                recovered_trk.update(det_box, feat=det_feat)
                self.trackers.append(recovered_trk)
            else:
                unmatched_after_recovery.append(det_idx)

        # Step 7: Suppress redundant duplicates and initialize new tracks
        for det_idx in unmatched_after_recovery:
            det_box = high_dets[det_idx].bounding_box
            det_feat = high_feats[det_idx]

            # Suppress spawning if detection significantly overlaps with an active tracker
            is_redundant = False
            for trk in self.trackers:
                if compute_iou(det_box, trk.get_bbox()) > 0.30:
                    is_redundant = True
                    if trk.time_since_update > 0:
                        trk.update(det_box, feat=det_feat)
                    break

            if not is_redundant:
                new_trk = KalmanBoxTracker(det_box, track_id=self._next_id)
                new_trk.update_appearance(det_feat)
                self._next_id += 1
                self.total_unique_seen += 1
                self.trackers.append(new_trk)

        # Step 8: Partition active trackers and migrate lost tracks
        new_active: List[KalmanBoxTracker] = []
        for trk in self.trackers:
            if trk.time_since_update == 0:
                new_active.append(trk)
            elif trk.time_since_update <= self.max_time_lost:
                self.lost_trackers.append(trk)

        # Prune dead lost trackers
        self.lost_trackers = [
            trk for trk in self.lost_trackers
            if trk.time_since_update <= self.max_time_lost
        ]

        # Step 9: Tracker NMS on active trackers to suppress multi-detection duplicates
        unique_active: List[KalmanBoxTracker] = []
        sorted_active = sorted(new_active, key=lambda t: (t.hits, -t.id), reverse=True)
        for trk in sorted_active:
            is_dup = False
            for u in unique_active:
                if compute_iou(trk.get_bbox(), u.get_bbox()) > 0.35:
                    is_dup = True
                    break
            if not is_dup:
                unique_active.append(trk)
        self.trackers = unique_active

        # Step 10: Build output TrackedPerson list
        active_tracked_persons: List[TrackedPerson] = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                conf = 0.85
                c = trk.get_bbox()
                for d in detections:
                    if compute_iou(c, d.bounding_box) > 0.30:
                        conf = d.confidence
                        break

                tracked_person = TrackedPerson(
                    track_id=trk.id,
                    bbox=trk.get_bbox(),
                    detection_confidence=conf,
                    timestamp=timestamp,
                    frame_id=current_frame,
                    state=TrackState.TRACKED,
                    age=trk.age,
                    time_since_update=0,
                    history=list(trk.history_centers)
                )
                active_tracked_persons.append(tracked_person)

        return active_tracked_persons

    def _associate_hybrid(
        self,
        dets: List[PersonDetection],
        feats: List[Optional[np.ndarray]],
        trks: List[KalmanBoxTracker],
        affinity_threshold: float = 0.30
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Bipartite matching combining spatial IoU and Visual Appearance Cosine Similarity.
        """
        if len(dets) == 0 or len(trks) == 0:
            return [], list(range(len(dets))), list(range(len(trks)))

        affinity_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            det_feat = feats[d]
            for t, trk in enumerate(trks):
                iou = compute_iou(det.bounding_box, trk.get_bbox())
                app_sim = trk.match_appearance(det_feat) if det_feat is not None else 0.0

                if trk.appearance is not None and det_feat is not None:
                    # Hybrid score: 60% spatial overlap + 40% visual appearance
                    score = 0.60 * iou + 0.40 * app_sim
                    # If high visual match and reasonable proximity, allow matching even if IoU dropped
                    if iou < 0.10 and app_sim >= 0.70:
                        score = max(score, 0.50 * app_sim)
                else:
                    score = iou

                affinity_matrix[d, t] = score

        # Greedy match based on highest affinity
        matched_pairs: List[Tuple[int, int]] = []
        matched_dets = set()
        matched_trks = set()

        indices = np.argsort(-affinity_matrix, axis=None)
        for idx in indices:
            d = idx // len(trks)
            t = idx % len(trks)
            if d in matched_dets or t in matched_trks:
                continue
            if affinity_matrix[d, t] < affinity_threshold:
                break
            matched_dets.add(d)
            matched_trks.add(t)
            matched_pairs.append((int(d), int(t)))

        unmatched_dets = [d for d in range(len(dets)) if d not in matched_dets]
        unmatched_trks = [t for t in range(len(trks)) if t not in matched_trks]
        return matched_pairs, unmatched_dets, unmatched_trks

    def _associate(
        self,
        det_boxes: List[Tuple[float, float, float, float]],
        trk_boxes: List[Tuple[float, float, float, float]],
        iou_threshold: float = 0.5
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Pure IoU matching kept for backward compatibility."""
        if len(det_boxes) == 0 or len(trk_boxes) == 0:
            return [], list(range(len(det_boxes))), list(range(len(trk_boxes)))

        iou_matrix = np.zeros((len(det_boxes), len(trk_boxes)), dtype=np.float32)
        for d, det in enumerate(det_boxes):
            for t, trk in enumerate(trk_boxes):
                iou_matrix[d, t] = compute_iou(det, trk)

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
    track_thresh = cfg.get("track_thresh", 0.4)
    match_thresh = cfg.get("match_thresh", 0.35)
    track_buffer = cfg.get("track_buffer", 45)
    return ByteTracker(track_thresh=track_thresh, match_thresh=match_thresh, max_time_lost=track_buffer)

