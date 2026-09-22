"""
Pose Representation & Normalization Layer for Guardian Matrix.
Standardized 2D and 3D pose representation as defined in Upgrade_Plan.md Section 9.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class KeypointName(IntEnum):
    """Standard COCO 17 Keypoint Indices with extended aliases."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


@dataclass
class PoseFrame:
    """
    Standardized Pose Representation for every person in a frame.
    Maintains both normalized coordinates (for single-person ML models)
    and scene/pixel coordinates (for pairwise interaction and surveillance geometry).
    """
    track_id: int
    timestamp: float
    keypoints_2d: np.ndarray  # Shape: (K, 2) normalized or pixel coordinates
    keypoints_3d: Optional[np.ndarray] = None  # Shape: (K, 3) relative coordinates in meters/normalized units
    visibility: Optional[np.ndarray] = None  # Shape: (K,) confidence/visibility per keypoint
    pose_confidence: float = 0.0
    frame_id: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None
    is_normalized: bool = False
    scene_keypoints_2d: Optional[np.ndarray] = None  # Exact pixel coordinates in image frame
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_keypoint_2d(self, idx: int) -> Optional[Tuple[float, float]]:
        """Get 2D keypoint coordinate (normalized if is_normalized is True)."""
        if self.keypoints_2d is not None and idx < len(self.keypoints_2d):
            kp = self.keypoints_2d[idx]
            if self.visibility is None or self.visibility[idx] > 0.2:
                return (float(kp[0]), float(kp[1]))
        return None

    def get_scene_keypoint_2d(self, idx: int) -> Optional[Tuple[float, float]]:
        """Get 2D keypoint coordinate in absolute image/scene space."""
        if self.scene_keypoints_2d is not None and idx < len(self.scene_keypoints_2d):
            kp = self.scene_keypoints_2d[idx]
            if self.visibility is None or self.visibility[idx] > 0.2:
                return (float(kp[0]), float(kp[1]))
        elif not self.is_normalized and self.keypoints_2d is not None and idx < len(self.keypoints_2d):
            kp = self.keypoints_2d[idx]
            if self.visibility is None or self.visibility[idx] > 0.2:
                return (float(kp[0]), float(kp[1]))
        elif self.is_normalized and self.bbox is not None and self.keypoints_2d is not None and idx < len(self.keypoints_2d):
            # Un-normalize using bbox
            x1, y1, x2, y2 = self.bbox
            h = max(1.0, y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            kp = self.keypoints_2d[idx]
            if self.visibility is None or self.visibility[idx] > 0.2:
                return (float(kp[0] * h + cx), float(kp[1] * h + cy))
        return None

    def get_keypoint_3d(self, idx: int) -> Optional[Tuple[float, float, float]]:
        """Get 3D keypoint coordinate if available."""
        if self.keypoints_3d is not None and idx < len(self.keypoints_3d):
            kp = self.keypoints_3d[idx]
            return (float(kp[0]), float(kp[1]), float(kp[2]))
        return None

    def get_torso_center(self) -> Tuple[float, float]:
        """Calculate midpoint between hips and shoulders in local coordinate system."""
        shoulders = []
        hips = []
        for s in [KeypointName.LEFT_SHOULDER, KeypointName.RIGHT_SHOULDER]:
            kp = self.get_keypoint_2d(s)
            if kp: shoulders.append(kp)
        for h in [KeypointName.LEFT_HIP, KeypointName.RIGHT_HIP]:
            kp = self.get_keypoint_2d(h)
            if kp: hips.append(kp)

        points = shoulders + hips
        if points:
            x = sum(p[0] for p in points) / len(points)
            y = sum(p[1] for p in points) / len(points)
            return (x, y)
        elif self.bbox is not None and not self.is_normalized:
            return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)
        return (0.0, 0.0)

    def get_scene_torso_center(self) -> Tuple[float, float]:
        """Calculate torso center in absolute image scene coordinates."""
        shoulders = []
        hips = []
        for s in [KeypointName.LEFT_SHOULDER, KeypointName.RIGHT_SHOULDER]:
            kp = self.get_scene_keypoint_2d(s)
            if kp: shoulders.append(kp)
        for h in [KeypointName.LEFT_HIP, KeypointName.RIGHT_HIP]:
            kp = self.get_scene_keypoint_2d(h)
            if kp: hips.append(kp)

        points = shoulders + hips
        if points:
            x = sum(p[0] for p in points) / len(points)
            y = sum(p[1] for p in points) / len(points)
            return (x, y)
        elif self.bbox is not None:
            return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)
        return (0.0, 0.0)


def normalize_pose(
    pose: PoseFrame,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    image_shape: Optional[Tuple[int, int]] = None
) -> PoseFrame:
    """
    Normalize pose coordinates relative to person bounding box, body center, and scale.
    Preserves original scene/pixel coordinates in `scene_keypoints_2d`.
    """
    if pose.is_normalized:
        return pose

    raw_scene_kps = pose.scene_keypoints_2d if pose.scene_keypoints_2d is not None else pose.keypoints_2d.copy()
    kps_2d = pose.keypoints_2d.copy()
    box = bbox or pose.bbox

    if box is not None:
        x1, y1, x2, y2 = box
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Center at torso / bbox center, scale by height
        norm_kps = np.zeros_like(kps_2d)
        norm_kps[:, 0] = (kps_2d[:, 0] - cx) / h
        norm_kps[:, 1] = (kps_2d[:, 1] - cy) / h
    elif image_shape is not None:
        img_h, img_w = image_shape[:2]
        norm_kps = np.zeros_like(kps_2d)
        norm_kps[:, 0] = kps_2d[:, 0] / max(1.0, img_w)
        norm_kps[:, 1] = kps_2d[:, 1] / max(1.0, img_h)
    else:
        norm_kps = kps_2d

    # Construct or normalize 3D coordinates (x, y, z)
    if pose.keypoints_3d is None:
        kps_3d = np.zeros((len(kps_2d), 3), dtype=np.float32)
        kps_3d[:, :2] = norm_kps[:, :2]
        kps_3d[:, 2] = 0.0
    else:
        kps_3d = pose.keypoints_3d.copy()

    return PoseFrame(
        track_id=pose.track_id,
        timestamp=pose.timestamp,
        keypoints_2d=norm_kps,
        keypoints_3d=kps_3d,
        visibility=pose.visibility,
        pose_confidence=pose.pose_confidence,
        frame_id=pose.frame_id,
        bbox=box,
        is_normalized=True,
        scene_keypoints_2d=raw_scene_kps,
        metadata=dict(pose.metadata)
    )
