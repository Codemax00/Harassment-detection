"""
Guardian Matrix - Modular Pose Estimator
Extracts 17 COCO body keypoints (nose, shoulders, elbows, wrists, hips, knees, ankles)
providing both normalized coordinates and pixel coordinates for downstream temporal/interaction reasoning.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from src.pose.pose_estimator import YOLOPoseEstimator, BasePoseEstimator
from src.pose.pose_representation import KeypointName, PoseFrame
from src.tracking.tracker import TrackedPerson
from config.settings import settings


class PoseEstimatorInterface(ABC):
    """Abstract interface for pose estimation models."""

    @abstractmethod
    def estimate(self, frame: np.ndarray, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Estimate pose for tracked persons.
        Returns: [ { "track_id": 1, "keypoints": [...], "confidence": 0.87 } ]
        """
        pass


class PoseEstimator(PoseEstimatorInterface):
    """
    Modular Pose Estimator wrapping YOLO-Pose.
    """

    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    # Standard COCO skeleton connections
    SKELETON_CONNECTIONS = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: Optional[float] = None,
        device: Optional[str] = None,
    ):
        m_path = model_path or settings.pose_model_path
        conf = confidence if confidence is not None else settings.pose_confidence
        dev = device or settings.device

        self.underlying_estimator = YOLOPoseEstimator(
            model_name=m_path,
            conf_threshold=conf,
            device=dev,
        )

    def estimate(self, frame: np.ndarray, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Estimates pose for each tracked person in the frame.
        Outputs standardized dicts matching Section 9:
        {
            "track_id": int,
            "confidence": float,
            "keypoints": [ {"name": "...", "x": float, "y": float, "norm_x": float, "norm_y": float, "confidence": float} ],
            "keypoints_raw": np.ndarray,
            "_pose_frame": PoseFrame
        }
        """
        if frame is None or frame.size == 0 or not tracks:
            return []

        h, w = frame.shape[:2]

        # Convert tracks to TrackedPerson list for underlying estimator
        tracked_people: List[TrackedPerson] = []
        for t in tracks:
            if "_tracked_person" in t and isinstance(t["_tracked_person"], TrackedPerson):
                tracked_people.append(t["_tracked_person"])
            else:
                bbox = t.get("bbox", [0, 0, 0, 0])
                tp = TrackedPerson(
                    track_id=t.get("track_id", 0),
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    detection_confidence=t.get("confidence", 0.5),
                    timestamp=0.0,
                )
                tracked_people.append(tp)

        pose_frames: List[PoseFrame] = self.underlying_estimator.estimate(frame, tracked_people)

        results: List[Dict[str, Any]] = []
        for pf in pose_frames:
            kps_list = []
            if pf.keypoints_2d is not None:
                for idx, (px, py) in enumerate(pf.keypoints_2d):
                    name = self.KEYPOINT_NAMES[idx] if idx < len(self.KEYPOINT_NAMES) else f"kp_{idx}"
                    vis = float(pf.visibility[idx]) if pf.visibility is not None and idx < len(pf.visibility) else 0.5
                    norm_x = px / float(w) if w > 0 else 0.0
                    norm_y = py / float(h) if h > 0 else 0.0

                    kps_list.append({
                        "name": name,
                        "x": round(float(px), 1),
                        "y": round(float(py), 1),
                        "norm_x": round(norm_x, 4),
                        "norm_y": round(norm_y, 4),
                        "confidence": round(vis, 3),
                    })

            results.append({
                "track_id": pf.track_id,
                "confidence": round(pf.pose_confidence, 3),
                "keypoints": kps_list,
                "keypoints_raw": pf.keypoints_2d,
                "_pose_frame": pf,
            })

        return results
