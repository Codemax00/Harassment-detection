"""
Guardian Matrix - Modular Person Detector
Wraps YOLO / RT-DETR models to provide a standardized detection interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from src.detection.detector import YOLOPersonDetector, PersonDetection
from config.settings import settings


class Detector(ABC):
    """Abstract interface for all person detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect people in frame.
        Returns: [ { "bbox": [x1, y1, x2, y2], "confidence": 0.91, "class": "person" } ]
        """
        pass


class PersonDetector(Detector):
    """
    Standard Person Detector implementation wrapping YOLO.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: Optional[float] = None,
        device: Optional[str] = None,
    ):
        model_p = model_path or settings.model_path
        conf = confidence if confidence is not None else settings.detection_confidence
        dev = device or settings.device

        self.underlying_detector = YOLOPersonDetector(
            model_path=model_p,
            conf_threshold=conf,
            device=dev,
        )

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect persons in the frame.
        Returns list of dicts matching Section 7 interface:
        [ { "bbox": [x1, y1, x2, y2], "confidence": 0.91, "class": "person" } ]
        """
        if frame is None or frame.size == 0:
            return []

        raw_detections = self.underlying_detector.detect(frame)
        if not raw_detections:
            return []

        # Sort detections by confidence descending
        sorted_dets = sorted(raw_detections, key=lambda d: d.confidence, reverse=True)
        keep_dets: List[PersonDetection] = []

        # Strict Non-Maximum Suppression (NMS) to ensure 1 person = 1 box
        for det in sorted_dets:
            x1_a, y1_a, x2_a, y2_a = det.bounding_box
            area_a = max(1.0, (x2_a - x1_a) * (y2_a - y1_a))
            suppressed = False

            for kept in keep_dets:
                x1_b, y1_b, x2_b, y2_b = kept.bounding_box
                area_b = max(1.0, (x2_b - x1_b) * (y2_b - y1_b))

                # Intersection
                xx1 = max(x1_a, x1_b)
                yy1 = max(y1_a, y1_b)
                xx2 = min(x2_a, x2_b)
                yy2 = min(y2_a, y2_b)

                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h

                if inter > 0:
                    iou = inter / (area_a + area_b - inter)
                    containment = inter / min(area_a, area_b)
                    if iou > 0.40 or containment > 0.65:
                        suppressed = True
                        break

            if not suppressed:
                keep_dets.append(det)

        results = []
        for d in keep_dets:
            results.append({
                "bbox": [round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)],
                "confidence": round(d.confidence, 3),
                "class": "person",
                "_raw": d,
            })
        return results
