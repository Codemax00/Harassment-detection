"""
Person Detection Layer for Guardian Matrix.
Evaluates modern detectors (YOLO, RT-DETR) providing clean PersonDetection objects.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import os

@dataclass
class PersonDetection:
    """
    Standardized Person Detection output as defined in Upgrade_Plan.md Section 5.1.
    """
    bounding_box: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    confidence: float
    frame_id: int
    track_candidate: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def x1(self) -> float:
        return self.bounding_box[0]

    @property
    def y1(self) -> float:
        return self.bounding_box[1]

    @property
    def x2(self) -> float:
        return self.bounding_box[2]

    @property
    def y2(self) -> float:
        return self.bounding_box[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height


class BasePersonDetector(ABC):
    """Abstract Base Class for all person detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int = 0) -> List[PersonDetection]:
        """Detect people in an input BGR or RGB image/frame."""
        pass


class YOLOPersonDetector(BasePersonDetector):
    """Ultralytics YOLO person detector implementation."""

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5, device: str = "cpu"):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            from ultralytics import YOLO

            # Handle torch weights_only compatibility safely if needed
            original_load = torch.load
            try:
                torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
                # Check potential locations for yolov8n.pt
                candidates = [
                    self.model_path,
                    os.path.join(os.getcwd(), self.model_path),
                    os.path.join(os.path.dirname(__file__), "..", "..", self.model_path),
                    os.path.join(os.path.dirname(__file__), "..", "..", "yolov8n.pt"),
                ]
                found_path = next((p for p in candidates if os.path.exists(p)), self.model_path)
                self.model = YOLO(found_path)
            finally:
                torch.load = original_load
        except Exception as e:
            print(f"⚠️ YOLOPersonDetector initialization warning: {e}. Running in simulation/fallback mode.")
            self.model = None

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> List[PersonDetection]:
        if frame is None or frame.size == 0:
            return []

        if self.model is None:
            # Fallback heuristic if model cannot be loaded in test/mock environment
            return []

        try:
            results = self.model(frame, verbose=False, device=self.device, iou=0.45)
            detections: List[PersonDetection] = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    # Class 0 is 'person' in COCO
                    if cls_id == 0 and conf >= self.conf_threshold:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                        detections.append(
                            PersonDetection(
                                bounding_box=(x1, y1, x2, y2),
                                confidence=conf,
                                frame_id=frame_id,
                                track_candidate=True
                            )
                        )
            return detections
        except Exception as e:
            print(f"Error during YOLO person detection: {e}")
            return []


class RTDETRPersonDetector(BasePersonDetector):
    """RT-DETR detector adapter interface for evaluation and benchmarking."""

    def __init__(self, model_name: str = "rtdetr-l.pt", conf_threshold: float = 0.5):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.model = None

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> List[PersonDetection]:
        # Interface adapter for RT-DETR evaluation
        return []


def load_detector(detector_type: str = "yolo", config: Optional[Dict[str, Any]] = None) -> BasePersonDetector:
    """Factory function to load configured person detector."""
    cfg = config or {}
    conf = cfg.get("conf", 0.5)
    model_name = cfg.get("model_name", "yolov8n.pt")
    device = cfg.get("device", "cpu")

    if detector_type.lower() in ("yolo", "ultralytics"):
        return YOLOPersonDetector(model_path=model_name, conf_threshold=conf, device=device)
    elif detector_type.lower() in ("rtdetr", "rt-detr"):
        return RTDETRPersonDetector(model_name=model_name, conf_threshold=conf)
    else:
        return YOLOPersonDetector(model_path=model_name, conf_threshold=conf, device=device)
