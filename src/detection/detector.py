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
    """Ultralytics YOLO person detector implementation with Motion-Gated ROI (M-ROI) & small-object CCTV tiling."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.20,
        device: Optional[str] = None,
        enable_tiling: bool = False,
        enable_mroi: bool = True,
        tile_size: int = 240,
        tile_overlap: float = 0.30
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        # Auto-detect CUDA if device not explicitly specified
        if device is None:
            try:
                import torch
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device

        self.enable_tiling = enable_tiling
        self.enable_mroi = enable_mroi
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.model = None
        self._bg_subtractor = None
        if self.enable_mroi:
            try:
                import cv2
                self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=120, varThreshold=20, detectShadows=False
                )
            except Exception:
                self._bg_subtractor = None

        self._load_model()

    def _load_model(self):
        try:
            os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
            from ultralytics import YOLO

            candidates = [
                self.model_path,
                os.path.join(os.getcwd(), self.model_path),
                os.path.join(os.path.dirname(__file__), "..", "..", self.model_path),
                os.path.join(os.path.dirname(__file__), "..", "..", "yolov8n.pt"),
            ]
            found_path = next((p for p in candidates if os.path.exists(p)), None)
            if found_path is None:
                raise FileNotFoundError(
                    f"YOLO detector model weights '{self.model_path}' not found. "
                    f"Searched candidates: {candidates}"
                )
            self.model = YOLO(found_path)
            # If CUDA is active, warm up on GPU
            if "cuda" in str(self.device):
                try:
                    self.model.to(self.device)
                except Exception:
                    pass
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to load YOLO person detector '{self.model_path}': {e}")
            raise RuntimeError(f"Failed to load YOLO person detector '{self.model_path}': {e}") from e

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> List[PersonDetection]:
        if frame is None or frame.size == 0:
            return []

        if self.model is None:
            return []

        try:
            detections: List[PersonDetection] = []
            H, W = frame.shape[:2]

            # 1. Full-frame standard inference (catches close and medium subjects)
            results = self.model(frame, verbose=False, device=self.device, conf=self.conf_threshold, iou=0.45)
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

            # 2. Motion-Gated Region of Interest (M-ROI) for distant moving pedestrians
            if self.enable_mroi and self._bg_subtractor is not None and H >= 300 and W >= 300:
                import cv2
                fg_mask = self._bg_subtractor.apply(frame)
                # Filter noise
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                fg_cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(fg_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    cx, cy, cw, ch = cv2.boundingRect(cnt)
                    # Look for small distant pedestrian motion (height between 12px and 120px)
                    if 12 <= ch <= 120 and 6 <= cw <= 100:
                        # Extract centered crop around the moving entity with padding
                        pad = 32
                        cx1 = max(0, cx - pad)
                        cy1 = max(0, cy - pad)
                        cx2 = min(W, cx + cw + pad)
                        cy2 = min(H, cy + ch + pad)
                        if (cx2 - cx1) >= 48 and (cy2 - cy1) >= 48:
                            m_crop = frame[cy1:cy2, cx1:cx2]
                            c_res = self.model(m_crop, verbose=False, device=self.device, conf=self.conf_threshold, iou=0.45)[0]
                            if c_res.boxes is not None:
                                for b in c_res.boxes:
                                    if int(b.cls[0].item()) == 0:
                                        conf = float(b.conf[0].item())
                                        if conf >= self.conf_threshold:
                                            bxy = b.xyxy[0].cpu().numpy()
                                            detections.append(
                                                PersonDetection(
                                                    bounding_box=(float(bxy[0] + cx1), float(bxy[1] + cy1), float(bxy[2] + cx1), float(bxy[3] + cy1)),
                                                    confidence=conf,
                                                    frame_id=frame_id,
                                                    track_candidate=True
                                                )
                                            )

            # 3. Dense Tiled inference for long-range / distant CCTV pedestrians (if explicitly enabled)
            if self.enable_tiling and H >= 300 and W >= 300:
                ts = self.tile_size
                step = max(64, int(ts * (1.0 - self.tile_overlap)))
                y_starts = list(range(0, max(1, H - ts + 1), step))
                if y_starts and y_starts[-1] + ts < H:
                    y_starts.append(H - ts)
                x_starts = list(range(0, max(1, W - ts + 1), step))
                if x_starts and x_starts[-1] + ts < W:
                    x_starts.append(W - ts)

                for y0 in y_starts:
                    for x0 in x_starts:
                        tile = frame[y0:y0+ts, x0:x0+ts]
                        t_res = self.model(tile, verbose=False, device=self.device, conf=self.conf_threshold, iou=0.45)[0]
                        if t_res.boxes is not None:
                            for b in t_res.boxes:
                                if int(b.cls[0].item()) == 0:
                                    conf = float(b.conf[0].item())
                                    if conf >= self.conf_threshold:
                                        t_xy = b.xyxy[0].cpu().numpy()
                                        gx1 = float(t_xy[0] + x0)
                                        gy1 = float(t_xy[1] + y0)
                                        gx2 = float(t_xy[2] + x0)
                                        gy2 = float(t_xy[3] + y0)
                                        detections.append(
                                            PersonDetection(
                                                bounding_box=(gx1, gy1, gx2, gy2),
                                                confidence=conf,
                                                frame_id=frame_id,
                                                track_candidate=True
                                            )
                                        )

            return self._suppress_duplicate_detections(detections)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error during YOLO person detection: {e}")
            return []

    @staticmethod
    def _suppress_duplicate_detections(detections: List[PersonDetection], iou_thresh: float = 0.45) -> List[PersonDetection]:
        if len(detections) <= 1:
            return detections
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[PersonDetection] = []
        for d in sorted_dets:
            is_dup = False
            for k in kept:
                area_d = max(1.0, (d.x2 - d.x1) * (d.y2 - d.y1))
                area_k = max(1.0, (k.x2 - k.x1) * (k.y2 - k.y1))
                inter_w = max(0.0, min(d.x2, k.x2) - max(d.x1, k.x1))
                inter_h = max(0.0, min(d.y2, k.y2) - max(d.y1, k.y1))
                inter_area = inter_w * inter_h
                union_area = area_d + area_k - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0
                if iou > iou_thresh or (inter_area / area_d > 0.70) or (inter_area / area_k > 0.70):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(d)
        return kept


class RTDETRPersonDetector(BasePersonDetector):
    """RT-DETR detector adapter interface for evaluation and benchmarking."""

    def __init__(self, model_name: str = "rtdetr-l.pt", conf_threshold: float = 0.5):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.model = None

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> List[PersonDetection]:
        raise NotImplementedError(
            f"RT-DETR person detector ({self.model_name}) is not yet integrated. "
            "Please use YOLOPersonDetector."
        )


def load_detector(detector_type: str = "yolo", config: Optional[Dict[str, Any]] = None) -> BasePersonDetector:
    """Factory function to load configured person detector."""
    cfg = config or {}
    conf = cfg.get("conf_threshold", cfg.get("conf", 0.20))
    model_name = cfg.get("model_path", cfg.get("model_name", "yolov8n.pt"))
    device = cfg.get("device", None)
    enable_tiling = cfg.get("enable_tiling", False)
    enable_mroi = cfg.get("enable_mroi", True)
    tile_size = cfg.get("tile_size", 240)
    tile_overlap = cfg.get("tile_overlap", 0.30)

    if detector_type.lower() in ("yolo", "ultralytics"):
        return YOLOPersonDetector(
            model_path=model_name,
            conf_threshold=conf,
            device=device,
            enable_tiling=enable_tiling,
            enable_mroi=enable_mroi,
            tile_size=tile_size,
            tile_overlap=tile_overlap
        )
    elif detector_type.lower() in ("rtdetr", "rt-detr"):
        return RTDETRPersonDetector(model_name=model_name, conf_threshold=conf)
    else:
        return YOLOPersonDetector(
            model_path=model_name,
            conf_threshold=conf,
            device=device,
            enable_tiling=enable_tiling,
            enable_mroi=enable_mroi,
            tile_size=tile_size,
            tile_overlap=tile_overlap
        )

