"""
Guardian Matrix Adapter.
Integrates the upgraded research-grade pipeline seamlessly with existing web and video analysis scripts.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Any, Callable
import cv2

# Ensure src is importable
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.pipeline import GuardianMatrixPipeline, PipelineFrameResult
from src.policy.safety_policy import PolicyState


class GuardianMatrixVideoAnalyzer:
    """
    Adapter for analyzing video files with the Guardian Matrix multi-stage architecture.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.pipeline = GuardianMatrixPipeline()

    def analyze_video(
        self,
        video_path: str,
        output_annotated_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, float, Dict[str, Any]], None]] = None,
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Processes video file through Guardian Matrix and returns comprehensive research results.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

        writer = None
        if output_annotated_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_annotated_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_annotated_path, fourcc, fps, (width, height))

        frame_results: List[Dict[str, Any]] = []
        alerts: List[Dict[str, Any]] = []
        review_events: List[Dict[str, Any]] = []
        frame_idx = 0
        t_start = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames is not None and frame_idx >= max_frames):
                    break

                frame_idx += 1
                current_timestamp = frame_idx / fps

                # Run full Guardian Matrix pipeline
                result: PipelineFrameResult = self.pipeline.process_frame(
                    frame=frame,
                    timestamp=current_timestamp,
                    frame_id=frame_idx,
                    annotate=(writer is not None)
                )

                if writer is not None and result.annotated_frame is not None:
                    writer.write(result.annotated_frame)

                frame_summary = {
                    "frame_id": frame_idx,
                    "timestamp": round(current_timestamp, 3),
                    "people_count": len(result.tracked_people),
                    "pattern": result.classified_event.value,
                    "confidence": round(result.event_confidence, 3),
                    "policy_state": result.policy_action.state.value,
                    "processing_time_ms": round(result.processing_time_ms, 2)
                }
                frame_results.append(frame_summary)

                if result.policy_action.state == PolicyState.ALERT:
                    alerts.append({
                        "frame_id": frame_idx,
                        "timestamp": round(current_timestamp, 3),
                        "pattern": result.classified_event.value,
                        "confidence": round(result.system1_decision.calibrated_confidence, 3),
                        "evidence": result.evidence.to_dict()
                    })
                elif result.policy_action.state == PolicyState.REVIEW:
                    review_events.append({
                        "frame_id": frame_idx,
                        "timestamp": round(current_timestamp, 3),
                        "pattern": result.classified_event.value,
                        "confidence": round(result.system1_decision.calibrated_confidence, 3)
                    })

                if progress_callback:
                    progress_callback(frame_idx, total_frames, current_timestamp, frame_summary)

        finally:
            cap.release()
            if writer:
                writer.release()

        total_elapsed = time.time() - t_start

        return {
            "video_path": video_path,
            "total_frames": frame_idx,
            "duration_seconds": round(duration, 2),
            "processed_fps": round(frame_idx / max(1e-4, total_elapsed), 2),
            "alerts_count": len(alerts),
            "review_count": len(review_events),
            "alerts": alerts,
            "reviews": review_events,
            "annotated_video": output_annotated_path,
            "frame_results_sample": frame_results[::max(1, len(frame_results)//50)] if frame_results else []
        }


class GuardianMatrixLiveStreamManager:
    """
    Live stream manager bridging camera devices, network CCTV feeds (RTSP/HTTP),
    and Guardian Matrix real-time multi-stage inference.
    """

    def __init__(self):
        from src.streaming.live_stream import LiveStreamSource, LiveStreamProcessor, LiveStreamConfig
        self._LiveStreamSource = LiveStreamSource
        self._LiveStreamProcessor = LiveStreamProcessor
        self._LiveStreamConfig = LiveStreamConfig
        self.processor: Optional[Any] = None
        self.pipeline = GuardianMatrixPipeline()
        self.active_source: Optional[Any] = None
        self.alert_history: List[Dict[str, Any]] = []

    def start_stream(
        self,
        source: Any = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 15.0,
        on_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> bool:
        if self.processor and self.processor.is_processing:
            self.stop_stream()

        cfg = self._LiveStreamConfig(
            source=source,
            width=width,
            height=height,
            target_fps=fps,
            buffer_size=1
        )
        stream_src = self._LiveStreamSource(config=cfg)
        self.processor = self._LiveStreamProcessor(source=stream_src, pipeline=self.pipeline)

        def event_handler(res: PipelineFrameResult):
            if res.policy_action.state in (PolicyState.REVIEW, PolicyState.ALERT):
                alert_entry = {
                    "frame_id": res.frame_id,
                    "timestamp": round(res.timestamp, 3),
                    "pattern": res.classified_event.value,
                    "confidence": round(res.system1_decision.calibrated_confidence, 3),
                    "policy_state": res.policy_action.state.value,
                    "rationale": res.policy_action.rationale
                }
                self.alert_history.append(alert_entry)
                if len(self.alert_history) > 200:
                    self.alert_history.pop(0)
                if on_event_callback:
                    on_event_callback(alert_entry)

        self.processor.add_event_listener(event_handler)
        self.processor.start_processing()
        self.active_source = source
        return True

    def get_mjpeg_stream(self):
        if not self.processor:
            return iter([])
        return self.processor.generate_mjpeg_stream()

    def get_latest_status(self) -> Dict[str, Any]:
        if not self.processor:
            return {
                "active": False,
                "people_detected": 0,
                "pattern": "INACTIVE",
                "confidence": 0.0,
                "policy_state": "NORMAL",
                "alerts_count": len(self.alert_history)
            }
        if not self.processor.latest_result:
            return {
                "active": self.processor.is_processing,
                "people_detected": 0,
                "pattern": "INITIALIZING",
                "confidence": 0.0,
                "policy_state": "NORMAL",
                "alerts_count": len(self.alert_history),
                "processing_time_ms": 0.0
            }
        res: PipelineFrameResult = self.processor.latest_result
        return {
            "active": self.processor.is_processing,
            "frame_id": res.frame_id,
            "timestamp": round(res.timestamp, 2),
            "people_detected": len(res.tracked_people),
            "pattern": res.classified_event.value,
            "confidence": round(res.system1_decision.calibrated_confidence, 3),
            "policy_state": res.policy_action.state.value,
            "alerts_count": len(self.alert_history),
            "processing_time_ms": round(res.processing_time_ms, 1)
        }

    def stop_stream(self):
        if self.processor:
            self.processor.stop_processing()
            self.processor = None
        self.active_source = None
