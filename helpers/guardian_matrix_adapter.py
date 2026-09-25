"""
Guardian Matrix Adapter.
Integrates the upgraded research-grade pipeline seamlessly with video analysis scripts,
evaluations, and live RTSP/CCTV streaming interfaces.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Any, Callable
import cv2
import numpy as np

# Ensure root is importable
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.pipeline import GuardianMatrixPipeline, PipelineFrameResult
from src.policy.safety_policy import PolicyState
from src.taxonomy.event_taxonomy import (
    EventCategory,
    ActionType,
    OperationalUrgency,
    EvidenceState
)


class GuardianMatrixVideoAnalyzer:
    """
    Adapter for analyzing video files with the Guardian Matrix multi-stage architecture.
    Fully understands 5-state hysteresis, safety vs aggression separation, and forensic evidence packaging.
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

        # Reset pipeline states for clean video evaluation
        self.pipeline.reset()

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
        active_incident_events: List[Dict[str, Any]] = []
        safety_incidents: List[Dict[str, Any]] = []
        aggressive_incidents: List[Dict[str, Any]] = []
        saved_bundles: List[str] = []

        frame_idx = 0
        t_start = time.time()
        latencies: List[float] = []

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

                latencies.append(result.processing_time_ms)

                if writer is not None and result.annotated_frame is not None:
                    writer.write(result.annotated_frame)

                state_res = result.state_machine_result
                temp_out = result.temporal_output

                state_val = state_res.state.value if state_res else result.policy_action.state.value
                cat_val = temp_out.category.value if temp_out else "NORMAL_ACTIVITY"
                act_val = temp_out.action.value if temp_out else result.classified_event.value
                risk_score = state_res.decision_score if state_res else result.system1_decision.calibrated_confidence

                if result.saved_incident_bundle:
                    saved_bundles.append(result.saved_incident_bundle)

                frame_summary = {
                    "frame_id": frame_idx,
                    "timestamp": round(current_timestamp, 3),
                    "people_count": len(result.tracked_people),
                    "incident_state": state_val,
                    "category": cat_val,
                    "action": act_val,
                    "pattern": act_val,
                    "risk_score": round(float(risk_score), 4),
                    "confidence": round(float(risk_score), 4),
                    "urgency": state_res.urgency.value if state_res else "BENIGN",
                    "policy_state": result.policy_action.state.value,
                    "processing_time_ms": round(result.processing_time_ms, 2)
                }
                frame_results.append(frame_summary)

                # Track active incidents
                if state_res and state_res.is_incident_active:
                    incident_info = {
                        "frame_id": frame_idx,
                        "timestamp": round(current_timestamp, 3),
                        "state": state_val,
                        "category": cat_val,
                        "action": act_val,
                        "risk": round(float(risk_score), 4),
                        "incident_id": state_res.incident_id,
                        "track_ids": state_res.active_track_ids
                    }
                    active_incident_events.append(incident_info)
                    if temp_out and temp_out.category == EventCategory.SAFETY_INCIDENT:
                        safety_incidents.append(incident_info)
                    elif temp_out and temp_out.category == EventCategory.AGGRESSIVE_INTERACTION:
                        aggressive_incidents.append(incident_info)

                # Backward-compatible alert collections
                if result.policy_action.state == PolicyState.ALERT or (state_res and state_res.is_incident_active):
                    alerts.append({
                        "frame_id": frame_idx,
                        "timestamp": round(current_timestamp, 3),
                        "pattern": act_val,
                        "category": cat_val,
                        "confidence": round(float(risk_score), 3),
                        "evidence": result.evidence.to_dict()
                    })
                elif result.policy_action.state == PolicyState.REVIEW or (state_res and state_res.requires_review):
                    review_events.append({
                        "frame_id": frame_idx,
                        "timestamp": round(current_timestamp, 3),
                        "pattern": act_val,
                        "category": cat_val,
                        "confidence": round(float(risk_score), 3)
                    })

                if progress_callback:
                    progress_callback(frame_idx, total_frames, current_timestamp, frame_summary)

        finally:
            cap.release()
            if writer:
                writer.release()
            final_bundle = self.pipeline.finalize()
            if final_bundle and final_bundle not in saved_bundles:
                saved_bundles.append(final_bundle)

        total_elapsed = time.time() - t_start
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

        unique_incident_ids = {e["incident_id"] for e in active_incident_events if e.get("incident_id")}
        unique_safety_ids = {e["incident_id"] for e in safety_incidents if e.get("incident_id")}
        unique_agg_ids = {e["incident_id"] for e in aggressive_incidents if e.get("incident_id")}

        # Determine dominant predicted category for the video
        if aggressive_incidents:
            predicted_category = EventCategory.AGGRESSIVE_INTERACTION.value
        elif safety_incidents:
            predicted_category = EventCategory.SAFETY_INCIDENT.value
        elif review_events:
            predicted_category = "REVIEW_REQUIRED"
        else:
            predicted_category = EventCategory.NORMAL_ACTIVITY.value

        return {
            "video_path": video_path,
            "total_frames": frame_idx,
            "duration_seconds": round(duration, 2),
            "processed_fps": round(frame_idx / max(1e-4, total_elapsed), 2),
            "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "p95_latency_ms": round(p95_lat, 2),
            "alerts_count": len(alerts),
            "review_count": len(review_events),
            "active_incidents_count": len(active_incident_events),
            "incident_episodes_count": len(unique_incident_ids),
            "safety_incidents_count": len(safety_incidents),
            "safety_episodes_count": len(unique_safety_ids),
            "aggressive_incidents_count": len(aggressive_incidents),
            "aggressive_episodes_count": len(unique_agg_ids),
            "predicted_category": predicted_category,
            "saved_bundles": saved_bundles,
            "alerts": alerts,
            "reviews": review_events,
            "annotated_video": output_annotated_path,
            "frame_results_sample": frame_results[::max(1, len(frame_results)//50)] if frame_results else []
        }


class GuardianMatrixLiveStreamManager:
    """
    Live stream manager bridging camera devices, network CCTV feeds (RTSP/HTTP),
    and Guardian Matrix real-time multi-stage inference with telemetry dashboard.
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
            state_res = res.state_machine_result
            temp_out = res.temporal_output

            if (state_res and state_res.is_incident_active) or res.policy_action.state in (PolicyState.REVIEW, PolicyState.ALERT):
                alert_entry = {
                    "frame_id": res.frame_id,
                    "timestamp": round(res.timestamp, 3),
                    "state": state_res.state.value if state_res else res.policy_action.state.value,
                    "category": temp_out.category.value if temp_out else "NORMAL_ACTIVITY",
                    "action": temp_out.action.value if temp_out else res.classified_event.value,
                    "urgency": state_res.urgency.value if state_res else "BENIGN",
                    "confidence": round(float(state_res.decision_score if state_res else res.system1_decision.calibrated_confidence), 3),
                    "policy_state": res.policy_action.state.value,
                    "rationale": state_res.rationale if state_res else res.policy_action.rationale,
                    "saved_bundle": res.saved_incident_bundle
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
                "state": "OFFLINE",
                "category": "NORMAL_ACTIVITY",
                "action": "STANDBY",
                "confidence": 0.0,
                "policy_state": "NORMAL",
                "alerts_count": len(self.alert_history),
                "fps": 0.0,
                "latency_ms": 0.0,
                "processing_time_ms": 0.0
            }
        proc_fps = getattr(self.processor, 'fps', getattr(self.processor, 'target_fps', 15.0))
        if not self.processor.latest_result:
            return {
                "active": self.processor.is_processing,
                "people_detected": 0,
                "state": "INITIALIZING",
                "category": "NORMAL_ACTIVITY",
                "action": "STANDBY",
                "confidence": 0.0,
                "policy_state": "NORMAL",
                "alerts_count": len(self.alert_history),
                "fps": round(float(proc_fps), 1),
                "latency_ms": 0.0,
                "processing_time_ms": 0.0
            }
        res: PipelineFrameResult = self.processor.latest_result
        state_res = res.state_machine_result
        temp_out = res.temporal_output

        return {
            "active": self.processor.is_processing,
            "frame_id": res.frame_id,
            "timestamp": round(res.timestamp, 2),
            "people_detected": len(res.tracked_people),
            "state": state_res.state.value if state_res else res.policy_action.state.value,
            "pattern": temp_out.action.value if temp_out else res.classified_event.value,
            "category": temp_out.category.value if temp_out else "NORMAL_ACTIVITY",
            "action": temp_out.action.value if temp_out else res.classified_event.value,
            "urgency": state_res.urgency.value if state_res else "BENIGN",
            "confidence": round(float(state_res.decision_score if state_res else res.system1_decision.calibrated_confidence), 3),
            "policy_state": res.policy_action.state.value,
            "alerts_count": len(self.alert_history),
            "fps": round(float(proc_fps), 1),
            "latency_ms": round(res.processing_time_ms, 1),
            "processing_time_ms": round(res.processing_time_ms, 1),
            "pose_quality": round(float(temp_out.quality_metrics.get("pose_quality", 0.0)), 2) if temp_out else 0.0,
            "tracking_quality": round(float(temp_out.quality_metrics.get("tracking_quality", 0.0)), 2) if temp_out else 0.0
        }

    def stop_stream(self):
        if self.processor:
            self.processor.stop_processing()
            self.processor = None
        self.active_source = None
