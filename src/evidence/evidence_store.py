"""
Auditable Evidence Store & Rolling Video Buffer for Guardian Matrix.
Packages every confirmed incident into:
incidents/
    YYYY-MM-DD_HH-MM-SS_<action>/
        metadata.json
        timeline.json
        keyframe.jpg
        clip.mp4

Maintains a bounded pre/post circular frame buffer to prevent memory leakage.
"""

from collections import deque
from dataclasses import dataclass, field
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from ..taxonomy.event_taxonomy import EventCategory, ActionType, OperationalUrgency, EvidenceState
from ..temporal.temporal_decision_engine import TemporalDecisionOutput
from ..policy.incident_state_machine import StateMachineResult, IncidentStateTransition


@dataclass
class EvidenceStoreConfig:
    """Configurable evidence storage settings matching Section 15 & 16."""
    base_incidents_dir: str = "incidents"
    pre_event_seconds: float = 3.0
    post_event_seconds: float = 3.0
    target_fps: float = 15.0
    save_video_clips: bool = True
    camera_id: str = "CAM_SURVEILLANCE_01"


class RollingVideoBuffer:
    """
    Circular frame buffer dynamically sized to stream FPS.
    Prevents memory exhaustion on long-running RTSP/HLS streams.
    """

    def __init__(self, buffer_seconds: float = 3.0, estimated_fps: float = 15.0):
        self.buffer_seconds = buffer_seconds
        self.fps = max(1.0, estimated_fps)
        self.max_frames = int(self.buffer_seconds * self.fps) + 10
        self.frames: deque[Tuple[float, int, np.ndarray]] = deque(maxlen=self.max_frames)

    def update_fps(self, measured_fps: float):
        if measured_fps > 1.0:
            self.fps = measured_fps
            new_max = int(self.buffer_seconds * self.fps) + 10
            if new_max != self.max_frames:
                self.max_frames = new_max
                current = list(self.frames)
                self.frames = deque(current[-new_max:], maxlen=new_max)

    def push_frame(self, frame: np.ndarray, timestamp: float, frame_id: int):
        # Store copy of frame or reference
        self.frames.append((timestamp, frame_id, frame.copy()))

    def get_pre_event_frames(self, start_timestamp: float) -> List[Tuple[float, int, np.ndarray]]:
        """Returns frames captured prior to the incident start."""
        cutoff = start_timestamp - self.buffer_seconds
        return [f for f in self.frames if f[0] >= cutoff and f[0] <= start_timestamp]

    def clear(self):
        self.frames.clear()


class EvidenceStore:
    """
    Forensic Incident Evidence Packaging Engine.
    """

    def __init__(self, config: Optional[EvidenceStoreConfig] = None):
        self.config = config or EvidenceStoreConfig()
        os.makedirs(self.config.base_incidents_dir, exist_ok=True)
        
        self.video_buffer = RollingVideoBuffer(
            buffer_seconds=self.config.pre_event_seconds,
            estimated_fps=self.config.target_fps
        )
        
        # Incident in progress tracking
        self.active_incident_id: Optional[str] = None
        self.current_incident_dir: Optional[str] = None
        self.incident_timeline: List[Dict[str, Any]] = []
        self.clip_writer: Optional[cv2.VideoWriter] = None
        self.peak_frame: Optional[np.ndarray] = None
        self.peak_risk: float = 0.0
        self.peak_timestamp: float = 0.0
        self.peak_frame_id: int = 0
        self.peak_decision: Optional[TemporalDecisionOutput] = None
        self.peak_state_result: Optional[StateMachineResult] = None
        self.incident_start_time: float = 0.0
        self.incident_category: EventCategory = EventCategory.NORMAL_ACTIVITY
        self.incident_action: ActionType = ActionType.WALKING
        self.incident_urgency: OperationalUrgency = OperationalUrgency.BENIGN
        self.involved_tracks: set[int] = set()
        self.post_event_deadline: Optional[float] = None
        self.pending_save: bool = False

    def on_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_id: int,
        decision: TemporalDecisionOutput,
        state_result: StateMachineResult,
        measured_fps: Optional[float] = None
    ) -> Optional[str]:
        """
        Processes frame into rolling buffer and captures active incident forensic data.
        Returns path to saved incident bundle when finalized, else None.
        """
        if measured_fps:
            self.video_buffer.update_fps(measured_fps)

        self.video_buffer.push_frame(frame, timestamp, frame_id)

        saved_path: Optional[str] = None

        # Case 1: Active incident is running
        if state_result.is_incident_active:
            if not self.active_incident_id:
                # Start recording new incident
                self._start_new_incident(state_result, timestamp, frame_id, frame)
            elif self.clip_writer is not None:
                self.clip_writer.write(frame)

            # Record frame into incident timeline
            self.incident_timeline.append({
                "timestamp": round(timestamp, 3),
                "frame_id": frame_id,
                "state": state_result.state.value,
                "risk": round(float(state_result.decision_score), 4),
                "action": state_result.action.value,
                "category": state_result.category.value
            })
            self.involved_tracks.update(state_result.active_track_ids)

            # Check for peak frame
            if self.peak_frame is None or state_result.decision_score >= self.peak_risk:
                self.peak_risk = float(state_result.decision_score)
                self.peak_frame = frame.copy()
                self.peak_timestamp = timestamp
                self.peak_frame_id = frame_id
                self.peak_decision = decision
                self.peak_state_result = state_result
                self.incident_action = state_result.action
                self.incident_category = state_result.category
                self.incident_urgency = state_result.urgency

        # Case 2: Incident just transitioned out of ACTIVE to COOLDOWN/NORMAL
        elif self.active_incident_id and not self.pending_save:
            # Set post-event deadline (e.g. 3.0s after incident ended)
            self.pending_save = True
            self.post_event_deadline = timestamp + self.config.post_event_seconds

        # Case 3: Capturing post-event context frames
        if self.pending_save:
            if self.clip_writer is not None:
                self.clip_writer.write(frame)
            self.incident_timeline.append({
                "timestamp": round(timestamp, 3),
                "frame_id": frame_id,
                "state": state_result.state.value,
                "risk": round(float(state_result.decision_score), 4),
                "action": state_result.action.value,
                "category": state_result.category.value
            })

            if self.post_event_deadline and timestamp >= self.post_event_deadline:
                # Finalize and export the incident bundle
                saved_path = self._finalize_incident_bundle(timestamp, decision, state_result)
                self._reset_incident_state()

        return saved_path

    def finalize_pending_incident(self) -> Optional[str]:
        """Finalizes any active or pending incident bundle immediately (e.g. on stream stop or EOF)."""
        if self.active_incident_id and self.peak_frame is not None:
            end_t = (self.peak_timestamp + 1.0) if self.peak_timestamp is not None else time.time()
            saved_path = self._finalize_incident_bundle(end_t, None, None)
            self._reset_incident_state()
            return saved_path
        return None

    def _start_new_incident(self, state_result: StateMachineResult, timestamp: float, frame_id: int, frame: np.ndarray):
        """Initializes a new incident directory and buffers pre-event context."""
        t_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(timestamp if timestamp > 100000 else time.time()))
        action_name = state_result.action.value
        safe_action = "".join(c for c in action_name if c.isalnum() or c in ("_", "-"))
        inc_dir_name = f"{t_str}_{safe_action}"
        self.current_incident_dir = os.path.join(self.config.base_incidents_dir, inc_dir_name)
        os.makedirs(self.current_incident_dir, exist_ok=True)

        self.active_incident_id = state_result.incident_id or f"INC-{t_str}_{safe_action}"
        self.incident_start_time = timestamp
        self.incident_category = state_result.category
        self.incident_action = state_result.action
        self.incident_urgency = state_result.urgency
        self.peak_risk = float(state_result.decision_score)
        self.peak_frame = frame.copy()
        self.peak_timestamp = timestamp
        self.peak_frame_id = frame_id
        self.involved_tracks = set(state_result.active_track_ids)
        self.pending_save = False

        # Prepend pre-event buffer frames and open streaming video writer
        pre_frames = self.video_buffer.get_pre_event_frames(timestamp)
        self.incident_timeline = [
            {
                "timestamp": round(f[0], 3),
                "frame_id": f[1],
                "state": EvidenceState.OBSERVING.value,
                "risk": 0.0,
                "action": ActionType.WALKING.value,
                "category": EventCategory.NORMAL_ACTIVITY.value
            }
            for f in pre_frames
        ]

        if self.config.save_video_clips and frame is not None:
            clip_path = os.path.join(self.current_incident_dir, "clip.mp4")
            h, w = frame.shape[:2]
            fps = max(5.0, self.video_buffer.fps)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.clip_writer = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))
            for _, _, pf_img in pre_frames:
                self.clip_writer.write(pf_img)
            self.clip_writer.write(frame)

    def _finalize_incident_bundle(
        self,
        end_timestamp: float,
        latest_decision: TemporalDecisionOutput,
        latest_state: StateMachineResult
    ) -> str:
        """Writes metadata.json, timeline.json, keyframe.jpg, and clip.mp4."""
        out_dir = self.current_incident_dir or self.config.base_incidents_dir
        os.makedirs(out_dir, exist_ok=True)

        # 1. Write metadata.json
        duration_s = max(0.0, end_timestamp - self.incident_start_time)
        dec = latest_decision if latest_decision is not None else self.peak_decision
        evidence_dict = {}
        quality_dict = {}
        if dec is not None:
            evidence_dict = {
                "peak_directional_impact": dec.temporal_features.get("peak_directional_impact", 0.0),
                "peak_fall_score": dec.temporal_features.get("peak_fall_score", 0.0),
                "persistence_score": dec.persistence_score,
                "temporal_features": dec.temporal_features
            }
            quality_dict = dec.quality_metrics

        metadata = {
            "incident_id": self.active_incident_id,
            "timestamp": round(self.incident_start_time, 3),
            "category": self.incident_category.value,
            "action": self.incident_action.value,
            "urgency": self.incident_urgency.value,
            "start_time": round(self.incident_start_time, 3),
            "end_time": round(end_timestamp, 3),
            "duration_seconds": round(duration_s, 2),
            "peak_risk": round(self.peak_risk, 4),
            "track_ids": sorted(list(self.involved_tracks)),
            "camera_id": self.config.camera_id,
            "evidence": evidence_dict,
            "quality": quality_dict,
            "state_transitions": [
                {
                    "state": entry.get("state"),
                    "timestamp": entry.get("timestamp"),
                    "risk": entry.get("risk")
                }
                for entry in self.incident_timeline[::max(1, len(self.incident_timeline) // 10)]
            ]
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 2. Write timeline.json
        with open(os.path.join(out_dir, "timeline.json"), "w", encoding="utf-8") as f:
            json.dump(self.incident_timeline, f, indent=2)

        # 3. Save keyframe.jpg (standard + raw forensic + annotated)
        if self.peak_frame is not None:
            # Pristine untouched keyframe for forensic provenance
            cv2.imwrite(os.path.join(out_dir, "keyframe_raw.jpg"), self.peak_frame)

            annotated_keyframe = self.peak_frame.copy()
            # Overlay audit stamp
            h, w = annotated_keyframe.shape[:2]
            cv2.rectangle(annotated_keyframe, (10, 10), (min(w - 10, 480), 85), (0, 0, 0), -1)
            cv2.putText(
                annotated_keyframe,
                f"INCIDENT: {self.incident_action.value} ({self.incident_category.value})",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255) if self.incident_category == EventCategory.AGGRESSIVE_INTERACTION else (0, 165, 255),
                2
            )
            cv2.putText(
                annotated_keyframe,
                f"Peak Risk: {self.peak_risk:.2f} | Time: {self.peak_timestamp:.2f}s | IDs: {list(self.involved_tracks)}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )
            cv2.imwrite(os.path.join(out_dir, "keyframe_annotated.jpg"), annotated_keyframe)
            cv2.imwrite(os.path.join(out_dir, "keyframe.jpg"), annotated_keyframe)

        # 4. Finalize clip.mp4
        if self.clip_writer is not None:
            self.clip_writer.release()
            self.clip_writer = None

        return out_dir

    def _reset_incident_state(self):
        """Clears active incident tracking after bundle write."""
        if self.clip_writer is not None:
            self.clip_writer.release()
            self.clip_writer = None
        self.active_incident_id = None
        self.current_incident_dir = None
        self.incident_timeline.clear()
        self.peak_frame = None
        self.peak_risk = 0.0
        self.involved_tracks.clear()
        self.post_event_deadline = None
        self.pending_save = False
