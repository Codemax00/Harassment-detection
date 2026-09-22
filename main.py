"""
Guardian Matrix - Live CCTV / RTSP AI Detection Pipeline
Main entry point for continuous live streaming, detection, tracking, pose analysis,
temporal interaction modeling, risk evaluation, HUD rendering, and incident recording.
"""

import sys
import os
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import psutil

# Ensure local packages are on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import settings
from stream.rtsp_stream import StreamProvider, RTSPStream
from detection.person_detector import PersonDetector
from tracking.tracker import MultiPersonTracker
from pose.pose_estimator import PoseEstimator
from analysis.interaction_analyzer import InteractionAnalyzer
from analysis.risk_analyzer import RiskAnalyzer
from alerts.alert_manager import AlertManager
from recording.incident_recorder import IncidentRecorder


class GuardianMatrixPipeline:
    """
    Unified end-to-end execution pipeline for live CCTV and multi-source video feeds.
    """

    def __init__(
        self,
        source_type: str = "rtsp",
        source_path: Optional[str] = None,
        headless: bool = False,
        device: Optional[str] = None,
    ):
        self.source_type = source_type
        self.source_path = source_path
        self.headless = headless
        self.device = device or settings.device

        # Initialize modular pipeline components
        print("[SETUP] Initializing Guardian Matrix Pipeline components...")
        self.stream: RTSPStream = StreamProvider.create(
            source_type=self.source_type,
            source_path=self.source_path,
        )
        self.detector = PersonDetector(device=self.device)
        self.tracker = MultiPersonTracker()
        self.pose_estimator = PoseEstimator(device=self.device)
        self.interaction_analyzer = InteractionAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.alert_manager = AlertManager()
        self.incident_recorder = IncidentRecorder()

        # Wire up alert listener to automatically record incidents
        self.alert_manager.add_listener(self.incident_recorder.trigger_recording)

        # Performance & Telemetry Tracking
        self.frame_idx = 0
        self.ai_fps = 0.0
        self.last_log_time = time.time()
        self.last_stream_status = ""
        self.last_risk_level = ""
        self.process = psutil.Process(os.getpid())

    def log(self, message: str):
        """Structured system logging."""
        now_str = datetime.now().strftime("%H:%M:%S")
        print(f"[{now_str}] {message}")

    def run(self, max_frames: Optional[int] = None):
        """Main continuous execution loop."""
        self.log(f"Starting pipeline from source: {self.source_type.upper()} ({self.stream.url})")
        self.stream.connect()

        window_name = "Guardian Matrix - Live CCTV AI Monitoring"
        if not self.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)

        prev_loop_time = time.time()

        try:
            while True:
                if max_frames and self.frame_idx >= max_frames:
                    self.log(f"Reached specified max_frames limit ({max_frames}). Gracefully stopping.")
                    break

                t_start = time.time()

                # 1. Fetch latest frame from stream receiver
                success, frame, meta = self.stream.read(timeout=0.2)

                # Check connection status transitions for logging
                current_status = meta.get("status", self.stream.status)
                if current_status != self.last_stream_status:
                    self.log(f"STREAM STATUS: {current_status.upper()}")
                    self.last_stream_status = current_status

                if not success or frame is None:
                    # Render offline / reconnecting HUD if in GUI mode
                    if not self.headless:
                        offline_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                        self._render_offline_hud(offline_frame, current_status)
                        cv2.imshow(window_name, offline_frame)
                        if cv2.waitKey(50) & 0xFF == ord("q"):
                            break
                    time.sleep(0.05)
                    continue

                self.frame_idx += 1

                # Feed raw frame to incident rolling buffer
                self.incident_recorder.add_frame(frame)

                # 2. Person Detection
                t_det = time.time()
                detections = self.detector.detect(frame)
                det_time_ms = (time.time() - t_det) * 1000.0

                # 3. Multi-Person Tracking
                t_track = time.time()
                tracks = self.tracker.update(detections, timestamp=t_start)
                track_time_ms = (time.time() - t_track) * 1000.0

                # 4. Pose Estimation
                t_pose = time.time()
                poses = self.pose_estimator.estimate(frame, tracks)
                pose_time_ms = (time.time() - t_pose) * 1000.0

                # Attach poses into tracker history
                for p in poses:
                    self.tracker.attach_pose(p["track_id"], p)

                # 5. Temporal Interaction Analysis
                interactions = self.interaction_analyzer.analyze(tracks, poses, timestamp=t_start)

                # 6. Multi-Signal Risk Scoring
                risk_result = self.risk_analyzer.evaluate(tracks, interactions)
                current_risk_level = risk_result["risk_level"]

                # 7. Alert Management with Temporal Confirmation
                alert = self.alert_manager.evaluate(risk_result, timestamp=t_start)
                if alert is not None:
                    self.log(f"INCIDENT CREATED: {alert['incident_id']}")
                    self.log(f"ALERT SENT: {alert['risk_level']} (Score: {alert['risk_score']})")

                # Compute loop AI FPS
                now = time.time()
                duration = now - prev_loop_time
                prev_loop_time = now
                if duration > 0:
                    self.ai_fps = 0.8 * self.ai_fps + 0.2 * (1.0 / duration) if self.ai_fps > 0 else (1.0 / duration)

                # Periodic structured telemetry logging (every ~3 seconds)
                if now - self.last_log_time >= 3.0:
                    self.log(f"FPS: {self.ai_fps:.1f} | PERSONS: {len(detections)} | TRACKS: {len(tracks)} | POSE: {'ACTIVE' if poses else 'IDLE'} | RISK: {current_risk_level}")
                    self.last_log_time = now

                # 8. Render Live HUD Display
                if not self.headless:
                    display_frame = frame.copy()
                    self._render_hud(
                        display_frame,
                        tracks=tracks,
                        poses=poses,
                        interactions=interactions,
                        risk_result=risk_result,
                        stream_meta=meta,
                        timings={
                            "det_ms": det_time_ms,
                            "track_ms": track_time_ms,
                            "pose_ms": pose_time_ms,
                            "e2e_ms": (time.time() - t_start) * 1000.0,
                        },
                    )
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.log("Quit requested by user.")
                        break

        except KeyboardInterrupt:
            self.log("KeyboardInterrupt detected. Shutting down gracefully...")
        finally:
            self.stream.release()
            if not self.headless:
                cv2.destroyAllWindows()
            self.log("Pipeline shutdown complete.")

    def _render_offline_hud(self, canvas: np.ndarray, status: str):
        """Draws offline / reconnecting screen."""
        h, w = canvas.shape[:2]
        cv2.putText(canvas, "GUARDIAN MATRIX - CCTV SYSTEM", (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        status_text = f"CAMERA STATUS: {status.upper()}"
        color = (0, 165, 255) if "RECONNECT" in status.upper() or "CONNECT" in status.upper() else (0, 0, 255)
        cv2.putText(canvas, status_text, (50, 160), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
        cv2.putText(canvas, f"URL: {self.stream.url}", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1)
        dots = "." * (int(time.time() * 2) % 4)
        cv2.putText(canvas, f"Attempting reconnection{dots}", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(canvas, "Press 'Q' in window to exit.", (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)

    def _render_hud(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        poses: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
        risk_result: Dict[str, Any],
        stream_meta: Dict[str, Any],
        timings: Dict[str, float],
    ):
        """Draws full interactive CCTV monitoring HUD on the video frame."""
        h, w = frame.shape[:2]

        # 1. Draw Skeleton on Pose Estimates
        for p in poses:
            kps = p.get("keypoints", [])
            kp_dict = {kp["name"]: (int(kp["x"]), int(kp["y"]), kp["confidence"]) for kp in kps}

            # Draw bones
            for p1_name, p2_name in PoseEstimator.SKELETON_CONNECTIONS:
                if p1_name in kp_dict and p2_name in kp_dict:
                    x1, y1, c1 = kp_dict[p1_name]
                    x2, y2, c2 = kp_dict[p2_name]
                    if c1 > 0.3 and c2 > 0.3:
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 200), 2)

            # Draw joints
            for kp in kps:
                if kp["confidence"] > 0.3:
                    cv2.circle(frame, (int(kp["x"]), int(kp["y"])), 4, (0, 200, 255), -1)

        # 2. Draw Bounding Boxes and Track Labels
        for t in tracks:
            bx = t["bbox"]
            x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
            tid = t["track_id"]
            vel = t.get("velocity", 0.0)

            # Highlight person color based on involvement in risk
            is_involved = tid in risk_result.get("track_ids", [])
            box_color = (0, 0, 255) if (is_involved and risk_result["risk_level"] in ("HIGH", "CRITICAL")) else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            label = f"Person {tid} ({vel:.0f} px/s)"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + lw + 6, max(0, y1)), box_color, -1)
            cv2.putText(frame, label, (x1 + 3, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # 3. Draw Interaction Indicator ONLY if an aggressive incident is detected
        if risk_result["risk_level"] in ("HIGH", "CRITICAL"):
            involved = set(risk_result.get("track_ids", []))
            for inter in interactions:
                p_a, p_b = inter.get("person_a", 0), inter.get("person_b", 0)
                if p_a in involved and p_b in involved:
                    track_a = next((t for t in tracks if t["track_id"] == p_a), None)
                    track_b = next((t for t in tracks if t["track_id"] == p_b), None)
                    if track_a and track_b:
                        ca = (int(track_a["center"][0]), int(track_a["center"][1]))
                        cb = (int(track_b["center"][0]), int(track_b["center"][1]))
                        dist = inter.get("distance", 0.0)
                        cv2.line(frame, ca, cb, (0, 0, 255), 2, cv2.LINE_AA)
                        mid = ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2)
                        cv2.putText(frame, f"ALERT: {dist:.0f}px", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 4. Top Header Banner
        cv2.rectangle(frame, (0, 0), (w, 42), (20, 20, 20), -1)
        cv2.putText(frame, "GUARDIAN MATRIX | LIVE CCTV AI SAFETY FEED", (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

        # 5. Right Status & Risk Panel
        panel_w = 340
        panel_x = max(0, w - panel_w)
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, 42), (w, 330), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Stream & System Telemetry
        status_str = stream_meta.get("status", self.stream.status)
        cam_fps = stream_meta.get("fps", self.stream.fps)
        lat = stream_meta.get("latency_ms", self.stream.latency_ms)
        dropped = self.stream.frames_dropped

        # Resource telemetry
        cpu_pct = psutil.cpu_percent()
        mem_mb = self.process.memory_info().rss / (1024 * 1024)

        y_pos = 70
        stat_color = (0, 255, 0) if status_str == "CONNECTED" else (0, 140, 255)
        cv2.putText(frame, f"Camera: {status_str}", (panel_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stat_color, 2)
        y_pos += 26
        cv2.putText(frame, f"Stream FPS: {cam_fps:.1f} | AI FPS: {self.ai_fps:.1f}", (panel_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        y_pos += 24
        cv2.putText(frame, f"Latency: {lat:.0f} ms | Dropped: {dropped}", (panel_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        y_pos += 24
        cv2.putText(frame, f"CPU: {cpu_pct:.1f}% | RAM: {mem_mb:.0f} MB", (panel_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        y_pos += 26
        cv2.putText(frame, f"People: {len(tracks)} | Active Tracks: {len(tracks)}", (panel_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
        y_pos += 30

        # Risk Level Display
        risk_level = risk_result.get("risk_level", "LOW")
        risk_score = risk_result.get("risk_score", 0.0)
        risk_color = (0, 255, 0)
        if risk_level in ("HIGH", "CRITICAL"):
            risk_color = (0, 0, 255)
        elif risk_level == "MEDIUM":
            risk_color = (0, 215, 255)

        cv2.putText(frame, f"Risk Level: {risk_level} ({int(risk_score * 100)}%)", (panel_x + 15, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.65, risk_color, 2)
        y_pos += 24

        # Evidence bullet points
        evidence_list = risk_result.get("evidence", [])
        for ev in evidence_list[:2]:
            cv2.putText(frame, f"- {ev}", (panel_x + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
            y_pos += 20

        # 6. Incident Alert Banner at Bottom if High Risk
        if risk_level in ("HIGH", "CRITICAL"):
            cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 180), -1)
            tracks_str = " & ".join(f"Person {tid}" for tid in risk_result.get("track_ids", []))
            incident_text = f"RISK: {risk_level} | Potential Safety Incident Detected | {tracks_str} | Conf: {int(risk_score * 100)}% | Dur: {risk_result.get('duration', 0.0):.1f}s"
            cv2.putText(frame, incident_text, (20, h - 22), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description="Guardian Matrix - Live CCTV AI Detection System")
    parser.add_argument(
        "--source",
        type=str,
        default="rtsp",
        choices=["rtsp", "webcam", "camera", "video"],
        help="Input video source type (default: rtsp)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="RTSP Stream URL (overrides .env CCTV_RTSP_URL)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to video file when --source video is selected",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device ('cpu', 'cuda')",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI window (ideal for servers or background tests)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames to process before exiting (useful for testing)",
    )
    parser.add_argument(
        "--public-cctv",
        action="store_true",
        help="Automatically stream from a verified live public outdoor CCTV stream from Live-Environment-Streams",
    )
    parser.add_argument(
        "--browse-streams",
        action="store_true",
        help="List verified global public CCTV / HLS camera streams from Live-Environment-Streams dataset",
    )

    args = parser.parse_args()

    if args.browse_streams:
        from helpers.live_environment_loader import get_urban_cctv_streams
        streams = get_urban_cctv_streams(limit=12)
        print("\n=== Verified Live Environment CCTV Streams (Live-Environment-Streams) ===")
        for i, s in enumerate(streams):
            safe_name = s['name'].encode('ascii', 'replace').decode('ascii')
            print(f"{i+1}. [{s['country']}] {safe_name} ({s['environment']})")
            print(f"   URL: {s['url']}\n")
        print("To test any stream, run:")
        print("  .venv\\Scripts\\python.exe main.py --source rtsp --url \"<STREAM_URL>\"\n")
        return

    # Determine source path
    source_path = None
    if args.public_cctv:
        source_path = "https://h057.video-stream-hosting.de/tourismusverbandlechtal-rtplive/_definst_/mp4:camera3.stream/chunklist_w1581120379.m3u8"
        print(f"[PUBLIC CCTV] Selected verified live street camera: {source_path}")
    elif args.source == "rtsp":
        source_path = args.url or settings.cctv_rtsp_url
    elif args.source in ("webcam", "camera"):
        source_path = args.url or "0"
    elif args.source == "video":
        source_path = args.file
        if not source_path:
            print("[ERROR] --source video requires --file <path_to_video.mp4>")
            sys.exit(1)

    pipeline = GuardianMatrixPipeline(
        source_type=args.source,
        source_path=source_path,
        headless=args.headless,
        device=args.device,
    )
    pipeline.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
