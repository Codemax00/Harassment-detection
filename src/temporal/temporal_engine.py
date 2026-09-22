"""
Temporal Motion Engine for Guardian Matrix.
Computes multi-frame kinematics (velocity, acceleration, joint angles, body orientation)
over sliding temporal windows (1-3 seconds) as specified in Upgrade_Plan.md Section 10 & 12.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..pose.pose_representation import PoseFrame, KeypointName


def compute_angle_3points(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """Compute 2D angle in degrees at vertex p2 formed by lines (p1-p2) and (p3-p2)."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float32)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cosine = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


@dataclass
class TemporalMotionFeatures:
    """
    Kinematic & geometric temporal features for an individual tracked person.
    """
    track_id: int
    window_duration_seconds: float
    num_frames: int

    # Mean and peak velocities (normalized units / sec)
    torso_velocity: Tuple[float, float, float]  # (dx/dt, dy/dt, dz/dt)
    torso_speed: float
    torso_acceleration: Tuple[float, float, float]  # (d2x/dt2, d2y/dt2, d2z/dt2)
    peak_acceleration: float

    # Limb velocities
    left_hand_velocity: float
    right_hand_velocity: float
    peak_hand_speed: float
    left_foot_velocity: float
    right_foot_velocity: float
    peak_foot_speed: float

    # Joint angles (degrees)
    left_elbow_angle: float
    right_elbow_angle: float
    left_knee_angle: float
    right_knee_angle: float
    torso_tilt_degrees: float

    # Orientation (heading angle in radians / degrees)
    body_orientation_degrees: float

    # Trajectory consistency
    motion_jitter: float
    is_rapid_motion: bool
    is_falling: bool


class SlidingWindowBuffer:
    """Maintains a rolling temporal window of PoseFrames per tracked person."""

    def __init__(self, window_seconds: float = 2.0, max_fps: float = 30.0):
        self.window_seconds = window_seconds
        self.max_frames = int(window_seconds * max_fps) + 5
        self.buffers: Dict[int, deque] = {}  # track_id -> deque[PoseFrame]

    def add_pose(self, pose: PoseFrame):
        tid = pose.track_id
        if tid not in self.buffers:
            self.buffers[tid] = deque(maxlen=self.max_frames)
        self.buffers[tid].append(pose)
        self._prune_expired(tid, pose.timestamp)

    def _prune_expired(self, track_id: int, current_timestamp: float):
        dq = self.buffers.get(track_id)
        if not dq:
            return
        cutoff = current_timestamp - self.window_seconds
        while len(dq) > 1 and dq[0].timestamp < cutoff:
            dq.popleft()

    def get_window(self, track_id: int) -> List[PoseFrame]:
        return list(self.buffers.get(track_id, []))

    def prune_inactive_tracks(self, active_track_ids: List[int]):
        active_set = set(active_track_ids)
        to_del = [tid for tid in self.buffers if tid not in active_set]
        for tid in to_del:
            del self.buffers[tid]


class TemporalMotionEngine:
    """
    Extracts kinematic and anatomical motion features from temporal pose trajectories.
    """

    def __init__(self, window_seconds: float = 2.0):
        self.buffer = SlidingWindowBuffer(window_seconds=window_seconds)

    def update(self, poses: List[PoseFrame]) -> Dict[int, TemporalMotionFeatures]:
        """Add current frame poses and compute temporal motion features for all active tracks."""
        for p in poses:
            self.buffer.add_pose(p)

        active_tids = [p.track_id for p in poses]
        self.buffer.prune_inactive_tracks(active_tids)

        features: Dict[int, TemporalMotionFeatures] = {}
        for tid in active_tids:
            window = self.buffer.get_window(tid)
            if len(window) >= 2:
                feat = self._compute_person_motion(tid, window)
                features[tid] = feat
        return features

    def _compute_person_motion(self, track_id: int, window: List[PoseFrame]) -> TemporalMotionFeatures:
        dt_total = max(1e-4, window[-1].timestamp - window[0].timestamp)
        num_frames = len(window)
        curr = window[-1]

        # Body height scale for normalized physical units (body heights / sec)
        curr_box = curr.bbox
        if curr_box is not None:
            box_w = max(1.0, curr_box[2] - curr_box[0])
            box_h = max(1.0, curr_box[3] - curr_box[1])
        else:
            box_w, box_h = 50.0, 100.0

        # Extract scene torso centers across the sliding window
        torso_positions = [p.get_scene_torso_center() for p in window]
        xs = [pos[0] / box_h for pos in torso_positions]
        ys = [pos[1] / box_h for pos in torso_positions]

        # Velocity dx/dt, dy/dt (units: body heights / sec)
        vx = (xs[-1] - xs[0]) / dt_total
        vy = (ys[-1] - ys[0]) / dt_total
        vz = 0.0
        torso_speed = float(np.sqrt(vx**2 + vy**2))

        # Acceleration d2x/dt2, d2y/dt2
        if num_frames >= 3:
            mid = num_frames // 2
            dt1 = max(1e-4, window[mid].timestamp - window[0].timestamp)
            dt2 = max(1e-4, window[-1].timestamp - window[mid].timestamp)
            v1x = (xs[mid] - xs[0]) / dt1
            v1y = (ys[mid] - ys[0]) / dt1
            v2x = (xs[-1] - xs[mid]) / dt2
            v2y = (ys[-1] - ys[mid]) / dt2
            dt_mid = (dt1 + dt2) / 2.0
            ax = (v2x - v1x) / dt_mid
            ay = (v2y - v1y) / dt_mid
            peak_acc = float(np.sqrt(ax**2 + ay**2))
        else:
            ax, ay, peak_acc = 0.0, 0.0, 0.0

        # Hand velocities (body heights / sec)
        lh_speeds, rh_speeds = [], []
        # Foot velocities (body heights / sec)
        lf_speeds, rf_speeds = [], []

        for i in range(1, num_frames):
            dt = max(1e-4, window[i].timestamp - window[i-1].timestamp)
            p_prev, p_curr = window[i-1], window[i]

            # Left & right wrists
            lw_prev = p_prev.get_scene_keypoint_2d(KeypointName.LEFT_WRIST)
            lw_curr = p_curr.get_scene_keypoint_2d(KeypointName.LEFT_WRIST)
            if lw_prev and lw_curr:
                lh_speeds.append((np.hypot(lw_curr[0] - lw_prev[0], lw_curr[1] - lw_prev[1]) / box_h) / dt)

            rw_prev = p_prev.get_scene_keypoint_2d(KeypointName.RIGHT_WRIST)
            rw_curr = p_curr.get_scene_keypoint_2d(KeypointName.RIGHT_WRIST)
            if rw_prev and rw_curr:
                rh_speeds.append((np.hypot(rw_curr[0] - rw_prev[0], rw_curr[1] - rw_prev[1]) / box_h) / dt)

            # Left & right ankles
            la_prev = p_prev.get_scene_keypoint_2d(KeypointName.LEFT_ANKLE)
            la_curr = p_curr.get_scene_keypoint_2d(KeypointName.LEFT_ANKLE)
            if la_prev and la_curr:
                lf_speeds.append((np.hypot(la_curr[0] - la_prev[0], la_curr[1] - la_prev[1]) / box_h) / dt)

            ra_prev = p_prev.get_scene_keypoint_2d(KeypointName.RIGHT_ANKLE)
            ra_curr = p_curr.get_scene_keypoint_2d(KeypointName.RIGHT_ANKLE)
            if ra_prev and ra_curr:
                rf_speeds.append((np.hypot(ra_curr[0] - ra_prev[0], ra_curr[1] - ra_prev[1]) / box_h) / dt)

        mean_lh_vel = float(np.mean(lh_speeds)) if lh_speeds else 0.0
        mean_rh_vel = float(np.mean(rh_speeds)) if rh_speeds else 0.0
        peak_hand_speed = max(
            float(np.max(lh_speeds)) if lh_speeds else 0.0,
            float(np.max(rh_speeds)) if rh_speeds else 0.0
        )

        mean_lf_vel = float(np.mean(lf_speeds)) if lf_speeds else 0.0
        mean_rf_vel = float(np.mean(rf_speeds)) if rf_speeds else 0.0
        peak_foot_speed = max(
            float(np.max(lf_speeds)) if lf_speeds else 0.0,
            float(np.max(rf_speeds)) if rf_speeds else 0.0
        )

        # Joint angles at latest frame
        ls = curr.get_scene_keypoint_2d(KeypointName.LEFT_SHOULDER)
        le = curr.get_scene_keypoint_2d(KeypointName.LEFT_ELBOW)
        lw = curr.get_scene_keypoint_2d(KeypointName.LEFT_WRIST)
        left_elbow_angle = compute_angle_3points(ls, le, lw) if (ls and le and lw) else 180.0

        rs = curr.get_scene_keypoint_2d(KeypointName.RIGHT_SHOULDER)
        re = curr.get_scene_keypoint_2d(KeypointName.RIGHT_ELBOW)
        rw = curr.get_scene_keypoint_2d(KeypointName.RIGHT_WRIST)
        right_elbow_angle = compute_angle_3points(rs, re, rw) if (rs and re and rw) else 180.0

        lh = curr.get_scene_keypoint_2d(KeypointName.LEFT_HIP)
        lk = curr.get_scene_keypoint_2d(KeypointName.LEFT_KNEE)
        la = curr.get_scene_keypoint_2d(KeypointName.LEFT_ANKLE)
        left_knee_angle = compute_angle_3points(lh, lk, la) if (lh and lk and la) else 180.0

        rh = curr.get_scene_keypoint_2d(KeypointName.RIGHT_HIP)
        rk = curr.get_scene_keypoint_2d(KeypointName.RIGHT_KNEE)
        ra = curr.get_scene_keypoint_2d(KeypointName.RIGHT_ANKLE)
        right_knee_angle = compute_angle_3points(rh, rk, ra) if (rh and rk and ra) else 180.0

        # Torso tilt
        torso_tilt = 0.0
        body_orientation = 0.0
        if ls and rs and lh and rh:
            mid_shoulder = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
            mid_hip = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
            dx_torso = mid_hip[0] - mid_shoulder[0]
            dy_torso = mid_hip[1] - mid_shoulder[1]
            torso_tilt = float(np.degrees(np.arctan2(abs(dx_torso), max(1e-4, abs(dy_torso)))))
            shoulder_w = rs[0] - ls[0]
            body_orientation = float(np.degrees(np.arctan2(0.0, shoulder_w)))

        # Fall / Lying Detection:
        # 1. Bounding box prominently wider than tall (box_w > box_h * 1.35)
        # 2. Sudden downward collapse (vy > 0.85 body heights/sec) with high torso tilt (> 45 deg)
        # 3. Prone posture (torso_tilt > 72 deg) without deep squat (to avoid confusing crouch/squat)
        is_squatting = (left_knee_angle < 80.0 and right_knee_angle < 80.0)
        is_prone = (box_w > box_h * 1.35) or (torso_tilt > 72.0 and not is_squatting)
        is_dynamic_fall = (vy > 0.85 and torso_tilt > 45.0)
        is_falling = bool(is_prone or is_dynamic_fall)

        # Rapid movement: fast sprint or violent strike motion (higher threshold to avoid normal walk)
        is_rapid = bool(torso_speed > 1.3 or peak_hand_speed > 2.2 or peak_foot_speed > 2.0)

        # Motion jitter / variance
        jitter = float(np.std(xs) + np.std(ys)) if len(xs) > 1 else 0.0

        return TemporalMotionFeatures(
            track_id=track_id,
            window_duration_seconds=float(dt_total),
            num_frames=num_frames,
            torso_velocity=(float(vx), float(vy), float(vz)),
            torso_speed=torso_speed,
            torso_acceleration=(float(ax), float(ay), 0.0),
            peak_acceleration=peak_acc,
            left_hand_velocity=mean_lh_vel,
            right_hand_velocity=mean_rh_vel,
            peak_hand_speed=peak_hand_speed,
            left_foot_velocity=mean_lf_vel,
            right_foot_velocity=mean_rf_vel,
            peak_foot_speed=peak_foot_speed,
            left_elbow_angle=left_elbow_angle,
            right_elbow_angle=right_elbow_angle,
            left_knee_angle=left_knee_angle,
            right_knee_angle=right_knee_angle,
            torso_tilt_degrees=torso_tilt,
            body_orientation_degrees=body_orientation,
            motion_jitter=jitter,
            is_rapid_motion=is_rapid,
            is_falling=is_falling
        )
