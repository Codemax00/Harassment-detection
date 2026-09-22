"""
Person-Person Interaction Engine for Guardian Matrix.
Computes comprehensive pairwise spatial, geometric, kinematic, and behavioral features
as defined in Upgrade_Plan.md Section 11.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import numpy as np

from ..pose.pose_representation import PoseFrame, KeypointName

if TYPE_CHECKING:
    from ..temporal.temporal_engine import TemporalMotionFeatures


@dataclass
class InteractionFeatures:
    """
    Standardized pairwise interaction evidence as defined in Upgrade_Plan.md Section 11.
    """
    person_a: int
    person_b: int
    distance: float  # Normalized Euclidean distance between body centers (in body-height units)
    relative_velocity: float  # Rate of distance change (negative = closing in, positive = separating)
    relative_direction: Tuple[float, float]  # Unit vector from A to B
    orientation_similarity: float  # -1.0 (opposite) to +1.0 (parallel)
    is_facing_each_other: bool
    hand_to_body_min_distance: float
    hand_to_hand_distance: float
    hand_contact_probability: float
    foot_to_body_min_distance: float = 1.0
    kick_contact_probability: float = 0.0
    aggressive_strike_score: float = 0.0
    movement_synchronization: float = 0.5  # Correlation of trajectories
    proximity_duration_seconds: float = 0.0
    repeated_contact_score: float = 0.0
    pursuit_score: float = 0.0
    separation_score: float = 0.0
    restraint_likelihood: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairwiseInteractionState:
    """Historical tracking state for a specific pair of people (A, B)."""
    pair_key: Tuple[int, int]
    first_seen: float
    last_seen: float
    proximity_history: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, distance)
    contact_events: int = 0
    consecutive_approach_frames: int = 0
    consecutive_retreat_frames: int = 0


class InteractionEngine:
    """
    Pairwise Interaction Engine. Computes interaction metrics across all combinations
    of detected and tracked people in a scene.
    """

    def __init__(self, proximity_threshold: float = 0.8, contact_threshold: float = 0.35):
        self.proximity_threshold = proximity_threshold
        self.contact_threshold = contact_threshold
        self.pair_states: Dict[Tuple[int, int], PairwiseInteractionState] = {}

    def update(
        self,
        poses: List[PoseFrame],
        motion_features: Dict[int, TemporalMotionFeatures],
        timestamp: float = 0.0
    ) -> List[InteractionFeatures]:
        """Compute pairwise interaction features for all pairs present in the frame."""
        if len(poses) < 2:
            return []

        # Sort poses by track_id for deterministic pairing
        sorted_poses = sorted(poses, key=lambda p: p.track_id)
        interaction_list: List[InteractionFeatures] = []

        n = len(sorted_poses)
        for i in range(n):
            for j in range(i + 1, n):
                p_a = sorted_poses[i]
                p_b = sorted_poses[j]
                if p_a.track_id == p_b.track_id:
                    continue

                # Filter duplicate bounding boxes of the same person (IoU > 0.40)
                if p_a.bbox and p_b.bbox:
                    b1, b2 = p_a.bbox, p_b.bbox
                    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                    inter_a = iw * ih
                    union_a = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter_a
                    iou = inter_a / max(1.0, union_a)
                    if iou > 0.40:
                        continue

                m_a = motion_features.get(p_a.track_id)
                m_b = motion_features.get(p_b.track_id)

                feat = self._analyze_pair(p_a, p_b, m_a, m_b, timestamp)
                interaction_list.append(feat)

        return interaction_list

    def _analyze_pair(
        self,
        pose_a: PoseFrame,
        pose_b: PoseFrame,
        motion_a: Optional[TemporalMotionFeatures],
        motion_b: Optional[TemporalMotionFeatures],
        timestamp: float
    ) -> InteractionFeatures:
        id_a, id_b = pose_a.track_id, pose_b.track_id
        pair_key = (min(id_a, id_b), max(id_a, id_b))

        # 1. Height normalization scale
        ha = max(1.0, (pose_a.bbox[3] - pose_a.bbox[1])) if pose_a.bbox else 100.0
        hb = max(1.0, (pose_b.bbox[3] - pose_b.bbox[1])) if pose_b.bbox else 100.0
        h_scale = (ha + hb) / 2.0

        # Scene torso centers
        pos_a = pose_a.get_scene_torso_center()
        pos_b = pose_b.get_scene_torso_center()
        dx_px = pos_b[0] - pos_a[0]
        dy_px = pos_b[1] - pos_a[1]
        dist_px = float(np.hypot(dx_px, dy_px))
        dist = float(dist_px / h_scale)  # normalized in body-height units

        # Relative unit direction from A to B
        if dist_px > 1e-4:
            rel_dir = (float(dx_px / dist_px), float(dy_px / dist_px))
        else:
            rel_dir = (0.0, 0.0)

        # 2. Pairwise state history and relative velocity
        state = self.pair_states.get(pair_key)
        if state is None:
            state = PairwiseInteractionState(
                pair_key=pair_key,
                first_seen=timestamp,
                last_seen=timestamp
            )
            self.pair_states[pair_key] = state

        state.last_seen = timestamp
        state.proximity_history.append((timestamp, dist))
        if len(state.proximity_history) > 60:
            state.proximity_history.pop(0)

        # Relative velocity (rate of distance change: negative = closing distance)
        rel_vel = 0.0
        if len(state.proximity_history) >= 2:
            t_prev, d_prev = state.proximity_history[-2]
            dt = max(1e-4, timestamp - t_prev)
            rel_vel = float((dist - d_prev) / dt)

        # Proximity duration (accumulated time within proximity threshold)
        in_prox_times = [t for t, d in state.proximity_history if d <= self.proximity_threshold]
        prox_duration = (in_prox_times[-1] - in_prox_times[0]) if len(in_prox_times) >= 2 else 0.0

        # 3. Hand-to-body and hand-to-hand distances in scene space
        a_wrists = [pose_a.get_scene_keypoint_2d(KeypointName.LEFT_WRIST), pose_a.get_scene_keypoint_2d(KeypointName.RIGHT_WRIST)]
        b_wrists = [pose_b.get_scene_keypoint_2d(KeypointName.LEFT_WRIST), pose_b.get_scene_keypoint_2d(KeypointName.RIGHT_WRIST)]
        a_w = [pt for pt in a_wrists if pt is not None]
        b_w = [pt for pt in b_wrists if pt is not None]

        h2h_dist = 99.0
        for wa in a_w:
            for wb in b_w:
                d_hh = float(np.hypot(wa[0] - wb[0], wa[1] - wb[1]) / h_scale)
                if d_hh < h2h_dist:
                    h2h_dist = d_hh

        h2b_dist = 99.0
        # Check A's hands to B's torso and head/shoulders
        b_targets = [pos_b]
        for idx in [KeypointName.NOSE, KeypointName.LEFT_SHOULDER, KeypointName.RIGHT_SHOULDER]:
            pt = pose_b.get_scene_keypoint_2d(idx)
            if pt: b_targets.append(pt)

        for wa in a_w:
            for bt in b_targets:
                d_hb = float(np.hypot(wa[0] - bt[0], wa[1] - bt[1]) / h_scale)
                if d_hb < h2b_dist:
                    h2b_dist = d_hb

        # Check B's hands to A's torso and head/shoulders
        a_targets = [pos_a]
        for idx in [KeypointName.NOSE, KeypointName.LEFT_SHOULDER, KeypointName.RIGHT_SHOULDER]:
            pt = pose_a.get_scene_keypoint_2d(idx)
            if pt: a_targets.append(pt)

        for wb in b_w:
            for at in a_targets:
                d_hb = float(np.hypot(wb[0] - at[0], wb[1] - at[1]) / h_scale)
                if d_hb < h2b_dist:
                    h2b_dist = d_hb

        # Hand contact probability (only valid if people are within realistic arm reach)
        min_hand_reach = min(h2h_dist, h2b_dist)
        if dist < 1.0 and min_hand_reach < self.contact_threshold:
            hand_contact_prob = float(np.clip(1.0 - (min_hand_reach / self.contact_threshold), 0.0, 1.0))
        else:
            hand_contact_prob = 0.0

        # 4. Kick / Foot-to-body detection
        a_ankles = [pose_a.get_scene_keypoint_2d(KeypointName.LEFT_ANKLE), pose_a.get_scene_keypoint_2d(KeypointName.RIGHT_ANKLE)]
        b_ankles = [pose_b.get_scene_keypoint_2d(KeypointName.LEFT_ANKLE), pose_b.get_scene_keypoint_2d(KeypointName.RIGHT_ANKLE)]
        a_f = [pt for pt in a_ankles if pt is not None]
        b_f = [pt for pt in b_ankles if pt is not None]

        f2b_dist = 99.0
        # A's foot to B's body or legs
        b_leg_targets = [pos_b]
        for idx in [KeypointName.LEFT_HIP, KeypointName.RIGHT_HIP, KeypointName.LEFT_KNEE, KeypointName.RIGHT_KNEE]:
            pt = pose_b.get_scene_keypoint_2d(idx)
            if pt: b_leg_targets.append(pt)

        for fa in a_f:
            for bt in b_leg_targets:
                d_fb = float(np.hypot(fa[0] - bt[0], fa[1] - bt[1]) / h_scale)
                if d_fb < f2b_dist:
                    f2b_dist = d_fb

        # B's foot to A's body or legs
        a_leg_targets = [pos_a]
        for idx in [KeypointName.LEFT_HIP, KeypointName.RIGHT_HIP, KeypointName.LEFT_KNEE, KeypointName.RIGHT_KNEE]:
            pt = pose_a.get_scene_keypoint_2d(idx)
            if pt: a_leg_targets.append(pt)

        for fb in b_f:
            for at in a_leg_targets:
                d_fb = float(np.hypot(fb[0] - at[0], fb[1] - at[1]) / h_scale)
                if d_fb < f2b_dist:
                    f2b_dist = d_fb

        kick_contact_prob = 0.0
        if dist < 1.2 and f2b_dist < 0.32:
            kick_contact_prob = float(np.clip(1.0 - (f2b_dist / 0.32), 0.0, 1.0))

        # 5. Strike Impact Score (Kinematic Velocity Spikes during proximity)
        peak_hand_v = max(
            motion_a.peak_hand_speed if motion_a else 0.0,
            motion_b.peak_hand_speed if motion_b else 0.0
        )
        peak_foot_v = max(
            motion_a.peak_foot_speed if motion_a else 0.0,
            motion_b.peak_foot_speed if motion_b else 0.0
        )

        hand_strike = hand_contact_prob * min(1.0, peak_hand_v / 2.2) if (hand_contact_prob > 0.3 and peak_hand_v > 1.8) else 0.0
        kick_strike = kick_contact_prob * min(1.0, peak_foot_v / 2.0) if (kick_contact_prob > 0.3 and peak_foot_v > 1.8) else 0.0
        aggressive_strike_score = float(max(hand_strike, kick_strike))

        if (hand_contact_prob > 0.65 or kick_contact_prob > 0.65 or aggressive_strike_score > 0.5) and dist < 0.7:
            state.contact_events += 1

        repeated_contact_score = float(np.clip(state.contact_events / 6.0, 0.0, 1.0))

        # 6. Body orientation similarity & facing
        orient_a = motion_a.body_orientation_degrees if motion_a else 0.0
        orient_b = motion_b.body_orientation_degrees if motion_b else 0.0
        angle_diff = abs(orient_a - orient_b)
        orient_similarity = float(np.cos(np.radians(angle_diff)))
        is_facing = bool(orient_similarity < -0.3 and dist < self.proximity_threshold * 1.5)

        # 7. Pursuit vs Separation vs Restraint kinematics
        pursuit_score = 0.0
        separation_score = 0.0
        restraint_score = 0.0
        sync_score = 0.5

        if motion_a and motion_b:
            va = np.array(motion_a.torso_velocity[:2])
            vb = np.array(motion_b.torso_velocity[:2])
            speed_a = np.linalg.norm(va)
            speed_b = np.linalg.norm(vb)

            if speed_a > 1e-3 and speed_b > 1e-3:
                sync_score = float((np.dot(va, vb) / (speed_a * speed_b) + 1.0) / 2.0)

            # Pursuit: requires true chase speeds (> 1.2 body heights/sec) and closing distance
            if speed_a > 1.2 and speed_b > 1.2 and sync_score > 0.8 and dist < 1.2:
                pursuit_score = float(min(1.0, (speed_a + speed_b) * sync_score * 0.4))

            # Separation: moving apart rapidly
            if rel_vel > 0.6 and (speed_a > 0.8 or speed_b > 0.8):
                separation_score = float(np.clip(rel_vel * 0.8, 0.0, 1.0))

            # Restraint: victim is nearly motionless while aggressor has high contact
            if dist < 0.45 and (hand_contact_prob > 0.65 or repeated_contact_score > 0.6):
                diff_motion = abs(motion_a.torso_speed - motion_b.torso_speed)
                min_motion = min(motion_a.torso_speed, motion_b.torso_speed)
                if min_motion < 0.4 and diff_motion > 0.4:
                    restraint_score = float(np.clip(hand_contact_prob * (0.5 + diff_motion), 0.0, 1.0))

        return InteractionFeatures(
            person_a=id_a,
            person_b=id_b,
            distance=dist,
            relative_velocity=rel_vel,
            relative_direction=rel_dir,
            orientation_similarity=orient_similarity,
            is_facing_each_other=is_facing,
            hand_to_body_min_distance=h2b_dist if h2b_dist < 90 else 1.0,
            hand_to_hand_distance=h2h_dist if h2h_dist < 90 else 1.0,
            hand_contact_probability=hand_contact_prob,
            foot_to_body_min_distance=f2b_dist if f2b_dist < 90 else 1.0,
            kick_contact_probability=kick_contact_prob,
            aggressive_strike_score=aggressive_strike_score,
            movement_synchronization=sync_score,
            proximity_duration_seconds=float(prox_duration),
            repeated_contact_score=repeated_contact_score,
            pursuit_score=pursuit_score,
            separation_score=separation_score,
            restraint_likelihood=restraint_score
        )
