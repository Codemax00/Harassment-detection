"""
Temporal Action Classifier Layer for Guardian Matrix.
Implements ST-GCN (Spatial-Temporal Graph Convolutional Network) and
Temporal Transformer skeleton models, plus a kinematic feature-based ensemble
to classify observable interaction patterns (Upgrade_Plan.md Section 14).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..taxonomy.event_taxonomy import EventLabel, EventTaxonomy
from ..pose.pose_representation import PoseFrame
from ..interaction.interaction_engine import InteractionFeatures
from .temporal_engine import TemporalMotionFeatures


# 17 COCO keypoint connectivity graph
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]


class BaseTemporalClassifier(ABC):
    @abstractmethod
    def classify(
        self,
        pose_sequences: Dict[int, List[PoseFrame]],
        interactions: List[InteractionFeatures],
        motion_features: Dict[int, TemporalMotionFeatures]
    ) -> Tuple[EventLabel, float, Dict[str, float]]:
        """Classify temporal action, returns (best_label, confidence, all_probabilities)."""
        pass


class STGCNModule(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network Block.
    Processes (Batch, Channels, Time, Joints).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10, num_joints: int = 17):
        super().__init__()
        self.num_joints = num_joints
        # Spatial adjacency matrix
        A = np.zeros((num_joints, num_joints), dtype=np.float32)
        for i, j in SKELETON_EDGES:
            if i < num_joints and j < num_joints:
                A[i, j] = 1.0
                A[j, i] = 1.0
        A += np.eye(num_joints, dtype=np.float32)  # Self-loops
        # Degree normalization
        D = np.diag(np.sum(A, axis=1) ** -0.5)
        norm_A = D @ A @ D
        self.register_buffer("A", torch.tensor(norm_A, dtype=torch.float32))

        # Spatial-Temporal conv stages
        self.spatial_conv1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.temporal_conv1 = nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(32)

        self.spatial_conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.temporal_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T, V)
        B, C, T, V = x.shape
        # Spatial graph multiply
        A = self.A[:V, :V]
        x = torch.einsum('bctv,vw->bctw', x, A)
        x = F.relu(self.bn1(self.temporal_conv1(self.spatial_conv1(x))))

        x = torch.einsum('bctv,vw->bctw', x, A)
        x = F.relu(self.bn2(self.temporal_conv2(self.spatial_conv2(x))))

        x = self.pool(x).view(B, -1)
        logits = self.fc(x)
        return logits


class STGCNClassifier(BaseTemporalClassifier):
    """
    ST-GCN temporal action classifier evaluating skeletal graph convolutions.
    """

    def __init__(self, num_classes: int = 10):
        self.model = STGCNModule(in_channels=3, num_classes=num_classes)
        self.model.eval()
        self.labels = [e for e in EventLabel]

    def classify(
        self,
        pose_sequences: Dict[int, List[PoseFrame]],
        interactions: List[InteractionFeatures],
        motion_features: Dict[int, TemporalMotionFeatures]
    ) -> Tuple[EventLabel, float, Dict[str, float]]:
        # If no pose sequences available, return default
        if not pose_sequences:
            return EventLabel.NORMAL_ACTIVITY, 0.9, {e.value: (0.9 if e == EventLabel.NORMAL_ACTIVITY else 0.01) for e in EventLabel}

        # Prepare tensor (B=1, C=3, T=window, V=17) from the most active track
        primary_tid = list(pose_sequences.keys())[0]
        window = pose_sequences[primary_tid]
        T = min(30, max(2, len(window)))
        x_np = np.zeros((1, 3, T, 17), dtype=np.float32)

        for t_idx, pf in enumerate(window[-T:]):
            kps = pf.keypoints_2d[:17]
            vis = pf.visibility[:17] if pf.visibility is not None else np.ones(len(kps))
            x_np[0, 0, :len(kps), t_idx] = kps[:, 0]
            x_np[0, 1, :len(kps), t_idx] = kps[:, 1]
            x_np[0, 2, :len(kps), t_idx] = vis

        with torch.no_grad():
            x_tensor = torch.from_numpy(x_np)
            logits = self.model(x_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        prob_dict = {self.labels[i].value: float(probs[i % len(self.labels)]) for i in range(len(self.labels))}
        best_idx = int(np.argmax(probs)) % len(self.labels)
        return self.labels[best_idx], float(probs[best_idx]), prob_dict


class TemporalTransformerModule(nn.Module):
    """
    Temporal Transformer Network Block.
    Processes joint embeddings across time steps with self-attention.
    """

    def __init__(self, in_features: int = 34, embed_dim: int = 64, num_heads: int = 4, num_classes: int = 10):
        super().__init__()
        self.embedding = nn.Linear(in_features, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, in_features)
        h = self.embedding(x)
        out = self.transformer(h)
        # Global temporal average pooling
        pooled = torch.mean(out, dim=1)
        return self.classifier(pooled)


class TemporalTransformerClassifier(BaseTemporalClassifier):
    """
    Temporal Transformer action classifier evaluating attention over pose sequences.
    """

    def __init__(self, num_classes: int = 10):
        self.model = TemporalTransformerModule(in_features=34, embed_dim=64, num_heads=4, num_classes=num_classes)
        self.model.eval()
        self.labels = [e for e in EventLabel]

    def classify(
        self,
        pose_sequences: Dict[int, List[PoseFrame]],
        interactions: List[InteractionFeatures],
        motion_features: Dict[int, TemporalMotionFeatures]
    ) -> Tuple[EventLabel, float, Dict[str, float]]:
        if not pose_sequences:
            return EventLabel.NORMAL_ACTIVITY, 0.9, {e.value: 0.1 for e in EventLabel}

        primary_tid = list(pose_sequences.keys())[0]
        window = pose_sequences[primary_tid]
        T = min(30, max(2, len(window)))
        x_np = np.zeros((1, T, 34), dtype=np.float32)

        for t_idx, pf in enumerate(window[-T:]):
            kps_flat = pf.keypoints_2d[:17, :2].flatten()
            x_np[0, t_idx, :len(kps_flat)] = kps_flat

        with torch.no_grad():
            x_tensor = torch.from_numpy(x_np)
            logits = self.model(x_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        prob_dict = {self.labels[i].value: float(probs[i % len(self.labels)]) for i in range(len(self.labels))}
        best_idx = int(np.argmax(probs)) % len(self.labels)
        return self.labels[best_idx], float(probs[best_idx]), prob_dict


class KinematicActionClassifier(BaseTemporalClassifier):
    """
    Physical Kinematic Interaction Classifier.
    Evaluates observable geometric, spatial, and motion dynamics
    (contact, pursuit, separation, restraint, strikes, falls) to produce
    grounded, auditable event classification.
    """

    def __init__(self):
        # Deterministic kinematic rule engine without fake/untrained neural weights
        pass

    def classify(
        self,
        pose_sequences: Dict[int, List[PoseFrame]],
        interactions: List[InteractionFeatures],
        motion_features: Dict[int, TemporalMotionFeatures]
    ) -> Tuple[EventLabel, float, Dict[str, float]]:
        # Base probabilities dictionary
        probs: Dict[str, float] = {e.value: 0.05 for e in EventLabel}
        probs[EventLabel.NORMAL_ACTIVITY.value] = 0.50

        # Physical heuristic scoring directly informed by pairwise interaction features
        max_contact = 0.0
        max_pursuit = 0.0
        max_separation = 0.0
        max_restraint = 0.0
        max_strike = 0.0
        max_kick = 0.0
        min_dist = 99.0

        for inter in interactions:
            min_dist = min(min_dist, inter.distance)
            max_contact = max(max_contact, inter.hand_contact_probability, inter.repeated_contact_score)
            max_pursuit = max(max_pursuit, inter.pursuit_score)
            max_separation = max(max_separation, inter.separation_score)
            max_restraint = max(max_restraint, inter.restraint_likelihood)
            max_strike = max(max_strike, inter.aggressive_strike_score)
            max_kick = max(max_kick, inter.kick_contact_probability)

        # Check for falls in motion features
        any_fall = any(mf.is_falling for mf in motion_features.values())
        any_rapid = any(mf.is_rapid_motion for mf in motion_features.values())

        if any_fall:
            probs[EventLabel.FALL_PATTERN.value] = 0.88
        elif max_strike > 0.40 or max_kick > 0.40:
            probs[EventLabel.AGGRESSIVE_MOTION_PATTERN.value] = 0.88
        elif max_contact > 0.65:
            probs[EventLabel.REPEATED_CONTACT_PATTERN.value] = 0.82
        elif max_restraint > 0.65:
            probs[EventLabel.RESTRAINT_PATTERN.value] = 0.80
        elif max_pursuit > 0.65:
            probs[EventLabel.PURSUIT_PATTERN.value] = 0.78
        elif max_separation > 0.65:
            probs[EventLabel.RAPID_SEPARATION.value] = 0.75
        elif min_dist < 0.65:
            probs[EventLabel.CLOSE_INTERACTION.value] = 0.72
        else:
            probs[EventLabel.NORMAL_ACTIVITY.value] = 0.88

        # Normalize probability distribution
        total_p = sum(probs.values())
        norm_probs = {k: v / total_p for k, v in probs.items()}

        best_label_str = max(norm_probs, key=norm_probs.get)
        best_label = EventLabel(best_label_str)
        confidence = float(norm_probs[best_label_str])

        return best_label, confidence, norm_probs


# Backward-compatibility alias
EnsembleTemporalClassifier = KinematicActionClassifier


def load_temporal_classifier(arch: str = "ensemble") -> BaseTemporalClassifier:
    a = arch.lower()
    if a == "st_gcn":
        return STGCNClassifier()
    elif a == "transformer":
        return TemporalTransformerClassifier()
    return KinematicActionClassifier()
