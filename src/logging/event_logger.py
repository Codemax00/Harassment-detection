"""
Reproducible Event Logging Layer for Guardian Matrix.
Preserves structured events, physical evidence, and execution metadata
as specified in Upgrade_Plan.md Section 31 & 32.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
import json
import os
import platform
import subprocess
import sys
from typing import Dict, List, Optional, Any

from ..evidence.structured_evidence import StructuredEvidence
from ..policy.safety_policy import PolicyAction


@dataclass
class EventRecord:
    """
    Standardized Reproducible Event Log Record matching Section 31.
    """
    event_id: str
    timestamp: str
    video_id: str
    track_ids: List[int]
    pose_model: str
    temporal_model: str
    decision_model: str
    event_state: str
    confidence: float
    evidence: Dict[str, Any]
    model_versions: Dict[str, str]
    reproducibility: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventLogger:
    """
    Manages structured event logging with full scientific provenance and reproducibility data.
    """

    def __init__(self, log_dir: str = "outputs/events", random_seed: int = 42):
        self.log_dir = log_dir
        self.random_seed = random_seed
        self.event_counter = 0
        os.makedirs(self.log_dir, exist_ok=True)
        self._cached_git_commit = self._get_git_commit()
        self._cached_framework_versions = self._get_framework_versions()

    def _get_git_commit(self) -> str:
        try:
            cmd = ["git", "rev-parse", "--short", "HEAD"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return res.stdout.strip()
        except Exception:
            return "git-commit-unknown"

    def _get_framework_versions(self) -> Dict[str, str]:
        versions = {
            "python": sys.version.split()[0],
            "os": platform.platform()
        }
        for pkg in ["torch", "ultralytics", "cv2", "numpy", "scipy"]:
            try:
                mod = __import__(pkg)
                versions[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                versions[pkg] = "not_installed"
        return versions

    def log_event(
        self,
        video_id: str,
        evidence: StructuredEvidence,
        policy_action: PolicyAction,
        confidence: float,
        pose_model_name: str = "YOLO_Pose",
        temporal_model_name: str = "ST-GCN+Transformer",
        decision_model_name: str = "Laya_System1"
    ) -> EventRecord:
        self.event_counter += 1
        event_id = f"GM-{self.event_counter:05d}"
        now_str = datetime.now().isoformat()

        tracked_ids = evidence.scene.get("tracked_ids", [])
        if evidence.interaction:
            pair = evidence.interaction.get("pair")
            if pair:
                tracked_ids = pair

        reproducibility = {
            "git_commit": self._cached_git_commit,
            "random_seed": self.random_seed,
            "python_version": sys.version.split()[0],
            "framework_versions": self._cached_framework_versions,
            "policy_rationale": policy_action.rationale
        }

        record = EventRecord(
            event_id=event_id,
            timestamp=now_str,
            video_id=video_id,
            track_ids=tracked_ids,
            pose_model=pose_model_name,
            temporal_model=temporal_model_name,
            decision_model=decision_model_name,
            event_state=policy_action.state.value,
            confidence=round(confidence, 3),
            evidence=evidence.to_dict(),
            model_versions={
                "pose": pose_model_name,
                "temporal": temporal_model_name,
                "system1": decision_model_name
            },
            reproducibility=reproducibility
        )

        # Write to event log file
        filepath = os.path.join(self.log_dir, f"{event_id}_{policy_action.state.value}.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(record.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Failed to persist event log {filepath}: {e}")

        return record
