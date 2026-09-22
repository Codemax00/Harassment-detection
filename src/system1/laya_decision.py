"""
Laya System-1 Fast Decision Layer for Guardian Matrix.
Evaluates structured evidence without raw video inputs, producing typed decisions
with multi-signal confidence quality gates (Upgrade_Plan.md Section 16 & 17).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np

from ..evidence.structured_evidence import StructuredEvidence
from ..taxonomy.event_taxonomy import EventTaxonomy, EventLabel


class DecisionState(str, Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    REVIEW = "REVIEW"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class LayaDecision:
    """
    Typed Decision emitted by Laya System-1.
    """
    decision: DecisionState
    calibrated_confidence: float
    raw_confidence: float
    passes_quality_gate: bool
    quality_gate_details: Dict[str, Any]
    escalate_to_system2: bool
    rationale: str
    timestamp: float
    frame_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class LayaSystem1Engine:
    """
    Laya System-1 Fast Inference Engine.
    Enforces the confidence architecture:
    Model Confidence + Pose Quality + Tracking Quality + Temporal Consistency + Evidence Completeness
    """

    def __init__(
        self,
        min_pose_quality: float = 0.25,
        min_tracking_quality: float = 0.25,
        min_temporal_consistency: float = 0.35,
        min_evidence_completeness: float = 0.35,
        suspicious_threshold: float = 0.55,
        review_threshold: float = 0.68,
        temperature: float = 1.05
    ):
        self.min_pose_quality = min_pose_quality
        self.min_tracking_quality = min_tracking_quality
        self.min_temporal_consistency = min_temporal_consistency
        self.min_evidence_completeness = min_evidence_completeness
        self.suspicious_threshold = suspicious_threshold
        self.review_threshold = review_threshold
        self.temperature = temperature

    def evaluate(self, evidence: StructuredEvidence) -> LayaDecision:
        """
        Evaluate structured evidence and return a typed decision.
        """
        # Extract quality metrics
        qm = evidence.quality_metrics
        pose_qual = qm.get("pose_quality", 0.0)
        track_qual = qm.get("tracking_quality", 0.0)
        temp_cons = qm.get("temporal_consistency", 0.0)
        ev_comp = qm.get("evidence_completeness", 0.0)

        # Check multi-factor quality gates
        gate_passed = (
            pose_qual >= self.min_pose_quality
            and track_qual >= self.min_tracking_quality
            and temp_cons >= self.min_temporal_consistency
            and ev_comp >= self.min_evidence_completeness
        )

        gate_details = {
            "pose_quality": {"value": pose_qual, "passed": pose_qual >= self.min_pose_quality},
            "tracking_quality": {"value": track_qual, "passed": track_qual >= self.min_tracking_quality},
            "temporal_consistency": {"value": temp_cons, "passed": temp_cons >= self.min_temporal_consistency},
            "evidence_completeness": {"value": ev_comp, "passed": ev_comp >= self.min_evidence_completeness},
            "all_gates_passed": gate_passed
        }

        # Temporal model predictions
        pattern_str = evidence.temporal_model.get("interaction_pattern", EventLabel.NORMAL_ACTIVITY.value)
        try:
            pattern = EventLabel(pattern_str)
        except ValueError:
            pattern = EventLabel.NORMAL_ACTIVITY

        raw_conf = float(evidence.temporal_model.get("confidence", 0.5))

        # Temperature scaling calibration
        calibrated_conf = float(1.0 / (1.0 + np.exp(- (np.log(max(1e-4, raw_conf / (1.0 - min(0.999, raw_conf)))) / self.temperature))))

        is_risk = EventTaxonomy.is_risk_event(pattern)

        # Apply Decision Logic
        if not gate_passed:
            if is_risk and calibrated_conf >= self.suspicious_threshold:
                decision = DecisionState.SUSPICIOUS
                escalate = True
                rationale = f"Observable physical risk pattern '{pattern.value}' detected under degraded surveillance conditions."
            else:
                decision = DecisionState.UNCERTAIN
                escalate = is_risk or (calibrated_conf >= self.suspicious_threshold)
                rationale = "Quality gate criteria not fully met; degraded pose/tracking/temporal consistency."
        else:
            if pattern == EventLabel.NORMAL_ACTIVITY:
                decision = DecisionState.NORMAL
                escalate = False
                rationale = "Routine activity observed; no physical risk patterns detected."
            elif is_risk and calibrated_conf >= self.review_threshold:
                decision = DecisionState.REVIEW
                escalate = True
                rationale = f"Observable high-risk interaction pattern '{pattern.value}' detected with high confidence ({calibrated_conf:.2f})."
            elif is_risk or calibrated_conf >= self.suspicious_threshold:
                decision = DecisionState.SUSPICIOUS
                escalate = True
                rationale = f"Suspicious interaction pattern '{pattern.value}' observed."
            else:
                decision = DecisionState.NORMAL
                escalate = False
                rationale = "Close or benign interaction within acceptable baseline variation."

        return LayaDecision(
            decision=decision,
            calibrated_confidence=round(calibrated_conf, 3),
            raw_confidence=round(raw_conf, 3),
            passes_quality_gate=gate_passed,
            quality_gate_details=gate_details,
            escalate_to_system2=escalate,
            rationale=rationale,
            timestamp=evidence.timestamp,
            frame_id=evidence.frame_id,
            metadata={"pattern": pattern.value}
        )
