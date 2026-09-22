"""
Deterministic Safety Policy Layer for Guardian Matrix.
Separates AI inference from operational interventions as specified in Upgrade_Plan.md Section 19.
Maps System-1 & System-2 outputs to states: NORMAL, MONITOR, REVIEW, ALERT.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from ..evidence.structured_evidence import StructuredEvidence
from ..system1.laya_decision import LayaDecision, DecisionState
from ..system2.verifier import System2VerificationResult
from ..taxonomy.event_taxonomy import EventTaxonomy, EventLabel


class PolicyState(str, Enum):
    NORMAL = "NORMAL"
    MONITOR = "MONITOR"
    REVIEW = "REVIEW"
    ALERT = "ALERT"


@dataclass
class PolicyAction:
    """
    Action determined by deterministic safety policy.
    """
    state: PolicyState
    trigger_alert: bool
    request_human_review: bool
    save_evidence_snapshot: bool
    rationale: str
    target_event_label: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SafetyPolicy:
    """
    Evaluates combined System-1 decisions and System-2 verification results against safety rules.
    """

    def __init__(self, auto_alert_requires_human_confirmation: bool = False):
        self.auto_alert_requires_human_confirmation = auto_alert_requires_human_confirmation

    def evaluate_policy(
        self,
        evidence: StructuredEvidence,
        system1_decision: LayaDecision,
        system2_result: Optional[System2VerificationResult] = None
    ) -> PolicyAction:
        pattern = evidence.temporal_model.get("interaction_pattern", "NORMAL_ACTIVITY")

        # Case 1: System-1 NORMAL -> Keep state NORMAL
        if system1_decision.decision == DecisionState.NORMAL:
            return PolicyAction(
                state=PolicyState.NORMAL,
                trigger_alert=False,
                request_human_review=False,
                save_evidence_snapshot=False,
                rationale="Routine interaction observed.",
                target_event_label=pattern
            )

        # Case 2: System-1 UNCERTAIN -> Elevate to MONITOR or REVIEW
        if system1_decision.decision == DecisionState.UNCERTAIN:
            if system2_result and system2_result.is_verified:
                return PolicyAction(
                    state=PolicyState.REVIEW,
                    trigger_alert=False,
                    request_human_review=True,
                    save_evidence_snapshot=True,
                    rationale=f"System-1 was uncertain due to quality gates, but System-2 confirmed observable pattern: {pattern}.",
                    target_event_label=pattern
                )
            elif EventTaxonomy.is_risk_event(pattern):
                return PolicyAction(
                    state=PolicyState.REVIEW,
                    trigger_alert=False,
                    request_human_review=True,
                    save_evidence_snapshot=True,
                    rationale=f"Uncertain surveillance quality, but observable risk pattern '{pattern}' flagged for human verification.",
                    target_event_label=pattern
                )
            else:
                return PolicyAction(
                    state=PolicyState.MONITOR,
                    trigger_alert=False,
                    request_human_review=False,
                    save_evidence_snapshot=False,
                    rationale="Uncertain interaction under active monitoring.",
                    target_event_label=pattern
                )

        # Case 3: System-1 SUSPICIOUS
        if system1_decision.decision == DecisionState.SUSPICIOUS:
            if system2_result and system2_result.verified_decision == "BENIGN_INTERACTION":
                return PolicyAction(
                    state=PolicyState.MONITOR,
                    trigger_alert=False,
                    request_human_review=False,
                    save_evidence_snapshot=False,
                    rationale="System-2 verified benign transient interaction. Retained in MONITOR.",
                    target_event_label=pattern
                )
            return PolicyAction(
                state=PolicyState.REVIEW,
                trigger_alert=False,
                request_human_review=True,
                save_evidence_snapshot=True,
                rationale=f"Suspicious physical pattern '{pattern}' flagged for human verification.",
                target_event_label=pattern
            )

        # Case 4: System-1 REVIEW -> Confirmed high confidence risk
        if system1_decision.decision == DecisionState.REVIEW:
            if not self.auto_alert_requires_human_confirmation or (system2_result and system2_result.verified_decision == "CONFIRMED_RISK") or pattern in (EventLabel.AGGRESSIVE_MOTION_PATTERN.value, EventLabel.FALL_PATTERN.value, EventLabel.REPEATED_CONTACT_PATTERN.value):
                return PolicyAction(
                    state=PolicyState.ALERT,
                    trigger_alert=True,
                    request_human_review=True,
                    save_evidence_snapshot=True,
                    rationale=f"High-priority physical interaction alert: {pattern}. Evidence preserved for human review.",
                    target_event_label=pattern
                )
            else:
                return PolicyAction(
                    state=PolicyState.REVIEW,
                    trigger_alert=False,
                    request_human_review=True,
                    save_evidence_snapshot=True,
                    rationale=f"Observable pattern '{pattern}' scheduled for human review.",
                    target_event_label=pattern
                )

        return PolicyAction(
            state=PolicyState.MONITOR,
            trigger_alert=False,
            request_human_review=False,
            save_evidence_snapshot=False,
            rationale="Default safety policy state.",
            target_event_label=pattern
        )
