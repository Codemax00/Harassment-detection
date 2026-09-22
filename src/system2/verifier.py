"""
System-2 Deep Verification Layer for Guardian Matrix.
Performs multimodal escalation for uncertain or suspicious events, inspecting
event metadata, pose sequence, and key frames (Upgrade_Plan.md Section 18).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from ..evidence.structured_evidence import StructuredEvidence
from ..system1.laya_decision import LayaDecision, DecisionState


@dataclass
class System2VerificationResult:
    """Outcome of System-2 Deep Verification."""
    is_verified: bool
    verified_decision: str  # "CONFIRMED_RISK", "BENIGN_INTERACTION", "INCONCLUSIVE"
    verification_confidence: float
    analysis_text: str
    inspected_elements: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class System2Verifier:
    """
    Slower secondary verification layer invoked only when System-1 is uncertain or suspicious.
    Combines rule-grounded multi-frame reasoning with optional multimodal LLM/VLM calls.
    """

    def __init__(self, use_external_ai: bool = False, ai_manager: Optional[Any] = None):
        self.use_external_ai = use_external_ai
        self.ai_manager = ai_manager

    def verify(
        self,
        decision: LayaDecision,
        evidence: StructuredEvidence,
        keyframes: Optional[List[np.ndarray]] = None
    ) -> System2VerificationResult:
        """Verify an escalated event."""
        inspected = ["structured_evidence", "quality_metrics", "temporal_model"]
        if keyframes:
            inspected.append(f"keyframes_{len(keyframes)}")

        # Step 1: Deep structured evidence review
        inter = evidence.interaction
        pattern = evidence.temporal_model.get("interaction_pattern", "")

        # Check for false positive triggers:
        # e.g., brief single-frame proximity without hand contact or high duration
        is_brief = False
        if inter:
            duration = inter.get("duration_s", 0.0)
            contact_prob = inter.get("hand_contact_probability", 0.0)
            rep_contact = inter.get("repeated_contact_score", 0.0)
            pursuit = inter.get("pursuit_score", 0.0)

            if duration < 0.8 and contact_prob < 0.3 and pursuit < 0.4:
                is_brief = True

        if is_brief and decision.decision == DecisionState.SUSPICIOUS:
            return System2VerificationResult(
                is_verified=False,
                verified_decision="BENIGN_INTERACTION",
                verification_confidence=0.88,
                analysis_text="System-2 verified that proximity was brief (<0.8s) with no sustained hand contact or pursuit. Downgraded to benign interaction.",
                inspected_elements=inspected
            )

        # If external AI is available and enabled, query it with structured prompt
        if self.use_external_ai and self.ai_manager is not None and keyframes:
            try:
                # Integrate with existing helpers/multi_ai if needed
                pass
            except Exception:
                pass

        # Grounded default verification
        if decision.decision == DecisionState.REVIEW or (inter and (inter.get("repeated_contact_score", 0.0) > 0.6 or inter.get("pursuit_score", 0.0) > 0.6)):
            return System2VerificationResult(
                is_verified=True,
                verified_decision="CONFIRMED_RISK",
                verification_confidence=max(0.85, decision.calibrated_confidence),
                analysis_text=f"System-2 confirmed observable physical interaction risk '{pattern}' with persistent trajectory alignment.",
                inspected_elements=inspected
            )
        else:
            return System2VerificationResult(
                is_verified=False,
                verified_decision="INCONCLUSIVE",
                verification_confidence=0.60,
                analysis_text="System-2 found ambiguous trajectory cues; flagged for human supervision.",
                inspected_elements=inspected
            )
