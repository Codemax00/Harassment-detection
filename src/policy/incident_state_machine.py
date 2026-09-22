"""
Incident State Machine for Guardian Matrix.
Implements a 5-state hysteresis machine:
NORMAL -> OBSERVING -> SUSPECTED -> ACTIVE_INCIDENT -> COOLDOWN -> NORMAL.
Enforces dual-threshold hysteresis (enter at 0.80, exit below 0.45) and an uncertainty gate
for robust surveillance monitoring without alert flickering.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import time

from ..taxonomy.event_taxonomy import (
    EventCategory,
    ActionType,
    OperationalUrgency,
    EvidenceState,
    EventClassification
)
from ..temporal.temporal_decision_engine import TemporalDecisionOutput


@dataclass
class IncidentStateMachineConfig:
    """Configurable state machine thresholds and timing parameters."""
    observing_threshold: float = 0.40
    suspected_threshold: float = 0.60
    active_threshold: float = 0.80
    exit_threshold: float = 0.45
    cooldown_seconds: float = 1.5
    min_observation_dwell_seconds: float = 0.20
    min_suspected_dwell_seconds: float = 0.20


@dataclass
class IncidentStateTransition:
    """Record of a state transition for forensic timeline audits."""
    timestamp: float
    from_state: EvidenceState
    to_state: EvidenceState
    decision_score: float
    category: EventCategory
    action: ActionType
    reason: str


@dataclass
class StateMachineResult:
    """End-to-end outcome of the Incident State Machine."""
    state: EvidenceState
    category: EventCategory
    action: ActionType
    urgency: OperationalUrgency
    decision_score: float
    is_incident_active: bool
    requires_review: bool
    incident_id: Optional[str]
    transition_occurred: bool
    state_dwell_seconds: float
    active_track_ids: List[int]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "category": self.category.value,
            "action": self.action.value,
            "urgency": self.urgency.value,
            "decision_score": round(float(self.decision_score), 4),
            "is_incident_active": self.is_incident_active,
            "requires_review": self.requires_review,
            "incident_id": self.incident_id,
            "transition_occurred": self.transition_occurred,
            "state_dwell_seconds": round(float(self.state_dwell_seconds), 2),
            "active_track_ids": self.active_track_ids,
            "rationale": self.rationale
        }


class IncidentStateMachine:
    """
    Deterministic Incident State Machine with Hysteresis & Uncertainty Gate.
    """

    def __init__(self, config: Optional[IncidentStateMachineConfig] = None):
        self.config = config or IncidentStateMachineConfig()
        self.current_state: EvidenceState = EvidenceState.NORMAL
        self.state_entry_time: float = 0.0
        self.cooldown_start_time: Optional[float] = None
        
        self.active_incident_id: Optional[str] = None
        self.incident_counter: int = 0
        self.peak_decision_score: float = 0.0
        self.incident_start_time: float = 0.0
        
        self.transition_history: List[IncidentStateTransition] = []

    def update(self, decision: TemporalDecisionOutput) -> StateMachineResult:
        """
        Updates the incident state given the temporal decision output.
        """
        now = decision.timestamp
        score = decision.decision_score
        prev_state = self.current_state
        transition_occurred = False
        reason = ""

        # Uncertainty Gate
        is_uncertain = decision.is_uncertain

        if self.current_state == EvidenceState.NORMAL:
            if score >= self.config.active_threshold and not is_uncertain:
                # Immediate burst escalation if confidence is very high
                self._transition_to(EvidenceState.ACTIVE_INCIDENT, now, score, decision, "Immediate high confidence incident confirmation")
                transition_occurred = True
            elif score >= self.config.suspected_threshold:
                self._transition_to(EvidenceState.SUSPECTED, now, score, decision, f"Evidence exceeds suspected threshold ({score:.2f} >= {self.config.suspected_threshold})")
                transition_occurred = True
            elif score >= self.config.observing_threshold:
                self._transition_to(EvidenceState.OBSERVING, now, score, decision, f"Evidence exceeds observation threshold ({score:.2f} >= {self.config.observing_threshold})")
                transition_occurred = True

        elif self.current_state == EvidenceState.OBSERVING:
            dwell = now - self.state_entry_time
            if score >= self.config.active_threshold and not is_uncertain and dwell >= self.config.min_observation_dwell_seconds:
                self._transition_to(EvidenceState.ACTIVE_INCIDENT, now, score, decision, "Rapid elevation to active incident from observing")
                transition_occurred = True
            elif score >= self.config.suspected_threshold and dwell >= self.config.min_observation_dwell_seconds:
                self._transition_to(EvidenceState.SUSPECTED, now, score, decision, "Persistent elevated evidence progressed to suspected")
                transition_occurred = True
            elif score < self.config.observing_threshold:
                self._transition_to(EvidenceState.NORMAL, now, score, decision, "Evidence dropped below observing threshold")
                transition_occurred = True

        elif self.current_state == EvidenceState.SUSPECTED:
            dwell = now - self.state_entry_time
            if score >= self.config.active_threshold and not is_uncertain and dwell >= self.config.min_suspected_dwell_seconds:
                self._transition_to(EvidenceState.ACTIVE_INCIDENT, now, score, decision, "Confirmed persistent evidence triggered active incident")
                transition_occurred = True
            elif score < self.config.observing_threshold:
                self._transition_to(EvidenceState.NORMAL, now, score, decision, "Evidence dissipated, returning to normal")
                transition_occurred = True
            elif score < self.config.suspected_threshold:
                self._transition_to(EvidenceState.OBSERVING, now, score, decision, "Evidence dropped back into observing range")
                transition_occurred = True

        elif self.current_state == EvidenceState.ACTIVE_INCIDENT:
            self.peak_decision_score = max(self.peak_decision_score, score)
            # Hysteresis: remain active while score >= exit_threshold (0.45)
            if score < self.config.exit_threshold:
                self._transition_to(EvidenceState.COOLDOWN, now, score, decision, f"Evidence dropped below exit threshold ({score:.2f} < {self.config.exit_threshold})")
                self.cooldown_start_time = now
                transition_occurred = True

        elif self.current_state == EvidenceState.COOLDOWN:
            # If evidence spikes back up during cooldown, immediately re-arm incident
            if score >= self.config.suspected_threshold:
                self._transition_to(EvidenceState.ACTIVE_INCIDENT, now, score, decision, "Evidence re-elevated during cooldown period")
                transition_occurred = True
            else:
                cd_duration = (now - self.cooldown_start_time) if self.cooldown_start_time else 0.0
                if cd_duration >= self.config.cooldown_seconds:
                    self._transition_to(EvidenceState.NORMAL, now, score, decision, f"Cooldown expired ({cd_duration:.1f}s), returning to normal")
                    transition_occurred = True

        # Determine Urgency & Review Requirements
        is_incident_active = (self.current_state == EvidenceState.ACTIVE_INCIDENT)
        requires_review = bool(is_uncertain and score >= self.config.suspected_threshold)

        urgency = decision.urgency
        if is_incident_active:
            if decision.category == EventCategory.SAFETY_INCIDENT:
                urgency = OperationalUrgency.SAFETY_ALERT
            elif decision.category == EventCategory.AGGRESSIVE_INTERACTION:
                urgency = OperationalUrgency.CRITICAL_ALERT
        elif requires_review:
            urgency = OperationalUrgency.REVIEW_REQUIRED

        dwell_time = max(0.0, now - self.state_entry_time)

        rationale = f"State: {self.current_state.value} | Action: {decision.action.value} | Risk: {score:.2f}"
        if is_incident_active:
            rationale += f" [ACTIVE INCIDENT #{self.active_incident_id}]"
        elif requires_review:
            rationale += " [REVIEW REQUIRED - Observation Uncertainty]"

        return StateMachineResult(
            state=self.current_state,
            category=decision.category,
            action=decision.action,
            urgency=urgency,
            decision_score=score,
            is_incident_active=is_incident_active,
            requires_review=requires_review,
            incident_id=self.active_incident_id if is_incident_active else None,
            transition_occurred=transition_occurred,
            state_dwell_seconds=dwell_time,
            active_track_ids=decision.dominant_track_ids,
            rationale=rationale
        )

    def _transition_to(
        self,
        new_state: EvidenceState,
        timestamp: float,
        score: float,
        decision: TemporalDecisionOutput,
        reason: str
    ):
        old_state = self.current_state
        self.current_state = new_state
        self.state_entry_time = timestamp

        # Create or close incident ID
        if new_state == EvidenceState.ACTIVE_INCIDENT and old_state != EvidenceState.ACTIVE_INCIDENT:
            self.incident_counter += 1
            t_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp if timestamp > 100000 else time.time()))
            self.active_incident_id = f"INC-{t_str}-{self.incident_counter:04d}_{decision.action.value}"
            self.incident_start_time = timestamp
            self.peak_decision_score = score
        elif new_state == EvidenceState.NORMAL:
            self.active_incident_id = None
            self.cooldown_start_time = None

        transition = IncidentStateTransition(
            timestamp=timestamp,
            from_state=old_state,
            to_state=new_state,
            decision_score=score,
            category=decision.category,
            action=decision.action,
            reason=reason
        )
        self.transition_history.append(transition)
        if len(self.transition_history) > 100:
            self.transition_history.pop(0)

    def reset(self):
        """Reset state machine to initial clean state."""
        self.current_state = EvidenceState.NORMAL
        self.state_entry_time = 0.0
        self.cooldown_start_time = None
        self.active_incident_id = None
        self.peak_decision_score = 0.0
        self.incident_start_time = 0.0
        self.transition_history.clear()
