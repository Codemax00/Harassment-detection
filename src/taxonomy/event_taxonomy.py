"""
Event Taxonomy Definition for Guardian Matrix.
Defines two-tiered event categorization, operational urgencies, and incident state representations.
Strictly isolates SAFETY_INCIDENT (falls, collapses) from AGGRESSIVE_INTERACTION (strikes, kicks).
"""

from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple


class EventCategory(str, Enum):
    NORMAL_ACTIVITY = "NORMAL_ACTIVITY"
    SAFETY_INCIDENT = "SAFETY_INCIDENT"
    AGGRESSIVE_INTERACTION = "AGGRESSIVE_INTERACTION"
    UNKNOWN = "UNKNOWN"


class ActionType(str, Enum):
    # Normal Actions
    WALKING = "WALKING"
    RUNNING = "RUNNING"
    STANDING = "STANDING"
    SITTING = "SITTING"
    TALKING = "TALKING"
    
    # Safety Actions (Isolated from Aggression)
    FALL = "FALL"
    SLIP = "SLIP"
    COLLAPSE = "COLLAPSE"
    
    # Aggressive Actions
    STRIKE = "STRIKE"
    KICK = "KICK"
    PUSH = "PUSH"
    GRAB = "GRAB"
    RESTRAINT = "RESTRAINT"
    PURSUIT = "PURSUIT"
    
    # Fallback
    UNKNOWN = "UNKNOWN"


class OperationalUrgency(str, Enum):
    BENIGN = "BENIGN"
    MONITOR = "MONITOR"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    SAFETY_ALERT = "SAFETY_ALERT"
    CRITICAL_ALERT = "CRITICAL_ALERT"


class EvidenceState(str, Enum):
    NORMAL = "NORMAL"
    OBSERVING = "OBSERVING"
    SUSPECTED = "SUSPECTED"
    ACTIVE_INCIDENT = "ACTIVE_INCIDENT"
    COOLDOWN = "COOLDOWN"


# Backwards compatibility enum for existing codebase modules
class EventLabel(str, Enum):
    NORMAL_ACTIVITY = "NORMAL_ACTIVITY"
    CLOSE_INTERACTION = "CLOSE_INTERACTION"
    RAPID_APPROACH = "RAPID_APPROACH"
    RAPID_SEPARATION = "RAPID_SEPARATION"
    REPEATED_CONTACT_PATTERN = "REPEATED_CONTACT_PATTERN"
    AGGRESSIVE_MOTION_PATTERN = "AGGRESSIVE_MOTION_PATTERN"
    PURSUIT_PATTERN = "PURSUIT_PATTERN"
    FALL_PATTERN = "FALL_PATTERN"
    RESTRAINT_PATTERN = "RESTRAINT_PATTERN"
    UNCERTAIN_INTERACTION = "UNCERTAIN_INTERACTION"


@dataclass
class EventClassification:
    """Structured, JSON-serializable classification representation."""
    category: EventCategory
    action: ActionType
    urgency: OperationalUrgency
    confidence: float = 0.0
    evidence_state: EvidenceState = EvidenceState.NORMAL
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "action": self.action.value,
            "urgency": self.urgency.value,
            "confidence": round(float(self.confidence), 4),
            "evidence_state": self.evidence_state.value,
            "rationale": self.rationale
        }


class EventTaxonomy:
    """Central taxonomy rules and mapping utilities."""

    # Explicit mapping of actions to strictly defined categories
    ACTION_TO_CATEGORY: Dict[ActionType, EventCategory] = {
        ActionType.WALKING: EventCategory.NORMAL_ACTIVITY,
        ActionType.RUNNING: EventCategory.NORMAL_ACTIVITY,
        ActionType.STANDING: EventCategory.NORMAL_ACTIVITY,
        ActionType.SITTING: EventCategory.NORMAL_ACTIVITY,
        ActionType.TALKING: EventCategory.NORMAL_ACTIVITY,
        
        ActionType.FALL: EventCategory.SAFETY_INCIDENT,
        ActionType.SLIP: EventCategory.SAFETY_INCIDENT,
        ActionType.COLLAPSE: EventCategory.SAFETY_INCIDENT,
        
        ActionType.STRIKE: EventCategory.AGGRESSIVE_INTERACTION,
        ActionType.KICK: EventCategory.AGGRESSIVE_INTERACTION,
        ActionType.PUSH: EventCategory.AGGRESSIVE_INTERACTION,
        ActionType.GRAB: EventCategory.AGGRESSIVE_INTERACTION,
        ActionType.RESTRAINT: EventCategory.AGGRESSIVE_INTERACTION,
        ActionType.PURSUIT: EventCategory.AGGRESSIVE_INTERACTION,
        
        ActionType.UNKNOWN: EventCategory.UNKNOWN
    }

    # Operational urgency defaults
    ACTION_TO_URGENCY: Dict[ActionType, OperationalUrgency] = {
        ActionType.WALKING: OperationalUrgency.BENIGN,
        ActionType.RUNNING: OperationalUrgency.BENIGN,
        ActionType.STANDING: OperationalUrgency.BENIGN,
        ActionType.SITTING: OperationalUrgency.BENIGN,
        ActionType.TALKING: OperationalUrgency.BENIGN,
        
        ActionType.FALL: OperationalUrgency.SAFETY_ALERT,
        ActionType.SLIP: OperationalUrgency.SAFETY_ALERT,
        ActionType.COLLAPSE: OperationalUrgency.SAFETY_ALERT,
        
        ActionType.STRIKE: OperationalUrgency.CRITICAL_ALERT,
        ActionType.KICK: OperationalUrgency.CRITICAL_ALERT,
        ActionType.PUSH: OperationalUrgency.CRITICAL_ALERT,
        ActionType.GRAB: OperationalUrgency.CRITICAL_ALERT,
        ActionType.RESTRAINT: OperationalUrgency.CRITICAL_ALERT,
        ActionType.PURSUIT: OperationalUrgency.REVIEW_REQUIRED,
        
        ActionType.UNKNOWN: OperationalUrgency.BENIGN
    }

    # Backward compatibility mapping from legacy EventLabel
    LEGACY_LABEL_MAP: Dict[EventLabel, Tuple[EventCategory, ActionType]] = {
        EventLabel.NORMAL_ACTIVITY: (EventCategory.NORMAL_ACTIVITY, ActionType.WALKING),
        EventLabel.CLOSE_INTERACTION: (EventCategory.NORMAL_ACTIVITY, ActionType.STANDING),
        EventLabel.RAPID_APPROACH: (EventCategory.NORMAL_ACTIVITY, ActionType.RUNNING),
        EventLabel.RAPID_SEPARATION: (EventCategory.NORMAL_ACTIVITY, ActionType.RUNNING),
        EventLabel.REPEATED_CONTACT_PATTERN: (EventCategory.AGGRESSIVE_INTERACTION, ActionType.STRIKE),
        EventLabel.AGGRESSIVE_MOTION_PATTERN: (EventCategory.AGGRESSIVE_INTERACTION, ActionType.STRIKE),
        EventLabel.PURSUIT_PATTERN: (EventCategory.AGGRESSIVE_INTERACTION, ActionType.PURSUIT),
        EventLabel.FALL_PATTERN: (EventCategory.SAFETY_INCIDENT, ActionType.FALL),
        EventLabel.RESTRAINT_PATTERN: (EventCategory.AGGRESSIVE_INTERACTION, ActionType.RESTRAINT),
        EventLabel.UNCERTAIN_INTERACTION: (EventCategory.UNKNOWN, ActionType.UNKNOWN)
    }

    RISK_LEVELS: Dict[EventLabel, str] = {
        EventLabel.NORMAL_ACTIVITY: "low",
        EventLabel.CLOSE_INTERACTION: "low",
        EventLabel.RAPID_APPROACH: "medium",
        EventLabel.RAPID_SEPARATION: "medium",
        EventLabel.PURSUIT_PATTERN: "high",
        EventLabel.REPEATED_CONTACT_PATTERN: "high",
        EventLabel.AGGRESSIVE_MOTION_PATTERN: "high",
        EventLabel.RESTRAINT_PATTERN: "critical",
        EventLabel.FALL_PATTERN: "critical",
        EventLabel.UNCERTAIN_INTERACTION: "uncertain"
    }

    @classmethod
    def get_category(cls, action: ActionType) -> EventCategory:
        return cls.ACTION_TO_CATEGORY.get(action, EventCategory.UNKNOWN)

    @classmethod
    def get_default_urgency(cls, action: ActionType) -> OperationalUrgency:
        return cls.ACTION_TO_URGENCY.get(action, OperationalUrgency.BENIGN)

    @classmethod
    def is_safety_incident(cls, category: EventCategory) -> bool:
        return category == EventCategory.SAFETY_INCIDENT

    @classmethod
    def is_aggressive_interaction(cls, category: EventCategory) -> bool:
        return category == EventCategory.AGGRESSIVE_INTERACTION

    @classmethod
    def from_legacy_label(cls, label: EventLabel) -> EventClassification:
        cat, act = cls.LEGACY_LABEL_MAP.get(label, (EventCategory.UNKNOWN, ActionType.UNKNOWN))
        urg = cls.ACTION_TO_URGENCY.get(act, OperationalUrgency.BENIGN)
        return EventClassification(category=cat, action=act, urgency=urg)

    @classmethod
    def get_all_labels(cls) -> List[str]:
        return [e.value for e in EventLabel]

    @classmethod
    def get_risk_level(cls, label: EventLabel) -> str:
        return cls.RISK_LEVELS.get(label, "unknown")

    @classmethod
    def is_risk_event(cls, label: EventLabel) -> bool:
        return cls.RISK_LEVELS.get(label) in ("high", "critical")
