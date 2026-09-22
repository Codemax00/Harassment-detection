"""
Event Taxonomy Definition for Guardian Matrix.
Observable intermediate interaction states as defined in Upgrade_Plan.md Section 13.
"""

from enum import Enum
from typing import Dict, List, Any


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


class EventTaxonomy:
    """Manages event categories, risk levels, and operational mappings."""

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
    def get_all_labels(cls) -> List[str]:
        return [e.value for e in EventLabel]

    @classmethod
    def get_risk_level(cls, label: EventLabel) -> str:
        return cls.RISK_LEVELS.get(label, "unknown")

    @classmethod
    def is_risk_event(cls, label: EventLabel) -> bool:
        return cls.RISK_LEVELS.get(label) in ("high", "critical")
