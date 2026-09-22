"""
Guardian Matrix - Alert Manager
Implements temporal frame confirmation (e.g. 10 consecutive high-risk frames),
cooldown suppression (30s), incident logging, and dispatching handlers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
import time
from datetime import datetime
from config.settings import settings


class AlertManagerInterface(ABC):
    @abstractmethod
    def evaluate(self, risk_result: Dict[str, Any], timestamp: Optional[float] = None) -> Optional[Dict[str, Any]]:
        pass


class AlertManager(AlertManagerInterface):
    """
    Alert Manager preventing single-frame false positives via consecutive frame confirmation
    and avoiding duplicate alert floods via cooldown timer.
    """

    def __init__(
        self,
        confirmation_frames: Optional[int] = None,
        cooldown_seconds: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
    ):
        self.confirmation_frames = (
            confirmation_frames if confirmation_frames is not None else settings.alert_confirmation_frames
        )
        self.cooldown_seconds = (
            cooldown_seconds if cooldown_seconds is not None else settings.alert_cooldown_seconds
        )
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else settings.alert_confidence
        )

        self.consecutive_qualifying_frames = 0
        self.last_alert_time: float = -999.0
        self.alert_history: List[Dict[str, Any]] = []
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []

    def add_listener(self, callback: Callable[[Dict[str, Any]], None]):
        """Register external listener/webhook/siren callback."""
        self._listeners.append(callback)

    def evaluate(self, risk_result: Dict[str, Any], timestamp: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Evaluates current risk evaluation result.
        Returns Alert dict if alert is triggered; otherwise None.
        """
        now = timestamp if timestamp is not None else time.time()
        score = risk_result.get("risk_score", 0.0)

        # Check if current frame qualifies as high risk
        if score >= self.confidence_threshold:
            self.consecutive_qualifying_frames += 1
        else:
            # Decay or reset counter if risk drops
            self.consecutive_qualifying_frames = max(0, self.consecutive_qualifying_frames - 2)
            return None

        # Check if consecutive confirmation threshold reached
        if self.consecutive_qualifying_frames < self.confirmation_frames:
            return None

        # Check cooldown period
        time_since_last_alert = now - self.last_alert_time
        if time_since_last_alert < self.cooldown_seconds:
            # Cooldown active, suppress alert spam
            return None

        # Trigger new confirmed alert!
        self.last_alert_time = now
        incident_id = f"INCIDENT_{int(now)}"
        alert = {
            "incident_id": incident_id,
            "timestamp": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            "risk_score": score,
            "risk_level": risk_result.get("risk_level", "HIGH"),
            "evidence": risk_result.get("evidence", []),
            "track_ids": risk_result.get("track_ids", []),
            "duration": risk_result.get("duration", 0.0),
            "consecutive_frames": self.consecutive_qualifying_frames,
        }

        self.alert_history.append(alert)

        # Dispatch to registered listeners
        for listener in self._listeners:
            try:
                listener(alert)
            except Exception as e:
                print(f"[WARNING] Alert listener error: {e}")

        # Structured log output
        tracks_str = " ".join(f"Person {tid}" for tid in alert["track_ids"])
        print(f"\n>>> [ALERT TRIGGERED] ID: {incident_id} | Risk: {alert['risk_level']} ({int(score*100)}%) | Tracks: {tracks_str} | Evidence: {', '.join(alert['evidence'])}")

        return alert

    def is_in_cooldown(self, timestamp: Optional[float] = None) -> bool:
        now = timestamp if timestamp is not None else time.time()
        return (now - self.last_alert_time) < self.cooldown_seconds

    def cooldown_remaining(self, timestamp: Optional[float] = None) -> float:
        now = timestamp if timestamp is not None else time.time()
        rem = self.cooldown_seconds - (now - self.last_alert_time)
        return max(0.0, round(rem, 1))
