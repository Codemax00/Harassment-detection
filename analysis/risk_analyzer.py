"""
Guardian Matrix - Multi-Signal Risk Scoring Module
Synthesizes detection confidence, pose dynamics, temporal kinematics,
pairwise interactions, and persistence into a calibrated risk score and evidence list.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from config.settings import settings


class RiskAnalyzerInterface(ABC):
    @abstractmethod
    def evaluate(self, tracks: List[Dict[str, Any]], interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass


class RiskAnalyzer(RiskAnalyzerInterface):
    """
    Evaluates multi-person safety risk from multi-signal telemetry.
    Ensures normal close proximity alone does NOT trigger high risk.
    Requires multiple corroborating signals: velocity, contact, pursuit, persistence.
    """

    def __init__(
        self,
        alert_confidence: Optional[float] = None,
        weight_proximity: float = 0.20,
        weight_approach: float = 0.25,
        weight_contact: float = 0.30,
        weight_pursuit: float = 0.25,
    ):
        self.alert_threshold = alert_confidence if alert_confidence is not None else settings.alert_confidence
        self.w_prox = weight_proximity
        self.w_app = weight_approach
        self.w_cont = weight_contact
        self.w_purs = weight_pursuit

    def evaluate(
        self,
        tracks: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluates current risk state across all detected interactions.
        Returns:
        {
            "risk_score": float (0.0 - 1.0),
            "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
            "evidence": List[str],
            "track_ids": List[int],
            "duration": float
        }
        """
        if not tracks or not interactions:
            return {
                "risk_score": 0.0,
                "risk_level": "LOW",
                "evidence": ["normal activity"],
                "track_ids": [t["track_id"] for t in tracks] if tracks else [],
                "duration": 0.0,
            }

        max_risk = 0.0
        primary_evidence: List[str] = []
        primary_tracks: List[int] = []
        max_duration = 0.0

        # Build track velocity lookup
        track_vels = {t["track_id"]: t.get("velocity", 0.0) for t in tracks}

        for inter in interactions:
            p_a = inter.get("person_a", 0)
            p_b = inter.get("person_b", 0)
            dist = inter.get("distance", 100.0)
            rel_vel = inter.get("relative_velocity", 0.0)
            duration = inter.get("duration", 0.0)
            contact_prob = inter.get("contact_probability", 0.0)
            pursuit_score = inter.get("pursuit_score", 0.0)
            repeated_contact = inter.get("repeated_contact_score", 0.0)

            vel_a = track_vels.get(p_a, 0.0)
            vel_b = track_vels.get(p_b, 0.0)
            max_vel = max(vel_a, vel_b)

            # CRITICAL GROUNDING FOR HUMAN SAFETY:
            # Peaceful coexistence (sitting, lying, resting together, looking at phones):
            # If both people are stationary or moving gently (velocity < 30 px/s and rel_vel > -15.0 px/s),
            # this is PEACEFUL ACTIVITY with ZERO safety risk.
            if max_vel < 30.0 and rel_vel > -15.0 and pursuit_score < 0.3:
                pair_risk = 0.0
                evidence_items = ["peaceful co-presence"]
            else:
                # Kinetic signals indicating genuine aggression or assault:
                # 1. Violent / Rapid Approach: closing distance at high speed (<= -20 px/s)
                app_score = 0.0
                if rel_vel <= -20.0:
                    app_score = min(1.0, abs(rel_vel) / 35.0)

                # 2. Aggressive contact requires high velocity motion or rapid closure
                cont_score = 0.0
                if (max_vel >= 30.0 or abs(rel_vel) >= 20.0) and (contact_prob > 0.3 or repeated_contact > 0.3):
                    cont_score = max(contact_prob, repeated_contact)

                # 3. Pursuit requires rapid chasing motion
                purs_score = pursuit_score if (max_vel >= 30.0 or abs(rel_vel) >= 20.0) else 0.0

                # 4. Proximity is only a risk multiplier if kinetic aggression exists
                prox_score = max(0.0, min(1.0, (150.0 - min(dist, 150.0)) / 150.0))

                if app_score == 0.0 and cont_score == 0.0 and purs_score == 0.0:
                    pair_risk = 0.0
                    evidence_items = ["normal movement"]
                else:
                    pair_risk = (
                        self.w_prox * prox_score
                        + self.w_app * app_score
                        + self.w_cont * cont_score
                        + self.w_purs * purs_score
                    )

                    # Escalate if multiple aggressive kinetic signals combine
                    if (app_score > 0.4 and cont_score > 0.4) or (cont_score > 0.6 and duration > 3.0):
                        pair_risk = min(1.0, pair_risk * 1.35)

                pair_risk = round(float(min(1.0, max(0.0, pair_risk))), 2)

                evidence_items = []
                if app_score > 0.4:
                    evidence_items.append("rapid aggressive approach")
                if purs_score > 0.4:
                    evidence_items.append("pursuit trajectory")
                if cont_score > 0.35:
                    evidence_items.append("forceful contact pattern")
                if not evidence_items:
                    evidence_items = ["normal activity"]

            if pair_risk > max_risk:
                max_risk = pair_risk
                primary_evidence = evidence_items
                primary_tracks = [p_a, p_b]
                max_duration = duration

        if not primary_evidence:
            primary_evidence = ["normal activity"]

        # Map to calibrated Risk Level
        if max_risk >= self.alert_threshold:
            risk_level = "CRITICAL" if max_risk >= 0.90 else "HIGH"
        elif max_risk >= 0.50:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "risk_score": max_risk,
            "risk_level": risk_level,
            "evidence": primary_evidence,
            "track_ids": primary_tracks,
            "duration": round(max_duration, 1),
        }
