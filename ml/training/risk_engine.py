"""
Risk Engine
===========
Risk scoring, risk grading, decision engine, and expected loss.
All business logic for transforming ML predictions into actionable outputs.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class RiskScorer:
    """
    Percentile-based risk scoring.

    Maps an applicant's PD to a 0–100 score based on where it falls
    in the training population's PD distribution.

    Score interpretation:
        "This applicant is riskier than {score}% of the training portfolio."

    This is fundamentally different from PD × 100 because:
    - It provides distributional context
    - It has better discrimination in the dense middle range
    - It is relative to the actual portfolio, not an arbitrary scale
    """

    def __init__(self, n_points: int = 1000):
        self.n_points = n_points
        self.percentile_values: Optional[np.ndarray] = None
        self.percentile_points: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, training_pds: np.ndarray) -> "RiskScorer":
        """
        Compute percentile breakpoints from the training PD distribution.
        Must be called once during training, then persisted.
        """
        self.percentile_points = np.linspace(0, 100, self.n_points + 1)
        self.percentile_values = np.percentile(training_pds, self.percentile_points)
        self._is_fitted = True
        return self

    def score(self, pd_value: float) -> int:
        """
        Map a PD to a risk score (0–100).
        Uses linear interpolation against training percentiles.
        """
        if not self._is_fitted:
            raise RuntimeError("RiskScorer must be fitted before scoring.")

        score = np.interp(pd_value, self.percentile_values, self.percentile_points)
        return int(np.clip(np.round(score), 0, 100))

    def score_batch(self, pd_values: np.ndarray) -> np.ndarray:
        """Score multiple PDs at once."""
        if not self._is_fitted:
            raise RuntimeError("RiskScorer must be fitted before scoring.")
        scores = np.interp(pd_values, self.percentile_values, self.percentile_points)
        return np.clip(np.round(scores), 0, 100).astype(int)

    def get_state(self) -> Dict:
        return {
            "percentile_points": self.percentile_points.tolist(),
            "percentile_values": self.percentile_values.tolist(),
            "n_points": self.n_points,
        }

    @classmethod
    def from_state(cls, state: Dict) -> "RiskScorer":
        scorer = cls(n_points=state["n_points"])
        scorer.percentile_points = np.array(state["percentile_points"])
        scorer.percentile_values = np.array(state["percentile_values"])
        scorer._is_fitted = True
        return scorer


class RiskGrader:
    """
    Maps risk scores (0–100) to letter grades (A through E).
    """

    DEFAULT_GRADES = {
        "A": (0, 20, "Very Low Risk"),
        "B": (21, 40, "Low Risk"),
        "C": (41, 60, "Moderate Risk"),
        "D": (61, 80, "High Risk"),
        "E": (81, 100, "Very High Risk"),
    }

    def __init__(self, grades: Optional[Dict] = None):
        self.grades = grades or self.DEFAULT_GRADES

    def grade(self, risk_score: int) -> Dict[str, str]:
        for letter, (low, high, label) in self.grades.items():
            if low <= risk_score <= high:
                return {"grade": letter, "label": label}
        return {"grade": "E", "label": "Very High Risk"}

    def get_risk_level(self, risk_score: int) -> str:
        """Return the risk level label for a given score."""
        return self.grade(risk_score)["label"]

    def get_color(self, grade: str) -> str:
        colors = {
            "A": "#10b981",  # green
            "B": "#22c55e",  # light green
            "C": "#f59e0b",  # amber
            "D": "#f97316",  # orange
            "E": "#ef4444",  # red
        }
        return colors.get(grade, "#6b7280")


class DecisionEngine:
    """
    Deterministic decision engine with a configurable policy layer.

    The decision flow:
        1. PD → threshold-based initial decision (APPROVE / REVIEW / REJECT)
        2. Policy rules → can ESCALATE the decision (never downgrade)
        3. All decisions and applied rules are logged for auditability

    This replaces hard-coded overrides with a documented, auditable policy system.
    """

    DECISION_HIERARCHY = {"APPROVE": 0, "REVIEW": 1, "REJECT": 2}

    def __init__(
        self,
        approve_threshold: float = 0.15,
        reject_threshold: float = 0.45,
        policy_rules: Optional[Dict] = None,
    ):
        self.approve_threshold = approve_threshold
        self.reject_threshold = reject_threshold
        self.policy_rules = policy_rules or {}

    def _pd_decision(self, pd_value: float) -> str:
        if pd_value < self.approve_threshold:
            return "APPROVE"
        elif pd_value >= self.reject_threshold:
            return "REJECT"
        else:
            return "REVIEW"

    def _escalate(self, current: str, target: str) -> str:
        """Escalate to the higher-severity decision."""
        if self.DECISION_HIERARCHY.get(target, 0) > self.DECISION_HIERARCHY.get(current, 0):
            return target
        return current

    def decide(self, pd_value: float, applicant_data: Dict) -> Dict[str, Any]:
        """
        Make a decision for an applicant.

        Returns:
            {
                "decision": "APPROVE" | "REVIEW" | "REJECT",
                "pd_decision": str,          # decision from PD alone
                "applied_rules": list,       # rules that triggered
                "escalated": bool,           # whether policy escalated
                "audit_trail": list,         # full audit trail
            }
        """
        pd_decision = self._pd_decision(pd_value)
        final_decision = pd_decision
        applied_rules = []
        audit_trail = [
            {
                "step": "pd_threshold",
                "pd": round(pd_value, 6),
                "thresholds": {
                    "approve": self.approve_threshold,
                    "reject": self.reject_threshold,
                },
                "result": pd_decision,
            }
        ]

        # Apply policy rules
        for rule_name, rule_config in self.policy_rules.items():
            field_name = rule_config.get("field")
            condition = rule_config.get("condition", "")
            min_decision = rule_config.get("min_decision", "REVIEW")

            if field_name not in applicant_data:
                continue

            value = applicant_data[field_name]
            triggered = False

            # Evaluate condition
            if condition.startswith("=="):
                target = condition[2:].strip().strip("'\"")
                triggered = str(value) == target
            elif condition.startswith(">="):
                triggered = float(value) >= float(condition[2:].strip())
            elif condition.startswith("<="):
                triggered = float(value) <= float(condition[2:].strip())
            elif condition.startswith(">"):
                triggered = float(value) > float(condition[1:].strip())
            elif condition.startswith("<"):
                triggered = float(value) < float(condition[1:].strip())

            if triggered:
                old_decision = final_decision
                final_decision = self._escalate(final_decision, min_decision)
                applied_rules.append({
                    "rule": rule_name,
                    "description": rule_config.get("description", ""),
                    "field": field_name,
                    "value": value,
                    "condition": condition,
                    "escalated_to": min_decision,
                })
                audit_trail.append({
                    "step": "policy_rule",
                    "rule": rule_name,
                    "triggered": True,
                    "field": field_name,
                    "value": str(value),
                    "before": old_decision,
                    "after": final_decision,
                })

        return {
            "decision": final_decision,
            "pd_decision": pd_decision,
            "applied_rules": applied_rules,
            "escalated": final_decision != pd_decision,
            "audit_trail": audit_trail,
        }


class ExpectedLossCalculator:
    """
    Expected Loss = PD × LGD × EAD

    LGD ASSUMPTION NOTICE:
        LGD is configurable and defaults to 0.45.
        This is an assumption, not an empirical estimate.
        See ml/config.py ExpectedLossConfig for full documentation.
    """

    def __init__(self, default_lgd: float = 0.45):
        self.default_lgd = default_lgd

    def calculate(
        self,
        pd_value: float,
        ead: float,
        lgd: Optional[float] = None,
    ) -> Dict[str, Any]:
        lgd_used = lgd if lgd is not None else self.default_lgd
        expected_loss = pd_value * lgd_used * ead

        return {
            "expected_loss": round(expected_loss, 2),
            "pd": round(pd_value, 6),
            "lgd": round(lgd_used, 4),
            "lgd_source": "user-provided" if lgd is not None else "assumed",
            "ead": round(ead, 2),
            "formula": "EL = PD × LGD × EAD",
            "assumption_notice": (
                "LGD is assumed at {:.0%} due to absence of recovery data. "
                "Actual loss may differ materially.".format(lgd_used)
            ) if lgd is None else None,
        }

    def calculate_batch(
        self,
        pd_values: np.ndarray,
        ead_values: np.ndarray,
        lgd: Optional[float] = None,
    ) -> np.ndarray:
        lgd_used = lgd if lgd is not None else self.default_lgd
        return pd_values * lgd_used * ead_values
