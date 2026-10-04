"""
ML Pipeline Configuration
=========================
Central configuration for the credit risk ML pipeline.
All thresholds, assumptions, and parameters are documented here.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from datetime import datetime
import uuid

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "loan_data.csv")
CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_data.csv")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "ml", "artifacts")

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Model version — stamped on every training run
# ---------------------------------------------------------------------------
def generate_model_version() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"v1.0_{timestamp}_{short_id}"

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    """
    Defines which columns to use, which to exclude, and why.
    """

    target: str = "loan_status"

    # Features excluded from modelling with documented rationale
    excluded_features: Dict[str, str] = field(default_factory=lambda: {
        "person_gender": (
            "Protected characteristic. Using gender in credit-risk models "
            "raises fairness and legal concerns (Equal Credit Opportunity Act / "
            "similar regulations). Excluded by design."
        ),
        "loan_status": "Target variable — must never appear in features.",
    })

    # Numeric features used for modelling
    numeric_features: List[str] = field(default_factory=lambda: [
        "person_age",
        "person_income",
        "person_emp_exp",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "credit_score",
    ])

    # Categorical features used for modelling
    categorical_features: List[str] = field(default_factory=lambda: [
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "previous_loan_defaults_on_file",
    ])

    # Engineered features created during preprocessing
    engineered_features: List[str] = field(default_factory=lambda: [
        "loan_percent_income",          # loan_amnt / person_income
        "income_to_loan_rate_ratio",    # person_income / (loan_amnt * loan_int_rate / 100)
        "credit_score_bin",             # binned credit score category
        "emp_stability",                # employment experience relative to age
    ])

    # Feature leakage audit notes
    leakage_audit: Dict[str, str] = field(default_factory=lambda: {
        "loan_int_rate": (
            "POTENTIAL CONCERN: In production lending, the interest rate may be "
            "determined AFTER the risk assessment (i.e., it is a consequence of risk, "
            "not a predictor). In this dataset it appears as an application-level "
            "feature. We include it with this documented caveat. If this feature "
            "shows disproportionate importance, investigate further."
        ),
        "loan_percent_income": (
            "Derived from loan_amnt and person_income, both of which are also "
            "features. This creates multicollinearity but is NOT target leakage. "
            "The ratio is more informative than the raw values for risk assessment. "
            "Retained deliberately."
        ),
        "previous_loan_defaults_on_file": (
            "Historical information available before the current lending decision. "
            "Not leakage. Strongly predictive — expected and acceptable."
        ),
    })

    @property
    def all_model_features(self) -> List[str]:
        return self.numeric_features + self.categorical_features


# ---------------------------------------------------------------------------
# Data validation rules
# ---------------------------------------------------------------------------
@dataclass
class DataValidationConfig:
    age_range: Tuple[int, int] = (18, 100)
    credit_score_range: Tuple[int, int] = (300, 850)
    min_income: float = 0
    max_loan_percent_income: float = 5.0   # flag, don't silently remove
    min_loan_amount: float = 0


# ---------------------------------------------------------------------------
# Risk scoring configuration
# ---------------------------------------------------------------------------
@dataclass
class RiskScoreConfig:
    """
    Risk Score Methodology
    ----------------------
    The risk score is a PERCENTILE-BASED mapping of the Probability of Default
    (PD) against the training population's PD distribution.

    Score = percentile_rank(applicant_PD, training_PD_distribution) on [0, 100]

    Interpretation: "This applicant is riskier than {score}% of the portfolio."

    This approach:
    - Provides genuine discrimination beyond raw PD
    - Is relative to the observed portfolio distribution
    - Has good separation across the risk spectrum
    - Is interpretable and defensible
    - Must be recalibrated when the portfolio materially changes

    The score feeds into Risk Grades, which are threshold-based categories.
    """

    # Risk grade thresholds (score boundaries)
    risk_grades: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "A": (0, 20),     # Very Low Risk
        "B": (21, 40),    # Low Risk
        "C": (41, 60),    # Moderate Risk
        "D": (61, 80),    # High Risk
        "E": (81, 100),   # Very High Risk
    })

    risk_grade_labels: Dict[str, str] = field(default_factory=lambda: {
        "A": "Very Low Risk",
        "B": "Low Risk",
        "C": "Moderate Risk",
        "D": "High Risk",
        "E": "Very High Risk",
    })

    # Number of percentile points to store for interpolation
    n_percentile_points: int = 1000


# ---------------------------------------------------------------------------
# Decision engine configuration
# ---------------------------------------------------------------------------
@dataclass
class DecisionConfig:
    """
    Decision Engine
    ---------------
    The decision engine combines ML-based PD thresholds with a configurable
    policy layer. The ML model NEVER directly returns APPROVE/REJECT.

    Flow:
        PD → Risk Score → PD Threshold Check → Policy Layer → Final Decision

    PD Thresholds (configurable):
        PD < approve_threshold          → APPROVE
        approve_threshold ≤ PD < reject → REVIEW
        PD ≥ reject_threshold           → REJECT

    Policy Layer:
        Rules that can ESCALATE a decision (never downgrade).
        Each rule is documented with its rationale.
        No hard overrides — everything is auditable.
    """

    approve_threshold: float = 0.15      # PD below this → APPROVE
    reject_threshold: float = 0.45       # PD at or above this → REJECT
    # Between approve and reject → MANUAL REVIEW

    # Policy rules: each can escalate the decision
    # Format: { rule_name: { description, condition_description, escalation } }
    policy_rules: Dict[str, Dict] = field(default_factory=lambda: {
        "previous_defaults_escalation": {
            "description": (
                "Applicants with previous defaults on file receive an escalation "
                "to at least MANUAL REVIEW regardless of model PD. Rationale: "
                "prior default history is a strong real-world risk signal that "
                "warrants human review even when the model's PD is low."
            ),
            "field": "previous_loan_defaults_on_file",
            "condition": "== 'Yes'",
            "min_decision": "REVIEW",   # escalate to at least this level
        },
        "very_low_credit_score": {
            "description": (
                "Applicants with credit scores below 500 receive escalation to "
                "at least MANUAL REVIEW. Rationale: extremely low credit scores "
                "indicate severe credit distress."
            ),
            "field": "credit_score",
            "condition": "< 500",
            "min_decision": "REVIEW",
        },
        "extreme_loan_burden": {
            "description": (
                "Applicants requesting loans exceeding 50% of annual income "
                "receive escalation to at least MANUAL REVIEW. Rationale: "
                "very high loan-to-income ratios indicate potential repayment stress."
            ),
            "field": "loan_percent_income",
            "condition": "> 0.5",
            "min_decision": "REVIEW",
        },
    })


# ---------------------------------------------------------------------------
# Expected Loss configuration
# ---------------------------------------------------------------------------
@dataclass
class ExpectedLossConfig:
    """
    Expected Loss = PD × LGD × EAD

    Where:
    - PD  = Probability of Default (from ML model)
    - LGD = Loss Given Default (assumed — see below)
    - EAD = Exposure at Default (= loan amount for this dataset)

    LGD ASSUMPTION:
        We use 45% as a CONFIGURABLE DEFAULT assumption.
        This is NOT presented as a universal truth.

        Context: Basel II IRB Foundation prescribes 45% LGD for senior
        unsecured exposures, but this is a regulatory floor, not an
        empirical estimate. Actual LGD varies by:
        - Collateral type and quality
        - Recovery process and jurisdiction
        - Economic cycle (downturn vs. upturn)
        - Seniority of the claim

        Since this dataset contains no recovery/collateral data,
        45% is used as a reasonable starting assumption for
        unsecured consumer lending. It MUST be clearly labelled
        as an assumption in all outputs.

        Users can override this value.
    """

    default_lgd: float = 0.45
    lgd_label: str = "Assumed (no recovery data in dataset)"


# ---------------------------------------------------------------------------
# Model training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    test_size: float = 0.20       # Held-out test set (protected, evaluated ONCE)
    val_size: float = 0.20        # Validation set (from remaining 80%)
    random_seed: int = RANDOM_SEED
    cv_folds: int = 5             # Cross-validation folds on training set
    scoring_metric: str = "roc_auc"

    # We do NOT pre-select a winner. All models are trained and compared.
    # The best model is selected based on validation set performance.
    model_selection_note: str = (
        "Model selection is based on validation set ROC-AUC as the primary metric, "
        "with PR-AUC and recall as secondary considerations. No model is assumed to "
        "win a priori. The final test set is used ONLY for the selected model's "
        "final performance report."
    )


# ---------------------------------------------------------------------------
# Calibration configuration
# ---------------------------------------------------------------------------
@dataclass
class CalibrationConfig:
    """
    Calibration is EVALUATED, not blindly applied.

    Process:
    1. Train model on training set
    2. Compute Brier score and reliability diagram on validation set
    3. If Brier score improves with Platt scaling or isotonic regression, apply it
    4. If calibration does NOT improve, use raw probabilities
    5. Document the decision either way

    We use the validation set (not test set) for calibration evaluation.
    """

    methods: List[str] = field(default_factory=lambda: ["sigmoid", "isotonic"])
    n_bins_reliability: int = 10
    # Minimum Brier score improvement to justify calibration
    min_brier_improvement: float = 0.001


# ---------------------------------------------------------------------------
# Aggregate config
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    validation: DataValidationConfig = field(default_factory=DataValidationConfig)
    risk_score: RiskScoreConfig = field(default_factory=RiskScoreConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    expected_loss: ExpectedLossConfig = field(default_factory=ExpectedLossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    model_version: str = field(default_factory=generate_model_version)
