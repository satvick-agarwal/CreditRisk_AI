"""
SQLAlchemy database models.
Stores applicants, predictions (with model version for auditability), and portfolio metrics.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, JSON, Boolean
from backend.app.core.database import Base


class Applicant(Base):
    """Loan applicant record. One row per loan application."""
    __tablename__ = "applicants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_age = Column(Float, nullable=False)
    person_gender = Column(String(20))           # stored but NOT used in model
    person_education = Column(String(50))
    person_income = Column(Float, nullable=False)
    person_emp_exp = Column(Float)
    person_home_ownership = Column(String(30))
    loan_amnt = Column(Float, nullable=False)
    loan_intent = Column(String(50))
    loan_int_rate = Column(Float)
    loan_percent_income = Column(Float)
    cb_person_cred_hist_length = Column(Float)
    credit_score = Column(Integer)
    previous_loan_defaults_on_file = Column(String(10))
    loan_status = Column(Integer)                # 0 = no default, 1 = default
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Prediction(Base):
    """
    Every prediction is stored with its model version for full auditability.
    This allows:
    - Tracing any past decision back to the exact model that made it
    - Comparing predictions across model versions
    - Regulatory/audit compliance
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    applicant_id = Column(Integer, nullable=True)  # null for ad-hoc predictions

    # Model auditability
    model_version = Column(String(100), nullable=False)
    model_name = Column(String(50))

    # Input features snapshot (for reproducibility)
    input_features = Column(JSON)

    # ML output
    probability_of_default = Column(Float, nullable=False)
    risk_score = Column(Integer)
    risk_grade = Column(String(5))
    risk_level = Column(String(30))

    # Decision engine output
    decision = Column(String(20))                 # APPROVE / REVIEW / REJECT
    pd_decision = Column(String(20))              # decision from PD alone
    decision_escalated = Column(Boolean, default=False)
    applied_policy_rules = Column(JSON)           # which rules triggered
    audit_trail = Column(JSON)                    # full decision audit trail

    # Financial metrics
    expected_loss = Column(Float)
    lgd_used = Column(Float)
    lgd_source = Column(String(30))               # "assumed" or "user-provided"
    ead = Column(Float)

    # SHAP explanation
    shap_contributions = Column(JSON)             # {feature: contribution}

    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_simulation = Column(Boolean, default=False)


class ModelRun(Base):
    """Records each model training run for version tracking."""
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(100), nullable=False, unique=True)
    model_name = Column(String(50))
    trained_at = Column(DateTime)
    test_roc_auc = Column(Float)
    test_pr_auc = Column(Float)
    test_f1 = Column(Float)
    test_recall = Column(Float)
    calibration_applied = Column(Boolean, default=False)
    calibration_method = Column(String(30))
    metadata_json = Column(JSON)
    is_active = Column(Boolean, default=True)    # currently serving model
