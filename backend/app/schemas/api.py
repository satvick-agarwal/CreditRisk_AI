"""
Pydantic schemas for API validation.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class ApplicantInput(BaseModel):
    """Input schema for a loan application."""
    person_age: float = Field(..., ge=18, le=100, description="Applicant age")
    person_gender: Optional[str] = Field(None, description="Gender (not used in ML model)")
    person_education: str = Field(..., description="Highest education level")
    person_income: float = Field(..., ge=0, description="Annual income")
    person_emp_exp: float = Field(..., ge=0, description="Employment experience in years")
    person_home_ownership: str = Field(..., description="Housing status (RENT, OWN, MORTGAGE, OTHER)")
    
    loan_amnt: float = Field(..., gt=0, description="Requested loan amount")
    loan_intent: str = Field(..., description="Purpose of the loan")
    loan_int_rate: float = Field(..., ge=0, description="Interest rate")
    
    cb_person_cred_hist_length: float = Field(..., ge=0, description="Credit history length in years")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    previous_loan_defaults_on_file: str = Field(..., description="'Yes' or 'No'")
    
    # LGD assumption can be overridden by user
    lgd_assumption: Optional[float] = Field(None, ge=0, le=1, description="Optional Loss Given Default assumption")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "person_age": 28,
                "person_education": "Bachelor",
                "person_income": 65000,
                "person_emp_exp": 5,
                "person_home_ownership": "RENT",
                "loan_amnt": 15000,
                "loan_intent": "PERSONAL",
                "loan_int_rate": 11.5,
                "cb_person_cred_hist_length": 4,
                "credit_score": 680,
                "previous_loan_defaults_on_file": "No"
            }
        }
    )


class SimulationInput(BaseModel):
    """Input schema for what-if simulation."""
    original_application: ApplicantInput
    modified_application: ApplicantInput


class RuleAudit(BaseModel):
    rule: str
    description: str
    field: str
    value: Any
    condition: str
    escalated_to: str


class PredictionOutput(BaseModel):
    """Structured output for a prediction."""
    # IDs
    applicant_id: Optional[int] = None
    prediction_id: Optional[int] = None
    
    # Model info
    model_version: str
    model_name: str
    
    # ML Risk Assessment
    probability_of_default: float = Field(..., description="Raw PD from ML model")
    risk_score: int = Field(..., description="Percentile-based score 0-100")
    risk_grade: str = Field(..., description="A-E grade")
    risk_level: str = Field(..., description="Human readable risk level")
    
    # Decision Engine
    decision: str = Field(..., description="Final decision: APPROVE, REVIEW, or REJECT")
    pd_decision: str = Field(..., description="Decision based purely on PD threshold")
    decision_escalated: bool = Field(..., description="True if policy layer escalated the decision")
    applied_policy_rules: List[RuleAudit] = Field(default_factory=list)
    
    # Financial Metrics
    expected_loss: float
    lgd_used: float
    lgd_source: str
    ead: float
    
    # Explainability
    shap_contributions: Dict[str, float] = Field(default_factory=dict)


class SimulationOutput(BaseModel):
    """Output for what-if simulation."""
    original_prediction: PredictionOutput
    simulated_prediction: PredictionOutput
    deltas: Dict[str, Any] = Field(
        ..., 
        description="Changes in key metrics (pd, risk_score, decision, expected_loss)"
    )


class PortfolioMetrics(BaseModel):
    """Aggregated portfolio metrics."""
    total_applications: int
    approval_rate: float
    review_rate: float
    rejection_rate: float
    average_pd: float
    average_risk_score: float
    total_exposure: float
    total_expected_loss: float
