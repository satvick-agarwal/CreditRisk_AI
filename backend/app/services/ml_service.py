"""
ML Inference Service.
Loads persisted artifacts and performs predictions.
"""

import os
import json
import logging
import joblib
import pandas as pd
from typing import Dict, Any, List

from backend.app.core.config import get_settings
from ml.training.risk_engine import RiskGrader, DecisionEngine, ExpectedLossCalculator

settings = get_settings()
logger = logging.getLogger(__name__)


class MLService:
    def __init__(self):
        self.is_loaded = False
        self.model = None
        self.preprocessor = None
        self.risk_scorer = None
        self.metadata = None
        self.feature_config = None
        
        # Risk Engine components
        self.risk_grader = RiskGrader()
        self.decision_engine = None
        self.el_calculator = ExpectedLossCalculator()

    def load_artifacts(self):
        """Load all ML artifacts into memory."""
        try:
            logger.info(f"Loading ML artifacts from {settings.ML_ARTIFACTS_DIR}")
            
            # Load metadata to verify version and get configs
            with open(os.path.join(settings.ML_ARTIFACTS_DIR, "metadata.json"), "r") as f:
                self.metadata = json.load(f)
                
            with open(os.path.join(settings.ML_ARTIFACTS_DIR, "feature_config.json"), "r") as f:
                self.feature_config = json.load(f)
            
            # Load models
            self.model = joblib.load(os.path.join(settings.ML_ARTIFACTS_DIR, "model.joblib"))
            self.preprocessor = joblib.load(os.path.join(settings.ML_ARTIFACTS_DIR, "preprocessor.joblib"))
            self.risk_scorer = joblib.load(os.path.join(settings.ML_ARTIFACTS_DIR, "risk_scorer.joblib"))
            
            # Initialize decision engine with config from training time
            dc = self.metadata.get("decision_config", {})
            self.decision_engine = DecisionEngine(
                approve_threshold=dc.get("approve_threshold", 0.15),
                reject_threshold=dc.get("reject_threshold", 0.45),
                policy_rules=dc.get("policy_rules", {})
            )
            
            self.is_loaded = True
            logger.info(f"Successfully loaded model version: {self.metadata.get('model_version')}")
            
        except Exception as e:
            logger.error(f"Failed to load ML artifacts: {e}")
            self.is_loaded = False

    def predict(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run full prediction pipeline on a single applicant."""
        if not self.is_loaded:
            self.load_artifacts()
            if not self.is_loaded:
                raise RuntimeError("ML artifacts not loaded.")

        # Prepare input dataframe
        df = pd.DataFrame([applicant_data])
        
        # Missing feature handling / basic feature engineering
        # Match training time engineering
        if "loan_percent_income" not in df.columns and "loan_amnt" in df.columns and "person_income" in df.columns:
            df["loan_percent_income"] = df["loan_amnt"] / df["person_income"].clip(lower=1)
            
        if "emp_stability" not in df.columns and "person_emp_exp" in df.columns and "person_age" in df.columns:
            df["emp_stability"] = (df["person_emp_exp"] / (df["person_age"] - 18).clip(lower=1)).clip(upper=1.0)
            
        if "income_to_loan_rate_ratio" not in df.columns and "person_income" in df.columns and "loan_amnt" in df.columns and "loan_int_rate" in df.columns:
            loan_cost = (df["loan_amnt"] * df["loan_int_rate"] / 100).clip(lower=1)
            df["income_to_loan_rate_ratio"] = (df["person_income"] / loan_cost).clip(upper=100)

        # Subset to required features
        input_features = self.feature_config["input_features"]
        df_model = df[input_features].copy()

        # Transform and Predict
        X_proc = self.preprocessor.transform(df_model)
        pd_value = float(self.model.predict_proba(X_proc)[0, 1])
        
        # Scoring
        risk_score = self.risk_scorer.score(pd_value)
        grade_info = self.risk_grader.grade(risk_score)
        
        # Decision
        decision_result = self.decision_engine.decide(pd_value, applicant_data)
        
        # Expected Loss
        ead = applicant_data.get("loan_amnt", 0)
        lgd_override = applicant_data.get("lgd_assumption")
        el_result = self.el_calculator.calculate(pd_value, ead, lgd_override)
        
        # SHAP calculation
        shap_contributions = self._compute_local_shap(X_proc)

        return {
            "model_version": self.metadata.get("model_version"),
            "model_name": self.metadata.get("selected_model"),
            "probability_of_default": pd_value,
            "risk_score": risk_score,
            "risk_grade": grade_info["grade"],
            "risk_level": grade_info["label"],
            "decision": decision_result["decision"],
            "pd_decision": decision_result["pd_decision"],
            "decision_escalated": decision_result["escalated"],
            "applied_policy_rules": decision_result["applied_rules"],
            "audit_trail": decision_result["audit_trail"],
            "expected_loss": el_result["expected_loss"],
            "lgd_used": el_result["lgd"],
            "lgd_source": el_result["lgd_source"],
            "ead": el_result["ead"],
            "shap_contributions": shap_contributions
        }

    def _compute_local_shap(self, X_proc) -> Dict[str, float]:
        """Compute SHAP values for a single prediction if available."""
        # For simplicity and speed in API, we'll use a fast approximation or pre-computed global importances 
        # combined with local feature values if exact SHAP is too slow.
        # But we'll try to load SHAP if available.
        import shap
        
        feature_names = self.feature_config["preprocessed_feature_names"]
        
        try:
            # TreeExplainer is fast enough for real-time
            if self.metadata.get("selected_model") in ("xgboost", "lightgbm", "random_forest"):
                # Needs the base model. If wrapped in CalibratedClassifierCV, we extract it.
                base_model = self.model
                if hasattr(self.model, "calibrated_classifiers_"):
                    # Use the first calibrated classifier's base model as an approximation
                    base_model = self.model.calibrated_classifiers_[0].estimator
                
                explainer = shap.TreeExplainer(base_model)
                shap_vals = explainer.shap_values(X_proc)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                
                contributions = dict(zip(feature_names, shap_vals[0].tolist()))
                # Sort by absolute magnitude
                return dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))
                
        except Exception as e:
            logger.warning(f"Failed to compute local SHAP: {e}")
            
        # Fallback to returning global feature importance
        return {}


# Singleton instance
ml_service = MLService()
