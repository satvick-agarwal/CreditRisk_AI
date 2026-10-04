"""
API endpoints for predictions, simulations, dashboard, model metrics, and portfolio analytics.
"""

import json
import os
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case, and_

from backend.app.core.database import get_db
from backend.app.core.config import get_settings
from backend.app.schemas.api import ApplicantInput, PredictionOutput, SimulationInput, SimulationOutput
from backend.app.models.database import Applicant, Prediction
from backend.app.services.ml_service import ml_service

settings = get_settings()
router = APIRouter(prefix="/api", tags=["api"])


# ===========================================================================
# PREDICTIONS
# ===========================================================================

@router.post("/predictions", response_model=PredictionOutput)
async def create_prediction(
    applicant: ApplicantInput,
    db: AsyncSession = Depends(get_db)
):
    """Predict credit risk for a new applicant and save the record."""
    try:
        result = ml_service.predict(applicant.model_dump())

        db_applicant = Applicant(**applicant.model_dump(exclude={"lgd_assumption"}))
        db.add(db_applicant)
        await db.flush()

        db_prediction = Prediction(
            applicant_id=db_applicant.id,
            model_version=result["model_version"],
            model_name=result["model_name"],
            input_features=applicant.model_dump(),
            probability_of_default=result["probability_of_default"],
            risk_score=result["risk_score"],
            risk_grade=result["risk_grade"],
            risk_level=result["risk_level"],
            decision=result["decision"],
            pd_decision=result["pd_decision"],
            decision_escalated=result["decision_escalated"],
            applied_policy_rules=result["applied_policy_rules"],
            audit_trail=result["audit_trail"],
            expected_loss=result["expected_loss"],
            lgd_used=result["lgd_used"],
            lgd_source=result["lgd_source"],
            ead=result["ead"],
            shap_contributions=result["shap_contributions"],
            is_simulation=False,
        )
        db.add(db_prediction)
        await db.commit()
        await db.refresh(db_prediction)

        return PredictionOutput(
            applicant_id=db_applicant.id,
            prediction_id=db_prediction.id,
            **result,
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# SIMULATION
# ===========================================================================

@router.post("/simulate", response_model=SimulationOutput)
async def simulate_risk(sim_input: SimulationInput):
    """Run a what-if simulation comparing two applicant states."""
    try:
        original_res = ml_service.predict(sim_input.original_application.model_dump())
        simulated_res = ml_service.predict(sim_input.modified_application.model_dump())

        deltas = {
            "pd_change": simulated_res["probability_of_default"] - original_res["probability_of_default"],
            "pd_change_pct": (
                (simulated_res["probability_of_default"] - original_res["probability_of_default"])
                / max(original_res["probability_of_default"], 0.0001)
            ),
            "risk_score_change": simulated_res["risk_score"] - original_res["risk_score"],
            "expected_loss_change": simulated_res["expected_loss"] - original_res["expected_loss"],
            "decision_changed": simulated_res["decision"] != original_res["decision"],
            "grade_changed": simulated_res["risk_grade"] != original_res["risk_grade"],
        }

        return SimulationOutput(
            original_prediction=PredictionOutput(**original_res),
            simulated_prediction=PredictionOutput(**simulated_res),
            deltas=deltas,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# DASHBOARD
# ===========================================================================

@router.get("/dashboard/summary")
async def get_dashboard_summary(db: AsyncSession = Depends(get_db)):
    """Portfolio KPIs from predictions or historical applicants."""
    try:
        pred_count = await db.scalar(select(func.count(Prediction.id)))

        if pred_count and pred_count > 0:
            query = select(
                func.count(Prediction.id).label("total"),
                func.avg(Prediction.probability_of_default).label("avg_pd"),
                func.avg(Prediction.risk_score).label("avg_score"),
                func.sum(Prediction.expected_loss).label("total_el"),
                func.sum(Prediction.ead).label("total_ead"),
                func.sum(case((Prediction.decision == "APPROVE", 1), else_=0)).label("approvals"),
                func.sum(case((Prediction.decision == "REJECT", 1), else_=0)).label("rejects"),
            ).where(Prediction.is_simulation == False)

            result = await db.execute(query)
            row = result.fetchone()
            total = row.total or 1
            approvals = row.approvals or 0
            rejects = row.rejects or 0

            return {
                "source": "live_predictions",
                "total_applications": total,
                "approval_rate": approvals / total,
                "rejection_rate": rejects / total,
                "review_rate": (total - approvals - rejects) / total,
                "average_pd": float(row.avg_pd or 0),
                "average_risk_score": float(row.avg_score or 0),
                "total_exposure": float(row.total_ead or 0),
                "total_expected_loss": float(row.total_el or 0),
            }
        else:
            query = select(
                func.count(Applicant.id).label("total"),
                func.avg(Applicant.loan_status).label("default_rate"),
                func.sum(Applicant.loan_amnt).label("total_exposure"),
            )
            result = await db.execute(query)
            row = result.fetchone()

            return {
                "source": "historical_baseline",
                "total_applications": row.total or 0,
                "approval_rate": 0.0,
                "rejection_rate": 0.0,
                "review_rate": 0.0,
                "average_pd": float(row.default_rate or 0),
                "average_risk_score": 0.0,
                "total_exposure": float(row.total_exposure or 0),
                "total_expected_loss": 0.0,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/risk-distribution")
async def get_risk_distribution(db: AsyncSession = Depends(get_db)):
    """Risk distribution across the applicant dataset."""
    try:
        # Use credit score bins to approximate risk distribution from applicants
        query = select(
            Applicant.credit_score,
            Applicant.loan_status,
            Applicant.loan_amnt,
            Applicant.person_income,
            Applicant.loan_intent,
            Applicant.person_home_ownership,
        )
        result = await db.execute(query)
        rows = result.fetchall()

        if not rows:
            return {"risk_bands": [], "by_intent": [], "by_credit": []}

        # Group by credit score bands
        credit_bands = {"Poor (300-579)": [], "Fair (580-669)": [], "Good (670-739)": [], "Very Good (740-799)": [], "Excellent (800-850)": []}
        intent_map = {}
        income_bands = {"<$30k": [], "$30k-$60k": [], "$60k-$100k": [], ">$100k": []}

        for row in rows:
            cs = row.credit_score or 0
            status = row.loan_status or 0
            intent = row.loan_intent or "Unknown"
            income = row.person_income or 0

            # Credit bands
            if cs < 580: credit_bands["Poor (300-579)"].append(status)
            elif cs < 670: credit_bands["Fair (580-669)"].append(status)
            elif cs < 740: credit_bands["Good (670-739)"].append(status)
            elif cs < 800: credit_bands["Very Good (740-799)"].append(status)
            else: credit_bands["Excellent (800-850)"].append(status)

            # Intent
            if intent not in intent_map:
                intent_map[intent] = []
            intent_map[intent].append(status)

            # Income bands
            if income < 30000: income_bands["<$30k"].append(status)
            elif income < 60000: income_bands["$30k-$60k"].append(status)
            elif income < 100000: income_bands["$60k-$100k"].append(status)
            else: income_bands[">$100k"].append(status)

        def band_stats(band_data):
            if not band_data:
                return {"count": 0, "default_rate": 0, "defaults": 0}
            return {
                "count": len(band_data),
                "default_rate": sum(band_data) / len(band_data),
                "defaults": sum(band_data),
            }

        return {
            "by_credit_score": {k: band_stats(v) for k, v in credit_bands.items()},
            "by_intent": {k: band_stats(v) for k, v in intent_map.items()},
            "by_income": {k: band_stats(v) for k, v in income_bands.items()},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# MODEL METRICS
# ===========================================================================

@router.get("/model/metrics")
async def get_model_metrics():
    """Return training metadata and evaluation metrics from saved artifacts."""
    try:
        metadata_path = os.path.join(settings.ML_ARTIFACTS_DIR, "metadata.json")
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail="Model metadata not found. Train the model first.")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return {
            "model_version": metadata.get("model_version"),
            "selected_model": metadata.get("selected_model"),
            "trained_at": metadata.get("trained_at"),
            "training_duration_seconds": metadata.get("training_duration_seconds"),
            "calibration": metadata.get("calibration"),
            "data_report": metadata.get("data_report"),
            "split_sizes": metadata.get("split_sizes"),
            "validation_metrics": metadata.get("validation_metrics"),
            "test_metrics": metadata.get("test_metrics"),
            "shap_feature_importance": metadata.get("shap_feature_importance"),
            "risk_grade_thresholds": metadata.get("risk_grade_thresholds"),
            "decision_config": metadata.get("decision_config"),
            "expected_loss_config": metadata.get("expected_loss_config"),
            "excluded_features": metadata.get("excluded_features"),
            "leakage_audit": metadata.get("leakage_audit"),
            "model_selection_note": metadata.get("model_selection_note"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/features")
async def get_model_features():
    """Return feature configuration used by the model."""
    try:
        config_path = os.path.join(settings.ML_ARTIFACTS_DIR, "feature_config.json")
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail="Feature config not found.")
        with open(config_path, "r") as f:
            return json.load(f)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# PORTFOLIO ANALYTICS
# ===========================================================================

@router.get("/portfolio/analytics")
async def get_portfolio_analytics(
    db: AsyncSession = Depends(get_db),
    credit_min: Optional[int] = Query(None),
    credit_max: Optional[int] = Query(None),
    income_min: Optional[float] = Query(None),
    income_max: Optional[float] = Query(None),
    intent: Optional[str] = Query(None),
    home_ownership: Optional[str] = Query(None),
    default_status: Optional[int] = Query(None),
):
    """Filterable portfolio analytics from the applicant dataset."""
    try:
        query = select(Applicant)
        conditions = []

        if credit_min is not None:
            conditions.append(Applicant.credit_score >= credit_min)
        if credit_max is not None:
            conditions.append(Applicant.credit_score <= credit_max)
        if income_min is not None:
            conditions.append(Applicant.person_income >= income_min)
        if income_max is not None:
            conditions.append(Applicant.person_income <= income_max)
        if intent:
            conditions.append(Applicant.loan_intent == intent)
        if home_ownership:
            conditions.append(Applicant.person_home_ownership == home_ownership)
        if default_status is not None:
            conditions.append(Applicant.loan_status == default_status)

        if conditions:
            query = query.where(and_(*conditions))

        result = await db.execute(query)
        applicants = result.scalars().all()

        if not applicants:
            return {
                "count": 0, "default_rate": 0, "avg_income": 0,
                "avg_loan": 0, "avg_credit_score": 0, "total_exposure": 0,
                "records": [],
            }

        defaults = sum(1 for a in applicants if a.loan_status == 1)
        total = len(applicants)

        return {
            "count": total,
            "default_rate": defaults / total,
            "defaults": defaults,
            "avg_income": sum(a.person_income for a in applicants) / total,
            "avg_loan": sum(a.loan_amnt for a in applicants) / total,
            "avg_credit_score": sum(a.credit_score for a in applicants if a.credit_score) / total,
            "total_exposure": sum(a.loan_amnt for a in applicants),
            "avg_interest_rate": sum(a.loan_int_rate or 0 for a in applicants) / total,
            "records": [
                {
                    "id": a.id,
                    "age": a.person_age,
                    "income": a.person_income,
                    "loan_amnt": a.loan_amnt,
                    "credit_score": a.credit_score,
                    "intent": a.loan_intent,
                    "home": a.person_home_ownership,
                    "default": a.loan_status,
                    "int_rate": a.loan_int_rate,
                }
                for a in applicants[:200]  # Limit to 200 for API performance
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# APPLICANT LOOKUP
# ===========================================================================

@router.get("/applicants/{applicant_id}")
async def get_applicant(applicant_id: int, db: AsyncSession = Depends(get_db)):
    """Fetch a single applicant by ID."""
    result = await db.execute(select(Applicant).where(Applicant.id == applicant_id))
    applicant = result.scalar_one_or_none()
    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")
    return {
        "id": applicant.id,
        "person_age": applicant.person_age,
        "person_income": applicant.person_income,
        "person_education": applicant.person_education,
        "person_emp_exp": applicant.person_emp_exp,
        "person_home_ownership": applicant.person_home_ownership,
        "loan_amnt": applicant.loan_amnt,
        "loan_intent": applicant.loan_intent,
        "loan_int_rate": applicant.loan_int_rate,
        "credit_score": applicant.credit_score,
        "cb_person_cred_hist_length": applicant.cb_person_cred_hist_length,
        "previous_loan_defaults_on_file": applicant.previous_loan_defaults_on_file,
        "loan_status": applicant.loan_status,
    }


@router.get("/applicants/{applicant_id}/risk")
async def get_applicant_risk(applicant_id: int, db: AsyncSession = Depends(get_db)):
    """Predict risk for an existing applicant in the database."""
    result = await db.execute(select(Applicant).where(Applicant.id == applicant_id))
    applicant = result.scalar_one_or_none()
    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")

    input_data = {
        "person_age": applicant.person_age,
        "person_income": applicant.person_income,
        "person_education": applicant.person_education,
        "person_emp_exp": applicant.person_emp_exp,
        "person_home_ownership": applicant.person_home_ownership,
        "loan_amnt": applicant.loan_amnt,
        "loan_intent": applicant.loan_intent,
        "loan_int_rate": applicant.loan_int_rate,
        "credit_score": applicant.credit_score,
        "cb_person_cred_hist_length": applicant.cb_person_cred_hist_length,
        "previous_loan_defaults_on_file": applicant.previous_loan_defaults_on_file,
    }

    prediction = ml_service.predict(input_data)
    return PredictionOutput(applicant_id=applicant_id, **prediction)
