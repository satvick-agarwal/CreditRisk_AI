"""
Credit Risk ML Training Pipeline
=================================
Trains, evaluates, calibrates, and persists credit risk models.

Process:
    1. Load and validate data
    2. Audit for feature leakage
    3. Engineer features
    4. Split: Train / Validation / Test (test is PROTECTED)
    5. Train 4 models (LR, RF, XGBoost, LightGBM)
    6. Evaluate all on VALIDATION set only
    7. Select best model (no pre-assumption about which wins)
    8. Evaluate calibration — apply ONLY if it measurably improves
    9. Final evaluation on TEST set (one time only)
    10. Compute SHAP explanations
    11. Fit risk scorer on training PD distribution
    12. Persist all artifacts with version metadata

Usage:
    python -m ml.training.pipeline
"""

import os
import sys
import json
import time
import warnings
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    log_loss,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed. Skipping XGBoost model.")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("WARNING: lightgbm not installed. Skipping LightGBM model.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. SHAP explanations will be skipped.")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.config import PipelineConfig, ARTIFACTS_DIR, RAW_DATA_PATH, RANDOM_SEED
from ml.training.risk_engine import RiskScorer, RiskGrader, DecisionEngine, ExpectedLossCalculator

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# 1. DATA LOADING & VALIDATION
# ===========================================================================

def load_and_validate_data(config: PipelineConfig) -> pd.DataFrame:
    """Load raw data and run validation checks."""
    logger.info(f"Loading data from {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # --- Data quality report ---
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "class_distribution": df[config.features.target].value_counts().to_dict(),
    }

    # Validate ranges
    val = config.validation
    issues = []

    age_invalid = df[(df["person_age"] < val.age_range[0]) | (df["person_age"] > val.age_range[1])]
    if len(age_invalid) > 0:
        issues.append(f"  {len(age_invalid)} rows with age outside {val.age_range}")

    cs_invalid = df[(df["credit_score"] < val.credit_score_range[0]) | (df["credit_score"] > val.credit_score_range[1])]
    if len(cs_invalid) > 0:
        issues.append(f"  {len(cs_invalid)} rows with credit_score outside {val.credit_score_range}")

    # Check employment experience vs age
    emp_age_issue = df[df["person_emp_exp"] > df["person_age"] - 16]
    if len(emp_age_issue) > 0:
        issues.append(f"  {len(emp_age_issue)} rows where emp_exp > age - 16 (suspicious)")

    # Check income
    income_zero = df[df["person_income"] <= 0]
    if len(income_zero) > 0:
        issues.append(f"  {len(income_zero)} rows with income <= 0")

    report["validation_issues"] = issues

    logger.info("=== DATA QUALITY REPORT ===")
    logger.info(f"  Rows: {report['total_rows']}")
    logger.info(f"  Duplicates: {report['duplicates']}")
    logger.info(f"  Missing: {sum(v for v in report['missing_values'].values() if v > 0)} total")
    logger.info(f"  Class 0 (no default): {report['class_distribution'].get(0, 0)}")
    logger.info(f"  Class 1 (default): {report['class_distribution'].get(1, 0)}")
    for issue in issues:
        logger.warning(f"  VALIDATION: {issue}")

    # --- Feature leakage audit ---
    logger.info("=== FEATURE LEAKAGE AUDIT ===")
    for feature, note in config.features.leakage_audit.items():
        logger.info(f"  [{feature}]: {note[:100]}...")

    logger.info("=== EXCLUDED FEATURES ===")
    for feature, reason in config.features.excluded_features.items():
        logger.info(f"  [{feature}]: {reason[:80]}...")

    return df, report


# ===========================================================================
# 2. FEATURE ENGINEERING
# ===========================================================================

def engineer_features(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Create engineered features. Operates on a copy."""
    df = df.copy()

    # Recompute loan_percent_income (in case raw data has stale values)
    df["loan_percent_income"] = df["loan_amnt"] / df["person_income"].clip(lower=1)

    # Employment stability: emp_exp / (age - 18), capped at 1.0
    df["emp_stability"] = (df["person_emp_exp"] / (df["person_age"] - 18).clip(lower=1)).clip(upper=1.0)

    # Income-to-loan-rate ratio: how well income covers loan cost
    loan_cost = (df["loan_amnt"] * df["loan_int_rate"] / 100).clip(lower=1)
    df["income_to_loan_rate_ratio"] = (df["person_income"] / loan_cost).clip(upper=100)

    # Credit score bins (for potential use, not as model feature to avoid multicollinearity)
    df["credit_score_bin"] = pd.cut(
        df["credit_score"],
        bins=[0, 580, 670, 740, 800, 850],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
        include_lowest=True,
    )

    return df


# ===========================================================================
# 3. PREPROCESSING
# ===========================================================================

def build_preprocessor(config: PipelineConfig) -> ColumnTransformer:
    """Build sklearn preprocessing pipeline."""
    numeric_features = [
        "person_age",
        "person_income",
        "person_emp_exp",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "credit_score",
        "emp_stability",
        "income_to_loan_rate_ratio",
    ]

    categorical_features = [
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "previous_loan_defaults_on_file",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features, categorical_features


# ===========================================================================
# 4. MODEL DEFINITIONS
# ===========================================================================

def get_models(config: PipelineConfig) -> dict:
    """
    Define candidate models.
    No model is assumed to win a priori.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            C=0.5,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3.5,  # approx ratio of negatives to positives
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

    if HAS_LIGHTGBM:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            verbose=-1,
        )

    return models


# ===========================================================================
# 5. EVALUATION
# ===========================================================================

def evaluate_model(y_true, y_pred, y_prob, model_name: str) -> dict:
    """Compute comprehensive evaluation metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
        "log_loss": round(log_loss(y_true, y_prob), 4),
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_negative_rate": round(fn / (fn + tp) if (fn + tp) > 0 else 0, 4),
        "false_positive_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
    }
    return metrics


def compute_calibration_data(y_true, y_prob, n_bins: int = 10) -> dict:
    """Compute calibration curve data for reliability diagram."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return {
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist(),
        "n_bins": n_bins,
    }


# ===========================================================================
# 6. CALIBRATION EVALUATION
# ===========================================================================

def evaluate_calibration(
    model,
    preprocessor,
    X_train, y_train,
    X_val, y_val,
    model_name: str,
    config: PipelineConfig,
) -> tuple:
    """
    Evaluate whether calibration improves the model.
    Returns (best_model_or_calibrated, calibration_report).
    """
    # Get raw probabilities on validation set
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    raw_probs = model.predict_proba(X_val_proc)[:, 1]
    raw_brier = brier_score_loss(y_val, raw_probs)

    calibration_report = {
        "raw_brier_score": round(raw_brier, 6),
        "calibration_applied": False,
        "calibration_method": None,
        "calibrated_brier_score": None,
        "improvement": None,
        "decision": None,
    }

    best_model = model
    best_brier = raw_brier
    best_method = None

    for method in config.calibration.methods:
        try:
            cal_model = CalibratedClassifierCV(
                model, method=method, cv=3
            )
            cal_model.fit(X_train_proc, y_train)
            cal_probs = cal_model.predict_proba(X_val_proc)[:, 1]
            cal_brier = brier_score_loss(y_val, cal_probs)

            logger.info(f"  Calibration [{method}]: Brier {raw_brier:.6f} → {cal_brier:.6f}")

            if cal_brier < best_brier - config.calibration.min_brier_improvement:
                best_brier = cal_brier
                best_model = cal_model
                best_method = method
        except Exception as e:
            logger.warning(f"  Calibration [{method}] failed: {e}")

    if best_method:
        calibration_report.update({
            "calibration_applied": True,
            "calibration_method": best_method,
            "calibrated_brier_score": round(best_brier, 6),
            "improvement": round(raw_brier - best_brier, 6),
            "decision": f"Applied {best_method} calibration (Brier improved by {raw_brier - best_brier:.6f})",
        })
        logger.info(f"  ✓ Calibration APPLIED ({best_method}): Brier {raw_brier:.6f} → {best_brier:.6f}")
    else:
        calibration_report["decision"] = (
            "Calibration NOT applied — no method improved Brier score beyond "
            f"threshold ({config.calibration.min_brier_improvement})"
        )
        logger.info(f"  ✗ Calibration NOT applied — raw probabilities are adequate")

    return best_model, calibration_report


# ===========================================================================
# 7. SHAP EXPLANATIONS
# ===========================================================================

def compute_shap_explanations(model, preprocessor, X_val, feature_names, model_name):
    """Compute SHAP values for model explanations."""
    if not HAS_SHAP:
        logger.warning("SHAP not available. Skipping explanations.")
        return None, None

    logger.info("Computing SHAP explanations...")
    X_val_proc = preprocessor.transform(X_val)

    try:
        if model_name in ("xgboost", "lightgbm", "random_forest"):
            # TreeExplainer is exact and fast for tree-based models
            explainer = shap.TreeExplainer(model)
        else:
            # Use a sample for LinearExplainer or KernelExplainer
            explainer = shap.LinearExplainer(model, X_val_proc)

        shap_values = explainer.shap_values(X_val_proc)

        # For binary classification, some explainers return a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 (default)

        # Global feature importance (mean absolute SHAP)
        shap_importance = np.abs(shap_values).mean(axis=0)
        feature_importance = sorted(
            zip(feature_names, shap_importance.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        logger.info(f"  SHAP computed for {len(feature_names)} features")
        return shap_values, feature_importance

    except Exception as e:
        logger.error(f"  SHAP computation failed: {e}")
        # Fallback: use model feature importances if available
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            feature_importance = sorted(
                zip(feature_names, importance.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
            return None, feature_importance
        return None, None


# ===========================================================================
# 8. MAIN PIPELINE
# ===========================================================================

def run_pipeline():
    """Execute the full training pipeline."""
    start_time = time.time()
    config = PipelineConfig()
    logger.info(f"=== CREDIT RISK ML PIPELINE ===")
    logger.info(f"Model Version: {config.model_version}")
    logger.info(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # --- 1. Load & validate ---
    df, data_report = load_and_validate_data(config)

    # --- 2. Clean data ---
    logger.info("=== DATA CLEANING ===")
    original_len = len(df)

    # Remove invalid ages (documented, not silent)
    age_mask = (df["person_age"] >= 18) & (df["person_age"] <= 100)
    df = df[age_mask]
    logger.info(f"  Removed {original_len - len(df)} rows with invalid age")

    # Handle missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.info(f"  Filling {missing.sum()} missing values")
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remove duplicates
    dups = df.duplicated().sum()
    if dups > 0:
        df = df.drop_duplicates()
        logger.info(f"  Removed {dups} duplicate rows")

    logger.info(f"  Final dataset: {len(df)} rows")

    # --- 3. Feature engineering ---
    logger.info("=== FEATURE ENGINEERING ===")
    df = engineer_features(df, config)

    # --- 4. Prepare features ---
    preprocessor, numeric_features, categorical_features = build_preprocessor(config)
    all_features = numeric_features + categorical_features

    # Verify all features exist
    missing_features = [f for f in all_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")

    X = df[all_features]
    y = df[config.features.target]

    logger.info(f"  Features: {len(all_features)} ({len(numeric_features)} numeric, {len(categorical_features)} categorical)")
    logger.info(f"  Target distribution: {y.value_counts().to_dict()}")

    # --- 5. Train / Validation / Test split ---
    # TEST SET IS PROTECTED — used ONLY for final evaluation
    logger.info("=== DATA SPLITTING ===")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config.training.test_size,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Split trainval into train and validation
    val_fraction = config.training.val_size / (1 - config.training.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction,
        random_state=RANDOM_SEED,
        stratify=y_trainval,
    )

    logger.info(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    logger.info(f"  Test set is PROTECTED — will only be used for final model evaluation")

    # --- 6. Fit preprocessor on TRAINING data only ---
    preprocessor.fit(X_train)
    feature_names_out = preprocessor.get_feature_names_out().tolist()
    logger.info(f"  Preprocessor fitted on training data → {len(feature_names_out)} output features")

    # --- 7. Train and evaluate all models on VALIDATION set ---
    logger.info("=== MODEL TRAINING & VALIDATION ===")
    models = get_models(config)
    results = {}
    trained_models = {}

    for name, model in models.items():
        logger.info(f"\n  Training: {name}")
        t0 = time.time()

        # Train
        X_train_proc = preprocessor.transform(X_train)
        model.fit(X_train_proc, y_train)

        # Evaluate on VALIDATION set (never test)
        X_val_proc = preprocessor.transform(X_val)
        y_val_prob = model.predict_proba(X_val_proc)[:, 1]

        # Use 0.5 threshold for basic metrics
        y_val_pred = (y_val_prob >= 0.5).astype(int)

        metrics = evaluate_model(y_val, y_val_pred, y_val_prob, name)
        calibration_data = compute_calibration_data(y_val, y_val_prob)
        metrics["calibration_curve"] = calibration_data
        metrics["training_time_seconds"] = round(time.time() - t0, 2)

        results[name] = metrics
        trained_models[name] = model

        logger.info(
            f"    ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"PR-AUC: {metrics['pr_auc']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"Brier: {metrics['brier_score']:.4f} | "
            f"FNR: {metrics['false_negative_rate']:.4f}"
        )

    # --- 8. Model selection ---
    # Primary: ROC-AUC, Secondary: PR-AUC, Tiebreaker: Recall
    logger.info("\n=== MODEL SELECTION ===")
    logger.info("  Selection criteria: ROC-AUC (primary), PR-AUC (secondary), Recall (tiebreaker)")
    logger.info("  No model assumed to win a priori.")

    best_name = max(
        results.keys(),
        key=lambda k: (results[k]["roc_auc"], results[k]["pr_auc"], results[k]["recall"]),
    )
    best_model = trained_models[best_name]
    best_metrics = results[best_name]

    logger.info(f"  ★ Selected: {best_name}")
    logger.info(f"    ROC-AUC: {best_metrics['roc_auc']} | PR-AUC: {best_metrics['pr_auc']} | Recall: {best_metrics['recall']}")

    # --- 9. Calibration evaluation ---
    logger.info("\n=== CALIBRATION EVALUATION ===")
    logger.info("  Evaluating whether calibration improves the selected model...")

    final_model, calibration_report = evaluate_calibration(
        best_model, preprocessor,
        X_train, y_train,
        X_val, y_val,
        best_name, config,
    )

    # --- 10. FINAL TEST SET EVALUATION (ONE TIME ONLY) ---
    logger.info("\n=== FINAL TEST SET EVALUATION ===")
    logger.info("  This is the ONLY evaluation on the protected test set.")

    X_test_proc = preprocessor.transform(X_test)
    if calibration_report["calibration_applied"]:
        y_test_prob = final_model.predict_proba(X_test_proc)[:, 1]
    else:
        y_test_prob = best_model.predict_proba(X_test_proc)[:, 1]

    y_test_pred = (y_test_prob >= 0.5).astype(int)
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_prob, f"{best_name}_test")
    test_calibration = compute_calibration_data(y_test, y_test_prob)
    test_metrics["calibration_curve"] = test_calibration

    logger.info(
        f"  TEST ROC-AUC: {test_metrics['roc_auc']:.4f} | "
        f"PR-AUC: {test_metrics['pr_auc']:.4f} | "
        f"F1: {test_metrics['f1']:.4f} | "
        f"Recall: {test_metrics['recall']:.4f} | "
        f"Brier: {test_metrics['brier_score']:.4f}"
    )

    # --- 11. SHAP explanations ---
    logger.info("\n=== SHAP EXPLANATIONS ===")
    shap_values, shap_feature_importance = compute_shap_explanations(
        best_model, preprocessor, X_val, feature_names_out, best_name,
    )

    # If SHAP failed, use model feature importances as fallback
    if shap_feature_importance is None and hasattr(best_model, "feature_importances_"):
        shap_feature_importance = sorted(
            zip(feature_names_out, best_model.feature_importances_.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        logger.info("  Using model's native feature importances as fallback")

    # --- 12. Risk scorer ---
    logger.info("\n=== RISK SCORING ===")
    X_train_proc = preprocessor.transform(X_train)
    if calibration_report["calibration_applied"]:
        train_pds = final_model.predict_proba(X_train_proc)[:, 1]
    else:
        train_pds = best_model.predict_proba(X_train_proc)[:, 1]

    risk_scorer = RiskScorer(n_points=config.risk_score.n_percentile_points)
    risk_scorer.fit(train_pds)
    logger.info(f"  Risk scorer fitted on {len(train_pds)} training PDs")
    logger.info(f"  PD range: [{train_pds.min():.4f}, {train_pds.max():.4f}]")
    logger.info(f"  PD median: {np.median(train_pds):.4f}")

    # --- 13. Save artifacts ---
    logger.info("\n=== SAVING ARTIFACTS ===")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Save model
    model_to_save = final_model if calibration_report["calibration_applied"] else best_model
    joblib.dump(model_to_save, os.path.join(ARTIFACTS_DIR, "model.joblib"))
    logger.info("  Saved: model.joblib")

    # Save preprocessor
    joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))
    logger.info("  Saved: preprocessor.joblib")

    # Save risk scorer
    joblib.dump(risk_scorer, os.path.join(ARTIFACTS_DIR, "risk_scorer.joblib"))
    logger.info("  Saved: risk_scorer.joblib")

    # Save SHAP values (sample for memory efficiency)
    if shap_values is not None:
        shap_sample_size = min(500, len(shap_values))
        shap_sample = {
            "values": shap_values[:shap_sample_size].tolist(),
            "feature_names": feature_names_out,
        }
        with open(os.path.join(ARTIFACTS_DIR, "shap_sample.json"), "w") as f:
            json.dump(shap_sample, f)
        logger.info("  Saved: shap_sample.json")

    # Save comprehensive metadata
    metadata = {
        "model_version": config.model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": round(time.time() - start_time, 2),
        "selected_model": best_name,
        "calibration": calibration_report,
        "data_report": {
            "total_rows_raw": data_report["total_rows"],
            "total_rows_cleaned": len(df),
            "class_distribution": data_report["class_distribution"],
            "validation_issues": data_report["validation_issues"],
        },
        "split_sizes": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "all_input": all_features,
            "preprocessed_output": feature_names_out,
        },
        "excluded_features": config.features.excluded_features,
        "leakage_audit": config.features.leakage_audit,
        "validation_metrics": results,
        "test_metrics": test_metrics,
        "shap_feature_importance": shap_feature_importance,
        "risk_scorer_state": risk_scorer.get_state(),
        "decision_config": {
            "approve_threshold": config.decision.approve_threshold,
            "reject_threshold": config.decision.reject_threshold,
            "policy_rules": {k: {kk: vv for kk, vv in v.items() if kk != "description"} for k, v in config.decision.policy_rules.items()},
        },
        "expected_loss_config": {
            "default_lgd": config.expected_loss.default_lgd,
            "lgd_label": config.expected_loss.lgd_label,
        },
        "risk_grade_thresholds": {k: {"low": v[0], "high": v[1], "label": v[2]} for k, v in RiskGrader.DEFAULT_GRADES.items()},
        "model_selection_note": config.training.model_selection_note,
    }

    with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("  Saved: metadata.json")

    # Save feature names for inference
    with open(os.path.join(ARTIFACTS_DIR, "feature_config.json"), "w") as f:
        json.dump({
            "input_features": all_features,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "preprocessed_feature_names": feature_names_out,
        }, f, indent=2)
    logger.info("  Saved: feature_config.json")

    # --- Summary ---
    elapsed = round(time.time() - start_time, 1)
    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE — {elapsed}s")
    logger.info(f"Model: {best_name} | Version: {config.model_version}")
    logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']} | Test PR-AUC: {test_metrics['pr_auc']}")
    logger.info(f"Calibration: {'Applied (' + calibration_report.get('calibration_method', '') + ')' if calibration_report['calibration_applied'] else 'Not needed'}")
    logger.info(f"Artifacts: {ARTIFACTS_DIR}")
    logger.info(f"{'='*60}\n")

    return metadata


if __name__ == "__main__":
    run_pipeline()
