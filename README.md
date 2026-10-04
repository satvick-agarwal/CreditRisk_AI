# CreditRisk AI — Decision Intelligence Platform

CreditRisk AI is an advanced, AI-powered credit risk decision intelligence platform. It transforms raw applicant data into actionable risk metrics, combining machine learning (LightGBM), explainable AI (SHAP), and a policy-driven decision engine.

---

## The Transformation Journey 🚀

This project began as a standard academic student project (**Credit Risk Analytics System**)—a basic Jupyter notebook wrapping a simple ML model with rudimentary UI. 

It has since undergone a complete architectural rewrite to evolve into an **enterprise-grade, full-stack ML application**, demonstrating advanced concepts in ML engineering, financial risk analysis, and software development:

### 1. From "Predict.py" to an Auditable ML Pipeline
*   **The Past:** A static script that blindly trained a model and pickled it.
*   **The Upgrade:** A robust ML pipeline (`ml/training/pipeline.py`) that handles cross-validation, hyperparameter tuning, model competition (RandomForest vs XGBoost vs LightGBM), and **model calibration** (using Sigmoid calibration to optimize Brier scores). 
*   **Compliance First:** The new pipeline explicitly enforces ethical ML by dropping protected classes (e.g., gender) and documents feature leakage audits to prevent look-ahead bias.

### 2. From "Black Box" to Explainable AI (XAI)
*   **The Past:** The model output a single integer: `1` (Default) or `0` (Paid).
*   **The Upgrade:** Every single inference is now passed through a TreeExplainer (`SHAP`) to generate a localized, feature-by-feature breakdown of exactly *why* the model made its decision. This is critical for regulatory compliance (e.g., FCRA adverse action notices).

### 3. From "Raw Model" to Decision Engine
*   **The Past:** A raw ML probability score determined the loan outcome.
*   **The Upgrade:** The ML model is now wrapped in a **Decision Engine** (`risk_engine.py`). If a borrower has a history of defaults, the policy layer escalates and intercepts the decision to automatically `REJECT` them, regardless of the ML score.

### 4. From "Jupyter Notebook" to Full-Stack Architecture
*   **The Past:** A scattered collection of Python scripts and a basic Streamlit dashboard.
*   **The Upgrade:** 
    *   **Backend:** A high-performance, asynchronous `FastAPI` layer using `SQLAlchemy` for full auditability (every prediction's exact inputs, model version, and SHAP values are committed to a SQLite/PostgreSQL database).
    *   **Frontend:** A stunning, dark-themed React + Vite Single Page Application (SPA). It uses Tailwind CSS and Recharts to visualize decision funnels, portfolio risk distribution, and simulate "what-if" risk scenarios in real time.

---

## Platform Features

- **Predictive Risk Engine**: Evaluates applicant Probability of Default (PD) using a calibrated LightGBM model.
- **Explainable AI (XAI)**: Provides SHAP-based feature importance for every prediction to ensure regulatory compliance and transparency.
- **Policy Enforcement**: A rules-based decision engine acts as an escalation layer over the ML model.
- **What-If Risk Simulator**: Allows analysts to adjust applicant characteristics and instantly view the impact on PD, Expected Loss, and the final decision.
- **Portfolio Analytics**: Interactive dashboards for monitoring portfolio exposure, default rates across credit and income bands, and risk distribution.
- **AI Risk Analyst**: A conversational interface that allows users to ask natural language questions about portfolio risk and model behavior, grounded purely in database evidence and model metrics.

## Setup & Installation

### 1. Model Training & Pipeline

The ML pipeline cleans data, engineers features, trains multiple models, selects the best performer, calibrates it, and persists the artifacts for inference.

```bash
# Install ML dependencies
pip install pandas numpy scikit-learn xgboost lightgbm shap joblib

# Run the training pipeline
python -m ml.training.pipeline
```

### 2. Backend API Server

The backend requires the ML artifacts to be present in `ml/artifacts/`.

```bash
# Install backend dependencies
pip install fastapi uvicorn pydantic pydantic-settings sqlalchemy aiosqlite python-dotenv

# Seed the local database with all ~45,000 historical applicants
python scripts/seed_db.py

# Start the API server
uvicorn backend.app.main:app --reload --port 8000
```
The API will be available at `http://localhost:8000`. Swagger documentation is at `http://localhost:8000/docs`.

### 3. Frontend Dashboard

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
The application will be available at `http://localhost:5173`.

## Financial Assumptions

- **Expected Loss (EL)**: Calculated as `PD * EAD * LGD`.
- **Exposure at Default (EAD)**: Assumed to be the total requested `loan_amnt`.
- **Loss Given Default (LGD)**: Configurable, but defaults to a baseline assumption of 45% (Basel standard proxy) unless overridden by the user.

## Auditability & Compliance

- **Model Versioning**: Every prediction saved in the database includes the exact model version and name used to generate it.
- **Input Snapshots**: The exact inputs used for inference are saved alongside the prediction.
- **Feature Exclusion**: Protected classes (e.g., gender) are explicitly excluded from the model to maintain fairness.
- **Leakage Auditing**: Features like interest rate and debt-to-income ratios are audited for potential temporal leakage relative to the lending decision.

## License
MIT
