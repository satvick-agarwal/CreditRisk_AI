# CreditRisk AI — Decision Intelligence Platform

CreditRisk AI is an enterprise-grade, AI-powered credit risk decision intelligence platform. It transforms raw borrower data into actionable risk metrics, combining high-performance machine learning (LightGBM), explainable AI (SHAP), a policy-driven decision engine, and an autonomous conversational AI Risk Analyst agent.

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
*   **The Upgrade:** Every single inference is now passed through a TreeExplainer (`SHAP`) to generate a localized, feature-by-feature breakdown of exactly *why* the model made its decision. This is critical for regulatory compliance (e.g., FCRA adverse action notices). Cached explainer instances guarantee sub-50ms explanation latency.

### 3. From "Raw Model" to Decision Engine
*   **The Past:** A raw ML probability score determined the loan outcome.
*   **The Upgrade:** The ML model is wrapped in a **Decision Engine** (`risk_engine.py`). If a borrower has a history of defaults, the policy layer escalates and intercepts the decision to automatically `REJECT` them, regardless of the ML score.

### 4. From Rigid Question-Matching to a True Conversational AI Agent
*   **The Past:** A hardcoded intent matcher limited to predefined questions and brittle string matching.
*   **The Upgrade:** An autonomous **Conversational AI Risk Analyst** built on the `google-genai` SDK with real-time tool calling:
    *   **Live Risk Scoring & SHAP Explanations (`get_applicant_risk`):** Dynamically evaluates applicants, calculates PD, assigns risk grades (A–F), runs policy rules, and extracts key SHAP risk drivers.
    *   **Borrower Profiles (`get_applicant`):** Retrieves borrower financials, debt ratios, and history from the database.
    *   **Interactive What-If Simulations (`simulate_applicant`):** Simulates parameter changes (loan amounts, interest rates, credit scores, income) and compares deltas against original assessments.
    *   **Portfolio Intelligence (`query_portfolio`):** Aggregates exposure, loss distributions, and segment comparisons across credit grades, home ownership, and education.
    *   **Model Governance & Metrics (`get_model_metrics`):** Audits global feature importance (SHAP summary), ROC-AUC, PR-AUC, and decision policies.
    *   **Low-Latency Multi-Model Resilience:** Built with automatic model fallbacks and backoff retries across `gemini-3.5-flash-lite`, `gemini-flash-lite-latest`, and `gemini-3.1-flash-lite`.

### 5. From "Jupyter Notebook" to Full-Stack Architecture
*   **The Past:** A scattered collection of Python scripts and a basic Streamlit dashboard.
*   **The Upgrade:** 
    *   **Backend:** A high-performance, asynchronous `FastAPI` layer using `SQLAlchemy` for full auditability (every prediction's exact inputs, model version, and SHAP values are committed to a SQLite/PostgreSQL database).
    *   **Frontend:** A modern, dark-themed React + Vite Single Page Application (SPA). It uses Tailwind CSS and Recharts to visualize decision funnels, portfolio risk distribution, and simulate "what-if" risk scenarios in real time.

---

## Platform Features

- **Predictive Risk Engine**: Evaluates applicant Probability of Default (PD) using a calibrated LightGBM model.
- **Explainable AI (XAI)**: Provides SHAP-based feature importance for every prediction to ensure regulatory compliance and transparency.
- **Policy Enforcement**: A rules-based decision engine acts as an escalation layer over the ML model.
- **What-If Risk Simulator**: Allows analysts to adjust applicant characteristics and instantly view the impact on PD, Expected Loss, and the final decision.
- **Portfolio Analytics**: Interactive dashboards for monitoring portfolio exposure, default rates across credit and income bands, and risk distribution.
- **AI Risk Analyst**: A genuine conversational agent capable of deep reasoning, multi-turn dialogue, what-if stress tests, and evidence-grounded answers.

---

## Project Structure

```text
CreditRisk_AI/
├── backend/
│   └── app/
│       ├── ai/                  # AI Agent & Gemini tool definitions
│       │   └── agent.py         # Function calling loop & tool registry
│       ├── api/                 # FastAPI router endpoints
│       │   └── endpoints.py     # REST endpoints (/applicants, /simulate, /chat, etc.)
│       ├── core/                # App configuration & database session
│       ├── models/              # SQLAlchemy ORM & Pydantic schemas
│       ├── schemas/             # Request & response schemas
│       └── services/            # ML inference, DB seeding, analytics
├── frontend/
│   ├── src/
│   │   ├── pages/               # Dashboard, Applicants, Simulator, AIAnalyst, etc.
│   │   ├── components/          # Reusable UI components & layouts
│   │   └── api/                 # Axios API clients
├── ml/
│   ├── artifacts/               # Serialized model, preprocessor, SHAP metadata
│   └── training/                # Training pipelines, model comparison, risk engine
├── data/                        # Raw & processed loan datasets
└── scripts/                     # Database seeders and maintenance utilities
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ & npm
- Gemini API Key ([Google AI Studio](https://aistudio.google.com/))

### 2. Environment Configuration
Copy `.env.example` to `.env` and provide your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```env
APP_NAME="CreditRisk AI"
APP_VERSION="2.0.0"
DEBUG=True
HOST="0.0.0.0"
PORT=8000
DATABASE_URL="sqlite+aiosqlite:///./creditrisk.db"
GEMINI_API_KEY="your_gemini_api_key_here"
```

### 3. Model Training & Pipeline (Optional if artifacts exist)
```bash
# Install ML dependencies
pip install pandas numpy scikit-learn xgboost lightgbm shap joblib

# Run the training pipeline
python -m ml.training.pipeline
```

### 4. Backend API Server
```bash
# Install backend dependencies
pip install fastapi uvicorn pydantic pydantic-settings sqlalchemy aiosqlite python-dotenv google-genai

# Seed the local database with historical applicants
python scripts/seed_db.py

# Start the API server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
The API is available at `http://localhost:8000`. Interactive Swagger documentation is at `http://localhost:8000/docs`.

### 5. Frontend Dashboard
```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
The application will be live at `http://localhost:5173`.

---

## Financial Assumptions

- **Expected Loss (EL)**: Calculated as `PD * EAD * LGD`.
- **Exposure at Default (EAD)**: Assumed to be the total requested `loan_amnt`.
- **Loss Given Default (LGD)**: Configurable baseline assumption of 45% (Basel standard proxy) unless overridden by policy.

---

## Auditability & Compliance

- **Model Versioning**: Every prediction saved in the database includes the exact model version and name used to generate it.
- **Input Snapshots**: The exact inputs used for inference are saved alongside the prediction.
- **Feature Exclusion**: Protected classes (e.g., gender) are explicitly excluded from the model to maintain fairness.
- **Leakage Auditing**: Features like interest rate and debt-to-income ratios are audited for potential temporal leakage relative to the lending decision.

---

## License
MIT
