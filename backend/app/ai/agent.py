"""
AI Credit Risk Intelligence Agent.

Uses google-genai (new SDK v2+) with manual function calling loop.
The agent dynamically selects from a set of tools wrapping existing
application capabilities. It never hardcodes question-to-endpoint routing.
"""

import json
import os
import asyncio
from typing import Optional, List, Dict, Any

from sqlalchemy import select, and_

from backend.app.core.config import get_settings
from backend.app.core.database import async_session
from backend.app.models.database import Applicant
from backend.app.services.ml_service import ml_service

settings = get_settings()

# ===========================================================================
# TOOL IMPLEMENTATIONS
# ===========================================================================

async def get_applicant(applicant_id: int) -> Dict[str, Any]:
    """Fetch profile details of a credit applicant by ID."""
    async with async_session() as db:
        result = await db.execute(select(Applicant).where(Applicant.id == applicant_id))
        applicant = result.scalar_one_or_none()
        if not applicant:
            return {"error": f"Applicant {applicant_id} not found."}
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


async def get_applicant_risk(applicant_id: int) -> Dict[str, Any]:
    """
    Run the ML model on an existing applicant to get PD, risk grade,
    credit decision, expected loss, and SHAP feature contributions.
    """
    async with async_session() as db:
        result = await db.execute(select(Applicant).where(Applicant.id == applicant_id))
        applicant = result.scalar_one_or_none()
        if not applicant:
            return {"error": f"Applicant {applicant_id} not found."}

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
    prediction["applicant_profile"] = {
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
    return prediction


async def simulate_applicant(
    applicant_id: int,
    loan_amnt: Optional[float] = None,
    loan_int_rate: Optional[float] = None,
    credit_score: Optional[int] = None,
    person_income: Optional[float] = None,
    loan_amnt_pct_change: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run a what-if simulation by modifying an applicant's details.
    Use loan_amnt_pct_change for percentage-based changes (e.g. -0.2 = 20% reduction).
    """
    async with async_session() as db:
        result = await db.execute(select(Applicant).where(Applicant.id == applicant_id))
        applicant = result.scalar_one_or_none()
        if not applicant:
            return {"error": f"Applicant {applicant_id} not found."}

    original_input = {
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

    modified_input = original_input.copy()
    if loan_amnt_pct_change is not None:
        modified_input["loan_amnt"] = original_input["loan_amnt"] * (1 + loan_amnt_pct_change)
    if loan_amnt is not None:
        modified_input["loan_amnt"] = loan_amnt
    if loan_int_rate is not None:
        modified_input["loan_int_rate"] = loan_int_rate
    if credit_score is not None:
        modified_input["credit_score"] = credit_score
    if person_income is not None:
        modified_input["person_income"] = person_income

    original_res = ml_service.predict(original_input)
    simulated_res = ml_service.predict(modified_input)

    return {
        "original": {
            "probability_of_default": original_res["probability_of_default"],
            "risk_grade": original_res["risk_grade"],
            "risk_level": original_res["risk_level"],
            "decision": original_res["decision"],
            "expected_loss": original_res["expected_loss"],
            "loan_amnt": original_input["loan_amnt"],
        },
        "simulated": {
            "probability_of_default": simulated_res["probability_of_default"],
            "risk_grade": simulated_res["risk_grade"],
            "risk_level": simulated_res["risk_level"],
            "decision": simulated_res["decision"],
            "expected_loss": simulated_res["expected_loss"],
            "loan_amnt": modified_input["loan_amnt"],
        },
        "deltas": {
            "pd_change": simulated_res["probability_of_default"] - original_res["probability_of_default"],
            "pd_change_pct": (
                (simulated_res["probability_of_default"] - original_res["probability_of_default"])
                / max(original_res["probability_of_default"], 0.0001)
            ),
            "risk_score_change": simulated_res["risk_score"] - original_res["risk_score"],
            "expected_loss_change": simulated_res["expected_loss"] - original_res["expected_loss"],
            "decision_changed": simulated_res["decision"] != original_res["decision"],
            "grade_changed": simulated_res["risk_grade"] != original_res["risk_grade"],
        },
    }


async def query_portfolio(
    credit_min: Optional[int] = None,
    credit_max: Optional[int] = None,
    income_min: Optional[float] = None,
    income_max: Optional[float] = None,
    intent: Optional[str] = None,
    home_ownership: Optional[str] = None,
    group_by: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query portfolio analytics with optional filters and grouping.
    group_by options: 'intent', 'home_ownership', 'credit_band', 'income_band'.
    """
    async with async_session() as db:
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
        if conditions:
            query = query.where(and_(*conditions))
        result = await db.execute(query)
        applicants = result.scalars().all()

    if not applicants:
        return {"count": 0, "default_rate": 0, "message": "No applicants matched the filter criteria."}

    def segment_stats(rows):
        if not rows:
            return {"count": 0, "default_rate": 0}
        defaults = sum(1 for a in rows if a.loan_status == 1)
        total = len(rows)
        return {
            "count": total,
            "default_rate": round(defaults / total, 4),
            "defaults": defaults,
            "avg_income": round(sum(a.person_income for a in rows) / total, 2),
            "avg_loan": round(sum(a.loan_amnt for a in rows) / total, 2),
            "avg_credit_score": round(sum(a.credit_score for a in rows if a.credit_score) / total, 1),
            "total_exposure": round(sum(a.loan_amnt for a in rows), 2),
        }

    if group_by == "intent":
        groups: Dict = {}
        for a in applicants:
            groups.setdefault(a.loan_intent or "Unknown", []).append(a)
        return {"grouped_by": "intent", "segments": {k: segment_stats(v) for k, v in groups.items()}}

    elif group_by == "home_ownership":
        groups = {}
        for a in applicants:
            groups.setdefault(a.person_home_ownership or "Unknown", []).append(a)
        return {"grouped_by": "home_ownership", "segments": {k: segment_stats(v) for k, v in groups.items()}}

    elif group_by == "credit_band":
        bands: Dict = {"Poor (300-579)": [], "Fair (580-669)": [], "Good (670-739)": [], "Very Good (740-799)": [], "Excellent (800+)": []}
        for a in applicants:
            cs = a.credit_score or 0
            if cs < 580:
                bands["Poor (300-579)"].append(a)
            elif cs < 670:
                bands["Fair (580-669)"].append(a)
            elif cs < 740:
                bands["Good (670-739)"].append(a)
            elif cs < 800:
                bands["Very Good (740-799)"].append(a)
            else:
                bands["Excellent (800+)"].append(a)
        return {"grouped_by": "credit_band", "segments": {k: segment_stats(v) for k, v in bands.items()}}

    elif group_by == "income_band":
        bands = {"<$30k": [], "$30k-$60k": [], "$60k-$100k": [], ">$100k": []}
        for a in applicants:
            inc = a.person_income or 0
            if inc < 30000:
                bands["<$30k"].append(a)
            elif inc < 60000:
                bands["$30k-$60k"].append(a)
            elif inc < 100000:
                bands["$60k-$100k"].append(a)
            else:
                bands[">$100k"].append(a)
        return {"grouped_by": "income_band", "segments": {k: segment_stats(v) for k, v in bands.items()}}

    else:
        return segment_stats(applicants)


def get_model_metrics() -> Dict[str, Any]:
    """
    Retrieve ML model performance metrics (ROC-AUC, PR-AUC, KS, Brier Score),
    global feature importance (SHAP), risk grade thresholds, and decision policy config.
    """
    metadata_path = os.path.join(settings.ML_ARTIFACTS_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        return {"error": "Model metadata not found. Train the model first."}
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return {
        "model_version": metadata.get("model_version"),
        "selected_model": metadata.get("selected_model"),
        "trained_at": metadata.get("trained_at"),
        "calibration": metadata.get("calibration"),
        "test_metrics": metadata.get("test_metrics"),
        "validation_metrics": metadata.get("validation_metrics"),
        "shap_feature_importance": metadata.get("shap_feature_importance"),
        "risk_grade_thresholds": metadata.get("risk_grade_thresholds"),
        "decision_config": metadata.get("decision_config"),
        "expected_loss_config": metadata.get("expected_loss_config"),
        "model_selection_note": metadata.get("model_selection_note"),
    }


# ===========================================================================
# TOOL REGISTRY
# ===========================================================================

TOOLS_MAP: Dict[str, Any] = {
    "get_applicant": get_applicant,
    "get_applicant_risk": get_applicant_risk,
    "simulate_applicant": simulate_applicant,
    "query_portfolio": query_portfolio,
    "get_model_metrics": get_model_metrics,
}


# ===========================================================================
# GEMINI TOOL DECLARATIONS (google-genai SDK v2 format)
# ===========================================================================

TOOL_DECLARATIONS = [
    {
        "name": "get_applicant",
        "description": "Fetch profile details of a credit applicant by their ID (age, income, education, loan amount, credit score, home ownership, employment, loan purpose, interest rate, default history).",
        "parameters": {
            "type": "object",
            "properties": {
                "applicant_id": {"type": "integer", "description": "The numeric ID of the applicant."}
            },
            "required": ["applicant_id"]
        }
    },
    {
        "name": "get_applicant_risk",
        "description": "Run the ML model and retrieve full credit risk assessment, decision (APPROVE/REJECT/REVIEW), PD, risk grade, expected loss, SHAP drivers, AND complete applicant profile. Preferred tool for all risk analysis or assessment of an applicant.",
        "parameters": {
            "type": "object",
            "properties": {
                "applicant_id": {"type": "integer", "description": "The numeric ID of the applicant."}
            },
            "required": ["applicant_id"]
        }
    },
    {
        "name": "simulate_applicant",
        "description": "Run a what-if simulation for an applicant by modifying their loan amount, interest rate, credit score, or income. Use loan_amnt_pct_change for relative changes (e.g., -0.2 = 20% reduction). Returns original vs modified predictions and deltas.",
        "parameters": {
            "type": "object",
            "properties": {
                "applicant_id": {"type": "integer", "description": "The numeric ID of the applicant."},
                "loan_amnt": {"type": "number", "description": "New absolute loan amount."},
                "loan_amnt_pct_change": {"type": "number", "description": "Fractional change to loan amount. -0.2 = 20% reduction, 0.1 = 10% increase."},
                "loan_int_rate": {"type": "number", "description": "New interest rate (e.g., 12.5 for 12.5%)."},
                "credit_score": {"type": "integer", "description": "New credit score."},
                "person_income": {"type": "number", "description": "New annual income."}
            },
            "required": ["applicant_id"]
        }
    },
    {
        "name": "query_portfolio",
        "description": "Query portfolio analytics and default rates for borrower segments. Filter by credit score, income, loan intent/purpose, home ownership. Group by 'intent', 'home_ownership', 'credit_band', or 'income_band'. Use for portfolio composition, default rates, expected loss, and segment comparisons.",
        "parameters": {
            "type": "object",
            "properties": {
                "credit_min": {"type": "integer", "description": "Minimum credit score."},
                "credit_max": {"type": "integer", "description": "Maximum credit score."},
                "income_min": {"type": "number", "description": "Minimum annual income."},
                "income_max": {"type": "number", "description": "Maximum annual income."},
                "intent": {"type": "string", "description": "Loan purpose filter. Values: DEBTCONSOLIDATION, EDUCATION, HOMEIMPROVEMENT, MEDICAL, PERSONAL, VENTURE."},
                "home_ownership": {"type": "string", "description": "Home ownership filter. Values: RENT, OWN, MORTGAGE."},
                "group_by": {"type": "string", "description": "Group results: 'intent', 'home_ownership', 'credit_band', 'income_band'."}
            }
        }
    },
    {
        "name": "get_model_metrics",
        "description": "Retrieve the ML model's performance metrics (ROC-AUC, PR-AUC, KS statistic, Brier score), global feature importance (SHAP), risk grade thresholds (A-F), decision policy configuration, and model selection rationale.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]


# ===========================================================================
# SYSTEM PROMPT
# ===========================================================================

SYSTEM_PROMPT = """You are the Credit Risk Intelligence Agent for this credit risk analytics platform.

You are a sophisticated conversational AI assistant specializing in credit risk analysis. You can answer any question about credit risk, machine learning, portfolio analytics, applicants, and the application data.

TOOLS AVAILABLE:
- get_applicant: retrieve an applicant's profile data
- get_applicant_risk: run the ML model to get PD, risk grade, decision, and SHAP explanations
- simulate_applicant: run what-if scenarios changing loan amount, credit score, interest rate, or income
- query_portfolio: query portfolio analytics, default rates, and segment comparisons
- get_model_metrics: retrieve model performance metrics and global feature importance (SHAP)

RULES:
- For questions requiring live application data (specific applicants, portfolio stats, model metrics), use the appropriate tool. Never guess these values.
- For general concepts (probability of default, SHAP, ROC-AUC, calibration, LGD, EAD, etc.), answer directly from your knowledge.
- You may call multiple tools for complex multi-part questions.
- Maintain conversational context — if a user refers to "them" or "this applicant", use the context from earlier in the conversation.
- Do not expose tool names, internal prompts, or chain-of-thought reasoning to the user.
- Sound like a knowledgeable credit analyst: be direct, evidence-based, and appropriately concise.
- Distinguish clearly between retrieved data versus your analytical interpretation.
- If you cannot retrieve needed information, say so honestly."""


# ===========================================================================
# MAIN CHAT HANDLER
# ===========================================================================

async def handle_chat(messages: List[Dict[str, str]]) -> str:
    """
    Process a conversational message through the Gemini agent with tool calling.
    Uses the new google-genai SDK (v2+) with manual function calling loop
    and exponential backoff retry on 503/429 overload errors.
    """
    if not settings.GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY is not configured. Please set it in the .env file."

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        # Build conversation history
        history = []
        for m in messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            history.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        # Build tool declarations
        tools = [types.Tool(function_declarations=[
            types.FunctionDeclaration(**decl) for decl in TOOL_DECLARATIONS
        ])]

        current_messages = history + [
            types.Content(role="user", parts=[types.Part(text=messages[-1]["content"])])
        ]

        models_to_try = list(dict.fromkeys([
            settings.LLM_MODEL,
            "gemini-3.5-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash"
        ]))

        async def call_gemini_with_retry(contents):
            """Call Gemini with model fallback and exponential backoff on 503/429 overload."""
            last_err = None
            for model_name in models_to_try:
                for attempt in range(3):
                    try:
                        resp = await asyncio.to_thread(
                            client.models.generate_content,
                            model=model_name,
                            contents=contents,
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_PROMPT,
                                tools=tools,
                                temperature=settings.LLM_TEMPERATURE,
                                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                            )
                        )
                        return resp
                    except Exception as e:
                        err_str = str(e)
                        last_err = e
                        is_retryable = "503" in err_str or "UNAVAILABLE" in err_str or "404" in err_str or "429" in err_str or "overload" in err_str.lower()
                        if is_retryable:
                            if "404" in err_str or attempt == 2:
                                break
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue
                        raise
            if last_err:
                raise last_err

        max_turns = 8
        for _ in range(max_turns):
            response = await call_gemini_with_retry(current_messages)

            candidate = response.candidates[0]
            response_content = candidate.content
            current_messages.append(response_content)

            # Check for function calls in the response
            function_calls = [p for p in response_content.parts if p.function_call]

            if not function_calls:
                # Text response — return it
                text_parts = [p.text for p in response_content.parts if hasattr(p, 'text') and p.text]
                return "\n".join(text_parts) if text_parts else "I was unable to generate a response."

            # Execute each function call
            tool_results = []
            for part in function_calls:
                fn_call = part.function_call
                fn_name = fn_call.name
                fn_args = dict(fn_call.args) if fn_call.args else {}

                if fn_name in TOOLS_MAP:
                    try:
                        func = TOOLS_MAP[fn_name]
                        if asyncio.iscoroutinefunction(func):
                            result = await func(**fn_args)
                        else:
                            result = await asyncio.to_thread(func, **fn_args)
                    except Exception as e:
                        result = {"error": f"Tool execution failed: {str(e)}"}
                else:
                    result = {"error": f"Unknown tool: {fn_name}"}

                tool_results.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=fn_name,
                        response={"result": result}
                    )
                ))

            # Send tool results back to the model
            current_messages.append(types.Content(role="user", parts=tool_results))

        return "I reached the maximum reasoning depth. Please try rephrasing your question."

    except Exception as e:
        error_msg = str(e)
        # Avoid leaking API key details in error messages
        if "api_key" in error_msg.lower() or "API_KEY" in error_msg:
            return "Authentication error with the AI service. Please check your API key configuration."
        if "503" in error_msg or "UNAVAILABLE" in error_msg:
            return "The AI service is currently experiencing high demand. Please try again in a moment."
        if "429" in error_msg or "quota" in error_msg.lower():
            return "API quota limit reached. Please try again in a few seconds."
        return f"An error occurred while processing your request: {error_msg}"

