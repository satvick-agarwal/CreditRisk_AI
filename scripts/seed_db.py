"""
Seed the database from the processed CSV.
"""
import sys
import os
import asyncio
import pandas as pd
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.core.database import async_session, init_db
from backend.app.models.database import Applicant
from backend.app.core.config import get_settings

settings = get_settings()

async def seed():
    print("Initializing DB...")
    await init_db()
    
    print(f"Loading data from {settings.RAW_DATA_PATH}")
    # Load processed data to match training
    processed_path = os.path.join(os.path.dirname(settings.RAW_DATA_PATH), "..", "processed", "cleaned_data.csv")
    if not os.path.exists(processed_path):
        print(f"File not found: {processed_path}")
        return
        
    df = pd.read_csv(processed_path)
    print(f"Loaded {len(df)} rows.")
    
    # We will seed all records. To avoid taking too long, we will do it in chunks.
    print(f"Seeding {len(df)} rows into database...")
    
    async with async_session() as db:
        # Clear existing to avoid duplicates if run multiple times
        await db.execute(Applicant.__table__.delete())
        await db.commit()
        
        for idx, row in df.iterrows():
            applicant = Applicant(
                person_age=float(row.get("person_age", 0)),
                person_gender=str(row.get("person_gender", "unknown")),
                person_education=str(row.get("person_education", "unknown")),
                person_income=float(row.get("person_income", 0)),
                person_emp_exp=float(row.get("person_emp_exp", 0)),
                person_home_ownership=str(row.get("person_home_ownership", "unknown")),
                loan_amnt=float(row.get("loan_amnt", 0)),
                loan_intent=str(row.get("loan_intent", "unknown")),
                loan_int_rate=float(row.get("loan_int_rate", 0)) if pd.notna(row.get("loan_int_rate")) else 0.0,
                loan_percent_income=float(row.get("loan_percent_income", 0)) if pd.notna(row.get("loan_percent_income")) else 0.0,
                cb_person_cred_hist_length=float(row.get("cb_person_cred_hist_length", 0)),
                credit_score=int(row.get("credit_score", 0)) if pd.notna(row.get("credit_score")) else 0,
                previous_loan_defaults_on_file=str(row.get("previous_loan_defaults_on_file", "No")),
                loan_status=int(row.get("loan_status", 0))
            )
            db.add(applicant)
            if idx % 1000 == 0:
                await db.commit()
                print(f"  Committed {idx} records...")
                
        await db.commit()
    print("Seeding complete.")

if __name__ == "__main__":
    asyncio.run(seed())
