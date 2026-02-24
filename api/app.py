from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
from pathlib import Path
import json
import boto3, uuid
from datetime import datetime, timezone
from .config import S3_BUCKET, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN

# ========== CONFIG ==========
s3_bucket = S3_BUCKET
aws_region = AWS_REGION
aws_access_key_id = AWS_ACCESS_KEY_ID
aws_secret_access_key = AWS_SECRET_ACCESS_KEY
aws_session_token = AWS_SESSION_TOKEN
# ============================

app = FastAPI()

# # Comment out the below line if not saving to S3 or if AWS credentials are not set (local testing)
# s3 = boto3.client("s3", region_name=s3_bucket,
#                   aws_access_key_id=aws_access_key_id,
#                   aws_secret_access_key=aws_secret_access_key,
#                   aws_session_token=aws_session_token
# )

# # Comment out the below line if not saving to S3 or if AWS credentials are not set (local testing)
# def upload_json_to_s3(obj: dict, key_prefix="logs/"):
#     key = f"{key_prefix}{datetime.now(timezone.utc).isoformat()}_{uuid.uuid4().hex}.json"
#     s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(obj).encode("utf-8"))
#     return key

# Request model
class PatientData(BaseModel):
    age: int
    gender: Optional[str] = None
    smoking_history: Optional[str] = None
    hypertension: int
    heart_disease: int
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

    gender_Female: Optional[int] = None
    smoking_history_current: Optional[int] = None
    smoking_history_former: Optional[int] = None
    smoking_history_never: Optional[int] = None
    smoking_history_unknown: Optional[int] = None

# Global artifacts (loaded at module level)
model = None
scaler = None
class_map = {}

def load_artifacts():
    """Load model artifacts at startup"""
    global model, scaler, class_map
    
    # Load model on startup
    try:
        model_path = Path("./model/best_diabetes_model_LightGBM.pkl")
        with open(model_path, "rb") as f:
            model = joblib.load(f)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("WARNING: model not found. Predictions will fail until model file is provided.")
    except Exception as e:
        print(f"ERROR loading model: {e}")

    # Load scaler (for numerical features)
    try:
        scaler_path = Path("./model/scaler.pkl")
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
    except Exception as e:
        print(f"ERROR loading scaler: {e}")

    # Set up class mapping - derive class info for human-readable output
    if model:
        classes = list(getattr(model, "classes_", []))
        if set(classes) == {0, 1}:
            class_map = {0: "Not Diabetic", 1: "Diabetic"}
        else:
            class_map = {c: str(c) for c in classes}

    ''' Debug info: print loaded feature names '''
    if hasattr(model, "feature_names_in_"):
        print(f"\n[DEBUG] Model features: {list(model.feature_names_in_)}")

# Load artifacts when module is imported
load_artifacts()

@app.post("/predict")
def predict(patient: PatientData):
    # Validate required artifacts
    if model is None or scaler is None:
        raise HTTPException(
            status_code=500,
            detail="Model artifacts not fully loaded.")
    
    ''' Below is for validating each artifact individually (debugging purposes)'''
    # # Check if model is loaded
    # if model is None:
    #     raise HTTPException(
    #         status_code=500,
    #         detail="Prediction model not loaded.")
    
    # # Check if scaler is loaded
    # if scaler is None:
    #     raise HTTPException(
    #         status_code=500,
    #         detail="Scaler not loaded.")


    try:
        # If caller provided simplified gender/smoking fields, derive one-hot flags
        # Compute gender_Female if not explicitly provided
        if getattr(patient, "gender_Female", None) is None:
            g = getattr(patient, "gender", None)
            if isinstance(g, str) and g.strip().lower() == "female":
                patient.gender_Female = 1
            else:
                try:
                    # allow numeric 0/1 passed as gender as well
                    patient.gender_Female = int(g) if g is not None else 0
                except Exception:
                    patient.gender_Female = 0

        # Compute smoking history one-hot flags if none provided
        if all(getattr(patient, f) is None for f in [
            "smoking_history_current", "smoking_history_former",
            "smoking_history_never", "smoking_history_unknown"]):
            s = getattr(patient, "smoking_history", None)
            s_norm = s.strip().lower() if isinstance(s, str) else None
            patient.smoking_history_current = 1 if s_norm == "current" else 0
            patient.smoking_history_former = 1 if s_norm == "former" else 0
            patient.smoking_history_never = 1 if s_norm == "never" else 0
            # treat anything else / None as unknown
            patient.smoking_history_unknown = 1 if s_norm not in {"current","former","never"} else 0

        features = {
            "age": patient.age,
            "bmi": patient.bmi,
            "HbA1c_level": patient.HbA1c_level,
            "blood_glucose_level": patient.blood_glucose_level,
            "hypertension": patient.hypertension,
            "heart_disease": patient.heart_disease,
            "gender_Female": patient.gender_Female,
            "smoking_history_current": patient.smoking_history_current,
            "smoking_history_former": patient.smoking_history_former,
            "smoking_history_never": patient.smoking_history_never,
            "smoking_history_unknown": patient.smoking_history_unknown
        }

        scaler_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]

        # Prepare data for scaling in EXACT order used during training
        numerical_df = pd.DataFrame([[
            patient.age,
            patient.bmi,
            patient.HbA1c_level,
            patient.blood_glucose_level
        ]], columns=scaler_cols)

        # Apply scaling
        scaled_nums = scaler.transform(numerical_df)[0]  # Get first (only) row
        
        # Update with scaled values
        features.update({
            "age": scaled_nums[0],
            "bmi": scaled_nums[1],
            "HbA1c_level": scaled_nums[2],
            "blood_glucose_level": scaled_nums[3]
        })

        # Prepare feature vector in training order
        feature_order = [
            "age", "hypertension", "heart_disease", "bmi", "HbA1c_level",
            "blood_glucose_level", "gender_Female", "smoking_history_current",
            "smoking_history_former", "smoking_history_never", "smoking_history_unknown"
        ]

        # Create DataFrame for prediction
        X_df = pd.DataFrame([[features[feat] for feat in feature_order]], columns=feature_order)

        # Make prediction
        pred = model.predict(X_df)[0]
        prob = model.predict_proba(X_df)[0][1] if hasattr(model, "predict_proba") else None
        prob_percentage = round(prob * 100, 2) if prob is not None else None
        label_name = class_map.get(pred, str(pred))

        # Comment out the below line if not saving to S3 or if AWS credentials are not set (local testing)
        # record = {
        #     "input": patient.model_dump(),
        #     "prediction": int(pred),
        #     "label": label_name,
        #     "probability": prob,
        #     "timestamp": datetime.now(timezone.utc).isoformat()
        # }

        # upload_key = upload_json_to_s3(record)

        return {
            "result": int(pred),
            "label": label_name,
            "probability": f"{prob_percentage}%" if prob_percentage is not None else None
            # "s3_key": upload_key
        }
    
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="Failed to load prediction model. Model may be incompatible."
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
