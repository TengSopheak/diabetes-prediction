# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle, boto3, json, uuid
from datetime import datetime, timezone
from .config import S3_BUCKET, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN

# ========== CONFIG ==========
MODEL_PATH = "./model/best_diabetes_model_LightGBM.pkl"
S3_BUCKET = S3_BUCKET
AWS_REGION = AWS_REGION
# ============================

class Person(BaseModel):
    # put feature names exactly as the model expects
    age: float
    bmi: float
    blood_glucose_level: float
    HbA1c_level: float
    hypertension: float
    heart_disease: float

app = FastAPI()

# Load model once on startup
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# derive class info for human-readable output
classes = list(getattr(model, "classes_", []))
if set(classes) == {0, 1}:
    CLASS_MAP = {0: "non-diabetic", 1: "diabetic"}
else:
    CLASS_MAP = {c: str(c) for c in classes}

# # Comment out the below line if not saving to S3 or if AWS credentials are not set (local testing)
# s3 = boto3.client("s3", region_name=AWS_REGION,
#                   aws_access_key_id=AWS_ACCESS_KEY_ID,
#                   aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#                   aws_session_token=AWS_SESSION_TOKEN
# )

# # Comment out the below line if not saving to S3 or if AWS credentials are not set (local testing)
# def upload_json_to_s3(obj: dict, key_prefix="logs/"):
#     key = f"{key_prefix}{datetime.now(timezone.utc).isoformat()}_{uuid.uuid4().hex}.json"
#     s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(obj).encode("utf-8"))
#     return key

@app.post("/predict")
def predict(p: Person):
    try:
        # convert p to model input shape
        # Example: if model expects a 2D list
        X = [[p.age, p.bmi, p.blood_glucose_level, p.HbA1c_level, p.hypertension, p.heart_disease]]
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None # Just class 1 probability
        prob_percentage = round(prob * 100, 2) if prob is not None else None
        label_name = CLASS_MAP.get(pred, str(pred))

        record = {
            "input": p.model_dump(),
            "prediction": int(pred),
            "label": label_name,
            "probability": prob,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # # Comment out the below line if not saving to S3 or if AWS credentials are not set (local testing)
        # upload_key = upload_json_to_s3(record)

        # # Comment out the below line if not saving to S3 or if AWS credentials are not set (local testing)
        # return {"result": int(pred), "label": label_name, "probability": f"{prob_percentage}%", "s3_key": upload_key}
    
        return {"result": int(pred), "label": label_name, "probability": f"{prob_percentage}%"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
