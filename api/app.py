from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import xgboost as xgb

# 1. Initialize API
app = FastAPI(
    title="Trend Detection API", 
    description="Real-time inference for BTC-USD trends",
    version="1.0"
)

# 2. Load Model (Load once at startup for speed)
# Ensure you are running this from the root 'trend-detection-ml' folder
MODEL_PATH = "models/xgboost_v1.joblib"

if not os.path.exists(MODEL_PATH):
    # Fallback for common path issues
    if os.path.exists(f"../{MODEL_PATH}"):
        MODEL_PATH = f"../{MODEL_PATH}"
    else:
        print(f"⚠️ WARNING: Model not found at {MODEL_PATH}")

try:
    # We load the wrapper class we created, or just the raw model if you saved that
    # For robustness, we'll assume it's the raw XGBoost model or Pipeline
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# 3. Define Input Schema (Data Validation)
class MarketData(BaseModel):
    ma_10: float
    ma_50: float
    rsi: float
    volatility: float
    log_return: float

# 4. Define Prediction Endpoint
@app.post("/predict")
def predict_trend(data: MarketData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # Convert input JSON to DataFrame (names must match training columns)
        input_data = {
            'ma_10': [data.ma_10],
            'ma_50': [data.ma_50],
            'rsi': [data.rsi],
            'volatility': [data.volatility],
            'log_return': [data.log_return]
        }
        features = pd.DataFrame(input_data)
        
        # XGBoost often creates a custom object in pipelines, but let's handle the raw prediction
        # If your saved model is the 'CryptoModel' class instance:
        if hasattr(model, 'predict_proba'):
            # It's likely a sklearn pipeline or our custom class
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            # Handle different proba shapes (sometimes returns [prob_0, prob_1])
            if isinstance(probability, (list, np.ndarray)) and len(probability) > 1:
                prob_up = probability[1]
            else:
                prob_up = probability
        else:
            # Raw Booster
            dmatrix = xgb.DMatrix(features)
            prob_up = model.predict(dmatrix)[0]
            prediction = 1 if prob_up > 0.5 else 0
        
        trend = "UP 🚀" if prediction == 1 else "DOWN 🔻"
        
        return {
            "prediction": trend,
            "confidence": float(prob_up if prediction == 1 else 1 - prob_up),
            "raw_probability_up": float(prob_up)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# 5. Root Endpoint (Health Check)
@app.get("/")
def home():
    return {"status": "healthy", "service": "Trend Detection API", "model_loaded": model is not None}