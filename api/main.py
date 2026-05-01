"""
api/main.py
-----------
FastAPI application.

Start with:
    uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas   import HouseFeatures, PredictionResponse
from api.predictor import predict

app = FastAPI(
    title="🏠 House Price Prediction API",
    description="Predicts house prices using XGBoost regression.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "House Price Prediction API is running 🚀"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_price(features: HouseFeatures):
    try:
        return predict(features)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/model/info", tags=["Model"])
def model_info():
    from api.predictor import _model_name, _feat_cols
    return {
        "active_model":  _model_name,
        "feature_count": len(_feat_cols),
        "features":      _feat_cols,
    }