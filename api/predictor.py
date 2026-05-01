import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from api.schemas import HouseFeatures

MODELS_PATH      = Path(__file__).resolve().parent.parent / "models" / "saved"
MODEL_PREFERENCE = ["XGBoost", "RandomForest", "LinearRegression"]


def _load_artifacts():
    scaler    = joblib.load(MODELS_PATH / "scaler.pkl")
    encoder   = joblib.load(MODELS_PATH / "encoder.pkl")
    feat_cols = joblib.load(MODELS_PATH / "feature_columns.pkl")
    model = None
    model_name = None
    for name in MODEL_PREFERENCE:
        p = MODELS_PATH / f"{name}.pkl"
        if p.exists():
            model      = joblib.load(p)
            model_name = name
            break
    if model is None:
        raise FileNotFoundError("No trained model found.")
    return model, model_name, scaler, encoder, feat_cols


_model, _model_name, _scaler, _encoder, _feat_cols = _load_artifacts()


def predict(features: HouseFeatures) -> dict:
    d = features.model_dump()

    # Step 1: Engineer features
    bathrooms      = max(d["bathrooms"], 1)
    bed_bath_ratio = round(d["bedrooms"] / bathrooms, 2)
    total_rooms    = d["bedrooms"] + bathrooms
    cbd_score      = round(1 / (d["distance_cbd_km"] + 1), 4)

    furnished_map   = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
    furnished_score = furnished_map.get(d["furnished"], 0)
    amenity_score   = d["garage"] * 1.5 + d["pool"] * 2.0 + furnished_score * 1.0

    age = d["age_years"]
    if age <= 5:
        age_group = "new"
    elif age <= 20:
        age_group = "modern"
    else:
        age_group = "old"

    # Step 2: Encode categoricals using saved encoder
    cat_values      = [[d["furnished"], age_group]]
    encoded         = _encoder.transform(cat_values)[0]
    furnished_enc   = encoded[0]
    age_group_enc   = encoded[1]

    # Step 3: Build feature dict matching trained columns exactly
    feature_dict = {
        "area_sqft":       d["area_sqft"],
        "bedrooms":        d["bedrooms"],
        "bathrooms":       bathrooms,
        "floors":          d["floors"],
        "age_years":       d["age_years"],
        "garage":          d["garage"],
        "pool":            d["pool"],
        "location_tier":   d["location_tier"],
        "distance_cbd_km": d["distance_cbd_km"],
        "school_rating":   d["school_rating"],
        "crime_index":     d["crime_index"],
        "bed_bath_ratio":  bed_bath_ratio,
        "total_rooms":     total_rooms,
        "cbd_score":       cbd_score,
        "furnished_score": furnished_score,
        "amenity_score":   amenity_score,
        "furnished_enc":   furnished_enc,
        "age_group_enc":   age_group_enc,
    }

    # Step 4: Build numpy array in exact column order (no DataFrame = no name check)
    values = np.array([[feature_dict.get(col, 0) for col in _feat_cols]], dtype=float)

    # Step 5: Scale using numpy array directly
    scaled = _scaler.transform(values)

    # Step 6: Predict
    price = float(_model.predict(scaled)[0])
    price = max(price, 10_000)
    band  = price * 0.10

    return {
        "predicted_price":  round(price, 2),
        "model_used":       _model_name,
        "confidence_range": {
            "low":  round(price - band, 2),
            "high": round(price + band, 2),
        },
        "input_summary": {
            "area_sqft":     d["area_sqft"],
            "bedrooms":      d["bedrooms"],
            "location_tier": d["location_tier"],
            "furnished":     d["furnished"],
        },
    }