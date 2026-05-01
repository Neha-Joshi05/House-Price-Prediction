import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

MODELS_PATH = Path(__file__).resolve().parents[2] / "models" / "saved"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bed_bath_ratio"] = (df["bedrooms"] / df["bathrooms"]).round(2)
    df["total_rooms"]    = df["bedrooms"] + df["bathrooms"]
    df["cbd_score"]      = (1 / (df["distance_cbd_km"] + 1)).round(4)

    furnished_map         = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
    df["furnished_score"] = df["furnished"].map(furnished_map)
    df["amenity_score"]   = (
        df["garage"] * 1.5 + df["pool"] * 2.0 + df["furnished_score"] * 1.0
    )

    df["age_group"] = pd.cut(
        df["age_years"],
        bins=[-1, 5, 20, 100],
        labels=["new", "modern", "old"]
    ).astype(str)

    return df


def run_full_pipeline(df: pd.DataFrame):
    df = engineer(df)

    y  = df["price"].copy()
    df = df.drop(columns=["price"])

    # Encode categoricals → converts to numbers, then DROP original cat cols
    cat_cols = [c for c in ["furnished", "age_group"] if c in df.columns]
    encoder  = OrdinalEncoder(
        categories="auto",
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    if cat_cols:
        encoded_vals          = encoder.fit_transform(df[cat_cols])
        encoded_col_names     = [f"{c}_enc" for c in cat_cols]
        df[encoded_col_names] = encoded_vals
        df                    = df.drop(columns=cat_cols)   # ← drop originals

    # Scale ALL remaining columns (all numeric now)
    num_cols = df.columns.tolist()
    scaler   = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Save artifacts
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler,              MODELS_PATH / "scaler.pkl")
    joblib.dump(encoder,             MODELS_PATH / "encoder.pkl")
    joblib.dump(df.columns.tolist(), MODELS_PATH / "feature_columns.pkl")
    print(f"💾 Saved → {MODELS_PATH}")
    print(f"   Feature columns: {df.columns.tolist()}")

    return df, y, scaler, encoder