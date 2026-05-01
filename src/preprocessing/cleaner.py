"""
cleaner.py
----------
Handles missing values, outlier removal, and type validation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "housing_clean.csv"

NUMERIC_COLS = [
    "area_sqft", "bedrooms", "bathrooms", "floors", "age_years",
    "garage", "pool", "location_tier", "distance_cbd_km",
    "school_rating", "crime_index",
]
CAT_COLS = ["furnished"]
TARGET   = "price"


def remove_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 3.0) -> pd.DataFrame:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr     = q3 - q1
    lower   = q1 - factor * iqr
    upper   = q3 + factor * iqr
    before  = len(df)
    df      = df[(df[col] >= lower) & (df[col] <= upper)]
    print(f"   Outlier removal [{col}]: {before - len(df)} rows dropped (kept {len(df):,})")
    return df


def clean(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    print(f"\n{'='*50}")
    print("🧹 Starting data cleaning ...")
    print(f"   Input shape : {df.shape}")

    # 1. Duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"   Duplicates removed : {before - len(df)}")

    # 2. Enforce dtypes
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"   Missing values:\n{missing[missing > 0]}")
        for col in NUMERIC_COLS:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        for col in CAT_COLS:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        df.dropna(subset=[TARGET], inplace=True)
    else:
        print("   No missing values found ✅")

    # 4. Domain rules
    if "bedrooms" in df.columns:
        df = df[df["bedrooms"].between(1, 10)]
    if "area_sqft" in df.columns:
        df = df[df["area_sqft"] > 100]
    if "age_years" in df.columns:
        df["age_years"] = df["age_years"].clip(lower=0)
    if TARGET in df.columns:
        df = df[df[TARGET] > 0]

    # 5. Outlier removal
    df = remove_outliers_iqr(df, TARGET)
    df = remove_outliers_iqr(df, "area_sqft")

    # 6. Reset index
    df = df.reset_index(drop=True)

    print(f"   Output shape: {df.shape}")
    print("✅ Cleaning complete!")

    if save:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"💾 Saved → {PROCESSED_PATH}")

    return df


if __name__ == "__main__":
    from src.utils.data_loader import load_or_generate

    raw     = load_or_generate()
    cleaned = clean(raw)

    print("\n--- Cleaned Data Types ---")
    print(cleaned.dtypes)
    print("\n--- Cleaned Stats ---")
    print(cleaned.describe().round(2))