"""
data_loader.py
--------------
Generates a synthetic housing dataset that mirrors real-world distributions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "housing.csv"


def generate_synthetic_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_samples

    area        = rng.integers(500, 5001, n).astype(float)
    bedrooms    = rng.integers(1, 7, n)
    bathrooms   = np.clip(rng.integers(1, bedrooms + 2, n), 1, 5)
    floors      = rng.integers(1, 4, n)
    age         = rng.integers(0, 51, n).astype(float)
    garage      = rng.integers(0, 2, n)
    pool        = rng.integers(0, 2, n)
    furnished   = rng.choice(["unfurnished", "semi-furnished", "furnished"], n,
                             p=[0.35, 0.40, 0.25])
    loc_tier    = rng.integers(1, 5, n)
    dist_cbd    = rng.uniform(1, 40, n).round(2)
    school_rtg  = rng.uniform(3, 10, n).round(1)
    crime_idx   = rng.uniform(10, 80, n).round(1)

    price = (
        50_000
        + area          * 120
        + bedrooms      * 8_000
        + bathrooms     * 5_000
        + floors        * 3_000
        - age           * 800
        + garage        * 12_000
        + pool          * 18_000
        + np.where(furnished == "furnished",      15_000,
          np.where(furnished == "semi-furnished",  6_000, 0))
        - (loc_tier - 1) * 20_000
        - dist_cbd      * 2_000
        + school_rtg    * 3_500
        - crime_idx     * 500
        + rng.normal(0, 15_000, n)
    )
    price = np.clip(price, 40_000, None).round(-2)

    return pd.DataFrame({
        "area_sqft":       area,
        "bedrooms":        bedrooms,
        "bathrooms":       bathrooms,
        "floors":          floors,
        "age_years":       age,
        "garage":          garage,
        "pool":            pool,
        "furnished":       furnished,
        "location_tier":   loc_tier,
        "distance_cbd_km": dist_cbd,
        "school_rating":   school_rtg,
        "crime_index":     crime_idx,
        "price":           price,
    })


def load_or_generate(force_generate: bool = False) -> pd.DataFrame:
    if RAW_PATH.exists() and not force_generate:
        print(f"📂 Loading existing dataset from {RAW_PATH}")
        return pd.read_csv(RAW_PATH)

    print("🔧 Generating synthetic housing dataset ...")
    df = generate_synthetic_dataset()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"✅ Dataset saved → {RAW_PATH}  ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    df = load_or_generate()
    print("\n--- First 5 rows ---")
    print(df.head())
    print("\n--- Shape ---")
    print(df.shape)
    print("\n--- Stats ---")
    print(df.describe().round(2))