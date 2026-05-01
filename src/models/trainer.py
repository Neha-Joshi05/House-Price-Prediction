"""
trainer.py
----------
Trains three models: Linear Regression, Random Forest, XGBoost.
Saves each model as a .pkl file.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from xgboost                 import XGBRegressor

MODELS_PATH  = Path(__file__).resolve().parents[2] / "models" / "saved"
RANDOM_STATE = 42


def get_models() -> dict:
    return {
        "LinearRegression": LinearRegression(),

        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }


def train_all(X, y, test_size: float = 0.2):
    """
    Splits data, trains all models, saves them.

    Returns
    -------
    trained_models : dict  { name: fitted_estimator }
    splits         : tuple (X_train, X_test, y_train, y_test)
    cv_scores      : dict  { name: mean_cv_r2 }
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    print(f"\n{'='*50}")
    print(f"📊 Train : {len(X_train):,} rows  |  Test : {len(X_test):,} rows")

    models    = get_models()
    trained   = {}
    cv_scores = {}
    kf        = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"\n🔧 Training {name} ...")
        model.fit(X_train, y_train)

        cv = cross_val_score(model, X_train, y_train,
                             cv=kf, scoring="r2", n_jobs=-1)
        cv_scores[name] = cv.mean()
        print(f"   CV R² : {cv.mean():.4f} ± {cv.std():.4f}")

        path = MODELS_PATH / f"{name}.pkl"
        joblib.dump(model, path)
        print(f"   💾 Saved → {path}")

        trained[name] = model

    return trained, (X_train, X_test, y_train, y_test), cv_scores