"""
evaluator.py
------------
Computes evaluation metrics and generates a comparison report.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

REPORTS_PATH = Path(__file__).resolve().parents[2] / "outputs" / "reports"


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"\n📈 {name}")
    print(f"   MAE  : ${mae:>12,.0f}      (avg prediction error)")
    print(f"   RMSE : ${rmse:>12,.0f}      (penalizes big errors)")
    print(f"   R²   : {r2:>13.4f}      (1.0 = perfect)")
    print(f"   MAPE : {mape:>12.2f}%     (% off on average)")

    return {
        "Model":    name,
        "MAE ($)":  round(mae, 2),
        "RMSE ($)": round(rmse, 2),
        "R² Score": round(r2, 4),
        "MAPE (%)": round(mape, 2),
    }


def evaluate_all(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    print(f"\n{'='*50}")
    print("📊 Model Evaluation Report")
    print(f"{'='*50}")

    rows = []
    for name, model in trained_models.items():
        rows.append(evaluate_model(name, model, X_test, y_test))

    report = pd.DataFrame(rows).set_index("Model")
    report = report.sort_values("R² Score", ascending=False)

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    save_path = REPORTS_PATH / "model_comparison.csv"
    report.to_csv(save_path)

    print(f"\n{'='*50}")
    print("🏆 Final Leaderboard")
    print(f"{'='*50}")
    print(report.to_string())

    best = report.index[0]
    print(f"\n🥇 Best Model : {best}")
    print(f"   R²   = {report.loc[best, 'R² Score']:.4f}")
    print(f"   MAE  = ${report.loc[best, 'MAE ($)']:,.0f}")
    print(f"   MAPE = {report.loc[best, 'MAPE (%)']:.2f}%")
    print(f"\n💾 Report saved → {save_path}")

    return report