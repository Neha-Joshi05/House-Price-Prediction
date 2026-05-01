import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.data_loader              import load_or_generate
from src.preprocessing.cleaner         import clean
from src.preprocessing.feature_engineer import run_full_pipeline
from src.models.trainer                 import train_all
from src.models.evaluator               import evaluate_all
from src.utils.visualizer               import (
    plot_price_distribution,
    plot_correlation_heatmap,
    plot_feature_vs_price,
    plot_categorical_boxplots,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_model_comparison,
    plot_feature_importance,
)

if __name__ == "__main__":
    # Data
    raw     = load_or_generate()
    cleaned = clean(raw, save=True)
    X, y, scaler, encoder = run_full_pipeline(cleaned)

    # Train
    trained_models, splits, cv_scores = train_all(X, y)
    X_train, X_test, y_train, y_test  = splits

    # Evaluate
    report = evaluate_all(trained_models, X_test, y_test)

    # EDA plots
    print("\n🎨 Generating EDA plots ...")
    plot_price_distribution(raw)
    plot_correlation_heatmap(raw)
    plot_feature_vs_price(raw)
    plot_categorical_boxplots(raw)

    # Model plots
    print("\n🎨 Generating model plots ...")
    plot_actual_vs_predicted(trained_models, X_test, y_test)
    plot_residuals(trained_models, X_test, y_test)
    plot_model_comparison(report)
    plot_feature_importance(
        trained_models["XGBoost"],
        feature_names=X.columns.tolist(),
        model_name="XGBoost"
    )

    print("\n✅ All done! Check outputs/plots/ for your 8 charts.")