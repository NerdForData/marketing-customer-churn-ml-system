import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

from data_prep import load_data, prepare_features_and_target


def campaign_threshold_analysis(
    data_path: str,
    model_path: str = "models/churn_pipeline_rf.joblib",
    target_fraction: float = 0.20
):
    # Load model and data
    pipeline = joblib.load(model_path)
    df = load_data(data_path)
    X, y = prepare_features_and_target(df)

    # Split
    _, X_val, _, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Predict probabilities
    proba = pipeline.predict_proba(X_val)[:, 1]

    # Determine threshold based on top X% customers
    threshold = np.quantile(proba, 1 - target_fraction)

    # Predictions
    preds = (proba >= threshold).astype(int)

    # Metrics
    precision = (preds[y_val == 1].sum()) / max(preds.sum(), 1)
    recall = (preds[y_val == 1].sum()) / y_val.sum()

    return {
        "threshold": threshold,
        "target_fraction": target_fraction,
        "precision": precision,
        "recall": recall,
        "customers_targeted": int(preds.sum()),
        "actual_churners_captured": int(preds[y_val == 1].sum())
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to Telco churn CSV"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/churn_pipeline_logreg.joblib",
        help="Path to saved model"
    )
    parser.add_argument(
        "--target-fraction",
        type=float,
        default=0.20,
        help="Fraction of customers to target (e.g., 0.20 for top 20%%)"
    )
    
    args = parser.parse_args()
    
    results = campaign_threshold_analysis(
        args.data_path,
        args.model_path,
        args.target_fraction
    )
    
    print("\n=== Campaign Threshold Analysis ===")
    print(f"Model: {args.model_path}")
    print(f"Target Fraction: {results['target_fraction']:.1%}")
    print(f"Threshold: {results['threshold']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"Customers Targeted: {results['customers_targeted']}")
    print(f"Actual Churners Captured: {results['actual_churners_captured']}")
