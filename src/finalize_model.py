import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from data_prep import load_data, prepare_features_and_target


def finalize_model(
    data_path: str,
    model_path: str = "models/churn_pipeline_rf.joblib",
    target_fraction: float = 0.20
):
    # Load trained model
    pipeline = joblib.load(model_path)

    # Load data
    df = load_data(data_path)
    X, y = prepare_features_and_target(df)

    # Validation split
    _, X_val, _, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Predict probabilities
    proba = pipeline.predict_proba(X_val)[:, 1]

    # Compute threshold
    threshold = float(np.quantile(proba, 1 - target_fraction))

    # Save final artifacts
    joblib.dump(pipeline, "models/final_churn_pipeline.joblib")

    with open("models/threshold.json", "w") as f:
        json.dump(
            {
                "threshold": threshold,
                "target_fraction": target_fraction
            },
            f,
            indent=4
        )

    print("Final model and threshold saved.")
    print(f"Threshold: {threshold:.4f}")


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
        default="models/churn_pipeline_rf.joblib",
        help="Path to trained model"
    )
    parser.add_argument(
        "--target-fraction",
        type=float,
        default=0.20,
        help="Target fraction for threshold"
    )
    
    args = parser.parse_args()
    
    finalize_model(args.data_path, args.model_path, args.target_fraction)
