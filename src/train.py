import argparse
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from data_prep import load_data, prepare_features_and_target, get_preprocessor


def train_model(data_path: str, model_type: str = "logreg"):
    # Load data
    df = load_data(data_path)
    X, y = prepare_features_and_target(df)

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Preprocessing
    preprocessor = get_preprocessor(X)

    # Choose model
    if model_type == "logreg":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        )
    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError("Unsupported model type")

    # Build pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)

    print(f"Model: {model_type}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/churn_pipeline_{model_type}.joblib"
    joblib.dump(pipeline, model_path)

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to Telco churn CSV"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="logreg",
        choices=["logreg", "rf"]
    )
    args = parser.parse_args()

    train_model(args.data_path, args.model_type)
