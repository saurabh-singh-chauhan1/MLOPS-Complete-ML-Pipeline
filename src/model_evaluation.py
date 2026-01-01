import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from dvclive import Live
from pathlib import Path

# =========================
# PATH SETUP
# =========================
LOG_DIR = Path("logs")
REPORTS_DIR = Path("reports")

LOG_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# =========================
# LOGGING CONFIGURATION
# =========================
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_DIR / "model_evaluation.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# =========================
# FUNCTIONS
# =========================
def load_params(params_path: str) -> dict:
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)
    logger.debug("Parameters retrieved from %s", params_path)
    return params


def load_model(file_path: str):
    with open(file_path, "rb") as file:
        model = pickle.load(file)
    logger.debug("Model loaded from %s", file_path)
    return model


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    logger.debug("Data loaded from %s", file_path)
    return df


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    logger.debug("Model evaluation metrics calculated")
    return metrics


def save_metrics(metrics: dict, file_path: Path) -> None:
    with open(file_path, "w") as file:
        json.dump(metrics, file, indent=4)
    logger.debug("Metrics saved to %s", file_path)


# =========================
# MAIN
# =========================
def main():
    try:
        params = load_params("params.yaml")
        clf = load_model("models/model.pkl")
        test_data = load_data("data/processed/test_tfidf.csv")

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)

        # 1️⃣ SAVE METRICS FIRST
        metrics_path = REPORTS_DIR / "metrics.json"
        save_metrics(metrics, metrics_path)

        # 2️⃣ THEN LOG WITH DVCLIVE
        with Live(save_dvc_exp=True) as live:
            for key, value in metrics.items():
                live.log_metric(key, value)
            live.log_params(params)

        logger.info("Model evaluation completed successfully")

    except Exception as e:
        logger.exception("Failed to complete the model evaluation process")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
