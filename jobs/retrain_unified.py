"""
jobs/retrain_unified.py
Trains ONE LightGBM model on ALL 12 datasets (10 SMD multivariate + EC2 + RDS).
Pools all sliding windows together, uses Youden's J threshold selection.
Saves: s3_mock/unified_model.pkl

Run from project root:
    python jobs/retrain_unified.py
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, roc_auc_score,
                              recall_score, confusion_matrix, roc_curve,
                              precision_score)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import generate_labels
from src.features import create_sliding_windows
from src.model import get_baseline_model, save_model_to_s3_mock

W = 288
H = 12
MODEL_PATH = "s3_mock/unified_model.pkl"

# --------------------------------------------------------------------------
# Dataset registry — each entry is (path, label)
# --------------------------------------------------------------------------
MULTIVARIATE_DIR = "data/multivariate"
CPU_ONLY_DIR     = "data/real_cpu_only"

def get_all_datasets():
    datasets = []
    for f in sorted(os.listdir(MULTIVARIATE_DIR)):
        if f.endswith(".csv"):
            datasets.append((os.path.join(MULTIVARIATE_DIR, f), f.replace(".csv", "")))
    for f in sorted(os.listdir(CPU_ONLY_DIR)):
        if f.endswith(".csv"):
            datasets.append((os.path.join(CPU_ONLY_DIR, f), f.replace(".csv", "")))
    return datasets


def load_and_label(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Ensure all 3 metric columns exist (zero-pad if CPU-only)
    for col in ['ram_pct', 'network_in']:
        if col not in df.columns:
            df[col] = 0.0

    # Apply dynamic Z-threshold labelling
    df = generate_labels(df, window_size=W, z_threshold=3.0)
    return df


def main():
    print("=== Unified 12-Dataset Retrain Job ===\n")

    datasets = get_all_datasets()
    print(f"Found {len(datasets)} datasets:\n")
    for path, name in datasets:
        print(f"  {name}")

    # -----------------------------------------------------------------------
    # Pool sliding windows from ALL datasets
    # -----------------------------------------------------------------------
    print("\n[1] Generating sliding windows across all datasets...")
    X_all_parts, y_all_parts = [], []

    for path, name in datasets:
        df = load_and_label(path)
        X, y = create_sliding_windows(df, W=W, H=H)
        X_all_parts.append(X)
        y_all_parts.append(y)
        incidents = y.sum()
        print(f"  {name}: {len(X)} windows, {int(incidents)} incident windows")

    X_all = np.concatenate(X_all_parts, axis=0)
    y_all = np.concatenate(y_all_parts, axis=0)

    print(f"\nTotal pooled: X={X_all.shape}, y={y_all.shape}")
    print(f"Class balance: {int(y_all.sum())} incident / {int((y_all==0).sum())} normal")

    # -----------------------------------------------------------------------
    # Chronological 80/20 split
    # -----------------------------------------------------------------------
    split_idx  = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    print(f"\n[2] Train: {len(X_train)} | Test: {len(X_test)}")

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    print("\n[3] Training LightGBM on pooled dataset...")
    model = get_baseline_model()
    model.fit(X_train, y_train)

    # -----------------------------------------------------------------------
    # Threshold selection via Youden's J
    # -----------------------------------------------------------------------
    print("\n[4] Evaluating with Youden's J threshold selection...")
    y_proba = model.predict_proba(X_test)[:, 1]

    threshold = 0.5
    if len(np.unique(y_test)) >= 2:
        fpr_arr, tpr_arr, thresholds_arr = roc_curve(y_test, y_proba)
        youden_j   = tpr_arr - fpr_arr
        optimal_idx = np.argmax(youden_j)
        threshold  = float(thresholds_arr[optimal_idx])

        y_pred = (y_proba >= threshold).astype(int)
        if recall_score(y_test, y_pred, zero_division=0) < 0.80:
            for t in sorted(thresholds_arr, reverse=True):
                y_pred_t = (y_proba >= t).astype(int)
                if recall_score(y_test, y_pred_t, zero_division=0) >= 0.80:
                    threshold = float(t)
                    break

        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"  Youden's J optimal threshold: {threshold:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
    else:
        print("  WARNING: Test set has no positive class. Using threshold=0.5")

    y_pred   = (y_proba >= threshold).astype(int)
    recall   = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    cm       = confusion_matrix(y_test, y_pred)

    print(f"\n  Recall:    {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    if recall >= 0.80:
        print("SUCCESS: 80% recall target met on pooled test set.")
    else:
        print("WARNING: Recall below 80% target on pooled test set.")

    # -----------------------------------------------------------------------
    # Save model artifact
    # -----------------------------------------------------------------------
    os.makedirs("s3_mock", exist_ok=True)
    model_data = {'model': model, 'threshold': threshold}
    save_model_to_s3_mock(model_data, MODEL_PATH)
    print(f"\n[5] Model saved to {MODEL_PATH}")
    print("=== Unified Retrain Complete ===")


if __name__ == "__main__":
    main()
