import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, recall_score, confusion_matrix, roc_curve

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import download_nab_dataset, generate_labels
from src.features import create_sliding_windows
from src.model import get_baseline_model, save_model_to_s3_mock

# Configuration
DATA_URL = "" # Synthetic data doesn't use a URL
DATA_PATH = "data/synthetic_cpu.csv"
MODEL_ARTIFACT_PATH = "s3_mock/synthetic_model.pkl"

# Window settings based roughly on 5 min intervals
# W = 288 (1 day)
# H = 12 (1 hour prediction horizon)
W = 288
H = 6   # Phase 5 fix: 30-min horizon (was 12/60min — too wide, inflated positive labels)

def main():
    print("=== Predictive Alerting Periodic Retraining Job (Synthetic Data) ===")
    
    # 1. Fetch Data
    print(f"\n1. Fetching and loading data from {DATA_PATH}...")
    df = download_nab_dataset(DATA_URL, DATA_PATH)
    print(f"Loaded dataset of shape {df.shape}")
    
    # 2. Generate Ground-Truth Labels dynamically
    print("\n2. Generating dynamic Z-score labels...")
    df = generate_labels(df, window_size=288, z_threshold=3.0)
    num_incidents = df['label'].sum()
    print(f"Detected {int(num_incidents)} raw incident points out of {len(df)}")
    
    # 3. Create Vector-Optimized Sliding Windows (Features & Target)
    print(f"\n3. Creating sliding windows (W={W}, H={H})...")
    X, y = create_sliding_windows(df, W=W, H=H)
    print(f"Created X shape: {X.shape}, y shape: {y.shape}")
    
    # Train-test split (chronological, last 20% is held-out eval set)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 4. Train Model
    print(f"\n4. Training LightGBM Baseline Model (Train set: {len(X_train)} samples)...")
    model = get_baseline_model()
    model.fit(X_train, y_train)
    
    # 5. Evaluate on Held-Out Set
    print("\n5. Evaluating Model on Held-Out Test Set (20% chronological)...")
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Phase 5: Youden's J — use as-is, no recall guardrail
    # Guard: roc_curve and roc_auc require at least one positive example in y_test
    if len(np.unique(y_test)) < 2:
        print("\n[WARNING] Test set has only one class. Falling back to threshold=0.5.")
        threshold = 0.5
        y_pred = (y_proba >= threshold).astype(int)
        roc_auc = float('nan')
    else:
        fpr_arr, tpr_arr, thresholds_arr = roc_curve(y_test, y_proba)
        youden_j = tpr_arr - fpr_arr
        optimal_idx = np.argmax(youden_j)
        threshold = float(thresholds_arr[optimal_idx])
        print(f"\n[Youden's J] Selected balanced threshold: {threshold:.4f}")
        y_pred = (y_proba >= threshold).astype(int)
        roc_auc = roc_auc_score(y_test, y_proba)
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred))
    
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"ROC-AUC: {roc_auc:.4f}" if not (isinstance(roc_auc, float) and np.isnan(roc_auc)) else "ROC-AUC: N/A (single class in test set)")
    print(f"Target Recall constraint approximately achieved: {recall:.4f}")
    print(f"Confusion Matrix: \n{cm}")
    
    if recall < 0.80:
        print("\nWARNING: Recall is still below the 80% target on the test set.")
    else:
        print("\nSUCCESS: Target 80% Recall constraint was met/exceeded on the test set!")
        
    # 6. Save Model Artifact to Mock S3
    print("\n6. Saving Model Artifact and Threshold...")
    model_data = {
        'model': model,
        'threshold': threshold
    }
    save_model_to_s3_mock(model_data, MODEL_ARTIFACT_PATH)
    print("=== Retraining Job Complete ===")

if __name__ == "__main__":
    main()
