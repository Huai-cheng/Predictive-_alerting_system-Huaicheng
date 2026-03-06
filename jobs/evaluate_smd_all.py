import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
import glob

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import create_sliding_windows
from src.model import get_baseline_model

DATA_DIR = "data/multivariate/"
W = 288
H = 6

def evaluate_file(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    # 1. Create Sliding Windows
    X, y = create_sliding_windows(df, W=W, H=H)
    
    # Check if there are any incidents at all
    if y.sum() == 0:
        return {"file": os.path.basename(file_path), "status": "No Incidents in data"}
        
    # 2. Train-test split (80% chronological)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    if len(np.unique(y_test)) < 2:
        return {"file": os.path.basename(file_path), "status": "No Incidents in Test Set"}
        
    # 3. Train Model
    model = get_baseline_model()
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    y_proba = model.predict_proba(X_test)[:, 1]
    
    if len(np.unique(y_test)) < 2:
         return {"file": os.path.basename(file_path), "status": "Only one class in y_test after proba"}   

    fpr_arr, tpr_arr, thresholds_arr = roc_curve(y_test, y_proba)
    youden_j = tpr_arr - fpr_arr
    optimal_idx = np.argmax(youden_j)
    threshold = float(thresholds_arr[optimal_idx])
    
    y_pred = (y_proba >= threshold).astype(int)
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    fpr = cm[0][1] / (cm[0][1] + cm[0][0]) if (cm[0][1] + cm[0][0]) > 0 else 0
    
    return {
        "file": os.path.basename(file_path),
        "status": "SUCCESS",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "threshold": threshold,
        "test_size": len(y_test),
        "test_incidents": int(y_test.sum()),
        "alerts_fired": int(y_pred.sum())
    }

def main():
    print("=== Predictive Alerting Batch Evaluation (SMD Multivariate) ===")
    print(f"Parameters: W={W}, H={H}, scale_pos_weight=5")
    print("-" * 105)
    print(f"{'File':<25} | {'Prec':<6} | {'Recall':<6} | {'FPR':<6} | {'ROC-AUC':<7} | {'Incidents':<15} | {'Notes'}")
    print("-" * 105)
    
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    
    results = []
    
    for f in files:
        res = evaluate_file(f)
        results.append(res)
        
        fname = res['file']
        if res['status'] == "SUCCESS":
            p = f"{res['precision']:.3f}"
            r = f"{res['recall']:.3f}"
            fpr = f"{res['fpr']:.3f}"
            roc = f"{res['roc_auc']:.3f}"
            counts = f"{res['test_incidents']}/{res['test_size']}"
            print(f"{fname:<25} | {p:<6} | {r:<6} | {fpr:<6} | {roc:<7} | {counts:<15} | Threshold: {res['threshold']:.3f} | Alerts: {res['alerts_fired']}")
        else:
            print(f"{fname:<25} | {'-':<6} | {'-':<6} | {'-':<6} | {'-':<7} | {'-':<15} | {res['status']}")
            
    # Calculate averages for successful files
    success = [r for r in results if r['status'] == 'SUCCESS']
    if success:
        avg_p = np.mean([r['precision'] for r in success])
        avg_r = np.mean([r['recall'] for r in success])
        avg_f = np.mean([r['fpr'] for r in success])
        avg_roc = np.mean([r['roc_auc'] for r in success])
        
        print("-" * 105)
        print(f"{'AVERAGE (Success)':<25} | {avg_p:.3f}  | {avg_r:.3f}  | {avg_f:.3f}  | {avg_roc:.3f}   |                 |")
        print("-" * 105)
        
        # Write to markdown table
        with open("notebooks/smd_results.md", "w") as f:
            f.write("# SMD Multivariate Evaluation Results\n\n")
            f.write(f"Parameters: `W={W}`, `H={H}`, `scale_pos_weight=5`\n\n")
            f.write("| File | Precision | Recall | FPR | ROC-AUC | Incidents (Test) | Threshold | Alerts |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in success:
                f.write(f"| {r['file']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['fpr']:.3f} | {r['roc_auc']:.3f} | {r['test_incidents']}/{r['test_size']} | {r['threshold']:.3f} | {r['alerts_fired']} |\n")
            f.write(f"| **AVERAGE** | **{avg_p:.3f}** | **{avg_r:.3f}** | **{avg_f:.3f}** | **{avg_roc:.3f}** | - | - | - |\n")

if __name__ == "__main__":
    main()
