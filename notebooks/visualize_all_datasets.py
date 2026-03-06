"""
notebooks/visualize_all_datasets.py
Loads the unified model and generates an individual 3-panel visualization
PNG for each of the 12 datasets. Saves a metrics_summary.csv at the end.

Output folder: notebooks/results/unified/

Run from project root:
    python notebooks/visualize_all_datasets.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import precision_score, recall_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import generate_labels
from src.features import create_sliding_windows
from src.model import load_model_from_s3_mock

W = 288
H = 12
MODEL_PATH  = "s3_mock/unified_model.pkl"
OUTPUT_DIR  = "notebooks/results/unified"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MULTIVARIATE_DIR = "data/multivariate"
CPU_ONLY_DIR     = "data/real_cpu_only"

CHART_TITLE_MAP = {
    "01_smd_machine_1_1": "SMD Server 01 (machine-1-1) — Real Production",
    "02_smd_machine_1_2": "SMD Server 02 (machine-1-2) — Real Production",
    "03_smd_machine_1_3": "SMD Server 03 (machine-1-3) — Real Production",
    "04_smd_machine_1_4": "SMD Server 04 (machine-1-4) — Real Production",
    "05_smd_machine_1_5": "SMD Server 05 (machine-1-5) — Real Production",
    "06_smd_machine_1_6": "SMD Server 06 (machine-1-6) — Real Production",
    "07_smd_machine_1_7": "SMD Server 07 (machine-1-7) — Real Production",
    "08_smd_machine_1_8": "SMD Server 08 (machine-1-8) — Real Production",
    "09_smd_machine_2_1": "SMD Server 09 (machine-2-1) — Real Production",
    "10_smd_machine_2_2": "SMD Server 10 (machine-2-2) — Real Production",
    "11_ec2_production":  "AWS EC2 CPU (Real CloudWatch) — CPU Only",
    "12_rds_database":    "AWS RDS CPU (Real CloudWatch) — CPU Only",
}


def get_all_datasets():
    datasets = []
    for f in sorted(os.listdir(MULTIVARIATE_DIR)):
        if f.endswith(".csv"):
            datasets.append((os.path.join(MULTIVARIATE_DIR, f), f.replace(".csv", "")))
    for f in sorted(os.listdir(CPU_ONLY_DIR)):
        if f.endswith(".csv"):
            datasets.append((os.path.join(CPU_ONLY_DIR, f), f.replace(".csv", "")))
    return datasets


def run_and_visualize(path: str, name: str, model, threshold: float):
    """Load dataset, run inference with smoothing, plot 3-panel chart."""

    # Load + label
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['ram_pct', 'network_in']:
        if col not in df.columns:
            df[col] = 0.0
    df = generate_labels(df, window_size=W, z_threshold=3.0)

    # 80/20 split — visualize the TEST portion (last 20%)
    X, y_true = create_sliding_windows(df, W=W, H=H)
    split_idx  = int(len(X) * 0.8)
    X_eval     = X[split_idx:]
    y_eval     = y_true[split_idx:]

    # Timestamps and CPU values for the eval window
    ts_arr  = df['timestamp'].values
    cpu_arr = df['value'].values
    ts_eval = ts_arr[W - 1 + split_idx : W - 1 + split_idx + len(X_eval)]
    cpu_eval = cpu_arr[W - 1 + split_idx : W - 1 + split_idx + len(X_eval)]
    label_eval = df['label'].values[W - 1 + split_idx : W - 1 + split_idx + len(X_eval)]

    # Predict
    y_proba  = model.predict_proba(X_eval)[:, 1]

    # Phase 4 Smoothing: 3 consecutive windows
    raw_pred = (y_proba >= threshold).astype(int)
    y_pred   = np.zeros_like(raw_pred)
    consec   = 0
    for i in range(len(raw_pred)):
        consec = consec + 1 if raw_pred[i] == 1 else 0
        if consec >= 3:
            y_pred[i] = 1

    # Compute metrics
    precision  = precision_score(y_eval, y_pred, zero_division=0)
    recall     = recall_score(y_eval, y_pred, zero_division=0)
    cm         = confusion_matrix(y_eval, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (int((y_eval==0).sum()), 0, 0, 0))
    total_neg  = tn + fp
    fpr        = fp / total_neg if total_neg > 0 else 0.0
    f1         = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # ---- Chart ----
    title   = CHART_TITLE_MAP.get(name, name)
    is_cpu_only = name.startswith("11_") or name.startswith("12_")
    metric_note = "Metrics: CPU only (RAM/Network zero-padded)" if is_cpu_only else "Metrics: CPU + RAM + Network In"

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[2, 2, 0.8], hspace=0.45)

    # --- TOP: CPU Signal + Actual Incidents ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(ts_eval, cpu_eval, color='#1f77b4', linewidth=1.0, label='CPU Utilization')
    incident_idx = np.where(label_eval == 1)[0]
    if len(incident_idx) > 0:
        ax1.scatter(ts_eval[incident_idx], cpu_eval[incident_idx],
                    color='red', marker='X', s=60, zorder=5, label='Actual Incident (Z-Score > 3)')
    ax1.set_title(f'Test Period: CPU Utilization vs Actual Incidents\n{title}',
                  fontsize=13, pad=8)
    ax1.set_ylabel('CPU Value')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # --- MIDDLE: Model Predictions ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(ts_eval, y_proba, color='purple', linewidth=1.2,
             label=f'Predicted Incident Probability (Next 1hr)')
    ax2.fill_between(ts_eval, y_proba, alpha=0.15, color='orange')
    ax2.axhline(y=threshold, color='orange', linestyle='--',
                linewidth=1.0, label=f'Alert Threshold ({threshold:.3f})')
    alert_idx = np.where(y_pred == 1)[0]
    if len(alert_idx) > 0:
        ax2.scatter(ts_eval[alert_idx], y_proba[alert_idx],
                    color='orange', s=18, zorder=5, label='Triggered Alert')
    ax2.set_title('Model Predictions (Probability of Incident in Next 1 Hour)', fontsize=12, pad=8)
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Timestamp')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- BOTTOM: Metrics Scorecard ---
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    scorecard = (
        f"  Precision: {precision:.3f}     "
        f"Recall: {recall:.3f}     "
        f"False Positive Rate: {fpr:.3f}     "
        f"F1-Score: {f1:.3f}     "
        f"Alerts Fired: {int(y_pred.sum())}     "
        f"True Positives: {int(tp)}   |   {metric_note}"
    )
    ax3.text(0.0, 0.6, scorecard, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.8))
    ax3.set_title('Evaluation Metrics (20% Held-Out Test Period)', fontsize=11, pad=4)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()

    return {
        'Dataset':          name,
        'Title':            title,
        'Precision':        round(precision, 4),
        'Recall':           round(recall, 4),
        'F1':               round(f1, 4),
        'FP Rate':          round(fpr, 4),
        'Alerts Fired':     int(y_pred.sum()),
        'True Positives':   int(tp),
        'False Positives':  int(fp),
        'Chart':            out_path,
    }


def main():
    print("=== Unified Dataset Visualization ===\n")

    model_data = load_model_from_s3_mock(MODEL_PATH)
    model      = model_data['model']
    threshold  = model_data.get('threshold', 0.5)
    print(f"Loaded unified model | Threshold: {threshold:.4f}\n")

    datasets = get_all_datasets()
    results  = []

    for path, name in datasets:
        print(f"Processing {name}...", end=" ", flush=True)
        try:
            row = run_and_visualize(path, name, model, threshold)
            results.append(row)
            print(f"Precision={row['Precision']:.3f} | Recall={row['Recall']:.3f} | FPR={row['FP Rate']:.3f}")
        except Exception as e:
            print(f"ERROR: {e}")

    # Save summary CSV
    df_summary = pd.DataFrame(results)
    csv_path   = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    print(f"\n{'='*80}")
    print("METRICS SUMMARY — All 12 Datasets")
    print(f"{'='*80}")
    display_cols = ['Dataset', 'Precision', 'Recall', 'F1', 'FP Rate', 'Alerts Fired', 'True Positives']
    print(df_summary[display_cols].to_string(index=False))
    print(f"{'='*80}")
    print(f"\nSummary CSV saved to: {csv_path}")
    print(f"Individual PNGs saved to: {OUTPUT_DIR}/")
    print("=== Visualization Complete ===")


if __name__ == "__main__":
    main()
