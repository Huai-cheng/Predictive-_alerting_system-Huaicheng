import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import download_nab_dataset, generate_labels
from src.features import create_sliding_windows
from src.model import load_model_from_s3_mock

DATA_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv"
DATA_PATH = "data/rds_cpu_cc0c53.csv"
MODEL_ARTIFACT_PATH = "s3_mock/rds_model.pkl"
PLOT_OUTPUT_PATH = "notebooks/results_visualization_rds.png"

W = 288
H = 12

def main():
    print("=== Predictive Alerting Visualization (AWS RDS) ===")
    
    # 1. Load Model
    try:
        model_data = load_model_from_s3_mock(MODEL_ARTIFACT_PATH)
        if isinstance(model_data, dict):
            model = model_data['model']
            threshold = model_data.get('threshold', 0.5)
        else:
            model = model_data
            threshold = 0.5
        print(f"Loaded model with dynamic threshold: {threshold:.3f}")
    except FileNotFoundError:
        print(f"Model artifact not found. Please run 'python jobs/retrain_rds.py' first.")
        sys.exit(1)
        
    # 2. Fetch and Prepare Data
    print(f"Fetching and preparing test data from {DATA_PATH}...")
    df = download_nab_dataset(DATA_URL, DATA_PATH)
    df = generate_labels(df, window_size=288, z_threshold=3.0)
    
    # To keep the plot readable, we'll only visualize the last 20% (the test set)
    split_idx = int(len(df) * 0.8)
    # Ensure we include the W window size for the first test point
    df_eval = df.iloc[split_idx - W:].copy().reset_index(drop=True)
    
    # 3. Create Windows and Predict
    print("Generating predictions over the evaluation set...")
    X, y_true = create_sliding_windows(df_eval, W=W, H=H)
    
    # y_pred will be binary based on the new dynamic threshold
    y_proba = model.predict_proba(X)[:, 1]
    raw_pred = (y_proba >= threshold).astype(int)
    
    # Phase 4 Smoothing: Require 3 consecutive warnings to trigger an alert
    y_pred = np.zeros_like(raw_pred)
    consecutive = 0
    for i in range(len(raw_pred)):
        if raw_pred[i] == 1:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= 3:
            y_pred[i] = 1
    # The timestamps corresponding to the END of each window W (current time t)
    timestamps = df_eval['timestamp'].values[W - 1 : W - 1 + len(X)]
    cpu_values = df_eval['value'].values[W - 1 : W - 1 + len(X)]
    
    # True incident points *occurring at time t* (not the H horizon boolean)
    current_incident_labels = df_eval['label'].values[W - 1 : W - 1 + len(X)]
    
    # 4. Plotting
    print("Plotting results...")
    # Plotting probabilities allows us to see how "confident" the model is before trigging a binary alert
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # --- Top Plot: Raw Data & Actual Incidents ---
    ax1.plot(timestamps, cpu_values, label='CPU Utilization', color='#1f77b4', linewidth=1)
    
    # Highlight actual incident points
    incident_indices = np.where(current_incident_labels == 1)[0]
    ax1.scatter(timestamps[incident_indices], cpu_values[incident_indices], 
                color='red', label='Actual Incident (Z-Score > 3)', marker='X', s=50, zorder=5)
    
    ax1.set_title('Test Period: CPU Utilization vs Actual Incidents', fontsize=14, pad=10)
    ax1.set_ylabel('CPU Value')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # --- Bottom Plot: Model Predictions & Alerts ---
    # Plot predicted probabilities over time
    ax2.plot(timestamps, y_proba, label='Predicted Incident Probability (Next 1hr)', 
             color='purple', linewidth=1.5)
    
    # Draw logic threshold line
    ax2.axhline(y=threshold, color='orange', linestyle='--', label=f'Alert Threshold ({threshold:.3f})')
    
    # Highlight predicted alert triggers
    alert_indices = np.where(y_pred == 1)[0]
    ax2.scatter(timestamps[alert_indices], y_proba[alert_indices], 
                color='orange', label='Triggered Alert', marker='o', s=30, zorder=5)
    
    # Add vertical lines to connect Alerts to Actual CPU bumps (optional, makes plot noisy if many alerts)
    # We can highlight regions where probabilities are high instead
    ax2.fill_between(timestamps, y_proba, threshold, where=(y_proba > threshold), color='orange', alpha=0.3)
    
    ax2.set_title('Model Predictions (Probability of Incident in Next 1 Hour)', fontsize=14, pad=10)
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Timestamp')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Format x-axis dates nicely
    date_form = DateFormatter("%m-%d %H:00")
    ax2.xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=45)
    
    # --- Evaluation Metrics Text Box ---
    y_true_horizon = np.max(y_true, axis=1) if len(y_true.shape) > 1 else y_true
    precision = precision_score(y_true_horizon, y_pred, zero_division=0)
    recall = recall_score(y_true_horizon, y_pred, zero_division=0)
    f1 = f1_score(y_true_horizon, y_pred, zero_division=0)
    
    if len(np.unique(y_true_horizon)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true_horizon, y_pred).ravel()
    else:
        cm = confusion_matrix(y_true_horizon, y_pred)
        tn, fp, fn, tp = (cm[0][0], 0, 0, 0) if y_true_horizon[0] == 0 else (0, 0, 0, cm[0][0])

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    alerts_fired = int(np.sum(y_pred))
    
    metrics_text = (f"Precision: {precision:.3f}    Recall: {recall:.3f}    "
                    f"False Positive Rate: {fpr:.3f}    F1-Score: {f1:.3f}    "
                    f"Alerts Fired: {alerts_fired}    True Positives: {tp}  |  "
                    f"Metrics: CPU Only")
                    
    fig.text(0.5, 0.08, "Evaluation Metrics (20% Held-Out Test Period)", ha='center', fontsize=12)
    fig.text(0.5, 0.04, metrics_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor='black', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(PLOT_OUTPUT_PATH, dpi=150)
    print(f"\nSaved visualization to {PLOT_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
