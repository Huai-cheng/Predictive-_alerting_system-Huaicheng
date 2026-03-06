import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import download_nab_dataset, generate_labels
from src.features import generate_streaming_features
from src.model import load_model_from_s3_mock

DATA_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"
DATA_PATH = "data/ec2_cpu_utilization_24ae8d.csv"
MODEL_ARTIFACT_PATH = "s3_mock/model.pkl"

W = 288
H = 12

def main():
    print("=== Predictive Alerting Streaming Inference Job ===")
    
    # 1. Load Model from Mock S3
    print(f"\n1. Loading Model from {MODEL_ARTIFACT_PATH}...")
    try:
        model_data = load_model_from_s3_mock(MODEL_ARTIFACT_PATH)
        # Handle backward compatibility if the old model format was saved
        if isinstance(model_data, dict):
            model = model_data['model']
            threshold = model_data.get('threshold', 0.5)
        else:
            model = model_data
            threshold = 0.5
        print(f"Loaded model with dynamic threshold: {threshold:.3f}")
    except FileNotFoundError:
        print("Model artifact not found. Please run jobs/retrain_ec2.py first.")
        sys.exit(1)
        
    # 2. Simulate streaming data (fetching recent points)
    print("\n2. Simulating streaming data...")
    df = download_nab_dataset(DATA_URL, DATA_PATH)
    df = generate_labels(df, window_size=288, z_threshold=3.0)
    
    # Simulate a stream by taking the last N points
    # We need at least W points to form a window
    stream_idx_start = len(df) - W - 50 # Simulate 50 streaming steps
    
    print("\n3. Running Streaming Inference...")
    alerts_triggered = 0
    consecutive_high_probs = 0 # Phase 4: Smoothing variable
    for i in range(stream_idx_start, len(df)):
        # At time t=i, we have data up to index i (inclusive)
        # We need the last W points [i-W+1 : i+1]
        window_raw_data = df.iloc[i-W+1 : i+1]['value'].values
        timestamp = df.iloc[i]['timestamp']
        
        # Ground truth (did an incident actually happen in the next H steps?)
        future_labels = df.iloc[i+1 : i+1+H]['label'].values
        actual_incident_in_horizon = int(np.max(future_labels)) if len(future_labels) > 0 else 0
        
        # Generate features dynamically for this single window
        features = generate_streaming_features(window_raw_data, current_timestamp=timestamp)
        
        # Predict
        y_proba = model.predict_proba(features)[0][1]
        
        # Phase 4: Consecutive Alerting Logic (Smoothing)
        if y_proba >= threshold:
            consecutive_high_probs += 1
        else:
            consecutive_high_probs = 0
            
        # Require 3 consecutive warnings to officially predict an incident
        if consecutive_high_probs >= 3:
            y_pred = 1
        else:
            y_pred = 0
        
        # Mock triggering an alert
        if y_pred == 1:
            alerts_triggered += 1
            print(f"[ALERT] Time: {timestamp} | P(Incident in next {H} steps) = {y_proba:.3f} (Threshold: {threshold:.3f}) | Consecutive Warnings: {consecutive_high_probs} | Actual: {actual_incident_in_horizon}")
            
    print(f"\n=== Streaming Job Complete. Triggered {alerts_triggered} alerts out of 50 steps. ===")

if __name__ == "__main__":
    main()
