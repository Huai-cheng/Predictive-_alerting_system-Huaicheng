import pandas as pd
import numpy as np
import os
import argparse

def generate_synthetic_data(output_path: str, num_days: int = 14, interval_min: int = 5):
    """
    Generates synthetic MULTIVARIATE cloud metrics data with 3 correlated signals:
    - cpu: CPU utilization (0-1 normalized). The primary target metric.
    - network_in: Network In bytes (normalized). Spikes 1-5 ticks BEFORE a CPU incident.
    - ram_pct: RAM utilization %. Slowly climbs in the hours BEFORE a CPU incident.
    
    Incidents have a slow, rolling build-up over several hours, allowing the 
    velocity, EMA and multivariate features to detect them before they breach threshold.
    """
    print(f"Generating synthetic MULTIVARIATE data ({num_days} days at {interval_min}m intervals)...")
    
    # Generate base timeline
    periods = int(num_days * 24 * 60 / interval_min)
    timestamps = pd.date_range(start='2024-01-01', periods=periods, freq=f'{interval_min}min')
    
    # 1. Base diurnal pattern (sine wave to simulate human traffic)
    time_numeric = np.arange(periods)
    daily_cycle = 24 * 60 / interval_min  # steps per day
    base_cpu = 0.3 + 0.1 * np.sin(2 * np.pi * time_numeric / daily_cycle)
    base_ram = 0.5 + 0.05 * np.sin(2 * np.pi * time_numeric / daily_cycle + np.pi / 4) # RAM lags slightly
    base_net = 0.2 + 0.08 * np.sin(2 * np.pi * time_numeric / daily_cycle)

    # 2. Add realistic, small random noise
    cpu_noise  = np.random.normal(0, 0.03, periods)
    ram_noise  = np.random.normal(0, 0.02, periods)
    net_noise  = np.random.normal(0, 0.04, periods)

    # 3. Inject Predictable Anomalies with correlated signals
    cpu_anomaly = np.zeros(periods)
    ram_anomaly = np.zeros(periods)
    net_anomaly = np.zeros(periods)

    # We will inject 20 incidents (enough so ~20% = 4 incidents land in test set)
    np.random.seed(42)
    incident_starts = np.random.choice(range(288, periods - 288), size=20, replace=False)
    
    for start_idx in incident_starts:
        buildup_duration = 36  # 3 hours of slow buildup
        peak_duration    = 12  # 1 hour of maxed-out failure

        # CPU: exponential buildup then spike
        buildup_curve = np.exp(np.linspace(0, 2.5, buildup_duration)) / 10
        cpu_anomaly[start_idx : start_idx + buildup_duration] += buildup_curve
        cpu_anomaly[start_idx + buildup_duration : start_idx + buildup_duration + peak_duration] += 1.5

        # RAM: starts climbing 2 hours BEFORE the CPU buildup (memory leak effect)
        ram_lead = 24  # 2 hours before start_idx
        ram_start = max(0, start_idx - ram_lead)
        ram_slope = np.linspace(0, 0.4, start_idx - ram_start + buildup_duration + peak_duration)
        ram_anomaly[ram_start : start_idx + buildup_duration + peak_duration] += ram_slope

        # Network: spikes sharply 2-5 ticks BEFORE the CPU incident (traffic flood)
        net_lead = np.random.randint(2, 6)
        net_start = max(0, start_idx - net_lead)
        # Short sharp spike in network traffic preceding the crash
        net_spike_len = net_lead + buildup_duration // 3
        net_spike = np.exp(np.linspace(0, 2.0, net_spike_len)) / 5
        net_anomaly[net_start : net_start + net_spike_len] += net_spike

    # Combine signals
    cpu_values = np.clip(base_cpu + cpu_noise + cpu_anomaly, 0.05, 2.5)
    ram_pct    = np.clip(base_ram + ram_noise + ram_anomaly, 0.10, 1.0)
    network_in = np.clip(base_net + net_noise + net_anomaly, 0.02, 3.0)
    
    df = pd.DataFrame({
        'timestamp':  timestamps,
        'value':      cpu_values,   # primary metric (CPU), kept as 'value' for compatibility
        'ram_pct':    ram_pct,
        'network_in': network_in,
    })
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved multivariate synthetic dataset to: {output_path}")
    print(f"  Columns: {list(df.columns)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic multivariate dataset")
    parser.add_argument("--output", type=str, default="data/synthetic_cpu.csv", help="Path to save CSV")
    args = parser.parse_args()
    
    generate_synthetic_data(args.output)
