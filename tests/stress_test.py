"""
Stress Test Battery for the Predictive Alerting System.

Generates 8 distinct cloud scenarios and runs them through the trained
Synthetic model's inference logic, reporting pass/fail for each.

Run from project root:
    python tests/stress_test.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import load_model_from_s3_mock
from src.features import create_sliding_windows
from src.data import generate_labels

MODEL_PATH = "s3_mock/synthetic_model.pkl"
W = 288
H = 12
OUTPUT_DIR = "notebooks/stress_test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Scenario Generators
# ---------------------------------------------------------------------------

def _base_frame(num_days: int = 3, interval_min: int = 5) -> pd.DataFrame:
    """Base quiet CPU signal: diurnal sine + tiny noise."""
    periods = int(num_days * 24 * 60 / interval_min)
    ts = pd.date_range(start='2024-01-01', periods=periods, freq=f'{interval_min}min')
    t  = np.arange(periods)
    daily = 24 * 60 / interval_min
    cpu   = 0.30 + 0.08 * np.sin(2 * np.pi * t / daily) + np.random.normal(0, 0.015, periods)
    ram   = 0.50 + 0.04 * np.sin(2 * np.pi * t / daily) + np.random.normal(0, 0.010, periods)
    net   = 0.20 + 0.05 * np.sin(2 * np.pi * t / daily) + np.random.normal(0, 0.012, periods)
    return pd.DataFrame({'timestamp': ts,
                         'value':      np.clip(cpu, 0.05, 2.5),
                         'ram_pct':    np.clip(ram, 0.10, 1.0),
                         'network_in': np.clip(net, 0.02, 3.0)})


def _inject_buildup(df, idx, buildup=36, peak=12, height=1.5):
    """Standard slow-buildup incident (same pattern model was trained on)."""
    curve = np.exp(np.linspace(0, 2.5, buildup)) / 10
    df.loc[idx:idx+buildup-1, 'value'] += curve
    df.loc[idx+buildup:idx+buildup+peak-1, 'value'] += height
    # Correlated RAM / Network
    df.loc[max(0,idx-24):idx+buildup+peak-1, 'ram_pct'] += np.linspace(0, 0.35, min(24+buildup+peak, len(df)-max(0,idx-24)))[:len(df.loc[max(0,idx-24):idx+buildup+peak-1])]
    net_lead = 3
    net_len  = net_lead + buildup // 3
    net_spike = np.exp(np.linspace(0, 2.0, net_len)) / 5
    start = max(0, idx - net_lead)
    df.loc[start:start+net_len-1, 'network_in'] += net_spike[:len(df.loc[start:start+net_len-1])]
    return df


def scenario_1_quiet_baseline():
    np.random.seed(1)
    return _base_frame(3), "No incidents (Quiet Baseline)", 0


def scenario_2_sparse_incidents():
    np.random.seed(2)
    df = _base_frame(3)
    periods = len(df)
    df = _inject_buildup(df, 200)
    df = _inject_buildup(df, 600)
    df['value']      = np.clip(df['value'], 0.05, 2.5)
    df['ram_pct']    = np.clip(df['ram_pct'], 0.10, 1.0)
    df['network_in'] = np.clip(df['network_in'], 0.02, 3.0)
    return df, "2 Sparse Incidents", 2


def scenario_3_dense_storm():
    np.random.seed(3)
    df = _base_frame(3)
    for start in [150, 240, 360, 480, 620]:
        df = _inject_buildup(df, start, buildup=20, peak=8)
    df['value']      = np.clip(df['value'], 0.05, 2.5)
    df['ram_pct']    = np.clip(df['ram_pct'], 0.10, 1.0)
    df['network_in'] = np.clip(df['network_in'], 0.02, 3.0)
    return df, "5 Incidents — Dense Storm", 5


def scenario_4_zero_day_spike():
    """Instant vertical spike, no buildup — the model's hardest case."""
    np.random.seed(4)
    df = _base_frame(3)
    spike_idx = 500
    df.loc[spike_idx:spike_idx+5, 'value']  += 1.8   # instant spike
    df.loc[spike_idx:spike_idx+5, 'network_in'] += 0.5
    df['value']      = np.clip(df['value'], 0.05, 2.5)
    df['ram_pct']    = np.clip(df['ram_pct'], 0.10, 1.0)
    df['network_in'] = np.clip(df['network_in'], 0.02, 3.0)
    return df, "Zero-Day Spike (No Buildup)", 1


def scenario_5_memory_leak_drift():
    """CPU creeps slowly from 30% to 95% over 12 hours via RAM pressure."""
    np.random.seed(5)
    df = _base_frame(3)
    drift_start = 200
    drift_len   = 144  # 12 hours at 5-min intervals
    drift_curve = np.linspace(0, 1.5, drift_len)
    df.loc[drift_start:drift_start+drift_len-1, 'value']   += drift_curve[:len(df.loc[drift_start:drift_start+drift_len-1])]
    df.loc[drift_start:drift_start+drift_len-1, 'ram_pct'] += np.linspace(0, 0.45, drift_len)[:len(df.loc[drift_start:drift_start+drift_len-1])]
    df['value']      = np.clip(df['value'], 0.05, 2.5)
    df['ram_pct']    = np.clip(df['ram_pct'], 0.10, 1.0)
    df['network_in'] = np.clip(df['network_in'], 0.02, 3.0)
    return df, "Memory Leak Drift (12-hour buildup)", 1


def scenario_6_seasonal_false_alarm_trap():
    """Regular daily spikes (scheduled batch jobs) that are NOT real incidents."""
    np.random.seed(6)
    df = _base_frame(3)
    daily = int(24 * 60 / 5)  # steps per day
    for day in range(3):
        job_start = day * daily + 84  # 7am each day
        spike_len = 12  # short 1-hour spike
        df.loc[job_start:job_start+spike_len-1, 'value'] += np.linspace(0.3, 0.0, spike_len)[:len(df.loc[job_start:job_start+spike_len-1])]
    df['value']      = np.clip(df['value'], 0.05, 2.5)
    df['ram_pct']    = np.clip(df['ram_pct'], 0.10, 1.0)
    df['network_in'] = np.clip(df['network_in'], 0.02, 3.0)
    return df, "Seasonal False Alarm Trap (Scheduled Jobs)", 0


def scenario_7_noisy_chaos():
    """RDS-style: high-frequency random noise, no real incidents."""
    np.random.seed(7)
    df = _base_frame(3)
    periods = len(df)
    df['value']      += np.random.normal(0, 0.15, periods)
    df['network_in'] += np.random.normal(0, 0.10, periods)
    df['value']      = np.clip(df['value'], 0.05, 2.5)
    df['ram_pct']    = np.clip(df['ram_pct'], 0.10, 1.0)
    df['network_in'] = np.clip(df['network_in'], 0.02, 3.0)
    return df, "Noisy Chaos (No True Incidents)", 0


def scenario_8_recovery():
    """Incident starts, system auto-heals — CPU ramps down mid-crash."""
    np.random.seed(8)
    df = _base_frame(3)
    start = 400
    buildup = 36
    # Partial crash: goes up then RECOVERS back to baseline
    ramp_up   = np.exp(np.linspace(0, 2.0, buildup)) / 10
    ramp_down = np.linspace(ramp_up[-1], 0, buildup)
    df.loc[start:start+buildup-1, 'value']            += ramp_up[:len(df.loc[start:start+buildup-1])]
    df.loc[start+buildup:start+2*buildup-1, 'value']  += ramp_down[:len(df.loc[start+buildup:start+2*buildup-1])]
    df['value']      = np.clip(df['value'], 0.05, 2.5)
    df['ram_pct']    = np.clip(df['ram_pct'], 0.10, 1.0)
    df['network_in'] = np.clip(df['network_in'], 0.02, 3.0)
    return df, "Recovery Scenario (Auto-Heal)", 0


# ---------------------------------------------------------------------------
# Inference + Smoothing
# ---------------------------------------------------------------------------

def run_inference(df, model, threshold):
    """Run the full inference pipeline and return probabilities + smoothed alerts."""
    df = generate_labels(df.copy(), window_size=W, z_threshold=3.0)
    X, y_true = create_sliding_windows(df, W=W, H=H)
    y_proba = model.predict_proba(X)[:, 1]

    # Phase 4 Consecutive Alert Smoothing
    raw_pred = (y_proba >= threshold).astype(int)
    y_pred   = np.zeros_like(raw_pred)
    consec   = 0
    for i in range(len(raw_pred)):
        consec = consec + 1 if raw_pred[i] == 1 else 0
        if consec >= 3:
            y_pred[i] = 1

    timestamps = df['timestamp'].values[W - 1: W - 1 + len(X)]
    cpu_values = df['value'].values[W - 1: W - 1 + len(X)]
    return timestamps, cpu_values, y_proba, y_pred, y_true


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Predictive Alerting Stress Test Battery ===\n")
    model_data = load_model_from_s3_mock(MODEL_PATH)
    model     = model_data['model']
    threshold = model_data.get('threshold', 0.5)
    print(f"Loaded model | Threshold: {threshold:.4f}\n")

    scenarios = [
        scenario_1_quiet_baseline,
        scenario_2_sparse_incidents,
        scenario_3_dense_storm,
        scenario_4_zero_day_spike,
        scenario_5_memory_leak_drift,
        scenario_6_seasonal_false_alarm_trap,
        scenario_7_noisy_chaos,
        scenario_8_recovery,
    ]

    results = []
    fig = plt.figure(figsize=(20, len(scenarios) * 3.5))
    gs  = gridspec.GridSpec(len(scenarios), 2, figure=fig, hspace=0.7, wspace=0.3)

    for i, gen_fn in enumerate(scenarios):
        df, name, expected_incidents = gen_fn()
        timestamps, cpu, y_proba, y_pred, y_true = run_inference(df, model, threshold)
        
        alerts_fired = int(y_pred.sum())
        true_positives = int((y_pred * y_true).sum())
        false_positives = int(((y_pred == 1) & (y_true == 0)).sum())
        
        # Determine PASS/FAIL
        if expected_incidents == 0:
            # No incidents: model should stay silent (0 consecutive alerts)
            status = "[PASS] Silent" if alerts_fired == 0 else f"[WARN] FP={alerts_fired}"
        else:
            # Incidents exist: model should fire at least once
            status = "[PASS] Detected" if alerts_fired > 0 else "[FAIL] Missed"

        results.append({
            'Scenario': name,
            'Expected Incidents': expected_incidents,
            'Alerts Fired': alerts_fired,
            'True Positives (windows)': true_positives,
            'False Positives (windows)': false_positives,
            'Status': status
        })
        print(f"[{i+1}] {name}")
        print(f"     Expected incidents: {expected_incidents} | Alerts fired: {alerts_fired} | {status}")

        # Left plot: CPU signal
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(range(len(cpu)), cpu, color='#1f77b4', linewidth=0.8)
        ax1.set_title(f"[{i+1}] {name}", fontsize=9, fontweight='bold')
        ax1.set_ylabel('CPU', fontsize=7)
        ax1.tick_params(labelsize=6)

        # Right plot: Model probability + alerts
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(range(len(y_proba)), y_proba, color='purple', linewidth=0.8)
        ax2.fill_between(range(len(y_proba)), y_proba, alpha=0.15, color='orange')
        ax2.axhline(y=threshold, color='orange', linestyle='--', linewidth=0.8)
        alert_idx = np.where(y_pred == 1)[0]
        ax2.scatter(alert_idx, y_proba[alert_idx], color='red', s=8, zorder=5)
        ax2.set_title(f"Probability | {status}", fontsize=9)
        ax2.set_ylabel('P(incident)', fontsize=7)
        ax2.tick_params(labelsize=6)

    plt.suptitle("Stress Test Battery — All 8 Cloud Scenarios", fontsize=14, fontweight='bold', y=1.01)
    out_path = os.path.join(OUTPUT_DIR, "stress_test_all_scenarios.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close()

    print(f"\nVisualization saved to: {out_path}")

    # Summary table
    print("\n" + "="*80)
    print("STRESS TEST SUMMARY")
    print("="*80)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print("="*80)

    passed = sum(1 for r in results if 'PASS' in r['Status'])
    print(f"\nFinal Score: {passed}/{len(results)} scenarios PASSED")

if __name__ == "__main__":
    main()
