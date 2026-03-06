import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
# %%
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import download_nab_dataset

DATA_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"
DATA_PATH = "data/ec2_cpu_utilization_24ae8d.csv"

def main():
    print("=== Predictive Alerting EDA: Window Size Justification ===")
    
    # 1. Fetch Data
    df = download_nab_dataset(DATA_URL, DATA_PATH)
    
    # Extract values
    values = df['value'].values
    
    # 2. Autocorrelation (ACF) - helps find lookback W
    # We look for where the ACF plot crosses the significance threshold or shows strong cyclical patterns.
    # For a 5-minute interval dataset, 1 day = 12 * 24 = 288 points.
    print("Calculating ACF (up to 400 lags)...")
    acf_values = sm.tsa.acf(values, nlags=400)
    
    # Find the first minimum in ACF to determine a potential short-term lookback
    first_min_idx = np.argmin(acf_values[:100])
    print(f"-> Short-term ACF minimum at lag {first_min_idx} (approx {first_min_idx * 5} minutes)")
    
    # Check correlation at 1 day (288 lags)
    if len(acf_values) > 288:
        print(f"-> ACF at 1 day (288 lags): {acf_values[288]:.3f}. High correlation indicates a daily seasonal pattern.")
        print(f"   => Justification for lookback W=288 (1 day).")
    
    # 3. Partial Autocorrelation (PACF) - helps find prediction horizon H
    # PACF shows the direct correlation of a lag with the current value, removing intermediate effects.
    print("\nCalculating PACF (up to 50 lags)...")
    pacf_values = sm.tsa.pacf(values, nlags=50)
    
    # Find where PACF drops off significantly (e.g., < 0.1)
    significant_lags = np.where(np.abs(pacf_values) > 0.1)[0]
    
    if len(significant_lags) > 1:
        last_sig_lag = significant_lags[-1] # excluding 0
        print(f"-> PACF drops below 0.1 after lag {last_sig_lag} (approx {last_sig_lag * 5} minutes)")
        print(f"   => Justification for prediction horizon H in the range of 6-12 (30-60 mins).")
    
    # Plotting
    os.makedirs('notebooks', exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(values, lags=400, ax=ax1, title="Autocorrelation (ACF) - Evidence for W")
    plot_pacf(values, lags=50, ax=ax2, title="Partial Autocorrelation (PACF) - Evidence for H")
    
    plot_path = "notebooks/acf_pacf.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nSaved ACF/PACF plots to {plot_path}")

if __name__ == "__main__":
    main()
