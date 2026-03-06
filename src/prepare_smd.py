"""
src/prepare_smd.py
Downloads and standardizes 10 Server Machine Dataset (SMD) files from GitHub.
- Source: NetManAIOps/OmniAnomaly (1-minute intervals, 38 columns, normalized)
- Output: data/multivariate/01_smd_server_XX.csv
  Columns: timestamp, value (CPU), ram_pct, network_in
- All datasets standardized to exactly 14 days at 5-minute intervals (4032 rows)

Column mapping from SMD 38-feature space:
  Col 0  = cpu_r          -> value       (CPU utilization ratio)
  Col 5  = mem_buff_cache -> ram_pct     (memory pressure proxy, inverted: high value = high RAM use)
  Col 12 = net_send       -> network_in  (network bytes sent, proxy for incoming load on server)
"""

import os
import sys
import io
import requests
import numpy as np
import pandas as pd

# 38-column SMD, 1-min sampling. We resample to 5-min → take every 5th row.
SMD_BASE_URL = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset/train"

# 10 machines selected from groups 1 and 2 — diverse server types
SMD_MACHINES = [
    "machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4",
    "machine-1-5", "machine-1-6", "machine-1-7", "machine-1-8",
    "machine-2-1", "machine-2-2",
]

TARGET_ROWS     = 4032          # 14 days × 24h × 12 intervals/h
RESAMPLE_FACTOR = 5             # 1-min → 5-min
OUTPUT_DIR      = "data/multivariate"

# SMD column indices we care about (0-indexed)
COL_CPU  = 0   # cpu_r  — CPU utilization (already 0-1 normalized)
COL_RAM  = 5   # mem_buff_cache — high = more RAM buffered = LESS pressure
                # We invert: ram_pct = 1 - col_5  so high value = high pressure
COL_NET  = 12  # net_send — network bytes proxy


def download_and_prepare(machine_name: str, idx: int) -> str:
    """Download one SMD machine file and save it as a standardized 14-day CSV."""
    url = f"{SMD_BASE_URL}/{machine_name}.txt"
    print(f"  Fetching {machine_name} from GitHub...", end=" ", flush=True)

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    # Parse raw text into DataFrame
    raw = pd.read_csv(io.StringIO(resp.text), header=None)
    print(f"{len(raw)} raw rows ({len(raw) // 60 // 24:.1f} days at 1-min)")

    # Select the 3 columns we need
    cpu_raw = raw.iloc[:, COL_CPU].values.astype(float)
    ram_raw = raw.iloc[:, COL_RAM].values.astype(float)
    net_raw = raw.iloc[:, COL_NET].values.astype(float)

    # Resample 1-min → 5-min by averaging blocks of 5
    n_blocks = len(cpu_raw) // RESAMPLE_FACTOR
    cpu_5m = cpu_raw[:n_blocks * RESAMPLE_FACTOR].reshape(n_blocks, RESAMPLE_FACTOR).mean(axis=1)
    ram_5m = ram_raw[:n_blocks * RESAMPLE_FACTOR].reshape(n_blocks, RESAMPLE_FACTOR).mean(axis=1)
    net_5m = net_raw[:n_blocks * RESAMPLE_FACTOR].reshape(n_blocks, RESAMPLE_FACTOR).mean(axis=1)

    # Invert RAM so high value = high memory pressure (consistent with our synthetic convention)
    ram_5m = 1.0 - ram_5m

    # Standardize to exactly TARGET_ROWS (14 days at 5-min)
    if len(cpu_5m) >= TARGET_ROWS:
        # Truncate: take the first 14 days
        cpu_5m = cpu_5m[:TARGET_ROWS]
        ram_5m = ram_5m[:TARGET_ROWS]
        net_5m = net_5m[:TARGET_ROWS]
    else:
        # Tile/repeat until we have enough, then truncate
        repeats = (TARGET_ROWS // len(cpu_5m)) + 2
        cpu_5m = np.tile(cpu_5m, repeats)[:TARGET_ROWS]
        ram_5m = np.tile(ram_5m, repeats)[:TARGET_ROWS]
        net_5m = np.tile(net_5m, repeats)[:TARGET_ROWS]

    # Generate timestamps (all starting from a common anchor, different months)
    start = pd.Timestamp("2024-01-01") + pd.DateOffset(days=14 * (idx - 1))
    timestamps = pd.date_range(start=start, periods=TARGET_ROWS, freq="5min")

    # Clip to valid ranges (SMD is already 0-1 normalized)
    cpu_5m = np.clip(cpu_5m, 0.0, 1.0)
    ram_5m = np.clip(ram_5m, 0.0, 1.0)
    net_5m = np.clip(net_5m, 0.0, 1.0)

    df = pd.DataFrame({
        "timestamp":  timestamps,
        "value":      cpu_5m,
        "ram_pct":    ram_5m,
        "network_in": net_5m,
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{idx:02d}_smd_{machine_name.replace('-', '_')}.csv"
    out_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(out_path, index=False)
    print(f"    Saved {TARGET_ROWS} rows → {out_path}")
    return out_path


def prepare_real_cpu_only():
    """Copy or verify EC2 and RDS datasets into data/real_cpu_only/ with zero-padded columns."""
    import shutil
    os.makedirs("data/real_cpu_only", exist_ok=True)

    sources = [
        ("data/ec2_cpu_utilization_24ae8d.csv", "11_ec2_production.csv"),
        ("data/rds_cpu_cc0c53.csv",              "12_rds_database.csv"),
    ]

    for src_path, dest_name in sources:
        dest_path = os.path.join("data/real_cpu_only", dest_name)
        if not os.path.exists(src_path):
            print(f"  WARNING: {src_path} not found — skipping")
            continue

        df = pd.read_csv(src_path)
        # Standardize column name to 'value' if needed
        if "value" not in df.columns:
            df = df.rename(columns={df.columns[1]: "value"})

        # Zero-pad RAM and Network (Option A: model learns to ignore flat cols)
        if "ram_pct" not in df.columns:
            df["ram_pct"] = 0.0
        if "network_in" not in df.columns:
            df["network_in"] = 0.0

        # Standardize to 14 days = 4032 rows
        if len(df) >= TARGET_ROWS:
            df = df.iloc[:TARGET_ROWS].reset_index(drop=True)
        else:
            repeats = (TARGET_ROWS // len(df)) + 2
            df = pd.concat([df] * repeats, ignore_index=True).iloc[:TARGET_ROWS]

        df.to_csv(dest_path, index=False)
        print(f"  Prepared {dest_name} ({len(df)} rows, zero-padded RAM/Network)")


def main():
    print("=== SMD Data Preparation ===\n")
    print(f"Downloading {len(SMD_MACHINES)} server machine files from GitHub...\n")

    for i, machine in enumerate(SMD_MACHINES, start=1):
        try:
            download_and_prepare(machine, i)
        except Exception as e:
            print(f"  ERROR downloading {machine}: {e}")

    print("\nPreparing real CPU-only datasets (EC2 + RDS)...")
    prepare_real_cpu_only()

    print(f"\nAll datasets prepared.")
    print(f"  Multivariate (10): {OUTPUT_DIR}/")
    print(f"  CPU-only (2):      data/real_cpu_only/")


if __name__ == "__main__":
    main()
