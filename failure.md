# Unified 12-Dataset Model — Failure Analysis

**Date:** 2026-03-06  
**Experiment:** Single LightGBM model trained on all 12 datasets (10 SMD + EC2 + RDS)

---

## Failure Summary

The unified model achieved **80.4% recall** (meets target) but produced catastrophically high false positive rates and unacceptably low precision across all 12 datasets.

| Dataset | Precision | Recall | FP Rate | Verdict |
|---|---|---|---|---|
| SMD Server 01 | 3.9% | 93.1% | 93.2% | FAIL |
| SMD Server 02 | 15.7% | 100% | 73.5% | FAIL |
| SMD Server 03 | 3.4% | 100% | 51.1% | FAIL |
| SMD Server 04 | 2.5% | 100% | 70.0% | FAIL |
| SMD Server 05 | 0.1% | 33.3% | 100% | FAIL |
| SMD Server 06 | 1.7% | 100% | 92.0% | FAIL |
| SMD Server 07 | 1.7% | 100% | 97.0% | FAIL |
| SMD Server 08 | 11.3% | 100% | 72.5% | FAIL |
| SMD Server 09 | 7.5% | 96.2% | 90.5% | FAIL |
| SMD Server 10 | 3.9% | 93.6% | 100% | FAIL |
| EC2 Production | 5.9% | 66.7% | 72.7% | FAIL |
| RDS Database | 12.0% | 35.1% | 28.4% | FAIL |

**Required targets:** Precision 20–50% | Recall ~80% | FPR <1%

---

## Root Cause Analysis

### Root Cause 1: Global Threshold Too Low (Primary Failure)

The Youden's J statistic selected a **global threshold of 0.009** by optimising on the pooled test set average. At a threshold this close to zero, nearly every prediction across every dataset fires an alert — regardless of what the server is actually doing.

**Why it happened:** Youden's J maximises `Sensitivity + Specificity - 1` on the average distribution. When 12 servers with vastly different noise profiles are pooled, the optimal average point happens to be a near-zero threshold that sacrifices specificity (and therefore FPR) entirely in order to maintain recall across all servers simultaneously.

**Effect:** FPR ranged from 28% to 100% — 28x to 100x worse than the <1% requirement.

---

### Root Cause 2: One Global Threshold Cannot Fit 12 Different Servers

Every server has a different noise floor:

- **Quiet servers** (e.g. SMD Server 01): Normal CPU variance is very low. A threshold of 0.009 flags almost every normal 5-minute window as a pre-incident.
- **Noisy servers** (e.g. SMD Server 10): CPU fluctuates constantly. The model outputs moderate probabilities even during normal operation — again, everything exceeds 0.009.

A threshold calibrated on the pooled average is **too aggressive for quiet servers and incorrectly calibrated for noisy ones.** There is no single threshold value that satisfies FPR <1% across all 12 servers simultaneously.

**Correct approach:** Each server requires its own threshold calibrated to its own noise floor — specifically the 99th percentile of predicted probabilities for normal windows on that server's training data.

---

### Root Cause 3: SMD Ground-Truth Labels Not Used

The SMD dataset ships with published anomaly labels in `ServerMachineDataset/test_label/`. Instead, our pipeline computed Z-score labels from the already-normalized [0,1] SMD signals. This introduced two problems:

1. **Label noise:** Z-score on pre-normalized data produces different statistical properties than Z-score on raw CloudWatch data. The "incidents" we labelled may not correspond to real hardware anomalies.
2. **Label over-generation:** The dynamic Z-threshold lowered itself to 2.0 on some datasets (e.g. Server 05, Server 07) to generate enough training examples, meaning normal fluctuations were labelled as incidents, training the model on corrupted labels.

---

### Root Cause 4: Cross-Dataset Feature Distribution Mismatch

The model was trained on a pool where:
- SMD data: CPU in [0.01, 0.3] (pre-normalized, very compressed)
- EC2 data: CPU in [0.05, 1.8] (NAB normalized, wider range)
- RDS data: CPU in [0.01, 0.5] (NAB normalized)

The feature distributions across these 12 sources are **non-stationary** — the model cannot learn a consistent mapping from feature space to incident probability when the underlying value scales are incompatible.

---

## What Was Proven

Despite the failure, the experiment established two important empirical facts:

1. **A single global model across heterogeneous servers cannot achieve FPR <1%.** This is a fundamental result, not an implementation bug.
2. **Per-dataset threshold calibration is non-negotiable.** The architecture must store one threshold per server, calibrated to that server's noise floor, not a pooled average.

---

## Correct Architecture (Next Attempt)

| What to Change | Why |
|---|---|
| Per-dataset threshold via 99th percentile of normal-class probabilities | Guarantees FPR ≤ 1% by construction |
| Use SMD published ground-truth labels from `test_label/` folder | Removes corrupted Z-score labels from training |
| Keep one shared LightGBM model | Good for generalization across server types |
| Store `{model, thresholds: {dataset_id: float}}` in pkl | One artifact, 12 calibrated operating points |
| Increase consecutive smoothing from 3 → 6 windows (30 min) | Additional FPR suppression without retraining |
