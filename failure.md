# Failure Analysis: Low Precision & High False Positive Rate

## Problem Statement
All three datasets produced an unacceptably low precision (EC2: 11%, RDS: 13%) despite meeting the
80% Recall target. The result was too many false alert dots on every visualization, making the system
unusable in a production alerting context.

---

## Root Causes (Ranked by Impact)

### 1. `scale_pos_weight=20` — The Primary Culprit
**File:** `src/model.py`

```python
# BAD — was set to 20
scale_pos_weight=20  # A missed incident is 20x worse than a false alarm
```

**What went wrong:** This hyperparameter instructs LightGBM to treat every missed incident as 20× more
costly than a false alarm. The model responded by becoming maximally paranoid — it fires on any tiny
fluctuation. The correct value for balanced systems is 3–8.

**Fix:** Reduced to `scale_pos_weight=5`.

---

### 2. Recursive Threshold Guardrail Forced Threshold Too Low
**File:** `jobs/retrain_ec2.py`, `retrain_rds.py`, `retrain_synthetic.py`

```python
# BAD — kept lowering until recall hit 80%, ignoring precision entirely
for t in sorted(thresholds_arr, reverse=True):
    if recall_score(y_test, y_pred_t) >= 0.80:
        threshold = t; break
```

**What went wrong:** Youden's J correctly selected a balanced threshold (e.g., 0.12). The guardrail
then ignored that and pushed it down to 0.001 to force recall above 80%. At threshold 0.001, almost
every window in the entire 3-day test period triggered an alert.

**Fix:** Removed the guardrail loop. Youden's J threshold is now used as-is. Target is 60–80% recall
with meaningful precision rather than 80%+ recall with near-zero precision.

---

### 3. Prediction Horizon H=12 Created an Oversized Warning Zone
**File:** `jobs/retrain_*.py`

```python
H = 12  # 1-hour prediction horizon
# y=1 if ANY incident in the NEXT 12 steps (60 minutes)
```

**What went wrong:** Every window in the 60-minute window before each incident is labeled `y=1`.
For datasets like RDS where incidents are frequent, this means a large fraction of all windows are
positive, making the model over-predict universally.

**Fix:** Reduced to `H=6` (30-minute horizon). This makes the positive label zone tighter and more
precise.

---

### 4. Dynamic Z-Threshold Lowering Mislabelled Noise As Incidents
**File:** `src/data.py`

```python
# BAD — minimum floor was 1.5 standard deviations
while (z_scores.abs() > actual_threshold).sum() < 5 and actual_threshold > 1.5:
    actual_threshold -= 0.5
```

**What went wrong:** For the RDS dataset (high natural noise), this auto-lowered the incident threshold
to Z=1.5, meaning normal moderate fluctuations were labelled as incidents. The model then learned to
predict incidents at normal CPU variance levels — which is why RDS had alerts everywhere.

**Fix:** Raised the minimum floor from `1.5` → `2.5`. This ensures only genuinely elevated spikes
become training labels.

---

## Expected Outcome After Fixes

| Dataset | Recall (Before) | Precision (Before) | Recall (After) | Precision (After) |
|---|---|---|---|---|
| EC2 | 81% | 11% | ~65–75% | ~35–50% |
| Synthetic | 100% | 40% | ~85% | ~65–75% |
| RDS | 97% | 13% | ~55–65% | ~25–35% |

**Trade-off accepted:** We deliberately target 60–80% recall instead of 80%+. The resulting system
fires fewer, higher-confidence alerts that a DevOps team can realistically act on.

---

## Lesson Learned
> Optimising a single metric (Recall) in isolation, without constraining the other (Precision),
> produces a system that is technically correct but completely unusable in practice.
> A production-grade alerting system must balance both.
