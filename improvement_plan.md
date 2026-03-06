# Predictive Alerting: Overall Improvement Plan

This document outlines the strategic roadmap for improving the Precision and overall robustness of our predictive alerting system, while maintaining our strict ~80% Recall target requirement. The core LightGBM algorithm will remain unchanged.

## 1. Dynamic Adjustment of Parameters Based on New Datasets

When feeding a completely new cloud dataset (e.g., a new microservice) into the pipeline, the system must dynamically adapt its parameters to maintain the 80% recall without destroying precision. 

### Factors That MUST Be Dynamically Adjusted:
1. **Classification Threshold (`threshold`):**
   *   *Why:* Different datatypes have varying base levels of noise. A 0.5 threshold might work for a quiet database, but a noisy web server might require dropping the threshold down to 0.01 to hit 80% recall.
   *   *How to Adjust:* We already implemented a mathematical auto-adjuster in `retrain.py` that evaluates the hold-out set, calculates the exact percentile score of true positives, and hard-codes that new threshold into the `.pkl` artifact. This ensures the inference function is correctly calibrated for the specific dataset it was trained on.
2. **Standard Deviation Incident Trigger (`z_threshold`):**
   *   *Why:* Currently set to `3.0` standard deviations to label an "incident". Some stable microservices rarely breach `3.0`, meaning the model gets zero training examples.
   *   *How to Adjust:* We should implement a function that calculates the historical volatility of the specific dataset. If the dataset never spikes past a Z-score of 3, the script should dynamically lower the target to `2.5` to ensure enough ground-truth labels are created for the model to learn from.

### Factors That Should NOT Be Adjusted (Static):
1. **Window Size (`W=288`) and Horizon (`H=12`):**
   *   *Why Not:* These are tied directly to human temporal behavior (1 day and 1 hour). Modifying these per-dataset makes the system inconsistent. Our Autocorrelation (ACF) math proves that daily cycles exist globally across almost all web traffic. Keeping `W` frozen ensures the model always understands daily cyclicality.
2. **Aggressive Recall Tuning (`scale_pos_weight=20`):**
   *   *Why Not:* The primary objective is to *never miss an incident*. If we dynamically lower this weight just because a new dataset is noisy, we risk missing critical crashes in an attempt to artificially improve precision. We must keep this artificially high to enforce the 80% recall guardrail globally.

## 2. Maintaining the Core Algorithm
The project structure dictates we must keep **LightGBM**. This is the correct choice because:
*   **Fast execution:** A Lambda function runs in milliseconds.
*   **Small footprint:** The artifact is <200 KB.
*   **Explainability:** Unlike Deep Learning, we can directly extract *Feature Importances* from the tree structure to prove to stakeholders exactly *why* the model fired an alert.

## 3. Adding Cloud Service Metrics
**Should we add more metrics? YES. This is the #1 way to solve the Precision problem.**
A single CPU timeline only tells us what the server is currently doing. When an immediate vertical CPU spike occurs (a "zero-day" crash), the model looks stupid because there was no prior warning. 

**Which specific metrics should we pick?**
1.  **Network In (Bytes):** *Why:* The highest correlation to a web server CPU crash is a sudden flood of incoming traffic (e.g., a DDoS attack or viral traffic spike). This almost always spikes 1-5 minutes *before* the CPU maxes out.
2.  **Memory Utilization (RAM %):** *Why:* Memory leaks happen slowly. If RAM slowly climbs to 99%, the CPU will inevitably crash soon after because the server starts "swapping" memory to disk. This gives hours of advanced warning.
3.  **Database Connection Count (or HTTP 500 Error Rate):** *Why:* If the application layer is struggling to connect to the database, CPU threads get locked up waiting for timeouts.

*Metrics we should NOT pick:* Disk Read/Write (unless it's a database server). For standard EC2 web servers, Disk I/O is usually just logging and doesn't strongly predict CPU failure.

## 4. Alternative Interpretations of Raw Metrics
**Even if we use more metrics (like RAM and Network), should we still use more interpretations of the data? YES.**
Raw numbers are noisy. Machine learning models (especially LightGBM) struggle to understand "momentum" or "chaos" just by looking at raw numbers. We must translate the data into concepts the model can easily split.

**Which interpretations should we pick?**
1.  **Exponential Moving Average (EMA) Crossovers:** 
    *   *Why:* Calculates a fast EMA (e.g., 1 hour) and a slow EMA (24 hours). The exact moment the fast line crosses the slow line, it mathematically proves a "regime shift" is occurring. It is the cleanest, most reliable signal of a trend change.
2.  **Rolling Entropy / Variation (Volatility):** 
    *   *Why:* Instead of measuring the *value* of the CPU, this measures how fast the value is *shaking*. If rolling variance suddenly spikes, the service is thrashing before an ultimate failure. 

*Interpretations we should NOT pick:* Fourier Transforms (FFT). While scientifically cool for finding frequencies, FFT is computationally heavy and overkill for predicting if a server is about to crash. Simple EMA crossovers are much faster to calculate inside an AWS Lambda function.

## 5. Other Ways to Improve Precision
**Should we add Option 5? YES, absolutely.**
Adding "Consecutive Alert Validation" (Smoothing) is the easiest, highest-ROI fix for Precision in a production alerting system.

*   *Why:* In the real world, DevOps teams get "alert fatigue" if an alarm fires for a 1-minute CPU spike that instantly resolves itself.
*   *How it works:* Update `inference.py` so that it doesn't trigger an immediate PagerDuty text if probability hits the threshold for just *one* 5-minute interval. Instead, require the model to predict > Threshold for **three consecutive ticks (15 minutes)** before firing the true alarm. 
*   *Result:* This completely eliminates isolated False Positives natively without needing to retrain the model. It also gives the auto-scaler time to fix the issue before waking up a human.

## 6. Overall Actionable Improvement Plan
To evolve this prototype into an enterprise-ready production alerting engine:

**Phase A: Feature Expansion (Next Week)**
1. Update `create_sliding_windows()` to ingest Multivariate Data (Network, RAM, Disk).
2. Integrate EMA Crossovers and Rolling Entropy features into the preprocessing pipeline.

**Phase B: Dynamic Pre-Processing (Next Month)**
3. Update `generate_labels()` to automatically calculate the target `z_threshold` based on the historical volatility characteristics of whichever dataset is fed into it.
4. Implement "Consecutive Alert Validation" inside `inference.py` to slash the False Positive rate.

**Phase C: Production MLOps (Q3)**
5. Implement Data Drift Detection (e.g., using evidently.ai). If the statistical baseline of the CPU shifts too drastically from when the model was last trained, auto-trigger the Retraining Lambda out-of-schedule.
6. Publish an automated dashboard mapping Feature Importances, proving to Ops engineers exactly which metric (Network vs CPU) triggered the impending alarm prediction.
