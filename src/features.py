import numpy as np
import pandas as pd

# Extra metric columns to extract features from alongside the primary CPU signal
EXTRA_METRIC_COLS = ['ram_pct', 'network_in']

def _extract_col_features(series_values: np.ndarray, X_raw: np.ndarray, num_samples: int,
                           W: int, current_time_idx: np.ndarray, col_df: pd.Series) -> np.ndarray:
    """
    Extract 6 summary features from a single metric column time series.
    Returns shape (num_samples, 6): mean, std, velocity, short/long ratio, EMA divergence, rolling variance.
    """
    fast_ema = col_df.ewm(span=12).mean().values
    slow_ema = col_df.ewm(span=288).mean().values
    ema_div  = fast_ema - slow_ema
    roll_var = col_df.rolling(window=24, min_periods=1).var().fillna(0).values

    shape_col = (num_samples, W)
    strides_col = (series_values.strides[0], series_values.strides[0])
    col_raw = np.lib.stride_tricks.as_strided(series_values, shape=shape_col, strides=strides_col)

    lookback_step = max(1, W // 4)
    col_mean = np.mean(col_raw, axis=1)
    col_std  = np.std(col_raw,  axis=1)
    col_vel  = col_raw[:, -1] - col_raw[:, -lookback_step]
    col_ratio = np.mean(col_raw[:, -12:], axis=1) / (col_mean + 1e-8)
    col_ema  = ema_div[current_time_idx]
    col_rvar = roll_var[current_time_idx]

    return np.stack([col_mean, col_std, col_vel, col_ratio, col_ema, col_rvar], axis=1)


def create_sliding_windows(df: pd.DataFrame, W: int, H: int,
                            value_col: str = 'value', label_col: str = 'label',
                            time_col: str = 'timestamp') -> tuple:
    """
    Builds a vector-optimized sliding window generator.
    Supports multivariate input: if the dataframe contains columns in EXTRA_METRIC_COLS
    (ram_pct, network_in) those are also featurised and appended to each sample.

    Returns:
        X (np.ndarray): Shape (num_samples, W + n_engineered_features)
        y (np.ndarray): Shape (num_samples,)  -- 1 if incident in next H steps
    """
    values     = df[value_col].values
    labels     = df[label_col].values
    timestamps = df[time_col].dt.hour.values

    # Pre-calculate full-sequence EMA & rolling variance for the primary CPU col
    values_series  = df[value_col]
    fast_ema       = values_series.ewm(span=12).mean().values
    slow_ema       = values_series.ewm(span=288).mean().values
    ema_divergence = fast_ema - slow_ema
    rolling_var    = values_series.rolling(window=24, min_periods=1).var().fillna(0).values

    n = len(values)
    if n < W + H:
        raise ValueError(f"Time series length {n} must be >= W + H ({W + H})")

    num_samples = n - W - H + 1

    # Primary CPU sliding windows via numpy striding
    shape_X   = (num_samples, W)
    strides_X = (values.strides[0], values.strides[0])
    X_raw     = np.lib.stride_tricks.as_strided(values, shape=shape_X, strides=strides_X)

    # Targets: 1 if any incident occurs in the next H steps
    shape_y   = (num_samples, H)
    strides_y = (labels.strides[0], labels.strides[0])
    y_windowed = np.lib.stride_tricks.as_strided(labels[W:], shape=shape_y, strides=strides_y)
    y_raw     = np.max(y_windowed, axis=1)

    # ---- Primary CPU Feature Engineering ----
    # 10 hand-crafted features per window
    X_features = np.zeros((num_samples, 10))
    current_time_idx = np.arange(W - 1, W - 1 + num_samples)

    lookback_step = max(1, W // 4)
    X_features[:, 0] = np.mean(X_raw, axis=1)
    X_features[:, 1] = np.std(X_raw,  axis=1)
    X_features[:, 2] = np.min(X_raw,  axis=1)
    X_features[:, 3] = np.max(X_raw,  axis=1)
    X_features[:, 4] = np.percentile(X_raw, 95, axis=1)
    X_features[:, 5] = X_raw[:, -1] - X_raw[:, -lookback_step]         # Velocity
    short_m          = np.mean(X_raw[:, -12:], axis=1)
    X_features[:, 6] = short_m / (X_features[:, 0] + 1e-8)             # Short/Long ratio
    X_features[:, 7] = timestamps[current_time_idx]                     # Hour of day
    X_features[:, 8] = ema_divergence[current_time_idx]                 # EMA divergence
    X_features[:, 9] = rolling_var[current_time_idx]                    # Rolling variance

    all_parts = [X_raw, X_features]

    # ---- Multivariate Extra Metrics (RAM, Network) ----
    # Only added when the dataframe actually contains those columns.
    for col in EXTRA_METRIC_COLS:
        if col in df.columns:
            col_feats = _extract_col_features(
                df[col].values.copy(), X_raw, num_samples, W,
                current_time_idx, df[col]
            )
            all_parts.append(col_feats)

    X_final = np.concatenate(all_parts, axis=1)
    return X_final, y_raw


def generate_streaming_features(window_data: np.ndarray,
                                 current_timestamp: pd.Timestamp = None,
                                 extra_windows: dict = None) -> np.ndarray:
    """
    Generate features for a single streaming window of size W.
    extra_windows: optional dict of {col_name: np.ndarray} for RAM/Network
    """
    w_mean = np.mean(window_data)
    w_std  = np.std(window_data)
    w_min  = np.min(window_data)
    w_max  = np.max(window_data)
    w_p95  = np.percentile(window_data, 95)

    lookback_step  = max(1, len(window_data) // 4)
    velocity       = window_data[-1] - window_data[-lookback_step]
    short_m        = np.mean(window_data[-12:])
    ratio          = short_m / (w_mean + 1e-8)
    hour           = current_timestamp.hour if current_timestamp else 0

    ws          = pd.Series(window_data)
    fast_ema    = ws.ewm(span=12).mean().iloc[-1]
    slow_ema    = ws.ewm(span=len(window_data)).mean().iloc[-1]
    ema_div     = fast_ema - slow_ema
    rolling_var = ws.rolling(window=24, min_periods=1).var().fillna(0).iloc[-1]

    cpu_features = np.array([w_mean, w_std, w_min, w_max, w_p95,
                              velocity, ratio, hour, ema_div, rolling_var])

    parts = [window_data, cpu_features]

    # Extra metrics features for streaming
    if extra_windows:
        for col in EXTRA_METRIC_COLS:
            if col in extra_windows:
                ew = extra_windows[col]
                ew_s     = pd.Series(ew)
                ew_mean  = np.mean(ew);  ew_std = np.std(ew)
                ew_vel   = ew[-1] - ew[max(0, len(ew) - lookback_step)]
                ew_ratio = np.mean(ew[-12:]) / (ew_mean + 1e-8)
                ew_ema   = ew_s.ewm(span=12).mean().iloc[-1] - ew_s.ewm(span=len(ew)).mean().iloc[-1]
                ew_rvar  = ew_s.rolling(window=24, min_periods=1).var().fillna(0).iloc[-1]
                parts.append(np.array([ew_mean, ew_std, ew_vel, ew_ratio, ew_ema, ew_rvar]))

    return np.concatenate(parts).reshape(1, -1)
