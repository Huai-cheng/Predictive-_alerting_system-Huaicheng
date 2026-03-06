import os
import pickle
import lightgbm as lgb
from typing import Any

def get_baseline_model() -> lgb.LGBMClassifier:
    """
    Returns an un-trained LightGBM classifier.
    Chosen for speed, handling of non-scaled features, and good baseline performance.
    We prioritize recall utilizing `scale_pos_weight` to aggressively penalize missed incidents.
    """
    # scale_pos_weight > 1 helps with an anomaly detection dataset 
    # where incident labels are inherently rare. Setting to 20 means
    # a missed incident is 20x worse than a false alarm.
    return lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=20,  # Phase 2 improvement: aggressive recall tuning
        random_state=42,
        n_jobs=-1
    )

def save_model_to_s3_mock(model: Any, filepath: str) -> None:
    """
    Serializable/deserializable model artifact saving (mocking S3).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Mock S3: Model saved to {filepath}")

def load_model_from_s3_mock(filepath: str) -> Any:
    """
    Serializable/deserializable model artifact loading (mocking S3).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model artifact not found at {filepath}")
        
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Mock S3: Model loaded from {filepath}")
    return model
