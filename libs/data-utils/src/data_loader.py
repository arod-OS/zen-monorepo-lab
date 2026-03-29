import json
import os

def load_sensor_config(config_path: str) -> dict:
    # Load sensor calibration config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)
    
def validate_frame_timestamps(timestamps: list[float]) -> bool:
    # Validate that sensor frame timestamps are monotonically increasing
    if len(timestamps) < 2:
        return True
    return all(timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1))

def compute_frame_rate(timestamps: list[float]) -> float:
    # Compute average frame rate from timestamps
    if len(timestamps) < 2:
        return 0.0
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return 0.0
    return (len(timestamps) - 1) / duration
