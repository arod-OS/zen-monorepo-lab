# Object detection module

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "libs", "data-utils", "src"))
from data_loader import validate_frame_timestamps

class RadarDetector:
    SUPPORTED_MODES = ["highway", "urban", "parking"]

    def __init__(self, mode: str = "highway", confidence_threshold: float = 0.5):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {mode}. Choose from {self.SUPPORTED_MODES}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1")
        self.mode = mode
        self.confidence_threshold = confidence_threshold

    def detect(self, radar_frame: dict) -> list[dict]:
        # Run detection on a single radar frame
        if "points" not in radar_frame:
            raise ValueError("Frame missing 'points' field")
        
        detections = []
        for point in radar_frame["points"]:
            confidence = point.get("intensity", 0) / 250.0
            if confidence >= self.confidence_threshold:
                detections.append({
                    "x": point["x"],
                    "y": point["y"],
                    "z": point.get("z", 0),
                    "velocity": point.get("doppler", 0),
                    "confidence": round(confidence, 3),
                    "class": self._classify(point),
                })
        return detections
    
    def _classify(self, point: dict) -> str:
        # Simple rule based classification placeholder
        doppler = abs(point.get("doppler", 0))
        if doppler > 5.0:
            return "vehicle"
        elif doppler > 0.5:
            return "pedestrian"
        else:
            return "static"
        
    def process_sequence(self, frames: list[dict]) -> dict:
        # Process a sequence of radar frames
        timestamps = [f.get("timestamp", 0) for f in frames]
        if not validate_frame_timestamps(timestamps):
            raise ValueError("Frame timestamps are not monotonically increasing")
        
        all_detections = []
        for frame in frames:
            all_detections.append(self.detect(frame))

        return {
            "num_frames": len(frames),
            "total_detections": sum(len(d) for d in all_detections),
            "detections_per_frame": all_detections,
        }