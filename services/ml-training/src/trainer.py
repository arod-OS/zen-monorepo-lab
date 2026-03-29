# Model training utilites

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "libs", "data-utils", "src"))
from data_loader import compute_frame_rate

class TrainingConfig:
    def __init__(self, model_name: str, epochs:int, batch_size: int, learning_rate: float):
        if epochs < 1:
            raise ValueError("Epochs must be >= 1")
        if batch_size < 1:
            raise ValueError("Batch size must be >= 1")
        if learning_rate <= 0:
            raise ValueError("Learing rate must be > 0")
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }
    
class DatasetValidator:
    @staticmethod
    def validate_split_ratios(train: float, val: float, test: float) -> bool:
        # check that split ratios sum to 1.0
        return abs((train + val + test) - 1.0) < 1e-6
    
    @staticmethod
    def check_minimum_samples(num_samples: int, batch_size: int, min_batches: int = 10) -> bool:
        # Ensire dataset has enough samples for meaningful training
        return num_samples >= batch_size * min_batches
    
    @staticmethod
    def validate_sensor_framerate(timestamps: list[float], min_fps: float = 10.0) -> bool:
        # Ensure sensor data meets minimum frame rate for trainning quality
        fps = compute_frame_rate(timestamps)
        return fps >= min_fps