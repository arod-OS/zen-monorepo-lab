import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trainer import TrainingConfig, DatasetValidator


class TestTrainingConfig:
    def test_valid_config(self):
        config = TrainingConfig("rf-perception-v2", epochs=100, batch_size=32, learning_rate=0.001)
        assert config.model_name == "rf-perception-v2"
        assert config.to_dict()["epochs"] == 100

    def test_invalid_epochs(self):
        with pytest.raises(ValueError):
            TrainingConfig("model", epochs=0, batch_size=32, learning_rate=0.001)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError):
            TrainingConfig("model", epochs=10, batch_size=0, learning_rate=0.001)

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError):
            TrainingConfig("model", epochs=10, batch_size=32, learning_rate=-0.01)


class TestDatasetValidator:
    def test_valid_split_ratios(self):
        assert DatasetValidator.validate_split_ratios(0.7, 0.15, 0.15) is True

    def test_invalid_split_ratios(self):
        assert DatasetValidator.validate_split_ratios(0.8, 0.15, 0.15) is False

    def test_sufficient_samples(self):
        assert DatasetValidator.check_minimum_samples(1000, batch_size=32) is True

    def test_insufficient_samples(self):
        assert DatasetValidator.check_minimum_samples(100, batch_size=32) is False

    def test_valid_framerate(self):
        # 20 fps
        timestamps = [i * 0.05 for i in range(100)]
        assert DatasetValidator.validate_sensor_framerate(timestamps, min_fps=10.0) is True

    def test_low_framerate(self):
        # 2 fps
        timestamps = [i * 0.5 for i in range(10)]
        assert DatasetValidator.validate_sensor_framerate(timestamps, min_fps=10.0) is False