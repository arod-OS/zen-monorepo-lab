import pytest
import json
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import load_sensor_config, validate_frame_timestamps, compute_frame_rate

class TestLoadSensorConfig:
    def test_loads_valid_config(self):
        config = {"sensor": "radar", "frequency": 77, "unit": "GHz"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            result = load_sensor_config(f.name)
        assert result == config
        os.unlink(f.name)

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_sensor_config("nonexistent/config.json")

class TestComputeFrameRate:
    def test_standard_frame_rate(self):
        # 10 frames over 1 second = 9 intervals = 9 fps
        timestamps = [i * (1.0 / 9.0) for i in range(10)]
        fps = compute_frame_rate(timestamps)
        assert abs(fps - 9.0) < 0.01

    def test_single_frame(self):
        assert compute_frame_rate([1.0]) == 0.0

    def test_empty(self):
        assert compute_frame_rate([]) == 0.0