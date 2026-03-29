import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from detector import RadarDetector

class TestRadarDetector:
    def setup_method(self):
        self.detector = RadarDetector(mode="highway", confidence_threshold=0.5)

    def test_init_valid_mode(self):
        d = RadarDetector(mode="urban")
        assert d.mode == "urban"

    def test_init_invalid_mode(self):
        with pytest.raises(ValueError, match="Unsupported mode"):
            RadarDetector(mode="offroad")

    def test_init_invalid_threshold(self):
        with pytest.raises(ValueError, match="Confidence threshold"):
            RadarDetector(confidence_threshold=1.5)

    def test_detect_filters_by_confidence(self):
        frame = {
            "points": [
                {"x": 10, "y": 5, "intensity": 200, "doppler": 8.0},   # confidence 0.78 -> included
                {"x": 20, "y": 10, "intensity": 50, "doppler": 1.0},   # confidence 0.20 -> excluded
                {"x": 30, "y": 15, "intensity": 180, "doppler": 0.0},  # confidence 0.71 -> included
            ]
        }
        results = self.detector.detect(frame)
        assert len(results) == 2

    def test_detect_missing_points(self):
        with pytest.raises(ValueError, match="missing 'points'"):
            self.detector.detect({"timestamp": 1.0})

    def test_classify_vehicle(self):
        point = {"x": 0, "y": 0, "doppler": 10.0, "intensity": 200}
        assert self.detector._classify(point) == "vehicle"

    def test_classify_pedestrian(self):
        point = {"x": 0, "y": 0, "doppler": 1.5, "intensity": 200}
        assert self.detector._classify(point) == "pedestrian"

    def test_classify_static(self):
        point = {"x": 0, "y": 0, "doppler": 0.1, "intensity": 200}
        assert self.detector._classify(point) == "static"

    def test_process_sequence(self):
        frames = [
            {"timestamp": 1.0, "points": [{"x": 10, "y": 5, "intensity": 200, "doppler": 8.0}]},
            {"timestamp": 2.0, "points": [{"x": 15, "y": 7, "intensity": 180, "doppler": 6.0}]},
        ]
        result = self.detector.process_sequence(frames)
        assert result["num_frames"] == 2
        assert result["total_detections"] == 2

    def test_process_sequence_bad_timestamps(self):
        frames = [
            {"timestamp": 2.0, "points": [{"x": 10, "y": 5, "intensity": 200, "doppler": 8.0}]},
            {"timestamp": 1.0, "points": [{"x": 15, "y": 7, "intensity": 180, "doppler": 6.0}]},
        ]
        with pytest.raises(ValueError, match="not monotonically increasing"):
            self.detector.process_sequence(frames)