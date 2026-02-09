import unittest

import numpy as np

from modules.calibration import (
    CalibrationInput,
    compute_scale_factor,
    convert_distance_to_meters,
    parse_calibration_input,
)

NUMPY_OK = hasattr(np, "array") and hasattr(np, "asarray")


class CalibrationTests(unittest.TestCase):
    def test_convert_distance(self):
        self.assertAlmostEqual(convert_distance_to_meters(10, "ft"), 3.048, places=6)
        self.assertAlmostEqual(convert_distance_to_meters(10, "in"), 0.254, places=6)
        self.assertAlmostEqual(convert_distance_to_meters(2, "m"), 2.0, places=6)

    @unittest.skipUnless(NUMPY_OK, "NumPy runtime is incomplete in this environment")
    def test_compute_scale_factor(self):
        floor_plan = {
            "x_range": (0.0, 10.0),
            "z_range": (0.0, 6.0),
            "occupancy": np.ones((1000, 1000), dtype=np.float32),
        }
        cal_input = CalibrationInput(
            point1_px=(0.0, 10.0),
            point2_px=(999.0, 10.0),
            known_distance=5.0,
            known_distance_unit="m",
            pixel_uncertainty=1.0,
        )

        result = compute_scale_factor(cal_input, floor_plan, image_shape=(1000, 1000))
        self.assertAlmostEqual(result["scale_factor"], 0.5, places=6)
        self.assertLess(result["uncertainty_mm"], 20.0)

    def test_parse(self):
        raw = {
            "point1_px": [10, 20],
            "point2_px": [30, 40],
            "known_distance": 12,
            "known_distance_unit": "ft",
        }
        cal = parse_calibration_input(raw)
        self.assertEqual(cal.point1_px, (10.0, 20.0))
        self.assertEqual(cal.known_distance_unit, "ft")


if __name__ == "__main__":
    unittest.main()
