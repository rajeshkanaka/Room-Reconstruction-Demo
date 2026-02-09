import unittest

import numpy as np

try:
    import cv2  # noqa: F401

    CV2_OK = True
except Exception:
    CV2_OK = False

if CV2_OK:
    from modules.floor_plan_generator import FloorPlanGenerator
else:
    FloorPlanGenerator = None

NUMPY_OK = hasattr(np, "array") and hasattr(np, "vstack")


@unittest.skipUnless(
    CV2_OK and NUMPY_OK and FloorPlanGenerator is not None,
    "Requires OpenCV + full NumPy runtime",
)
class FloorPlanGeometryTests(unittest.TestCase):
    def test_rectangular_room_polygon(self):
        rng = np.random.default_rng(123)
        left = np.column_stack(
            [np.full(300, -2.0), rng.uniform(0, 2.6, 300), rng.uniform(0, 4.0, 300)]
        )
        right = np.column_stack(
            [np.full(300, 2.0), rng.uniform(0, 2.6, 300), rng.uniform(0, 4.0, 300)]
        )
        back = np.column_stack(
            [rng.uniform(-2, 2, 300), rng.uniform(0, 2.6, 300), np.full(300, 4.0)]
        )
        floor = np.column_stack(
            [rng.uniform(-2, 2, 800), rng.uniform(0.0, 0.2, 800), rng.uniform(0.0, 4.0, 800)]
        )

        points = np.vstack([left, right, back, floor])
        gen = FloorPlanGenerator(manhattan_snap=True)
        out = gen.generate_floor_plan(points, absolute_scale_m_per_unit=1.0, accurate_mode=True)

        self.assertGreaterEqual(len(out["boundary_world_m"]), 4)
        self.assertGreater(out["measurements"]["area_sqm"], 5.0)


if __name__ == "__main__":
    unittest.main()
