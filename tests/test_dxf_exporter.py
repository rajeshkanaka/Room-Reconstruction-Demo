import os
import tempfile
import unittest

import numpy as np

from modules.dxf_exporter import DXFExporter

NUMPY_OK = hasattr(np, "array") and hasattr(np, "asarray")


class DXFExporterTests(unittest.TestCase):
    @unittest.skipUnless(NUMPY_OK, "NumPy runtime is incomplete in this environment")
    def test_export_floor_plan(self):
        exporter = DXFExporter()
        floor_plan = {
            "boundary_world_m": [[0, 0], [4, 0], [4, 3], [0, 3]],
            "measurements": {
                "width_m": 4.0,
                "depth_m": 3.0,
                "area_sqm": 12.0,
                "area_sqft": 129.166,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plan.dxf")
            out = exporter.export_floor_plan(floor_plan, path)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)


if __name__ == "__main__":
    unittest.main()
