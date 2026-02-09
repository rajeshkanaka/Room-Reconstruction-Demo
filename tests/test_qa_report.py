import unittest

from modules.compliance_profile import get_compliance_profile
from modules.qa_report import build_qa_report


class _QualityStub:
    passed = True
    failures = []
    metrics = {"registration_ratio": 0.9}


class QAReportTests(unittest.TestCase):
    def test_build_report(self):
        profile = get_compliance_profile("us_residential_v1")
        report = build_qa_report(
            profile=profile,
            compliance_status="PASS",
            quality_result=_QualityStub(),
            calibration={"uncertainty_mm": 4.0},
            sfm_result={"success": True, "num_registered": 7, "reprojection_errors": [0.5, 0.8]},
            floor_plan_measurements={"width_m": 4.2},
        )

        self.assertEqual(report["compliance_profile"], "us_residential_v1")
        self.assertIn("accuracy_metrics", report)
        self.assertIn("pass_flags", report)


if __name__ == "__main__":
    unittest.main()
