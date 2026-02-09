import unittest

from modules.compliance_profile import get_compliance_profile


class ComplianceProfileTests(unittest.TestCase):
    def test_known_profile(self):
        profile = get_compliance_profile("us_residential_v1")
        self.assertEqual(profile.name, "us_residential_v1")
        self.assertGreater(profile.tolerance.critical_mm, 0)

    def test_unknown_profile_defaults(self):
        profile = get_compliance_profile("does_not_exist")
        self.assertEqual(profile.name, "us_residential_v1")


if __name__ == "__main__":
    unittest.main()
