import unittest
from critical_values import check


class CriticalValuesTests(unittest.TestCase):
    def test_sign_change(self):
        self.assertFalse(check('The value is -5 m.','The value is +5 m.')['passed'])

    def test_unit_change(self):
        self.assertFalse(check('The sample measured 5 mm.','The sample measured 5 m.')['passed'])

    def test_trailing_zeros_and_scientific_notation(self):
        self.assertTrue(check('The value is 2×10⁻³ m.','The value is 0.0020 m.')['passed'])

    def test_exponent_change(self):
        self.assertFalse(check('We used x² = 4.','We used x³ = 4.')['passed'])

    def test_cosmetic_formula(self):
        self.assertTrue(check('We used x² = 4.','We used x^2=4.')['passed'])

    def test_inequality_reversal(self):
        self.assertFalse(check('We found p < 0.05.','We found p > 0.05.')['passed'])

    def test_subscript_preservation(self):
        self.assertTrue(check('We used x₁ = 4.','We used x_1=4.')['passed'])
        self.assertFalse(check('We used x₁ = 4.','We used x_2=4.')['passed'])

    def test_chemical_formula_change(self):
        self.assertFalse(check('CO2 was detected.','CO3 was detected.')['passed'])


if __name__=='__main__':unittest.main()
