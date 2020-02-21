
# unit test framework
import unittest
# the package API
from tsoracle import factor
from tsoracle.factor import TSPolynomial
# custom required type
from numpy import ndarray

class TestMultiply(unittest.TestCase):

    def test_Root(self):
        """Unit tests for Root
        """
        # verify a polynomial that has a real root
        rt = factor.Root(-0.8, 0)
        self.assertAlmostEqual(0.5, rt.frequency, 6)
        self.assertAlmostEqual(1.25, rt.abs_recip, 6)
        self.assertEqual('-0.80000', rt.as_string)
        self.assertEqual('1.0+0.8000B', rt.polynomial)
        
        # verify a polynomial that has a real root
        rt = factor.Root(0.8, 0)
        self.assertAlmostEqual(0.0, rt.frequency, 6)
        self.assertAlmostEqual(1.25, rt.abs_recip, 6)
        self.assertEqual('0.80000', rt.as_string)
        self.assertEqual('1.0-0.8000B', rt.polynomial)

        # verify complex roots
        rt = factor.Root(1.2, -0.4)
        self.assertAlmostEqual(0.05120819117478338, rt.frequency, 6)
        self.assertAlmostEqual(0.7906, rt.abs_recip, 6)
        self.assertEqual('1.20000+-0.40000i', rt.as_string)
        self.assertEqual('1.0-2.4000B+1.6000B^2', rt.polynomial)

    def test_get_roots(self):
        """Unit tests for get_roots
        """
        # verify a polynomial that has real roots
        roots = factor.get_roots([1.2, 0.4])
        self.assertAlmostEqual(-3.679449, roots[0], 6)
        self.assertAlmostEqual(0.679449, roots[1], 6)
        # verify a polynomial that has complex roots
        roots = factor.get_roots([1.2, -0.4])
        self.assertAlmostEqual(-0.5, roots[0].imag, 6)
        self.assertAlmostEqual(1.5, roots[0].real, 6)
        self.assertAlmostEqual(0.5, roots[1].imag, 6)
        self.assertAlmostEqual(1.5, roots[1].real, 6)

    def test_roots_in_unit_circle(self):
        """Unit tests for roots_in_unit_circle
        """
        ## AR only
        # verify 2 real roots outside of unit circle
        self.assertFalse(factor.roots_in_unit_circle([0.6, -0.4], None))
        # verify 2 real roots one inside of unit circle
        self.assertTrue(factor.roots_in_unit_circle([1.2, 0.4], None))
        # verify 2 imag roots both inside of unit circle
        self.assertFalse(factor.roots_in_unit_circle([1.2, -0.4], None))
        
        ## MA only
        # verify 2 real roots outside of unit circle
        self.assertFalse(factor.roots_in_unit_circle(None, [0.6, -0.4]))
        # verify 2 real roots one inside of unit circle
        self.assertTrue(factor.roots_in_unit_circle(None, [1.2, 0.4]))
        # verify 2 imag roots both inside of unit circle
        self.assertFalse(factor.roots_in_unit_circle(None, [1.2, -0.4]))

        ## Test Combinations
        # verify 2 real roots one inside of unit circle in the AR part
        # verify 2 real roots outside of unit circle in the MA part
        self.assertTrue(factor.roots_in_unit_circle([1.2, 0.4], [0.6, -0.4]))

        # verify 2 real roots outside of unit circle in the AR part
        # verify 2 real roots outside of unit circle in the MA part
        self.assertFalse(factor.roots_in_unit_circle([0.6, -0.4], [0.6, -0.4]))

        # verify 2 real roots outside of unit circle in the AR part
        # verify 2 real roots outside of unit circle in the MA part
        self.assertFalse(factor.roots_in_unit_circle([0.6, -0.4], [1.2, -0.4]))

    def test_get_system_frequency(self):
        """Unit tests for get_system_frequency
        """

        self.assertAlmostEqual(0.05120819117478338, 
                               factor.get_system_freq(*[1.2, -0.4]), 10)

    def test_to_glp(self):
        """Unit tests for to_glp
        """

        # no polynomials should produce a one followed by zeros
        psis = factor.to_glp(lags = 8)

        self.assertAlmostEqual(psis[0], 1, 10)
        self.assertAlmostEqual(psis[1], 0, 10)
        self.assertAlmostEqual(psis[2], 0, 10)
        self.assertAlmostEqual(psis[3], 0, 10)
        self.assertAlmostEqual(psis[4], 0, 10)
        self.assertAlmostEqual(psis[5], 0, 10)
        self.assertAlmostEqual(psis[6], 0, 10)
        self.assertAlmostEqual(psis[7], 0, 10)

        # a functional test
        psis = factor.to_glp([0.9], [ -0.3, -0.4], 8)

        self.assertAlmostEqual(psis[0], 1.0, 10)
        self.assertAlmostEqual(psis[1], 1.2, 10)
        self.assertAlmostEqual(psis[2], 1.48, 10)
        self.assertAlmostEqual(psis[3], 1.332, 10)
        self.assertAlmostEqual(psis[4], 1.1988, 10)
        self.assertAlmostEqual(psis[5], 1.07892, 10)
        self.assertAlmostEqual(psis[6], 0.971028, 10)
        self.assertAlmostEqual(psis[7], 0.8739252, 10)

    def test_multiply(self):
        """Unit tests for multiply
        """
        # basic test
        ret = factor.multiply( [ [0.4], [-0.5] ] )
        self.assertAlmostEqual(-0.1, ret[0], 6)
        self.assertAlmostEqual(0.2, ret[1], 6)

        # slightly more complicated
        ret = factor.multiply( [ [-0.8], [1.2, -0.4] ] )
        self.assertAlmostEqual(0.4, ret[0], 6)
        self.assertAlmostEqual(0.56, ret[1], 6)
        self.assertAlmostEqual(-0.32, ret[2], 6)

        # return what was put in
        ret = factor.multiply([[-1]])
        self.assertAlmostEqual(-1, ret[0], 6)

    def test_poly_to_string(self):
        """Unit test for poly_to_string
        """
        # just one factor case
        self.assertEqual('1.0 + 0.1*x', 
                         factor.poly_to_string([-0.1]))
        
        # mode than one factor
        self.assertEqual('1.0 + 0.1*x - 0.2*x^2', 
                         factor.poly_to_string([-0.1,  0.2]))

    def test_tspolynomial(self):
        """Unit test for TSPolynomial
        """
        # basic test
        tspoly = TSPolynomial([-0.1,  0.2])

        # get back what you put in
        self.assertEqual(-0.1, tspoly.coef[0])
        self.assertEqual(0.2, tspoly.coef[1])
        # get the string
        self.assertEqual('1.0 + 0.1*x - 0.2*x^2', tspoly.poly_str)
