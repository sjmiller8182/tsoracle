
# unit test framework
import unittest
# the package API
from tsoracle import factor
from tsoracle.factor import TSPolynomial
# custom required type
from numpy import ndarray

class TestMultiply(unittest.TestCase):

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
