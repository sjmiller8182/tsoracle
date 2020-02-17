"""Unit tests for generate
"""
# unit test framework
import unittest
# the package API
from tsoracle import generate
# custom required type
from numpy import ndarray

class TestNoise(unittest.TestCase):

    def test_noise_generation(self):
        """Test noise functional API
        """
        normal_noise = generate.noise(0.01, 100, 42)

        # verify correct ret type
        self.assertIsInstance(normal_noise, ndarray, msg='Verify correct ret type')
        # verify correct size
        self.assertEqual(100, normal_noise.size, msg='Verify correct size')
        # test error case
        self.assertRaises(ValueError, generate.noise, 0.01, -1)

        # verify that no additive noise is returned with var is set to 0
        signal = generate.noise(0, 3)
        self.assertEqual(3, signal.size)
        self.assertEqual(0, signal[0])
        self.assertEqual(0, signal[1])
        self.assertEqual(0, signal[2])
    
    def test_noise_generator(self):
        """Test noise generator object
        """
        Generator = generate.Noise()

        realize_1 = Generator.gen(2, 42)
        realize_2 = Generator.gen(2, 42)

        # verify sizes
        self.assertEqual(2, realize_1.size, msg='verify sizes')
        self.assertEqual(2, realize_2.size, msg='verify sizes')
        # verify random seed works
        self.assertAlmostEqual(realize_1[0], realize_2[0], 12, msg='verify random seed works')
        self.assertAlmostEqual(realize_1[1], realize_2[1], 12, msg='verify random seed works')
        # test error case
        self.assertRaises(ValueError, Generator.gen, 0.01, -1)
