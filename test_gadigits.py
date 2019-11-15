#!/usr/bin/env python3

"""Run with pytest."""

import unittest
import unittest.mock

from gadigits import Weight
from gadigits import DigitRecognizer


class TestDigitRecognizer(unittest.TestCase):
    def setUp(self):
        self.random_patcher = unittest.mock.patch("gadigits.random")
        self.random_mock = self.random_patcher.start()

    def tearDown(self):
        self.random_patcher.stop()

    def test_fit_with_max_weights_on_full_white_image(self):
        self.random_mock.uniform.return_value = Weight.max_value
        mnist_image_mock = unittest.mock.Mock(bw_pixels=[1, 1, 1, 1])
        dr = DigitRecognizer(0, dim=2)
        self.assertEqual(1, dr.fit(mnist_image_mock))

    def test_fit_with_min_weights_on_full_white_image(self):
        self.random_mock.uniform.return_value = Weight.min_value
        mnist_image_mock = unittest.mock.Mock(bw_pixels=[1, 1, 1, 1])
        dr = DigitRecognizer(0, dim=2)
        self.assertEqual(-1, dr.fit(mnist_image_mock))

    def test_fit_with_max_weights_on_half_white_image(self):
        self.random_mock.uniform.return_value = Weight.max_value
        mnist_image_mock = unittest.mock.Mock(bw_pixels=[0, 1, 0, 1])
        dr = DigitRecognizer(0, dim=2)
        self.assertAlmostEqual(0.5, dr.fit(mnist_image_mock), delta=0.01)

    def test_zero_pixels_do_not_matter(self):
        mnist_image_mock = unittest.mock.Mock(bw_pixels=[0, 1, 0, 1])

        self.random_mock.uniform.side_effect = [
            Weight.max_value, 0,
            Weight.max_value, 0
        ]
        dr = DigitRecognizer(0, dim=2)
        self.assertAlmostEqual(0, dr.fit(mnist_image_mock), delta=0.01)

        self.random_mock.uniform.side_effect = [
            0, 0,
            0, 0
        ]
        dr = DigitRecognizer(0, dim=2)
        self.assertAlmostEqual(0, dr.fit(mnist_image_mock), delta=0.01)

        self.random_mock.uniform.side_effect = [
            Weight.min_value, 0,
            Weight.min_value, 0
        ]
        dr = DigitRecognizer(0, dim=2)
        self.assertAlmostEqual(0, dr.fit(mnist_image_mock), delta=0.01)

    def test_fit_with_mixed_weights_on_half_white_image(self):
        self.random_mock.uniform.side_effect = [
            0, Weight.min_value,
            0, Weight.min_value
        ]
        mnist_image_mock = unittest.mock.Mock(bw_pixels=[0, 1, 0, 1])
        dr = DigitRecognizer(0, dim=2)
        self.assertAlmostEqual(-0.5, dr.fit(mnist_image_mock), delta=0.01)


class TestWeight(unittest.TestCase):
    def test_weight_transformations(self):
        self.assertEqual('0000000000000000', Weight(-32.766).get_string())
        self.assertEqual(-32.766, Weight('0000000000000000').get_number())

        self.assertEqual('0111111111111101', Weight(-0.001).get_string())
        self.assertEqual(-0.001, Weight('0111111111111101').get_number())
        # print('-32765: {}'.format(Weight(-32765)))
        #
        self.assertEqual('0111111111111110', Weight(0).get_string())
        self.assertEqual(0, Weight('0111111111111110').get_number())
        #
        # print('1: {}'.format(Weight(1)))
        # print('2: {}'.format(Weight(2)))
        # print('3: {}'.format(Weight(3)))
        #
        # print('32766: {}'.format(Weight(32766)))
        # print('32766: {}'.format(Weight('1111111111111100')))
        #
        # print('32767: {}'.format(Weight(32767)))
        # print('32768: {}'.format(Weight(32768)))
        #
        self.assertEqual('1111111111111111', Weight(32.769).get_string())
        self.assertEqual(32.769, Weight('1111111111111111').get_number())

        pass
