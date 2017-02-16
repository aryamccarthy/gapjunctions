# Author: Arya McCarthy
"""
Test our Hermite polynomial creator.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import (
    assert_almost_equal, TestCase, run_module_suite)

from gapjunctions.interpolate import interpolate


class TestInterpolate(TestCase):
    # Test whichever interpolation method we use.
    def setUp(self):
        self.true_poly = np.poly1d([-2, 3, 1, 5])
        self.test_xs = np.linspace(-1, 1, 100)
        self.xs = np.linspace(-1, 1, 2)
        self.ys = self.true_poly(self.xs)

    def test_hermite(self):
        # Can we make a Hermite polynomial?
        xs = [0.1, 0.1, 1, 1]
        ys = [self.true_poly(0.1),
              self.true_poly.deriv(1)(0.1),
              self.true_poly(1),
              self.true_poly.deriv(1)(1)]
        P = interpolate(ys[0], ys[2], ys[1], ys[3], xs[0], xs[-1])
        assert_almost_equal(self.true_poly(self.test_xs), P(self.test_xs))


if __name__ == '__main__':
    run_module_suite()
