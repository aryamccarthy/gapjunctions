# Author: Arya McCarthy
"""
Test our Hermite polynomial creator.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import (
    assert_almost_equal, TestCase, run_module_suite, assert_allclose,
    assert_equal)
from scipy.interpolate import KroghInterpolator

from gapjunctions.interpolate import interpolate


def check_shape(interpolator_cls, x_shape, y_shape, deriv_shape=None, axis=0,
                extra_args={}):
    np.random.seed(1234)

    x = [-1, 0, 1]
    s = list(range(1, len(y_shape) + 1))
    s.insert(axis % (len(y_shape) + 1), 0)
    y = np.random.rand(*((3,) + y_shape)).transpose(s)

    # Cython code chokes on y.shape = (0, 3) etc, skip them
    if y.size == 0:
        return

    xi = np.zeros(x_shape)
    yi = interpolator_cls(x, y, axis=axis, **extra_args)(xi)

    target_shape = ((deriv_shape or ()) + y.shape[:axis] +
                    x_shape + y.shape[axis:][1:])
    assert_equal(yi.shape, target_shape)

    # check it works also with lists
    if x_shape and y.size > 0:
        interpolator_cls(list(x), list(y), axis=axis, **extra_args)(list(xi))

    # check also values
    if xi.size > 0 and deriv_shape is None:
        bs_shape = y.shape[:axis] + (1,) * len(x_shape) + y.shape[axis:][1:]
        yv = y[((slice(None,),) * (axis % y.ndim)) + (1,)]
        yv = yv.reshape(bs_shape)

        yi, y = np.broadcast_arrays(yi, yv)
        assert_allclose(yi, y)


SHAPES = [(), (0,), (1,), (3, 2, 5)]


def _check_complex(ip):
    x = [1, 2, 3, 4]
    y = [1, 2, 1j, 3]
    p = ip(x, y)
    assert_allclose(y, p(x))


def test_complex():
    for ip in [KroghInterpolator]:
        yield _check_complex, ip


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
