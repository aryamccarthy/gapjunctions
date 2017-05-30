
"""
Interpolation algorithms using cubic Hermite polynomials.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.interpolate import KroghInterpolator


def interpolate(x0, x1, m0, m1, t0, t1):
    x = np.repeat([t0, t1], 2)
    y = np.ravel(np.dstack(([x0, x1], [m0, m1])))
    return KroghInterpolator(x, y)
