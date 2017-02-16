# Author: Arya McCarthy
"""
The magic behind our method.

Uses waveform relaxation and piecewise polynomial approximations to solve
a system of differential equations.
"""
from __future__ import division, print_function, absolute_import

from numpy import allclose, empty_like, isscalar, ones_like
from scipy.integrate import ode

from .interpolate import interpolate


def make_initial_approximation(y0):
    return [lambda t: y0i for y0i in y0]

def updated_approximation(f, y0, y1, t0, t1, P):
    return P


def call_scipy_integrator(g_i, y0_i, t0, t1, solout, f_params):
    # assert isscalar(y0_i)
    assert isscalar(t0)
    assert isscalar(t1)
    assert callable(g_i)

    r = ode(g_i).set_integrator('dopri5')
    r.set_solout(solout)
    r.set_f_params(*f_params)
    r.set_initial_value(y0_i, t0)
    result = r.integrate(t1)
    assert r.successful()
    return result


def compute_update(f, y0, t0, t1, P, solout, f_params):
    return call_scipy_integrator(f, y0, t0, t1, solout, f_params)


def runner(_, f, y0, t0, t1, rtol, atol, solout, n_iters, verbosity, f_params):
    P = make_initial_approximation(y0)  # Initialize to a flat line.
    y1_previous = ones_like(y0)

    for _ in range(n_iters):
        y1 = compute_update(f, y0, t0, t1, P, solout, f_params)

        if allclose(y1, y1_previous, rtol=rtol, atol=atol):
            break
        else:
            P = updated_approximation(f, y0, y1, t0, t1, P)
            y1_previous = y1
    else:
        raise ArithmeticError("Failed to converge in {} iterations.".format(n_iters))

    return t1, y1
    r = ode(f).set_integrator('dopri5', rtol=rtol, atol=atol,
                              nsteps=n_iters, verbosity=verbosity)
    r.set_solout(solout)
    r.set_f_params(*f_params)
    r.set_initial_value(y0, t0)
    r.integrate(t1)
    assert r.successful()
    return r.t, r.y
