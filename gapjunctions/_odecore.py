# Author: Arya McCarthy
"""
The magic behind our method.

Uses waveform relaxation and piecewise polynomial approximations to solve
a system of differential equations.
"""
from __future__ import division, print_function, absolute_import
try:
    from collections.abc import Sequence
except ImportError:  # Python 2.7
    from collections import Sequence
from functools import partial

from numpy import allclose, empty_like, isscalar, ones_like
from scipy.integrate import ode

from .interpolate import interpolate


class Neuron(object):
    def __init__(self, y0, i):
        self.index = i
        self.system_state = y0
        self._approximation = None

    @property
    def approximation(self):
        if not self._approximation:
            self._approximation = lambda t: self.current_state
        return self._approximation

    @property
    def current_state(self):
        return self.system_state[self.index]


class NeuronPool(Sequence):
    def __init__(self, f, y0, t0, t1, solout, verbosity, f_params):
        self.f = f
        self.system_state = y0
        self.t0 = t0
        self.t1 = t1
        self.neurons = [Neuron(y0, i) for i, _ in enumerate(y0)]
        self._solve = partial(call_scipy_integrator, solout=solout, verbosity=verbosity, f_params=f_params)

    def __getitem__(self, item):
        return self.neurons[item]

    def __len__(self):
        return len(self.neurons)

    def initial_approximations(self):
        return [n.approximation for n in self.neurons]

    def update_approximations(self):
        pass

    def solve_approximated_system(self):
        return self._solve(self.f, self.system_state, self.t0, self.t1)


def call_scipy_integrator(g_i, y0_i, t0, t1, solout, verbosity, f_params):
    # assert isscalar(y0_i)
    assert isscalar(t0)
    assert isscalar(t1)
    assert callable(g_i)

    r = ode(g_i).set_integrator('dopri5', verbosity=verbosity)
    r.set_solout(solout)
    r.set_f_params(*f_params)
    r.set_initial_value(y0_i, t0)
    result = r.integrate(t1)
    assert r.successful()
    return result


def compute_update(f, y0, t0, t1, P, solout, f_params):
    return call_scipy_integrator(f, y0, t0, t1, solout, f_params)


def runner(_, f, y0, t0, t1, rtol, atol, solout, n_iters, verbosity, f_params):
    neurons = NeuronPool(f, y0, t0, t1, solout, verbosity, f_params)
    P = neurons.initial_approximations()
    y1_previous = ones_like(y0)

    for _ in range(n_iters):
        y1 = neurons.solve_approximated_system()

        if allclose(y1, y1_previous, rtol=rtol, atol=atol):
            break
        else:
            neurons.update_approximations()
            y1_previous = y1
    else:
        raise ArithmeticError("Failed to converge in {} iterations.".format(n_iters))

    return t1, y1
