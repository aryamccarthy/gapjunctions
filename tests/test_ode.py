# Author: Arya McCarthy
"""
Tests for numerical integration.
"""
from __future__ import division, print_function, absolute_import

from numpy import allclose, array, arange, dot, pi, sqrt, sin, cos, zeros
from numpy.testing import (assert_, TestCase, run_module_suite,
                           assert_array_almost_equal, assert_allclose,
                           assert_array_equal, assert_equal)
from scipy.integrate import odeint, ode

from gapjunctions import ode as myode


class TestOde(TestCase):
    # Check integrate.ode
    def _do_problem(self, problem, integrator, method='adams'):

        # ode has callback arguments in different order than odeint
        f = lambda t, z: problem.f(z, t)
        jac = None
        if hasattr(problem, 'jac'):
            jac = lambda t, z: problem.jac(z, t)

        ig = ode(f, jac)
        ig.set_integrator(integrator,
                          atol=problem.atol / 10,
                          rtol=problem.rtol / 10,
                          method=method)
        ig.set_initial_value(problem.z0, t=0.0)
        z = ig.integrate(problem.stop_t)

        assert_(ig.successful(), (problem, method))
        assert_(problem.verify(array([z]), problem.stop_t), (problem, method))

    def test_relaxation(self):
        # Check the relaxation solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            if problem.stiff:
                continue
            if hasattr(problem, 'jac'):
                continue
            self._do_problem(problem, 'relax')

    def test_concurrent_ok(self):
        f = lambda t, y: 1.0

        for k in range(3):
            for sol in ('relax',):
                r = ode(f).set_integrator(sol)
                r.set_initial_value(0, 0)

                r2 = ode(f).set_integrator(sol)
                r2.set_initial_value(0, 0)

                r.integrate(r.t + 0.1)
                r2.integrate(r2.t + 0.1)
                r2.integrate(r2.t + 0.1)

                assert_allclose(r.y, 0.1)
                assert_allclose(r2.y, 0.2)

            for sol in ('relax',):
                r = ode(f).set_integrator(sol)
                r.set_initial_value(0, 0)

                r2 = ode(f).set_integrator(sol)
                r2.set_initial_value(0, 0)

                r.integrate(r.t + 0.1)
                r.integrate(r.t + 0.1)
                r2.integrate(r2.t + 0.1)
                r.integrate(r.t + 0.1)
                r2.integrate(r2.t + 0.1)

                assert_allclose(r.y, 0.3)
                assert_allclose(r2.y, 0.2)


class TestSolout(TestCase):
    # Check integrate.ode correctly handles solout for dopri5 and dop853
    def _run_solout_test(self, integrator):
        # Check correct usage of solout
        ts = []
        ys = []
        t0 = 0.0
        tend = 10.0
        y0 = [1.0, 2.0]

        def solout(t, y):
            ts.append(t)
            ys.append(y.copy())

        def rhs(t, y):
            return [y[0] + y[1], -y[1]**2]
        ig = ode(rhs).set_integrator(integrator)
        ig.set_solout(solout)
        ig.set_initial_value(y0, t0)
        ret = ig.integrate(tend)
        assert_array_equal(ys[0], y0)
        assert_array_equal(ys[-1], ret)
        assert_equal(ts[0], t0)
        assert_equal(ts[-1], tend)

    def test_solout(self):
        for integrator in ('relax',):
            self._run_solout_test(integrator)

    def _run_solout_break_test(self, integrator):
        # Check correct usage of stopping via solout
        ts = []
        ys = []
        t0 = 0.0
        tend = 10.0
        y0 = [1.0, 2.0]

        def solout(t, y):
            ts.append(t)
            ys.append(y.copy())
            if t > tend / 2.0:
                return -1

        def rhs(t, y):
            return [y[0] + y[1], -y[1]**2]
        ig = ode(rhs).set_integrator(integrator)
        ig.set_solout(solout)
        ig.set_initial_value(y0, t0)
        ret = ig.integrate(tend)
        assert_array_equal(ys[0], y0)
        assert_array_equal(ys[-1], ret)
        assert_equal(ts[0], t0)
        assert_(ts[-1] > tend / 2.0)
        assert_(ts[-1] < tend)

    def test_solout_break(self):
        for integrator in ('relax',):
            self._run_solout_break_test(integrator)


# ------------------------------------------------------------------------------
# Test problems
# ------------------------------------------------------------------------------


class ODE:
    """
    ODE problem

    Borrowed from scipy's test_integrate.py
    """
    stiff = False
    cmplx = False
    stop_t = 1
    z0 = []

    atol = 1e-6
    rtol = 1e-5


class SimpleOscillator(ODE):
    r"""
    Free vibration of a simple oscillator::
        m \ddot{u} + k u = 0, u(0) = u_0 \dot{u}(0) \dot{u}_0
    Solution::
        u(t) = u_0*cos(sqrt(k/m)*t)+\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)
    """
    stop_t = 1 + 0.09
    z0 = array([1.0, 0.1], float)

    k = 4.0
    m = 1.0

    def f(self, z, t):
        tmp = zeros((2, 2), float)
        tmp[0, 1] = 1.0
        tmp[1, 0] = -self.k / self.m
        return dot(tmp, z)

    def verify(self, zs, t):
        omega = sqrt(self.k / self.m)
        u = (self.z0[0] * cos(omega * t) +
             self.z0[1] * sin(omega * t) / omega)
        return allclose(u, zs[:, 0], atol=self.atol, rtol=self.rtol)


PROBLEMS = [SimpleOscillator]


# ------------------------------------------------------------------------------


def f(t, x):
    dxdt = [x[1], -x[0]]
    return dxdt


def jac(t, x):
    j = np.array([[0.0, 1.0],
                  [-1.0, 0.0]])
    return j


def f1(t, x, omega):
    dxdt = [omega * x[1], -omega * x[0]]
    return dxdt


def jac1(t, x, omega):
    j = array([[0.0, omega],
               [-omega, 0.0]])
    return j


def f2(t, x, omega1, omega2):
    dxdt = [omega1 * x[1], -omega2 * x[0]]
    return dxdt


def jac2(t, x, omega1, omega2):
    j = array([[0.0, omega1],
               [-omega2, 0.0]])
    return j


def fv(t, x, omega):
    dxdt = [omega[0] * x[1], -omega[1] * x[0]]
    return dxdt


def jacv(t, x, omega):
    j = array([[0.0, omega[0]],
               [-omega[1], 0.0]])
    return j


class ODECheckParameterUse(object):
    """Call an ode-class solver with several cases of parameter use."""

    # This class is intentionally not a TestCase subclass.
    # solver_name must be set before tests can be run with this class.

    # Set these in subclasses.
    solver_name = ''
    solver_uses_jac = False

    def _get_solver(self, f, jac):
        solver = ode(f, jac)
        if self.solver_uses_jac:
            solver.set_integrator(self.solver_name, atol=1e-9, rtol=1e-7,
                                  with_jacobian=self.solver_uses_jac)
        else:
            # XXX Shouldn't set_integrator *always* accept the keyword arg
            # 'with_jacobian', and perhaps raise an exception if it is set
            # to True if the solver can't actually use it?
            solver.set_integrator(self.solver_name, atol=1e-9, rtol=1e-7)
        return solver

    def _check_solver(self, solver):
        ic = [1.0, 0.0]
        solver.set_initial_value(ic, 0.0)
        solver.integrate(pi)
        assert_array_almost_equal(solver.y, [-1.0, 0.0])

    def test_no_params(self):
        solver = self._get_solver(f, jac)
        self._check_solver(solver)

    def test_one_scalar_param(self):
        solver = self._get_solver(f1, jac1)
        omega = 1.0
        solver.set_f_params(omega)
        if self.solver_uses_jac:
            solver.set_jac_params(omega)
        self._check_solver(solver)

    def test_two_scalar_params(self):
        solver = self._get_solver(f2, jac2)
        omega1 = 1.0
        omega2 = 1.0
        solver.set_f_params(omega1, omega2)
        if self.solver_uses_jac:
            solver.set_jac_params(omega1, omega2)
        self._check_solver(solver)

    def test_vector_param(self):
        solver = self._get_solver(fv, jacv)
        omega = [1.0, 1.0]
        solver.set_f_params(omega)
        if self.solver_uses_jac:
            solver.set_jac_params(omega)
        self._check_solver(solver)


class RelaxationCheckParameterUse(ODECheckParameterUse, TestCase):
    solver_name = 'relax'
    solver_uses_jac = False


def test_odeint_banded_jacobian():
    # Test the use of the `Dfun`, `ml` and `mu` options of odeint.

    def func(y, t, c):
        return c.dot(y)

    def jac(y, t, c):
        return c

    def bjac_cols(y, t, c):
        return np.column_stack((np.r_[0, np.diag(c, 1)], np.diag(c)))

    def bjac_rows(y, t, c):
        return np.row_stack((np.r_[0, np.diag(c, 1)], np.diag(c)))

    c = array([[-50, 75, 0],
               [0, -0.1, 1],
               [0, 0, -1e-4]])

    y0 = arange(3)
    t = np.linspace(0, 50, 6)

    # The results of the following three calls should be the same.
    sol0, info0 = odeint(func, y0, t, args=(c,), full_output=True,
                         Dfun=jac)

    sol1, info1 = odeint(func, y0, t, args=(c,), full_output=True,
                         Dfun=bjac_cols, ml=0, mu=1, col_deriv=True)

    sol2, info2 = odeint(func, y0, t, args=(c,), full_output=True,
                         Dfun=bjac_rows, ml=0, mu=1)

    # These could probably be compared using `assert_array_equal`.
    # The code paths might not be *exactly* the same, so `allclose` is used
    # to compare the solutions.
    assert_allclose(sol0, sol1)
    assert_allclose(sol0, sol2)

    # Verify that the number of jacobian evaluations was the same
    # for all three calls of odeint.  This is a regression test--there
    # was a bug in the handling of banded jacobians that resulted in
    # an incorrect jacobian matrix being passed to the LSODA code.
    # That would cause errors or excessive jacobian evaluations.
    assert_array_equal(info0['nje'], info1['nje'])
    assert_array_equal(info0['nje'], info2['nje'])


if __name__ == "__main__":
    run_module_suite()
