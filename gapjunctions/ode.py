# Author: Arya McCarthy
"""
First-order integrator using waveform relaxation.
TODO: Implement the waveform relaxation.

class RelaxationIntegrator
---------

A subclass of the SciPy ODE class.
"""

from scipy.integrate import _ode
from scipy.integrate import ode


def runner(_, f, y0, t0, t1, rtol, atol, solout, nsteps, verbosity, f_params):
    r = ode(f).set_integrator('dopri5', rtol=rtol, atol=atol,
                              nsteps=nsteps, verbosity=verbosity)
    r.set_solout(solout)
    r.set_f_params(*f_params)
    r.set_initial_value(y0, t0)
    r.integrate(t1)
    assert r.successful()
    return r.t, r.y


class RelaxationIntegrator(_ode.IntegratorBase):
    """Integrator capable of handling discontinuous, coupled equations."""

    runner = runner  # TODO: Switch to WR runner.
    name = 'relax'
    supports_solout = True

    def __init__(self,
                 rtol=1e-6, atol=1e-12,
                 nsteps=500,
                 method=None,
                 verbosity=1,
                 ):
        self.rtol = rtol
        self.atol = atol
        self.nsteps = nsteps
        self.verbosity = verbosity
        self.success = True
        self.set_solout(None)

    def set_solout(self, solout, complex=False):
        self._solout = solout

    def reset(self, n, has_jac):
        self.call_args = [self.rtol, self.atol, self._solout,
                          self.nsteps, self.verbosity]
        self.success = True

    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        args = ((f, y0, t0, t1) + tuple(self.call_args) + (f_params,))
        try:
            t, y = self.runner(*args)
        except ValueError as e:
            print("Something went wrong with integration!")
            self.success = False
            raise
        return y, t


if RelaxationIntegrator.runner is not None:
    _ode.IntegratorBase.integrator_classes.append(RelaxationIntegrator)
