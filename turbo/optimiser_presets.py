#!/usr/bin/env python3
'''
Sensible preset configurations for the optimiser
'''

import sklearn.gaussian_process as gp
from . import modules as tm


def load_optimiser_preset(optimiser, name):
    if name == 'default':
        # this default should hopefully provide reasonable results in most situations
        optimiser.latent_space = tm.NoLatentSpace()
        optimiser.plan = tm.Plan(pre_phase_trials=10)
        optimiser.pre_phase_select = tm.random_selector()
        optimiser.maximise_acq = tm.random_quasi_newton()
        optimiser.async_eval = None
        optimiser.surrogate_factory = tm.SciKitGPSurrogate.Factory(gp_params=dict(
            alpha = 1e-5, # larger => more noise. Default = 1e-10
            kernel = 1.0 * gp.kernels.Matern(nu=2.5) + gp.kernels.WhiteKernel(),
            n_restarts_optimizer = 10,
        ))
        optimiser.acq_func_factory = tm.EI.Factory(xi=0.01)
    else:
        raise ValueError('unknown preset name: {}'.format(name))

