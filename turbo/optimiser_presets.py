#!/usr/bin/env python3
""" Sensible preset configurations for the optimiser """

import sklearn.gaussian_process as sk_gp
from . import modules as tm


def load_optimiser_preset(optimiser, name):
    if name == 'default':
        # this default should hopefully provide reasonable results in most situations
        # the Optimiser.Plan defaults are left alone
        optimiser.latent_space = tm.NoLatentSpace()
        optimiser.pre_phase_select = tm.LHS_selector(num_total=optimiser.pre_phase_trials)
        optimiser.fallback = tm.Fallback(selector=tm.random_selector())
        optimiser.aux_optimiser = tm.RandomAndQuasiNewton()
        optimiser.surrogate = tm.GPySurrogate()
        '''
        optimiser.surrogate = tm.SciKitGPSurrogate(model_params=dict(
            kernel = 1.0 * sk_gp.kernels.Matern(nu=2.5) + sk_gp.kernels.WhiteKernel(),
            normalize_y = True,
        ), training_iterations=10)
        '''
        optimiser.acquisition = tm.EI(xi=0.01)

    elif name == 'random_search':
        # a degenerate optimiser which never leaves the pre-phase and so
        # performs random search rather than Bayesian optimisation.
        optimiser.pre_phase_trials = float('inf')
        optimiser.latent_space = tm.NoLatentSpace()
        optimiser.pre_phase_select = tm.random_selector()
        optimiser.fallback = None
        optimiser.aux_optimiser = None
        optimiser.surrogate = None
        optimiser.acquisition = None

    else:
        raise ValueError('unknown preset name: {}'.format(name))

