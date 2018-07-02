#!/usr/bin/env python3
'''
Implementations of the various surrogate model options for use with Bayesian
optimisation. Currently, only different Gaussian process implementations are
available, however other machine learning models and variations of Gaussian
processes are possible so long as they can predict uncertainty as well as a mean
prediction.
'''
# python 2 compatibility
from __future__ import (absolute_import, division, print_function, unicode_literals)
from .py2 import *

try:
    import sklearn.gaussian_process as sk_gp
except ImportError:
    pass # not required if not used

try:
    import gpy
except ImportError:
    pass # not required if not used

# local imports
from .utils import *

#TODO: tests

class Surrogate(object):
    '''
    A wrapper around specific models or libraries suitable for being used as a
    surrogate model for Bayesian optimisation.
    '''

    def fit(self, X, y, hyper_params=None, max_its=None):
        '''
        train the model to fit the given data set {X, y}. If hyperparameters are
        not provided then they are obtained by optimising the data likelihood,
        guided by the given gp_parmas. If max_its is provided, then it
        overwrites the default maximum iterations parameter.
        '''
        raise NotImplementedError()
    def predict(self, X, std_dev=False):
        '''
        return the mean y-prediction for the given Xs, and also the variance if
        std_dev=True
        '''
        raise NotImplementedError()
    def get_hyper_params(self):
        '''
        return the hyperparameters of the model in a format suitable for storage
        and passing to train.

        When training with the same hyperparameters and dataset, the resulting
        model should be identical. Alternatively, training with the same
        hyperparameters with a different dataset is also possible.
        '''
        raise NotImplementedError()
    def sample(self, x, n):
        '''
        return n y-samples for given the input x
        '''
        raise NotImplementedError()
    def get_training_set(self):
        '''
        return the X and y arrays that fit() was called with
        '''
        raise NotImplementedError()


class SciKitGPSurrogate(Surrogate):

    @staticmethod
    def Custom(gp_params):
        '''
        Specialise a SciKitGPSurrogate with the given parameters which are passed to
        the scikit GaussianProcessRegressor constructor
        gp_params: a dictionary of parameters
        '''
        class SciKitSurrogate_Specialised(SciKitGPSurrogate):
            def __init__(self, optimiser):
                super().__init__(optimiser, gp_params)
        return SciKitSurrogate_Specialised

    def __init__(self, optimiser, gp_params=None):
        if gp_params is None:
            self.gp_params = dict(
                alpha = 1e-10, # larger => more noise. Default = 1e-10
                # nu=1.5 assumes the target function is once-differentiable
                kernel = 1.0 * sk_gp.kernels.Matern(nu=1.5) + sk_gp.kernels.WhiteKernel(),
                #kernel = 1.0 * sk_gp.kernels.RBF(),
                n_restarts_optimizer = 10,
                # make the mean 0 (theoretically a bad thing, see docs, but can help)
                # with the constant offset in the kernel this shouldn't be required
                # this may be a dangerous option, seems to make worse predictions
                #normalize_y = True,
                copy_X_train = True # whether to make a copy of the training data (in-case it is modified)
            )
        else:
            self.gp_params = gp_params
        self.log_warning = lambda warn: optimiser._log('GP warning: {}'.format(warn))

    def fit(self, X, y, hyper_params=None, max_its=None):

        # max_its overwrites the default parameters
        if max_its is None:
            gp_params = self.gp_params
        else:
            gp_params = self.gp_params.copy()
            gp_params['n_restarts_optimizer'] = max_its

        self.model = sk_gp.GaussianProcessRegressor(**gp_params)
        with WarningCatcher(self.log_warning):
            if hyper_params is None:
                    self.model.fit(X, y)
            else:
                # gp_params may not have everything defined
                p = self.model.get_params()
                kernel = p['kernel']
                trained_kernel = kernel.clone_with_theta(np.array(hyper_params))
                opt = p['optimizer']
                self.model.set_params(optimizer=None)
                # don't want to modify the kernel which is part of gp_params, so modify a clone
                self.model.set_params(kernel=trained_kernel)
                self.model.fit(X, y)
                self.model.set_params(kernel=kernel, optimizer=opt)

    def predict(self, X, std_dev=False):
        with WarningCatcher(self.log_warning):
            return self.model.predict(X, return_std=std_dev)

    def get_hyper_params(self):
        return np.copy(self.model.kernel_.theta)

    def sample(self, x, n):
        with WarningCatcher(self.log_warning):
            # by default random_state uses a fixed seed! Setting to None uses the
            # current numpy random state.
            return self.model.sample_y(x, n, random_state=None)

    def get_training_set(self):
        return (self.model.X_train_, self.model.y_train_)

