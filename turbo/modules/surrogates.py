#!/usr/bin/env python3

import numpy as np

try:
    import sklearn.gaussian_process as sk_gp
except ImportError:
    sk_gp = None # not required if not used

try:
    import gpy
except ImportError:
    gpy = None # not required if not used


#TODO: for models which support it, the factory can use the hyperparameters for the last model to initialise the next one (e.g. for MCMC)

class Surrogate(object):
    ''' A probabilistic model for approximating the objective function

    A wrapper around specific models or libraries suitable for being used as a
    surrogate model for Bayesian optimisation.
    '''

    def fit(self, X, y, hyper_params=None):
        '''fit the model to the given data set

        If hyperparameters are not provided then they are estimated
        algorithmically (e.g. through optimisation of data likelihood or
        marginalisation etc.)
        '''
        raise NotImplementedError()

    def predict(self, X, return_std_dev=False):
        '''
        Args:
            X: a point or matrix of points (as rows)

        Returns:
            The mean `y` prediction for the given `X`'s, and also the standard
            deviation if `return_std_dev=True`.
            `mus.shape == (X_height,)`
            `sigmas.shape == (X_height,)`
        '''
        raise NotImplementedError()

    def sample(self, x, n):
        '''
        Returns:
            `n` samples for `y` given the input `x`
        '''
        raise NotImplementedError()

    def get_hyper_params(self):
        '''
        Returns:
            the hyperparameters of the model in a format suitable for storage
            and passing to :meth:`train()`.

        Note:
            When training with the same hyperparameters and dataset, the
            resulting model should be identical. Alternatively, training with
            the same hyperparameters with a different dataset is also possible.
        '''
        raise NotImplementedError()

    def get_training_set(self):
        '''
        Returns:
            the `X, y` training set which the model is trained on.
        '''
        raise NotImplementedError()

    class Factory:
        ''' Passed to the optimiser and used to create Surrogate instances for
        each iteration of the optimisation

        The factory is instantiated by the user so that the surrogate model can
        be configured. The optimiser then uses the factory to generate and train
        a new model each iteration.
        '''
        def __call__(self, X=None, y=None, hyper_params_hint=None):
            ''' generate a surrogate model trained on the given data

            Args:
                hyper_params_hint: when provided, use the given hyper parameters
                    as a starting point during the optimisation process.
                    Obtained using `get_hyper_params()` on a trained model.

            Note:
                can either omit `X, y` in which case the surrogate will not be
                fitted to anything, or provide both.
            '''
            raise NotImplementedError()



class SciKitGPSurrogate(Surrogate):
    def __init__(self, gp_params):
        self.gp_params = gp_params
        self.model = sk_gp.GaussianProcessRegressor(**self.gp_params)

    def fit(self, X, y, hyper_params=None):
        if hyper_params is None:
            self.model.fit(X, y)
        else:
            # gp_params may not have kernel or optimizer defined
            p = self.model.get_params()
            kernel = p['kernel']
            opt = p['optimizer']
            # hyper_params are the parameters for the _kernel_ only.
            trained_kernel = kernel.clone_with_theta(np.array(hyper_params))
            # don't want to modify the kernel which is part of gp_params, so modify a clone
            self.model.set_params(kernel=trained_kernel)
            self.model.set_params(optimizer=None)
            self.model.fit(X, y)
            self.model.set_params(kernel=kernel, optimizer=opt)

    def predict(self, X, return_std_dev=False):
        res = self.model.predict(X, return_std=return_std_dev)
        if return_std_dev:
            # both mus and sigmas should have shape (X_height,)
            return res[0].flatten(), res[1]
        else:
            return res.flatten()

    def sample(self, x, n):
        # by default random_state uses a fixed seed! Setting to None uses the
        # current numpy random state.
        return self.model.sample_y(x, n, random_state=None)

    def get_hyper_params(self):
        '''
        Note:
            theta is log transformed
        '''
        return np.copy(self.model.kernel_.theta)


    class Factory(Surrogate.Factory):
        default_params = {
            # the multiplier allows the kernel to have a peak value different from
            # 1, and therefore can fit better to data of different scales.
            # nu=1.5 assumes the target function is once-differentiable
            # WhiteKernel assumes that the objective function contains some noise
            'kernel' : 1.0 * sk_gp.kernels.Matern(nu=1.5) + sk_gp.kernels.WhiteKernel(),
            'n_restarts_optimizer' : 10,
            # make a copy of the training data (in-case it is modified)
            'copy_X_train' : True
        }

        def __init__(self, gp_params=None, use_hint=True):
            '''
            Args:
                gp_params (dict): parameters to pass to scikit
                    see: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
                use_hint: whether to use the hyper parameters hint (usually the
                    hyperparameters of the previous model) as a starting point
                    for the hyper parameter optimisation.
            '''
            assert sk_gp is not None, 'failed to import sklearn.'
            self.gp_params = self.default_params if gp_params is None else gp_params
            self.use_hint = use_hint

        def __call__(self, X=None, y=None, hyper_params_hint=None):
            '''
            Note:
                can either omit `X, y` in which case the surrogate will not be
                fitted to anything, or provide both.
            '''
            sur = SciKitGPSurrogate(self.gp_params)
            if X is not None and y is not None:
                if self.use_hint and hyper_params_hint is not None:
                    # the hyperparameter values provided here are used as a
                    # starting point during restart 0 of the optimiser.
                    sur.model.kernel = sur.model.kernel.clone_with_theta(hyper_params_hint)
                sur.fit(X, y)
            return sur


