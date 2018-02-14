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

    def fit(self, X, y, hyper_params_hints=None, fixed_hyper_params=None):
        '''fit the model to the given data set

        If fixed hyperparameters are not provided then they are estimated
        algorithmically (e.g. through optimisation of data likelihood or
        marginalisation etc.)

        Args:
            hyper_params_hints (list): an optional list of hyperparameters to use
                as starting points during the model training process
            fixed_hyper_params: a set of hyperparameters to use instead of training

        Note:
            only one of `hyper_params_hints` and `fixed_hyper_params` should be provided at once
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
        def reset(self):
            ''' called when the optimiser is reset '''
            raise NotImplementedError()
        def __call__(self, trial_num, X, y):
            ''' instantiate a new surrogate model

            Returns: (model, fitting_info)
            '''
            raise NotImplementedError()



class SciKitGPSurrogate(Surrogate):
    def __init__(self, gp_params):
        self.gp_params = gp_params
        self.model = sk_gp.GaussianProcessRegressor(**self.gp_params)

    def fit(self, X, y, hyper_params_hints=None, fixed_hyper_params=None):
        assert fixed_hyper_params is None or hyper_params_hints is None, \
            'cannot provided fixed hyperparameters and training hints at the same time'
        if fixed_hyper_params is None:
            if hyper_params_hints is not None:
                assert len(hyper_params_hints) == 1, 'multiple hints not implemented' #TODO
                # the hyperparameter values provided here are used as a starting
                # point during restart 0 of the optimiser.
                self.model.kernel = self.model.kernel.clone_with_theta(hyper_params_hints[0])
            self.model.fit(X, y)
        else:
            # gp_params may not have kernel or optimizer defined
            p = self.model.get_params()
            kernel = p['kernel']
            opt = p['optimizer']
            # hyper_params are the parameters for the _kernel_ only.
            trained_kernel = kernel.clone_with_theta(fixed_hyper_params)
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

        def __init__(self, gp_params=None, train_interval=0, hint_with_last_hyper_params=True):
            '''
            Args:
                gp_params (dict): parameters to pass to scikit
                    see: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
                surrogate_train_interval: how often to optimise the hyperparameters
                    of the surrogate model. 0 => every iteration. When not training,
                    the hyperparameters are reused from the last time they were
                    trained.
                hint_with_last_hyper_params: whether to use the last trained
                    surrogate hyperparameters as a starting point when optimising
                    them the next time.
            '''
            assert sk_gp is not None, 'failed to import sklearn.'
            self.gp_params = self.default_params if gp_params is None else gp_params
            self.train_interval = train_interval
            self.hint_with_last_hyper_params = hint_with_last_hyper_params
            self._last_trained = -1
            self._last_model_params = None

        def reset(self):
            self._last_trained = -1
            self._last_model_params = None

        def _should_train_model(self, trial_num):
            trials_since = trial_num - self._last_trained - 1
            return self._last_trained == -1 or trials_since >= self.train_interval

        def _get_hyper_params_hints(self):
            if self.hint_with_last_hyper_params:
                return None if self._last_model_params is None else [self._last_model_params]
            else:
                return None

        def __call__(self, trial_num, X, y):
            model = SciKitGPSurrogate(self.gp_params)
            train = self._should_train_model(trial_num)
            fitting_info = {'trained': train}
            if train:
                hints = self._get_hyper_params_hints()
                model.fit(X, y, hyper_params_hints=hints)
                self._last_trained = trial_num
                self._last_model_params = model.get_hyper_params()
                fitting_info.update({'hints': hints})
            else:
                model.fit(X, y, fixed_hyper_params=self._last_model_params)
                fitting_info.update({'re_used': (self._last_trained, self._last_model_params)})
            return model, fitting_info


