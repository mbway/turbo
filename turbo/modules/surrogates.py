#!/usr/bin/env python3

import warnings
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
        def __call__(self, trial_num, X, y):
            ''' instantiate a new surrogate model

            Returns: (model, fitting_info)
            '''
            raise NotImplementedError()



class SciKitGPSurrogate(Surrogate):
    '''A surrogate model which uses a `GaussianProcessRegressor` from scikit learn

    Note: see http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
    '''
    def __init__(self, **kwargs):
        self.model = sk_gp.GaussianProcessRegressor(**kwargs)

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
            'kernel' : 1.0 * sk_gp.kernels.Matern(nu=2.5) + sk_gp.kernels.WhiteKernel(),
            'n_restarts_optimizer' : 10,
            # make a copy of the training data (in-case it is modified)
            'copy_X_train' : True,
            'normalize_y' : False
        }

        def __init__(self, gp_params=None, variable_iterations=None, hint_with_last_hyper_params=True):
            '''
            Args:
                gp_params (dict): parameters to pass to the `GaussianProcessRegressor` constructor
                    see: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
                variable_iterations: a function from `trial_num` to the number of
                    training iterations to use for fitting the model for this trial. If
                    provided, this supersedes `n_restarts_optimizer` from
                    `gp_params`.

                    If the function returns 0 then then no training
                    is performed and the hyperparameters are fixed to either the
                    last model's parameters or those defined in `gp_params`
                    (depending on the value of `hint_with_last_hyper_params`).

                    If the function returns `n>0` then
                    `n_restarts_optimizer=n-1` is used (note that
                    `n_restarts_optimizer=0` means that 1 iteration is
                    performed).

                    Use cases for this parameter include not training every
                    iteration and instead copying the hyperparameters forward.
                    Alternatively, few iterations can be used for most trials,
                    then occasionally more can be used.

                    Example:
                        `lambda trial_num: 4 if (trial_num-pre_phase) % 3 == 0 else 1`
                        will generate the sequence 4,1,1,4,1,1,4,...
                        starting with 4 on the first trial after the pre-phase

                    Example:
                        `lambda trial_num: [10,5,2][(trial_num-pre_phase) % 3]`
                        will generate a sequence of repeating 10,5,2,10,5,2,...
                hint_with_last_hyper_params: whether to use the last trained
                    surrogate hyperparameters as a starting point when optimising
                    them the next time.
            '''
            assert sk_gp is not None, 'failed to import sklearn.'
            self.gp_params = self.default_params if gp_params is None else gp_params
            self.variable_iterations = variable_iterations
            self.hint_with_last_hyper_params = hint_with_last_hyper_params

            self._last_model_params = None

        def _get_hyper_params_hints(self):
            if self.hint_with_last_hyper_params:
                return None if self._last_model_params is None else [self._last_model_params]
            else:
                return None

        def __call__(self, trial_num, X, y):
            model = SciKitGPSurrogate(**self.gp_params)

            iterations = self.variable_iterations(trial_num) if self.variable_iterations is not None \
                else model.model.n_restarts_optimizer + 1
            assert iterations >= 0, 'invalid number of iterations: {}'.format(iterations)

            fitting_info = {'iterations': iterations}

            if iterations > 0:
                model.model.n_restarts_optimizer = iterations - 1
                hints = self._get_hyper_params_hints()
                with warnings.catch_warnings(record=True) as ws:
                    model.fit(X, y, hyper_params_hints=hints)
                self._last_model_params = model.get_hyper_params()
                fitting_info.update({'hints': hints})
                if len(ws) > 0:
                    fitting_info.update({'warnings': [w.message for w in ws]})

            else:
                model.fit(X, y, fixed_hyper_params=self._last_model_params)
                fitting_info.update({'re_used': (self._last_trained, self._last_model_params)})
            return model, fitting_info


