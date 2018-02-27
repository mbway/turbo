#!/usr/bin/env python3

import warnings
import numpy as np
import copy

try:
    import sklearn.gaussian_process as sk_gp
except ImportError:
    sk_gp = None # not required if not used

try:
    import GPy
except ImportError:
    GPy = None # not required if not used

import turbo as tb

#TODO: MCMC?

class Surrogate:
    ''' A probabilistic model (predicts uncertainty as well as the mean) for approximating the objective function

    The surrogate persists throughout the Bayesian optimisation run and may
    store some state (such as the last model parameters). It is a factory which
    constructs a new model instance for each trial/iteration.

    This class provides a wrapper around specific models or libraries suitable
    for being used as a surrogate model for Bayesian optimisation.
    '''

    def construct_model(self, trial_num, X, y):
        '''create a model instance trained on the data set for the given trial

        Returns: (model, fitting_info)
        '''
        raise NotImplementedError()


    class ModelInstance:
        '''An instance of the surrogate model which is trained on the data set
        for a particular trial.

        This class provides a consistent interface independent of the underlying library.
        '''
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

        def get_hyper_params(self):
            '''
            Returns:
                the hyperparameters of the model in a format suitable for
                storage and using for starting points for future models.

            Note:
                When training with the same hyperparameters and dataset, the
                resulting model should be identical. Alternatively, training with
                the same hyperparameters with a different dataset is also possible.
            '''
            raise NotImplementedError()

        def get_hyper_param_names(self):
            ''' get the names of the parameters corresponding to get_hyper_params '''
            raise NotImplementedError()

        def get_log_likelihood(self):
            ''' get the data log likelihood of the model '''
            raise NotImplementedError()




class GPySurrogate(Surrogate):
    '''A surrogate model which uses GPy for Gaussian process regression

    Note: see https://gpy.readthedocs.io/en/deploy/

    differences between GPy and scikit learn
    - 0 restarts of n_restarts_optimizer in scikit means 1 iteration (but no
        restarts). In GPy num_restarts=0 means 0 iterations.
    - GPy includes a Gaussian noise term in the constructor which is preferred
        over using a white kernel (https://github.com/SheffieldML/GPy/issues/506)
    - GPy kernels do not require multiplication by a constant kernel, since this
        constant is included in the kernel already.
    '''

    default_model_params = {'normalizer' : True}
    default_optimise_params = {'parallel' : True, 'verbose' : False}

    def __init__(self, model_params=None, optimise_params=None,
                 training_iterations=10, param_continuity=True, sparse=False):
        '''
        Args:
            model_params (dict): arguments to pass to the model constructor
                (GPRegression or SparseGPRegression) (see GPy documentation)
            optimise_params (dict): arguments to be passed to
                optimize_restarts() (see GPy or paramz documentation)
            training_iterations (None, int, or function): the number of
                optimiser iterations to perform. Can be a constant or a function of
                the trail number. Set to None to specify with optimise_params
                instead.

                0 => no training is performed (fixed to the starting values)

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
            param_continuity (bool): whether to use the trained hyper parameters
                from the previous model as a starting point when training the
                next model (otherwise chosen randomly).
            sparse (bool): whether to use SparseGPRegression instead of
                GPRegression as the model (see GPy documentation)
        '''
        assert GPy is not None, 'failed to import GPy.'
        self.model_params = model_params or self.default_model_params
        self.optimise_params = optimise_params or self.default_optimise_params
        self.training_iterations = training_iterations
        assert training_iterations is None or self.optimise_params.get('num_restarts') is None, \
            'cannot specify num_restarts and training_iterations at the same time'
        self.param_continuity = param_continuity
        self.sparse = sparse

        self._last_model_params = None

    def _get_training_iterations(self, trial_num):
        if self.training_iterations is None:
            iterations = self.optimise_params.get('num_restarts') # may be not present or None
        elif callable(self.training_iterations):
            iterations = self.training_iterations(trial_num)
        else:
            iterations = self.training_iterations
        assert iterations is not None, 'must specify the number of training iterations'
        assert iterations >= 0, 'invalid number of iterations: {}'.format(iterations)
        return iterations

    def construct_model(self, trial_num, X, y):
        iterations = self._get_training_iterations(trial_num)
        fitting_info = {'iterations' : iterations}

        # the kernel parameters are altered by the model, so give a copy each
        # time. Also kernels cause pickling issues once they have been passed to
        # a model.
        model_params = copy.deepcopy(self.model_params)

        model_class = GPy.models.SparseGPRegression if self.sparse else GPy.models.GPRegression

        # don't initialise the model until the initial hyperparameters have been set
        # will always raise RuntimeWarning("Don't forget to initialize by self.initialize_parameter()!")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*initialize_parameter.*')
            model = model_class(X, tb.utils.col2D(y), initialize=False, **model_params)

        model.update_model(False) # prevents the GP from fitting to the data until we are ready to enable it manually
        model.initialize_parameter() # initialises the hyperparameter objects
        if self.param_continuity and self._last_model_params is not None:
            model[:] = self._last_model_params
        model.update_model(True)

        if iterations == 0: # fixed
            fitting_info.update({'fixed' : model[:]})
        else:
            # the current parameters are used as one of the starting locations (as of the time of writing)
            # https://github.com/sods/paramz/blob/master/paramz/model.py
            optimise_params = self.optimise_params.copy()
            # for this function restarts == iterations (whereas with scikit learn restarts = iterations-1)
            optimise_params['num_restarts'] = iterations

            with warnings.catch_warnings(record=True) as ws:
                model.optimize_restarts(**optimise_params)

            if len(ws) > 0:
                fitting_info.update({'warnings': [w.message for w in ws]})

            self._last_model_params = model[:]

        return GPySurrogate.ModelInstance(model), fitting_info


    class ModelInstance(Surrogate.ModelInstance):
        def __init__(self, model):
            self.model = model

        def predict(self, X, return_std_dev=False):
            mean, var = self.model.predict(X)
            # both mus and sigmas should have shape (X_height,)
            if return_std_dev:
                return mean.flatten(), np.sqrt(var).flatten()
            else:
                return mean.flatten()

        def get_hyper_params(self):
            return self.model[:]

        def get_hyper_param_names(self):
            return self.model.parameter_names()

        def get_log_likelihood(self):
            return self.model.log_likelihood()




class SciKitGPSurrogate(Surrogate):
    '''A surrogate model which uses a `GaussianProcessRegressor` from scikit learn

    Note: see http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
    '''

    if sk_gp is not None:
        default_model_params = {
            # the 1.0* allows the kernel to have a peak value different from
            # 1, and therefore can fit better to data of different scales.
            # nu=1.5 assumes the target function is once-differentiable
            # WhiteKernel assumes that the objective function contains some noise
            'kernel' : 1.0 * sk_gp.kernels.Matern(nu=2.5) + sk_gp.kernels.WhiteKernel(),
            # from the scikit documentation: When enabled, the normalization
            # effectively modifies the GP's prior based on the data, which
            # contradicts the likelihood principle; normalization is thus disabled
            # per default.
            'normalize_y' : True
        }


    def __init__(self, model_params=None, training_iterations=None, param_continuity=True):
        '''
        Args:
            model_params (dict): parameters to pass to the `GaussianProcessRegressor` constructor
                see: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
            training_iterations (None, int, or function): the number of
                optimiser iterations to perform. Can be a constant or a function of
                the trail number. Set to None to specify with n_restarts_optimizer
                instead.

                0 => no training is performed (fixed to the starting values)

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
            param_continuity (bool): whether to use the trained hyper parameters
                from the previous model as a starting point when training the
                next model (otherwise chosen randomly).
        '''
        assert sk_gp is not None, 'failed to import sklearn.'
        self.model_params = model_params or self.default_model_params
        self.training_iterations = training_iterations
        assert training_iterations is None or self.model_params.get('n_restarts_optimizer') is None, \
            'cannot specify n_restarts_optimizer and training_iterations at the same time'
        self.param_continuity = param_continuity

        self._last_model_params = None

    def _get_training_iterations(self, trial_num):
        if self.training_iterations is None:
            iterations = self.model_params.get('n_restarts_optimizer') # may be not present or None
        elif callable(self.training_iterations):
            iterations = self.training_iterations(trial_num)
        else:
            iterations = self.training_iterations
        assert iterations is not None, 'must specify the number of training iterations'
        assert iterations >= 0, 'invalid number of iterations: {}'.format(iterations)
        return iterations

    def construct_model(self, trial_num, X, y):
        iterations = self._get_training_iterations(trial_num)
        fitting_info = {'iterations' : iterations}

        assert 'kernel' in self.model_params, 'you must specify a kernel for the GP'
        # don't want the initial parameter values to be changed, so make a copy
        model_params = copy.deepcopy(self.model_params)

        if self.param_continuity and self._last_model_params is not None:
            # theta is log-transformed
            model_params['kernel'].theta = np.log(self._last_model_params.copy())

        if iterations == 0: # fixed
            model_params['optimizer'] = None
            model_params['n_restarts_optimizer'] = 0
            fitting_info.update({'fixed' : model.kernel.theta})
        else:
            # for scikit: 0 restarts => 1 iteration
            model_params['n_restarts_optimizer'] = iterations - 1

        model = sk_gp.GaussianProcessRegressor(**model_params)

        with warnings.catch_warnings(record=True) as ws:
            model.fit(X, y)
        if len(ws) > 0:
            fitting_info.update({'warnings': [w.message for w in ws]})

        if iterations > 0:
            # theta is log-transformed
            self._last_model_params = np.exp(model.kernel_.theta.copy())

        return SciKitGPSurrogate.ModelInstance(model), fitting_info


    class ModelInstance(Surrogate.ModelInstance):
        def __init__(self, model):
            self.model = model

        def predict(self, X, return_std_dev=False):
            res = self.model.predict(X, return_std=return_std_dev)
            if return_std_dev:
                # both mus and sigmas should have shape (X_height,)
                return res[0].flatten(), res[1]
            else:
                return res.flatten()

        def get_hyper_params(self):
            # theta is log transformed
            return np.exp(self.model.kernel_.theta.copy())

        def get_hyper_param_names(self):
            k = self.model.kernel_
            return [h.name for h in k.hyperparameters]

        def get_log_likelihood(self):
            return self.model.log_marginal_likelihood()

