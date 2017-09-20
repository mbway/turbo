#!/usr/bin/env python3
'''
The Bayesian Optimisation specific code
'''
# python 2 compatibility
from __future__ import (absolute_import, division, print_function, unicode_literals)
from .py2 import *

import numpy as np

import sklearn.gaussian_process as gp

# local modules
from .core import Optimiser, Sample
from .utils import *
from .bayesian_utils import *
from .plot import BayesianOptimisationOptimiserPlotting
from . import acquisition_functions as ac_funs


class AcquisitionStrategy(object):
    '''
    A configuration object for specifying the high-level behaviour of the
    Bayesian optimisation algorithm that the optimiser should perform.

    Note: all string arguments are case insensitive and will be converted to a
    normal form for internal use
    '''

    def __init__(self, pre_phase_steps, acquisition_function,
                 GP_hyperparameter_strategy='optimise', parallel_strategy='KB',
                 random_proportion=0):
        '''
        pre_phase_steps: int >1
            number of steps to choose random configurations to evaluate before
            starting Bayesian optimisation (required so that the surrogate GP
            has something to fit to).

        acquisition_function:  (fun : str|callable[, args : dict])
            fun: a function from acquisition_functions or the name of an
                acquisition function can be passed to specify the acquisition
                function to use (PI/PoI, EI, TS, CB/UCB/LCB). See the relevant
                function definition for a description of the function.
            args: a dict specifying any parameters relevant to the chosen
                acquisition function. May be omitted completely or passed without
                specifying any arguments ie `{}`.
                PI/PoI, EI: 'xi' : >0, larger => more exploration
                CB/UCB/LCB:
                    'kappa' : >0, larger => more exploration
                    OR
                    'beta_t' : function of t (number of steps) #TODO:
                TS: no args

        GP_hyperparameter_strategy : (strategy : str[, args : dict])
            strategy: the name of the GP hyperparameter strategy to use:
                'optimise'/'optimize'/'PE'/'Point Estimate': optimise the GP
                    hyperparameters by maximizing the log-marginal-likelihood
                    according to the specified gp_params. Then calculate the
                    acquisition function from the prediction of this single GP.
                'marginalise'/'MCMC'/'integrated acquisition': take N samples of
                    hyperparameters for the GP, fit the GP with each sample and
                    calculate the acquisition function, then average the
                    acquisitions to give a Monte-carlo estimate for the
                    'integrated acquisition function' which marginalises the GP
                    hyperparameters.
            args: a dict specifying any parameters relevant to the chosen strategy
                PE: 'train_interval': default 1, how frequently (in number of
                    steps) to re-optimise the hyperparameters. If the
                    hyperparameters aren't re-trained for a step then the previous
                    parameters are used (the GP is still fitted to the up-to-date
                    dataset). 1 => re-optimise every step. You may want to
                    increase the interval between re-training for performance
                    reasons.
                'marginalise':
                    'N': the number of hyperparameter samples to take

        parallel_strategy:  (strategy : str[, args : dict])
            strategy: the name of the parallel strategy to use:
                'None'/None/'Serial': disallow parallel evaluations. Each
                    evaluation must finish before the next sample is suggested.
                    args: None
                'CL'/'Constant Liar': use a constant value as an estimate for
                    the cost of unfinished evaluations
                'KB'/'Kriging Believer': use the surrogate predicted mean as an
                    estimate for the true cost value of any unfinished
                    evaluations. Re-fit the GP with the dataset augmented with
                    these 'hypothesised samples' then calculate the acquisition
                    function as normal.
                    args: None
                'MC'/'Monte Carlo'/'Monte-Carlo': sample N possible cost values
                    from the surrogate posterior for each unfinished evaluation.
                    Then fit the GP for every combination of sampled cost values
                    and calculate the acquisition function value for each of
                    these simulations, then average the result. Because each
                    simulation is independent, they may themselves be performed
                    in parallel across multiple threads.
                    args:
                        'N': the number of simulations to use.
                        'num_threads': #TODO
                        'sim_train_its': the number of optimiser restarts to
                            allow the GPs for each simulation. 0 => re-use the
                            hyperparameters of the main suggestion GP.
                'asyTS': asynchronous Thompson sampling. Ignore unfinished
                    evaluations and perform Thompson sampling on the GP
                    posterior as normal. Only possible when using the TS
                    acquisition function.
                    args: None
            args: a dict specifying any parameters relevant to the chosen strategy.

        #TODO:
        input_warping:
            hyper-cube
            learn warping
            log-linear

        #TODO:
        random_proportion: 0-1, proportion of the time to choose a sample at random
            rather than with an acquisition function. This is a form of
            'harmless Bayesian optimisation' because when taking samples at
            random occasionally, it can do no worse than totally random search
            (asymptotically). 0 to disable this feature.
        '''
        assert pre_phase_steps > 1, 'not enough pre-phase steps'
        self.pre_phase_steps = pre_phase_steps
        self.random_proportion = random_proportion

        self.acq_fun, self.acq_fun_args = self._load_tuple_arg(
            acquisition_function,
            allowed_values=[
                (ac_funs.probability_of_improvement,
                    ('pi', 'poi', 'probability of improvement'),
                    {'xi' : 0.01},
                    lambda args,keys: keys <= {'xi'}),
                (ac_funs.expected_improvement,
                    ('ei', 'expected improvement'),
                    {'xi' : 0.01},
                    lambda args,keys: keys <= {'xi'}),
                (ac_funs.thompson_sample,
                    ('ts', 'thompson sample'),
                    {},
                    lambda args,keys: not keys),
                (ac_funs.confidence_bound,
                    ('cb', 'ucb', 'lcb', 'confidence bound',
                     'upper confidence bound', 'lower confidence bound'),
                    {'kappa' : 2.0},
                    lambda args,keys: keys <= {'kappa'}),
            ],
            # do not allow custom acquisition functions, instead custom
            # functions should be implemented alongside the existing ones and
            # handled properly.
            no_match_check=lambda val, args, keys: False
        )

        self.gp_strategy, self.gp_strategy_args = self._load_tuple_arg(
            GP_hyperparameter_strategy,
            allowed_values=[
                ('optimise', ('optimize', 'pe', 'point estimate'),
                    {'train_interval' : 1},
                    lambda args,keys: keys <= {'train_interval'}),
                ('marginalise', ('mcmc', 'integrated acquisition'),
                    {},#TODO
                    lambda args,keys: keys <= {'N'})
            ],
            # do not allow values which do not match the allowed
            no_match_check=lambda val, args, keys: False
        )

        self.parallel_strategy, self.parallel_strategy_args = self._load_tuple_arg(
            parallel_strategy,
            allowed_values=[
                ('none', (None, 'serial'), {},
                    lambda args,keys: not keys),
                ('cl', ('constant liar'), {'L' : 'mean'},
                    lambda args,keys: keys <= {'L'}),
                ('kb', ('kriging believer'), {},
                    lambda args,keys: not keys),
                ('mc', ('monte carlo', 'monte-carlo'),
                    {'N' : 20, 'sim_train_its' : 0},
                    lambda args,keys: keys <= {'N', 'num_threads', 'sim_train_its'}),
                ('asyts', (), {},
                    lambda args,keys: not keys)
            ],
            # do not allow values which do not match the allowed
            no_match_check=lambda val, args, keys: False
        )

        # check strategy sanity
        if self.parallel_strategy == 'asyts':
            assert self.acq_fun == ac_funs.thompson_sample, \
                'asyTS only compatible with Thompson Sampling acquisition function'

    def __str__(self):
        s = ('AcquisitionStrategy(\n\tpre_phase_steps={},\n\tacquisition_function=({}, {}),\n\t'
             'GP_hyperparameter_strategy=("{}", {}),\n\tparallel_strategy=("{}", {}),\n\trandom_proportion={})\n)')
        return s.format(self.pre_phase_steps, self.acq_fun, self.acq_fun_args,
                 self.gp_strategy, self.gp_strategy_args,
                 self.parallel_strategy, self.parallel_strategy_args,
                 self.random_proportion)

    def _load_tuple_arg(self, t, allowed_values, no_match_check):
        '''
        load an argument which is allowed to be either a single value, or a
        tuple with 1 or 2 values where the second element is an optional
        dictionary of arguments relevant to the choice of first parameter.

        When loaded, the value is converted to a 'canonical form'
        t: the tuple argument to load
        allowed_values: list of the form
                [(canonical_value1, ('other', values), default_args, check_args),
                 (canonical_value2, ('other', values), default_args, check_args), ...]
            if the first part of t matches any canonical or other values, then
            it will be set to the canonical value for that choice.
            the provided arguments will be laid on top of the provided default
            args then check_args will be called. check_args is a predicate
            which takes the args dict and a set of the dict keys, and checks
            whether the arguments are valid for the given choice
        no_match_check: a predicate called if the value does not match any of
        the allowed values. Takes the value, arguments dict and set of argument
        keys. If the predicate matches then the value is allowed, otherwise an
        error is raised

        returns: val, dict
        '''
        if isinstance(t, tuple):
            if len(t) == 1:
                val, args = t[0], {}
            elif len(t) == 2:
                val, args = t
            else:
                raise ValueError('invalid value: {}'.format(t))
        else:
            val, args = t, {}

        if isinstance(val, str): # perform string comparisons in lower case
            val = val.lower()

        found = False
        for canonical, other, default_args, check_args in allowed_values:
            if val == canonical or val in other:
                val = canonical
                args = dict(default_args, **args) # overlay over defaults
                arg_keys = set(args.keys())
                assert check_args(args, arg_keys), 'arguments invalid: {} {}'.format(val, args)
                return val, args

        if no_match_check(val, args, keys):
            return val, args
        else:
            raise ValueError('invalid value: {}'.format(t))


    #TODO: move to optimiser
    def get_name(self, maximise_cost):
        if self.parallel_strategy == 'mc':
            prefix = 'Mean-'
        else:
            prefix = ''

        if self.acq_fun == ac_funs.probability_of_improvement:
            ac_name = 'PI'
        elif self.acq_fun == ac_funs.expected_improvement:
            ac_name = 'EI'
        elif self.acq_fun == ac_funs.thompson_sample:
            ac_name = 'TS'
        elif self.acq_fun == ac_funs.confidence_bound:
            ac_name = 'UCB' if maximise_cost else 'LCB'
        else:
            ac_name = 'Custom'

        return prefix + ac_name


class BayesianOptimisationOptimiser(BayesianOptimisationOptimiserPlotting, Optimiser):
    '''
    Bayesian Optimisation Strategy:
    1. Sample some random initial points
    2. Using a surrogate function to stand in for the cost function (which is
       unknown) define an acquisition function which estimates how good sampling at
       each point would be. Then maximise this value to obtain the next point to
       sample.
    3. With each obtained sample, the accuracy of the surrogate function with
       respect to the true cost function increases, which in turn allows better
       choices of next samples to test.

    Technicalities:
    - If the next suggested sample is too close to an existing sample, then this
      would not give much new information, so instead the next point is chosen
      randomly. This results in a better picture of the cost function which in
      turn may make new points look more desirable, allowing the algorithm to
      progress and not get stuck in local optima
    - Bayesian optimisation is inherently a serial process. To parallelise the
      algorithm, the results for ongoing jobs are estimated by trusting the
      expected value of the surrogate function. After the job has finished the
      correct value is used, but in the meantime it allows more jobs to be
      started based on the estimated results.
    - Logarithmically spaced parameters (where the order of magnitude is more
      important than the absolute value) must be sampled log-uniformly rather than
      uniformly.
    - Discrete parameters are not compatible with Bayesian optimisation since
      the chosen surrogate function is a Gaussian Process, which fits real-valued
      data, and the acquisition function also relies on real-number calculations.
      Modifications to the algorithm may allow for discrete valued parameters however.
    '''

    def __init__(self, ranges, maximise_cost, acquisition_strategy,
                 gp_params=None, maximisation_args=None, close_tolerance=1e-5):
        '''
        ranges: dict
            A dictionary of parameter names to arrays which span the range of
            values to search for the optimum within.
        maximise_cost: bool
            True => larger cost is better. False => smaller cost is better
        acquisition_strategy: AcquisitionStrategy. None => default
            describes the high-level behaviour for the Bayesian optimisation
            algorithm, such as the acquisition function to use and how to handle
            parallelism.
        gp_params: dict/None
            parameters for the Gaussian Process surrogate function, None will
            choose some sensible defaults. (See "sklearn gaussian process
            regressor")
        maximisation_args: dict/None
            parameters for determining the procedure for maximising the
            acquisition function. None to use default values, or a dictionary
            with integer values for:
                'num_random': number of uniform-random samples to take when
                    searching for the maximum of the acquisition function. (0 to
                    only use the gradient based optimiser)
                'num_grad_restarts': the number of restarts to run the
                    gradient-based optimiser for. (0 to only use random samples)
                'take_best_random': specify the number of best points to take
                    from the random optimisation phase and use as start points
                    for the gradient-based optimisation phase. This counts
                    towards the num_grad_restarts total.
                'num_threads': TODO
                larger values for each of these parameters means that the
                optimisation is more likely to find the global maximum of the
                acquisition function, however the optimisation becomes more
                costly (however this will probably be insignificant in
                comparison to the time to evaluate a configuration).
                0 may be passed for one of the two parameters to ignore that
                step of the optimisation.
        close_tolerance: in some situations Bayesian optimisation may get stuck
            on local optima and will continue to sample points roughly in the
            same location. When this happens the GP can break (as input values
            must be unique within some tolerance). It is also a waste of
            resources to sample lots of times in a very small neighbourhood.
            Instead, when the next sample is to be 'close' to any of the points
            sampled before (ie squared Euclidean distance <= close_tolerance),
            sample a random point instead.
        '''
        ranges = {param:np.array(range_) for param, range_ in ranges.items()} # numpy arrays are required
        super().__init__(ranges, maximise_cost)

        if acquisition_strategy is None:
            self.strategy = AcquisitionStrategy(
                pre_phase_steps = 4,
                acquisition_function = ac_funs.confidence_bound,
                GP_hyperparameter_strategy = 'optimise',
                parallel_strategy = 'KB',
                random_proportion = 0
            )
        else:
            # the AcquisitionStrategy object processes the user input into a
            # canonical format and validates it.
            self.strategy = acquisition_strategy

        if gp_params is None:
            self.gp_params = dict(
                alpha = 1e-10, # larger => more noise. Default = 1e-10
                # nu=1.5 assumes the target function is once-differentiable
                kernel = 1.0 * gp.kernels.Matern(nu=1.5) + gp.kernels.WhiteKernel(),
                #kernel = 1.0 * gp.kernels.RBF(),
                n_restarts_optimizer = 10,
                # make the mean 0 (theoretically a bad thing, see docs, but can help)
                # with the constant offset in the kernel this shouldn't be required
                #normalize_y = True,
                copy_X_train = True # whether to make a copy of the training data (in-case it is modified)
            )
        else:
            self.gp_params = gp_params


        self.close_tolerance = close_tolerance

        if maximisation_args is None:
            self.maximisation_args = dotdict({
                'num_random' : 10000,
                'num_grad_restarts' : 10,
                'take_best_random' : 3
            })
        else:
            assert set(maximisation_args.keys()) == {'num_random', 'num_grad_restarts', 'take_best_random'}
            # convert each parameter to an integer
            self.maximisation_args = dotdict({k:int(v) for k, v in maximisation_args.items()})
            # at least one of the methods has to be used (non-zero)
            assert self.maximisation_args.num_random > 0 or self.maximisation_args.num_grad_restarts > 0
            assert 0 <= self.maximisation_args.take_best_random <= self.maximisation_args.num_grad_restarts

        # ranges
        if not ranges:
            raise ValueError('empty ranges not allowed with Bayesian optimisation')

        self.ranges = {}
        for param, range_ in ranges.items():
            type_ = range_type(range_)

            if type_ == RangeType.Arbitrary:
                raise ValueError('arbitrary ranges: {} are not allowed with Bayesian optimisation'.format(param))

            bounds = (min(range_), max(range_))

            self.ranges[param] = Range(values=range_, type_=type_, bounds=bounds)

        for param, range_ in self.ranges.items():
            low, high = range_.bounds
            self._log('param "{}": detected type: {}, bounds: [{}, {}]'.format(
                param, range_.type_, low, high))

        # deals with converting to and from configuration dictionaries and points
        # in space which the GP is trained in.
        self.point_space = PointSpace(self.params, self.ranges, self.close_tolerance)

        # [(job_ID, Step)]: steps which have been started but have not finished
        self.outstanding_steps = []
        # a GP trained on only the concrete samples, only used for some
        # acquisition strategies.
        self.concrete_gp = None

        self.step_log = {}
        self.step_log_keep = 100 # max number of steps to keep


    def configuration_space_size(self):
        return inf # continuous

    def _ready_for_next_configuration(self):
        if self.strategy.parallel_strategy == 'none': # force serial
            more_pre_phase = self.num_started_jobs < self.strategy.pre_phase_steps
            pre_phase_done = self.num_finished_jobs >= self.strategy.pre_phase_steps
            waiting_for_job = any(job_ID not in self.finished_job_ids for
                                  job_ID, step in self.outstanding_steps)

            return more_pre_phase or (pre_phase_done and not waiting_for_job)
        else: # allow parallel
            # wait for all of the pre-phase samples to be taken before starting
            # the Bayesian optimisation steps. Otherwise always ready.
            waiting_for_pre_phase = (self.num_started_jobs >= self.strategy.pre_phase_steps and
                                     self.num_finished_jobs < self.strategy.pre_phase_steps)
            return not waiting_for_pre_phase

    def trim_step_log(self, keep=-1):
        '''
        remove old steps from the step_log to save space. Keep the N steps with
        largest job IDs
        keep: the number of steps to keep (-1 => keep = self.step_log_keep)
        '''
        if keep == -1:
            keep = self.step_log_keep

        steps = self.step_log.keys()
        if len(steps) > keep:
            removing = sorted(steps, reverse=True)[keep:]
            for step in removing:
                del self.step_log[step]

    def _maximise_acquisition(self, acq_fun):
        log_warning = lambda warn: self._log('warning in argmax acquisition: {}'.format(warn))
        with WarningCatcher(log_warning):
            x, y = maximise_function(
                f=acq_fun,
                bounds=self.point_space.point_bounds,
                gen_random=self.point_space.random_points,
                **self.maximisation_args
            )
        if x is None:
            self._log('all attempts at acquisition function maximisation failed')
            return None, -inf
        return x, y

    def _get_data_set(self):
        '''
        calculate the data set for an optimisation step
        sx, sy: points (in point_space) corresponding to concrete samples
            (samples which have finished evaluating).
        hx: points (in point_space) corresponding to hypothesised samples
            (samples which have _not_ finished evaluating). hy is not calculated
            here because it is dependent on the particular acquisition strategy.
        '''
        if len(self.samples) > 0:
            sx = np.vstack(self.point_space.config_to_point(s.config) for s in self.samples)
            sy = np.array([[s.cost] for s in self.samples])
        else:
            sx = np.empty(shape=(0, self.point_space.num_attribs))
            sy = np.empty(shape=(0, 1))

        assert sx.shape == (len(self.samples), self.point_space.num_attribs)
        assert sy.shape == (len(self.samples), 1)

        # remove steps whose jobs have since finished
        self.outstanding_steps = [(job_ID, step) for job_ID, step in self.outstanding_steps
                                        if job_ID not in self.finished_job_ids]

        if len(self.outstanding_steps) > 0:
            hx = np.vstack(step.chosen_x() for job_ID, step in self.outstanding_steps)
        else:
            hx = np.empty(shape=(0, self.point_space.num_attribs))
        assert hx.shape == (len(self.outstanding_steps), self.point_space.num_attribs)

        return sx, sy, hx

    def _train_gp(self, xs, ys, theta=None):
        '''
        return a GP trained on the given data
        xs, ys: the data set to train the GP on
        theta: optional, specify the parameters for the GP to skip the
            optimisation section of the training
        '''
        if theta is not None:
            gp_model = restore_GP(theta, self.gp_params, xs, ys)
        else:
            gp_model = gp.GaussianProcessRegressor(**self.gp_params)
            log_warning = lambda warn: self._log('warning training GP: {}'.format(warn))
            with WarningCatcher(log_warning):
                gp_model.fit(xs, ys)
        return gp_model

    def _get_acq_fun(self, gp_model, ys):
        '''
        return a partially applied acquisition function which takes a single
        argument: an ndarray of points to evaluate the acquisition function at.

        gp_model: the trained GP to use for evaluating
        ys: cost values for the concrete and hypothesised samples
        '''
        acq = self.strategy.acq_fun
        acq_params = self.strategy.acq_fun_args

        if acq in [ac_funs.probability_of_improvement,
                   ac_funs.expected_improvement]:
            chooser = np.max if self.maximise_cost else np.min
            best_cost = chooser(ys)
            return lambda xs: acq(xs, gp_model, self.maximise_cost, best_cost, **acq_params)

        elif acq == ac_funs.thompson_sample:
            raise NotImplementedError()

        elif acq == ac_funs.confidence_bound:
            if 'beta_t' in acq_params:
                raise NotImplementedError()
                # t = job_ID?
                acq_params = {'kappa' : 123}
            return lambda xs: acq(xs, gp_model, self.maximise_cost, **acq_params)

        else:
            raise ValueError()

    def _max_acq_suggestion(self, data_set):
        sx, sy, hx = data_set

        # choose values for hy depending on the strategy
        if len(hx) == 0:
            # no parallel strategy required
            hy = np.empty(shape=(0, 1))
            xs, ys = sx, sy

        elif self.strategy.parallel_strategy == 'kb': # Kriging Believer
            log_warning = lambda w: self._log('warning predicting hy: {}'.format(warn))
            # use the data saved in the outstanding steps to get the
            # hypothesised cost for the points in hx.
            # there will be at least one step since len(hx) != 0
            hy = []
            for job_ID, step in self.outstanding_steps:
                s0 = step.suggestions[0]
                assert isinstance(s0, Step.MaxAcqSuggestion)
                with WarningCatcher(log_warning):
                    gp = restore_GP(s0.gp, self.gp_params,
                                    np.vstack((step.sx, step.hx)), np.vstack((step.sy, s0.hy)))
                    hy.append(gp.predict(step.chosen_x()))
            hy = np.vstack(hy)

            xs, ys = np.vstack((sx, hx)), np.vstack((sy, hy))

        else:
            raise NotImplementedError()

        gp_model = self._train_gp(xs, ys)
        acq = self._get_acq_fun(gp_model, ys) # partially apply

        # suggest_x may be None if the maximisation failed
        suggest_x, suggest_ac = self._maximise_acquisition(acq)

        return Step.MaxAcqSuggestion(
            hy=hy, gp=store_GP(gp_model), x=suggest_x, ac=suggest_ac)


    def _max_mc_acq_suggestion(self, data_set):
        sx, sy, hx = data_set
        assert self.strategy.parallel_strategy == 'mc'
        assert len(hx) != 0 # should only be called when there are outstanding steps

        # if there have been no new concrete samples, just re-use the last GP
        if self.concrete_gp is None or np.any(self.concrete_gp.y_train_ != sy):
            self.concrete_gp = self._train_gp(sx, sy)
        concrete_params = store_GP(self.concrete_gp)

        log_warning = lambda w: self._log('GP warning: {}'.format(warn))
        with WarningCatcher(log_warning):
            # the number of samples to draw for each outstanding step
            N = self.strategy.parallel_strategy_args['N']
            hy = []
            for job_ID, step in self.outstanding_steps:
                # by default random_state uses a fixed seed! Setting to None
                # uses the current numpy random state.
                chosen_hys = self.concrete_gp.sample_y(step.chosen_x(), N, random_state=None)
                assert chosen_hys.shape == (1, 1, N)
                hy.append(chosen_hys.reshape(1, N))

        # hys is a list where each element is a set of sampled values
        # corresponding to hx (ie multiple versions of hy)
        hys = [make2D(np.array(row)) for row in np.vstack(hy).T.tolist()]
        xs = np.vstack((sx, hx))

        # run each simulation
        train_its = self.strategy.parallel_strategy_args['sim_train_its']
        if train_its != 0:
            sim_gp_params = self.gp_params.copy()
            sim_gp_params['n_restarts_optimizer'] = train_its
        # [(hy, sim_gp)]
        sims = []
        # list of acquisition functions for each simulation
        sim_acq_funs = []
        with WarningCatcher(log_warning):
            for hy in hys:
                ys = np.vstack((sy, hy))

                # fit the GP to the points of this simulation
                if train_its == 0: # re-use the main GP
                    sim_gp = restore_GP(concrete_params, self.gp_params, xs, ys)
                    sims.append((hy, None)) # None => re-using the concrete GP params
                else:
                    sim_gp = gp.GaussianProcessRegressor(**sim_gp_params)
                    sim_gp.fit(xs, ys)
                    sims.append((hy, store_GP(sim_gp)))

                acq = self._get_acq_fun(sim_gp, ys) # partially apply
                sim_acq_funs.append(acq)

            # average acquisition across every simulation
            acq_fun = lambda xs: 1.0/len(sim_acq_funs) * np.sum(acq(xs) for acq in sim_acq_funs)

            # suggest_x may be None if the maximisation failed
            suggest_x, suggest_ac = self._maximise_acquisition(acq_fun)

        return Step.MC_MaxAcqSuggestion(
            gp=concrete_params,
            simulations=sims,
            x=suggest_x,
            ac=suggest_ac
        )

    def _random_suggestion(self, data_set):
        sx, sy, hx = data_set
        xs = np.vstack((sx, hx))
        chosen_x = self.point_space.unique_random_point(different_from=xs, num_attempts=1000)
        return Step.RandomSuggestion(x=chosen_x)

    def _next_configuration(self, job_ID):
        data_set = self._get_data_set()

        # pre-phase
        if self.num_started_jobs < self.strategy.pre_phase_steps:
            # still in the pre-phase where samples are chosen at random
            # make sure that each configuration is sufficiently different from all previous samples
            random_suggestion = self._random_suggestion(data_set)

            self._log('in pre-phase: choosing random configuration {}/{}'.format(
                job_ID, self.strategy.pre_phase_steps))
            #TODO: add to step log
            return self.point_space.point_to_config(random_suggestion.x)


        # Bayesian optimisation
        sx, sy, hx = data_set
        xs = np.vstack((sx, hx))
        suggestions = []

        if self.strategy.parallel_strategy == 'none':
            # not allowing parallel evaluations is handled in
            # _ready_for_next_configuration, this method should not be called
            # with evaluations still in progress when there is no parallel
            # strategy.
            assert len(hx) == 0 # no hypothesised samples

        # suggestion by maximising an acquisition function
        if self.strategy.parallel_strategy == 'mc' and len(hx) != 0:
            suggestion = self._max_mc_acq_suggestion(data_set)
        else:
            suggestion = self._max_acq_suggestion(data_set)
        suggestions.append(suggestion)

        # maximising acquisition function failed or was too close to an existing configuration
        if suggestion.x is None:
            self._log('choosing random sample because maximising acquisition function failed')
            random_suggestion = self._random_suggestion(data_set)
            suggestions.append(random_suggestion)

        elif close_to_any(suggestion.x, xs, self.close_tolerance):
            # cannot have multiple samples too close to one another (may break GP and not useful anyway)
            self._log('argmax(acquisition function) too close to an existing sample: choosing randomly instead')
            #TODO: what about if the evaluator changes the config and happens to create a duplicate point?
            random_suggestion = self._random_suggestion(data_set)
            suggestions.append(random_suggestion)

        step = Step(
            job_ID=job_ID,
            sx=sx, sy=sy,
            hx=hx,
            suggestions=suggestions
        )

        self.outstanding_steps.append((job_ID, step))
        assert job_ID not in self.step_log
        self.step_log[job_ID] = step
        self.trim_step_log()

        return self.point_space.point_to_config(step.chosen_x())



    def _consistent_and_quiescent(self):
        # super class checks general properties
        sup = super()._consistent_and_quiescent()
        # either there are no outstanding steps, or they are for jobs which
        # have already finished and just haven't been removed yet.
        no_outstanding_steps = (
            len(self.outstanding_steps) == 0 or
            all(job_id in self.finished_job_ids for job_id, x in
                 self.outstanding_steps)
        )
        # 1. make sure that the arrays are exactly 2 dimensional
        # 2. make sure that there are equal numbers of rows in the samples (and
        # hypothesised samples) xs and ys (ie every x must have a y).
        # 3. make sure that there is only 1 output attribute for each y
        eq_rows = lambda a, b: a.shape[0] == b.shape[0]
        is_2d = lambda x: len(x.shape) == 2
        try:
            step_log_valid = True
            for _, step in self.step_log.items():
                step_log_valid &= all((
                    is_2d(step.sx), is_2d(step.sy), is_2d(step.hx),
                    eq_rows(step.sx, step.sy), step.sy.shape[1] == 1
                ))
                for s in step.suggestions:
                    step_log_valid &= is_2d(s.x)
                    if isinstance(s, Step.RandomSuggestion):
                        pass
                    elif isinstance(s, Step.MaxAcqSuggestion):
                        step_log_valid &= all((
                            eq_rows(step.hx, s.hy), np.isscalar(s.ac), s.hy.shape[1] == 1
                        ))
                    elif isinstance(s, Step.MC_MaxAcqSuggestion):
                        step_log_valid &= all(
                            all((eq_rows(step.hx, sim[0]), sim[0].shape[1] == 1))
                            for sim in s.simulations
                        )
                        step_log_valid &= np.isscalar(s.ac)
                    else:
                        step_log_valid = False
                if not step_log_valid:
                    break
        except IndexError:
            # one of the shapes has <2 elements
            step_log_valid = False

        return sup and no_outstanding_steps and step_log_valid

    def _save_dict(self):
        save = super()._save_dict()

        # outstanding_steps is not needed to be saved since the optimiser should
        # be quiescent.

        save['step_log'] = []
        # convert the step log (dict) to an array of tuples sorted by job_ID
        for n, step in sorted(self.step_log.items(), key=lambda x: x[0]):
            step = step.to_dict()
            step['suggestions'] = [[s.__class__.__name__, s.to_dict()] for s in step['suggestions']]
            save['step_log'].append((n, step))
        return save

    def _load_dict(self, save):
        super()._load_dict(save)

        for n, step in save['step_log']:
            # convert lists back to numpy arrays
            for key in ['sx', 'sy', 'hx']:
                step[key] = np.array(step[key])
            # ensure the shapes are correct
            # sx,sy will never be empty since there will always be pre-samples
            # hx,hy may be empty
            if len(step['hx'])== 0:
                step['hx'] = np.empty(shape=(0, self.point_space.num_attribs))

            convert_hy = lambda hy: np.empty(shape=(0, 1)) if len(hy) == 0 else np.array(hy)
            suggestions = []
            for type_, s in step['suggestions']:
                s['x'] = np.array(s['x'])
                if type_ == 'RandomSuggestion':
                    suggestions.append(Step.RandomSuggestion(**s))
                elif type_ == 'MaxAcqSuggestion':
                    s['hy'] = convert_hy(s['hy'])
                    suggestions.append(Step.MaxAcqSuggestion(**s))
                elif type_ == 'MC_MaxAcqSuggestion':
                    #TODO: handle ac_random_state
                    s['simulations'] = [(convert_hy(hy), sim_gp) for hy, sim_gp in s['simulations']]
                    suggestions.append(Step.MC_MaxAcqSuggestion(**s))
                else:
                    raise ValueError(type_)
            step['suggestions'] = suggestions


        # convert list of tuples to dictionary
        self.step_log = {n : Step(**s) for n, s in save['step_log']}

        # reset any progress attributes
        self.outstanding_steps = []
        self.concrete_gp = None

