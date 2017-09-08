#!/usr/bin/env python3
'''
The Bayesian Optimisation specific code
'''
# python 2 compatibility
from __future__ import (absolute_import, division, print_function, unicode_literals)
from .py2 import *

import numpy as np

import sklearn.gaussian_process as gp
import scipy.optimize
from scipy.stats import norm # Gaussian/normal distribution

# local modules
from .core import Optimiser, Sample
from .utils import *
from .plot import BayesianOptimisationOptimiserPlotting


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
    def __init__(self, ranges, maximise_cost=False,
                 acquisition_function='UCB', acquisition_function_params=None,
                 gp_params=None, pre_samples=4, ac_max_params=None,
                 close_tolerance=1e-5, allow_parallel=True):
        '''
        acquisition_function: the function to determine where to sample next
            either a function or a string with the name of the function (eg 'EI')
        acquisition_function_params: a dictionary of parameter names and values
            to be passed to the acquisition function. (see specific acquisition
            function for details on what parameters it takes)
        gp_params: parameter dictionary for the Gaussian Process surrogate
            function, None will choose some sensible defaults. (See "sklearn
            gaussian process regressor")
        pre_samples: the number of jobs (not samples, despite the name) to run
            before starting Bayesian optimisation
        ac_max_params: parameters for maximising the acquisition function. None
            to use default values, or a dictionary  with integer values for:
                'num_random': number of random samples to take when maximising
                    the acquisition function
                'num_restarts': number of restarts to use for the gradient-based
                    maximisation of the acquisition function
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
        allow_parallel: whether to hypothesise about the results of ongoing jobs
            in order to start another job in parallel. (useful when running a
            server with multiple client evaluators).
        '''
        ranges = {param:np.array(range_) for param, range_ in ranges.items()} # numpy arrays are required
        super().__init__(ranges, maximise_cost)

        self.acquisition_function_params = ({} if acquisition_function_params is None
                                            else acquisition_function_params)
        ac_param_keys = set(self.acquisition_function_params.keys())

        if acquisition_function == 'PI':
            self.acquisition_function_name = 'PI'
            self.acquisition_function = self.probability_of_improvement
            # <= is subset. Not all params must be provided, but those given must be valid
            assert ac_param_keys <= set(['xi']), 'invalid acquisition function parameters'

        elif acquisition_function == 'EI':
            self.acquisition_function_name = 'EI'
            self.acquisition_function = self.expected_improvement
            # <= is subset. Not all params must be provided, but those given must be valid
            assert ac_param_keys <= set(['xi']), 'invalid acquisition function parameters'

        elif acquisition_function == 'UCB':
            self.acquisition_function_name = 'UCB'
            self.acquisition_function = self.upper_confidence_bound
            # <= is subset. Not all params must be provided, but those given must be valid
            assert ac_param_keys <= set(['kappa']), 'invalid acquisition function parameters'

        elif callable(acquisition_function):
            self.acquisition_function_name = 'custom acquisition function'
            self.acquisition_function = acquisition_function
        else:
            raise ValueError('invalid acquisition_function')

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

        if ac_max_params is None:
            self.ac_max_params = dotdict({'num_random' : 10000, 'num_restarts' : 10})
        else:
            assert set(ac_max_params.keys()) <= set(['num_random', 'num_restarts'])
            # convert each parameter to an integer
            self.ac_max_params = dotdict({k:int(v) for k, v in ac_max_params.items()})
            # at least one of the methods has to be used (non-zero)
            assert self.ac_max_params.num_random > 0 or self.ac_max_params.num_restarts > 0

        assert pre_samples > 1, 'not enough pre-samples'
        self.pre_samples = pre_samples
        self.close_tolerance = close_tolerance

        self.range_types = {param : range_type(range_) for param, range_ in self.ranges.items()}

        if RangeType.Arbitrary in self.range_types.values():
            bad_ranges = [param for param, type_ in self.range_types.items()
                          if type_ == RangeType.Arbitrary]
            raise ValueError('arbitrary ranges: {} are not allowed with Bayesian optimisation'.format(bad_ranges))
        elif not self.ranges:
            raise ValueError('empty ranges not allowed with Bayesian optimisation')

        # record the bounds only for the linear and logarithmic ranges
        self.range_bounds = {param: (min(self.ranges[param]), max(self.ranges[param])) for param in self.params}

        for param in self.params:
            low, high = self.range_bounds[param]
            self._log('param "{}": detected type: {}, bounds: [{}, {}]'.format(
                param, self.range_types[param], low, high))

        # Only provide bounds for the parameters that are included in
        # self.config_to_point. Provide the log(lower), log(upper) bounds for
        # logarithmically spaced ranges.
        # IMPORTANT: use range_bounds when dealing with configs and point_bounds
        # when dealing with points
        self.point_bounds = []
        for param in self.params: # self.params is ordered
            type_ = self.range_types[param]
            low, high = self.range_bounds[param]
            if type_ == RangeType.Linear:
                self.point_bounds.append((low, high))
            elif type_ == RangeType.Logarithmic:
                self.point_bounds.append((np.log(low), np.log(high)))


        self.allow_parallel = allow_parallel
        # not ready for a next configuration until the job with id ==
        # self.wait_until has been processed. Not used when allow_parallel.
        self.wait_for_job = None
        # estimated samples for ongoing jobs. list of (job_ID, Sample). Not
        # used when (not allow_parallel)
        self.hypothesised_samples = []

        self.step_log = {}
        self.step_log_keep = 100 # max number of steps to keep

    class Step(DataHolder):
        '''
        Data regarding a single Bayesian optimisation step.

        gp: trained Gaussian process. Depending on the parallel strategy the
            hypothesised samples may or may not be taken into account.
        sx,sy: numpy arrays corresponding to points of samples taken thus far.
            x = configuration as a point
            y = true evaluated cost
        hx,hy: numpy arrays corresponding to _hypothesised_ points of ongoing.
            jobs while the step was being calculated.
            x = configuration as a point
            y = mean estimate of the surrogate function trained on all previous samples
        best_sample: the best sample so far
        next_x: the next configuration to test#TODO: type?

        #TODO: store other data like parallel strategy so it can be changed during optimisation and still plot correctly
        #TODO: add
        type: pre_phase|bayes|random_fallback, the type of step describes how
            the next configuration was chosen

        next_ac: used when type=bayes|random_fallback
            the value of the acquisition function evaluated at next_x
            #TODO: rename to argmax_ac
        argmax_ac: used when type=random_fallback
            the next config to test (as a point) as determined by maximising the
            acquisition function, may be different to next_x
        #TODO: add
        gp_posterior_sample: used when type=bayes|random_fallback and using TS acquisition function
            a sample from the GP posterior which is used as an acquisition
            function when using Thomson sampling.
        '''
        __slots__ = ('gp', 'sx', 'sy', 'hx', 'hy', 'best_sample', 'next_x',
                     'chosen_at_random', 'next_ac', 'argmax_acquisition')
        __defaults__ = {
            'next_ac' : 0.0,
            'argmax_ac' : 0.0
            #'gp_posterior_sample' : None
        }

    def configuration_space_size(self):
        return inf # continuous

    def _ready_for_next_configuration(self):
        if self.allow_parallel:
            # wait for all of the pre-phase samples to be taken before starting
            # the Bayesian optimisation steps. Otherwise always ready.
            waiting_for_pre_phase = (self.num_started_jobs >= self.pre_samples and
                                     self.num_finished_jobs < self.pre_samples)
            return not waiting_for_pre_phase
        else:
            in_pre_phase = self.num_started_jobs < self.pre_samples
            # all jobs from the pre-phase are finished, need the first Bayesian
            # optimisation sample
            pre_phase_finished = (self.wait_for_job is None and
                                  self.num_finished_jobs >= self.pre_samples)
            # finished waiting for the last Bayesian optimisation job to finish
            bayes_job_finished = self.wait_for_job in self.finished_job_ids

            return in_pre_phase or pre_phase_finished or bayes_job_finished

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

    def _maximise_acquisition(self, gp_model, best_cost):
        '''
        maximise the acquisition function to obtain the next configuration to test
        gp_model: a Gaussian process trained on the samples taken so far
        best_cost: the cost function value of the best known sample so far
        returns: config (as a point/numpy array), acquisition value (not negative)
            or None,0 if all attempts to optimise the acquisition_function fail

        Important note: This is a _local_ optimisation. This means that _any_
        local optimum is acceptable. There may be some slight variations in the
        function even if it looks flat when plotted, and the 'maximum' sometimes
        rests there, and not at the obvious global maximum. This is fine.
        '''
        # Maximise the acquisition function by random sampling
        if self.ac_max_params.num_random > 0:
            random_points = self._random_config_points(self.ac_max_params.num_random)
            random_ac = self.acquisition_function(random_points, gp_model,
                            self.maximise_cost, best_cost, **self.acquisition_function_params)
            best_random_i = random_ac.argmax()

            # keep track of the current best
            best_next_x = make2D_row(random_points[best_random_i])
            best_neg_ac = -random_ac[best_random_i] # negative acquisition function value for best_next_x
        else:
            best_next_x = None
            best_neg_ac = inf

        # Maximise the acquisition function by minimising the negative acquisition function

        # scipy has no maximise function, so instead minimise the negation of the acquisition function
        # reshape(1,-1) => 1 sample (row) with N attributes (cols). Needed because x is passed as shape (N,)
        # unpacking the params dict is harmless if the dict is empty
        neg_acquisition_function = lambda x: -self.acquisition_function(
            make2D_row(x), gp_model, self.maximise_cost, best_cost,
            **self.acquisition_function_params)

        if self.ac_max_params.num_restarts > 0:
            # it doesn't matter if these points are close to any existing samples
            starting_points = self._random_config_points(self.ac_max_params.num_restarts)
            if self.ac_max_params.num_random > 0:
                # see if gradient-based optimisation can improve upon the best
                # randomly chosen sample.
                starting_points = np.vstack([best_next_x, starting_points])

            for j in range(starting_points.shape[0]):
                starting_point = make2D_row(starting_points[j])

                # note: nested WarningCatchers work as expected
                log_warning = lambda warn: self._log('warning when maximising the acquisition function: {}'.format(warn))
                with WarningCatcher(log_warning):
                    # result is an OptimizeResult object
                    result = scipy.optimize.minimize(
                        fun=neg_acquisition_function,
                        x0=starting_point,
                        bounds=self.point_bounds,
                        method='L-BFGS-B', # Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Bounded
                        options=dict(maxiter=15000) # maxiter=15000 is default
                    )
                if not result.success:
                    self._log('restart {}/{} of negative acquisition minimisation failed'.format(
                        j, starting_points.shape[0]))
                    continue

                # result.fun == negative acquisition function evaluated at result.x
                if result.fun < best_neg_ac:
                    best_next_x = result.x # shape=(num_attribs,)
                    best_neg_ac = result.fun # shape=(1,1)

        # acquisition function optimisation finished:
        # best_next_x = argmax(acquisition_function)

        if best_next_x is None:
            self._log('all attempts at acquisition function maximisation failed')
            return None, 0
        else:
            # reshape to make shape=(1,num_attribs) and negate best_neg_ac to make
            # it the positive acquisition function value
            best_next_x = make2D_row(best_next_x)
            # ensure that the chosen value lies within the bounds (which may not
            # be the case due to floating point error)
            best_next_x = np.clip(best_next_x, [lo for lo, hi in self.point_bounds], [hi for lo, hi in self.point_bounds])
            return best_next_x, -np.asscalar(best_neg_ac)

    def _bayes_step(self, job_ID):
        '''
        generate the next configuration to test using Bayesian optimisation
        '''
        #TODO: extract to _get_data()
        # samples converted to points which can be used in calculations
        # shape=(num_samples, num_attribs)
        sx = np.vstack([self.config_to_point(s.config) for s in self.samples])
        # shape=(num_samples, 1)
        sy = np.array([[s.cost] for s in self.samples])

        # if running parallel jobs: add hypothesised samples to the data set to
        # fit the surrogate cost function to. If running serial: mark this job
        # as having to finish before proceeding
        if self.allow_parallel:
            # remove samples whose jobs have since finished
            self.hypothesised_samples = [(ID, s) for ID, s in self.hypothesised_samples
                                         if ID not in self.finished_job_ids]

            if len(self.hypothesised_samples) > 0:
                hx = np.vstack([self.config_to_point(s.config)
                                for ID, s in self.hypothesised_samples])
                hy = np.array([[s.cost] for ID, s in self.hypothesised_samples])
            else:
                hx = np.empty(shape=(0, sx.shape[1]))
                hy = np.empty(shape=(0, 1))
        else:
            self.wait_for_job = job_ID # do not add a new job until this job has been processed

            hx = np.empty(shape=(0, sx.shape[1]))
            hy = np.empty(shape=(0, 1))

        # combination of the true samples (sx, sy) and the hypothesised samples
        # (hx, hy) if there are any
        xs = np.vstack([sx, hx])
        ys = np.vstack([sy, hy])

        # setting up a new model each time shouldn't be too wasteful and it
        # has the benefit of being easily reproducible (eg for plotting)
        # because the model is definitely 'clean' each time. In my tests,
        # there was no perceptible difference in timing.
        gp_model = gp.GaussianProcessRegressor(**self.gp_params)
        # the optimiser may fail for various reasons, one being that 'the
        # function is dominated by noise'. In one example I looked at the GP was
        # still sensible even with the warning, so ignoring it should be fine.
        # Worst case scenario is that a few bad samples are taken before the GP
        # sorts itself out again.
        # warnings may be triggered for fitting or predicting
        log_warning = lambda warn: self._log('warning with the gp: {}'.format(warn))
        #log_warning = lambda warn: print(warn)
        with WarningCatcher(log_warning):
            # NOTE: fitting only optimises _certain_ kernel parameters with given
            # bounds, see gp_model.kernel_.theta for the optimised kernel
            # parameters.
            # NOTE: RBF(...) has NO parameters to optimise, however 1.0 * RBF(...) does!
            gp_model.fit(xs, ys)

            # gp_model.kernel_ is a copy of gp_model.kernel with the parameters optimised
            #self._log('GP params={}'.format(gp_model.kernel_.theta))

            # best known configuration and the corresponding cost of that configuration
            best_sample = self.best_sample()

            next_x, next_ac = self._maximise_acquisition(gp_model, best_sample.cost)

            # next_x as chosen by the acquisition function maximisation (for the step log)
            argmax_acquisition = next_x


        # maximising the acquisition function failed
        if next_x is None:
            self._log('choosing random sample because maximising acquisition function failed')
            next_x = self._unique_random_config(different_from=xs, num_attempts=1000)
            next_ac = 0
            chosen_at_random = True
        # acquisition function successfully maximised, but the resulting configuration would break the GP.
        # having two samples too close together will 'break' the GP
        elif close_to_any(next_x, xs, self.close_tolerance):
            self._log('argmax(acquisition function) too close to an existing sample: choosing randomly instead')
            #TODO: what about if the evaluator changes the config and happens to create a duplicate point?
            next_x = self._unique_random_config(different_from=xs, num_attempts=1000)
            next_ac = 0
            chosen_at_random = True
        else:
            next_x = self.point_to_config(next_x)
            chosen_at_random = False

        if self.allow_parallel:
            # use the GP to estimate the cost of the configuration, later jobs
            # can use this guess to determine where to sample next
            with WarningCatcher(log_warning):
                est_cost = gp_model.predict(self.config_to_point(next_x))
            est_cost = np.asscalar(est_cost) # ndarray of shape=(1,1) is returned from predict()
            self.hypothesised_samples.append((job_ID, Sample(next_x, est_cost, job_ID=job_ID)))

        assert job_ID not in self.step_log.keys()
        self.step_log[job_ID] = BayesianOptimisationOptimiser.Step(
            gp = gp_model,
            sx = sx, sy = sy,
            hx = hx, hy = hy,
            best_sample = best_sample,
            next_x = next_x,
            next_ac = next_ac, # chosen_at_random => next_ac=0
            chosen_at_random = chosen_at_random,
            argmax_acquisition = argmax_acquisition # different to next_x when chosen_at_random
        )
        self.trim_step_log()

        return next_x


    def _next_configuration(self, job_ID):
        if self.num_started_jobs < self.pre_samples:
            # still in the pre-phase where samples are chosen at random
            # make sure that each configuration is sufficiently different from all previous samples
            if len(self.samples) == 0:
                config = self._random_config()
            else:
                sx = np.vstack([self.config_to_point(s.config) for s in self.samples])
                config = self._unique_random_config(different_from=sx, num_attempts=1000)

            if config is None: # could not find a unique configuration
                return None # finished
            else:
                self._log('in pre-phase: choosing random configuration {}/{}'.format(job_ID, self.pre_samples))
                return config
        else:
            # Bayesian optimisation
            return self._bayes_step(job_ID)


    @staticmethod
    def probability_of_improvement(xs, gp_model, maximise_cost, best_cost, xi=0.01):
        r'''
        This acquisition function is similar to EI
        $$PI(\mathbf x)\quad=\quad\mathrm P\Big(f(\mathbf x)\ge f(\mathbf x^+)+\xi\Big)\quad=\quad\Phi\left(\frac{\mu(\mathbf x)-f(\mathbf x^+)-\xi}{\sigma(\mathbf x)}\right)$$
        '''
        mus, sigmas = gp_model.predict(xs, return_std=True)
        sigmas = make2D(sigmas)

        sf = 1 if maximise_cost else -1   # scaling factor
        diff = sf * (mus - best_cost - xi)  # mu(x) - f(x+) - xi

        with np.errstate(divide='ignore'):
            Zs = diff / sigmas # produces Zs[i]=inf for all i where sigmas[i]=0.0
        Zs[sigmas == 0.0] = 0.0 # replace the infs with 0s

        return norm.cdf(Zs)

    @staticmethod
    def expected_improvement(xs, gp_model, maximise_cost, best_cost, xi=0.01):
        r''' expected improvement acquisition function
        xs: array of points to evaluate the GP at. shape=(num_points, num_attribs)
        gp_model: the GP fitted to the past configurations
        maximise_cost: True => higher cost is better, False => lower cost is better
        best_cost: the (actual) cost of the best known configuration (either
            smallest or largest depending on maximise_cost)
        xi: a parameter >0 for exploration/exploitation trade-off. Larger =>
            more exploration. The default value of 0.01 is recommended.#TODO: citation needed

        Theory:

        $$EI(\mathbf x)=\mathbb E\left[max(0,\; f(\mathbf x)-f(\mathbf x^+))\right]$$
        where $f$ is the surrogate objective function and $\mathbf x^+=$ the best known configuration so far.

        Maximising the expected improvement will result in the next configuration to test ($\mathbf x$) being better ($f(\mathbf x)$ larger) than $\mathbf x^+$ (but note that $f$ is only an approximation to the real objective function).
        $$\mathbf x_{\mathrm{next}}=\arg\max_{\mathbf x}EI(\mathbf x)$$

        If $f$ is a Gaussian Process (which it is in this case) then $EI$ can be calculated analytically:

        $$EI(\mathbf x)=\begin{cases}
        \left(\mu(\mathbf x)-f(\mathbf x^+)\right)\mathbf\Phi(Z) \;+\; \sigma(\mathbf x)\phi(Z)  &  \text{if } \sigma(\mathbf x)>0\\
        0 & \text{if } \sigma(\mathbf x) = 0
        \end{cases}$$

        $$Z=\frac{\mu(\mathbf x)-f(\mathbf x^+)}{\sigma(\mathbf x)}$$

        Where
        - $\phi(\cdot)=$ standard multivariate normal distribution PDF (ie $\boldsymbol\mu=\mathbf 0$, $\Sigma=I$)
        - $\Phi(\cdot)=$ standard multivariate normal distribution CDF

        a parameter $\xi$ can be introduced to control the exploitation-exploration trade-off ($\xi=0.01$ works well in almost all cases (Lizotte, 2008))

        $$EI(\mathbf x)=\begin{cases}
        \left(\mu(\mathbf x)-f(\mathbf x^+)-\xi\right)\mathbf\Phi(Z) \;+\; \sigma(\mathbf x)\phi(Z)  &  \text{if } \sigma(\mathbf x)>0\\
        0 & \text{if } \sigma(\mathbf x) = 0
        \end{cases}$$

        $$Z=\begin{cases}
        \frac{\mu(\mathbf x)-f(\mathbf x^+)-\xi}{\sigma(\mathbf x)}  &  \text{if }\sigma(\mathbf x)>0\\
        0 & \text{if }\sigma(\mathbf x) = 0
        \end{cases}$$
        '''
        mus, sigmas = gp_model.predict(xs, return_std=True)
        sigmas = make2D(sigmas)

        sf = 1 if maximise_cost else -1   # scaling factor
        diff = sf * (mus - best_cost - xi)  # mu(x) - f(x+) - xi

        with np.errstate(divide='ignore'):
            Zs = diff / sigmas # produces Zs[i]=inf for all i where sigmas[i]=0.0

        EIs = diff * norm.cdf(Zs)  +  sigmas * norm.pdf(Zs)
        EIs[sigmas == 0.0] = 0.0 # replace the infs with 0s

        return EIs

    #TODO: should rename? make CB/UCB/LCB all refer to this function
    # has been called lower confidence bound when minimising: https://scikit-optimize.github.io/notebooks/bayesian-optimization.html
    @staticmethod
    def upper_confidence_bound(xs, gp_model, maximise_cost, best_cost, kappa=2.0):
        r'''
        upper confidence bound when maximising, lower confidence bound when minimising
        $$\begin{align*}
        UCB(\mathbf x)&=\mu(\mathbf x)+\kappa\sigma(\mathbf x)\\
        LCB(\mathbf x)&=\mu(\mathbf x)-\kappa\sigma(\mathbf x)
        \end{align*}$$

        xs: array of points to evaluate the GP at. shape=(num_points, num_attribs)
        gp_model: the GP fitted to the past configurations
        maximise_cost: True => higher cost is better, False => lower cost is better
        best_cost: not used in this acquisition function
        kappa: parameter which controls the trade-off between exploration and
            exploitation. Larger values favour exploration more. (geometrically,
            the uncertainty is scaled more so is more likely to look better than
            known good locations). 'often kappa=2 is used' (Bijl et al., 2016)
            kappa=0 => 'Expected Value' (EV) acquisition function
            $$EV(\mathbf x)=\mu(\mathbf x)$$ (pure exploitation, not very useful)
        '''
        mus, sigmas = gp_model.predict(xs, return_std=True)
        sigmas = make2D(sigmas)
        sf = 1 if maximise_cost else -1   # scaling factor
        return sf * (mus + sf * kappa * sigmas)

    def _unique_random_config(self, different_from, num_attempts=1000):
        ''' generate a random config which is different from any configurations tested in the past
        different_from: numpy array of points shape=(num_points, num_attribs)
            which the resulting configuration must not be identical to (within a
            very small tolerance). This is because identical configurations
            would break the GP (it models a function, so each 'x' corresponds to
            exactly one 'y')
        num_attempts: number of re-tries before giving up
        returns: a random configuration, or None if num_attempts is exceeded
        '''
        for _ in range(num_attempts):
            config = self._random_config()
            if not close_to_any(self.config_to_point(config), different_from, tol=self.close_tolerance):
                return config
        self._log('could not find a random configuration sufficiently different from previous samples, parameter space must be (almost) fully explored.')
        return None

    def _random_config(self):
        '''
        generate a random configuration, sampling each parameter appropriately
        based on its type (uniformly or log-uniformly)
        '''
        config = {}
        for param in self.params:
            type_ = self.range_types[param]

            if type_ == RangeType.Linear:
                low, high = self.range_bounds[param]
                config[param] = np.random.uniform(low, high)

            elif type_ == RangeType.Logarithmic:
                low, high = self.range_bounds[param]
                # not exponent, but a value in the original space
                config[param] = log_uniform(low, high)

            elif type_ == RangeType.Constant:
                config[param] = self.ranges[param][0] # only 1 choice

            else:
                raise ValueError('invalid range type: {}'.format(type_))

        return dotdict(config)

    def _random_config_points(self, num_points):
        '''
        generate an array of vertically stacked configuration points equivalent to self.config_to_point(self._random_config())
        num_points: number of points to generate (height of output)
        returns: numpy array with shape=(num_points,num_attribs)
        '''
        cols = [] # generate points column/parameter-wise
        for param in self.params: # self.params is sorted
            type_ = self.range_types[param]
            low, high = self.range_bounds[param]

            if type_ == RangeType.Linear:
                cols.append(np.random.uniform(low, high, size=(num_points, 1)))
            elif type_ == RangeType.Logarithmic:
                # note: NOT log_uniform because that computes a value in the
                # original space but distributed logarithmically. We are looking
                # for just the exponent here, not the value.
                cols.append(np.random.uniform(np.log(low), np.log(high), size=(num_points, 1)))
        return np.hstack(cols)


    def config_to_point(self, config):
        '''
        convert a configuration (dictionary of param:val) to a point (numpy
        array) in the parameter space that the Gaussian process uses.

        As a point, constant parameters are ignored, and values from logarithmic
        ranges are the exponents of the values. ie a value of 'n' as a point
        corresponds to a value of e^n as a configuration.

        config: a dictionary of parameter names to values
        returns: numpy array with shape=(1,number of linear or logarithmic parameters)
        '''
        assert set(config.keys()) == set(self.ranges.keys())
        elements = []
        for param in self.params: # self.params is sorted
            type_ = self.range_types[param]
            if type_ == RangeType.Linear:
                elements.append(config[param])
            elif type_ == RangeType.Logarithmic:
                elements.append(np.log(config[param]))
        return np.array([elements])

    def _index_for_param(self, param):
        '''
        return the index of the given parameter name in a point created by
        config_to_point(). None if the parameter is not present.
        '''
        assert param in self.params
        i = 0
        for p in self.params: # self.params is sorted
            if param == p:
                return i
            if self.range_types[p] in [RangeType.Linear, RangeType.Logarithmic]:
                i += 1
        return None

    def point_to_config(self, point):
        '''
        convert a point (numpy array) used by the Gaussian process into a
        configuration (dictionary of param:val).

        As a point, constant parameters are ignored, and values from logarithmic
        ranges are the exponents of the values. ie a value of 'n' as a point
        corresponds to a value of e^n as a configuration.

        returns: a configuration dict with all parameters included
        '''
        assert len(point.shape) == 2, 'must be a 2D point'
        assert point.shape[0] == 1, 'only 1 point can be converted at a time'
        config = {}
        pi = 0 # current point index
        for param in self.params: # self.params is sorted
            type_ = self.range_types[param]

            if type_ == RangeType.Constant:
                config[param] = self.ranges[param][0] # only 1 choice
            else:
                if pi >= point.shape[1]:
                    raise ValueError('point has too few attributes')
                val = point[0, pi]
                pi += 1

                if type_ == RangeType.Linear:
                    config[param] = val
                elif type_ == RangeType.Logarithmic:
                    config[param] = np.exp(val)

        if pi != point.shape[1]:
            raise ValueError('point has too many attributes')

        return dotdict(config)

    def _consistent_and_quiescent(self):
        # super class checks general properties
        sup = super()._consistent_and_quiescent()
        # either not waiting for a job, or waiting for a job which has finished
        not_waiting = (
            self.wait_for_job is None or
            self.wait_for_job in self.finished_job_ids
        )
        # either there are no hypothesised samples, or they are for jobs which
        # have already finished and just haven't been removed yet.
        no_hypotheses = (
            len(self.hypothesised_samples) == 0 or
            all(job_id in self.finished_job_ids for job_id, sample in
                 self.hypothesised_samples)
        )
        # 1. make sure that the arrays are exactly 2 dimensional
        # 2. make sure that there are equal numbers of rows in the samples (and
        # hypothesised samples) xs and ys (ie every x must have a y).
        # 3. make sure that there is only 1 output attribute for each y
        eq_rows = lambda a, b: a.shape[0] == b.shape[0]
        is_2d = lambda x: len(x.shape) == 2
        try:
            step_log_valid = (
                all(all([
                    is_2d(step.sx), is_2d(step.sy),
                    is_2d(step.hx), is_2d(step.hy),
                    np.isscalar(step.next_ac), is_2d(step.argmax_acquisition),

                    eq_rows(step.sx, step.sy),
                    eq_rows(step.hx, step.hy),

                    step.hy.shape[1] == step.sy.shape[1] == 1,

                ]) for job_id, step in self.step_log.items())
            )
        except IndexError:
            # one of the shapes has <2 elements
            step_log_valid = False

        return sup and not_waiting and no_hypotheses and step_log_valid

    def _save_dict(self):
        save = super()._save_dict()

        # hypothesised samples and wait_for_job are not needed to be saved since
        # the optimiser should be quiescent

        # save GP models compressed and together since they contain a lot of
        # redundant information and are not human readable anyway.
        # for a test run with ~40 optimisation steps, naive storage (as part of
        # step log): 1MB, separate with compression: 200KB
        save['step_log'] = []
        gps = {}
        # convert the step log (dict) to an array of tuples sorted by job_ID
        for n, s in sorted(self.step_log.items(), key=lambda x: x[0]):
            step = {k:v for k, v in s if k != 'gp'} # convert Step to dict
            step['best_sample'] = step['best_sample'].to_encoded_tuple()
            save['step_log'].append((n, step))
            gps[n] = s.gp
        save['gps'] = JSON_encode_binary(gps)
        return save

    def _load_dict(self, save):
        super()._load_dict(save)

        gps = JSON_decode_binary(save['gps'])

        for n, s in save['step_log']:
            s['gp'] = gps[n]
            s['best_sample'] = Sample.from_encoded_tuple(s['best_sample'])

            # convert lists back to numpy arrays
            for key in ['sx', 'sy', 'hx', 'hy', 'argmax_acquisition']:
                s[key] = np.array(s[key])
            # ensure the shapes are correct
            # sx,sy will never be empty since there will always be pre-samples
            # hx,hy may be empty
            if s['hx'].size == 0:
                s['hx'] = np.empty(shape=(0, s['sx'].shape[1]))
                s['hy'] = np.empty(shape=(0, 1))

        # convert list of tuples to dictionary
        self.step_log = {n : BayesianOptimisationOptimiser.Step(**s) for n, s in save['step_log']}

        # reset any progress attributes
        self.wait_for_job = None
        self.hypothesised_samples = []

