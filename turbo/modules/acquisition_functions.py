#!/usr/bin/env python3
from math import isinf
from scipy.stats import norm # Gaussian/normal distribution

# local modules
from turbo.utils import *

#TODO: look at gpyopt for guidance doing acquisition function gradients
#TODO: by looking at gpyopt, it looks like acquisition functions have to change calculations when doing MCMC, I think this is best detected by whether surrogate.predict returns a single answer or several.
#TODO: like the surrogates, make just EI, UCB ect be the factories and have 'Instances'


class AcquisitionFunction:
    """ A function which is used to determine the best place to sample next

    A single AcquisitionFunction instance is specialised to the current
    iteration and a new instance will be generated next iteration.
    """
    def __init__(self, model, desired_extremum):
        """
        Args:
            model: the surrogate model fitted to the past trials.
            desired_extremum: `'max' =>` higher cost is better, `'min' =>` lower
                cost is better.
        """
        self.model = model
        assert desired_extremum in ('min', 'max')
        self.desired_extremum = desired_extremum

    def get_name(self):
        """ get a human readable abbreviated name for the acquisition function """
        raise NotImplementedError()

    def __call__(self, X):
        """ query the acquisition function at the given points

        Args:
            X: the array of points to evaluate at. `shape=(num_points, num_attribs)`
        """
        raise NotImplementedError()

    class Factory:
        """ Passed to the optimiser and used to create AcquisitionFunction
        instances for each iteration of the optimisation

        The factory is instantiated by the user so that the acquisition
        functions can be configured. The optimiser then uses the factory to
        generate acquisition functions specialised to the current iteration.
        """
        def get_type(self):
            """ get the type of acquisition function that this factory constructs

            In the literature there are several families of acquisition
            function, each requiring different data from the optimiser. The
            following are the currently implemented possibilities:

            - 'improvement': improvement based acquisition function (how much
              does the point improve on the incumbent/current best)
            - 'optimism': based on optimism, where there is uncertainty, assume
              a positive outcome.

            Returns:
                the acquisition function type as a string
            """
            raise NotImplementedError()

        def __call__(self, trial_num, model, desired_extremum, *args):
            """ generate an acquisition function for the current iteration

            Args:
                trial_num: the trial/iteration number. Used as the 'time'
                    variable for acquisition functions which vary over time.
                model: the trained surrogate model to use for querying
                desired_extremum: `'max' =>` higher cost is better, `'min' =>` lower
                    cost is better.
                args: acquisition functions may require more data depending on their type
            """
            raise NotImplementedError()

        def plot_parameter(self, start_trial, end_trial):
            """A utility function to help determine the shape of the acquisition
            function parameter over time to help refine it.

            Args:
                start_trial (int): the trial number to start plotting at
                end_trial (int): the trial number to start plotting at
            """
            raise NotImplementedError()


class UCB(AcquisitionFunction):
    r"""Confidence Bounds Acquisition Function

    .. math::
        \begin{align*}
        UCB(\mathbf{x})&=\mu(\mathbf{x})+\beta\sigma(\mathbf{x})\\
        -LCB(\mathbf{x})&=-(\mu(\mathbf{x})-\beta\sigma(\mathbf{x}))
        \end{align*}

    Note:
        since the acquisition function is always maximised, when minimising the
        objective function the *negative* lower confidence bound is used.

    Note:
        Although technically UCB refers only to the maximisation version of the
        acquisition function, the name UCB is more common in the literature and
        so used over something more generic like 'confidence_bound' for
        simplicity.
    """
    def __init__(self, model, desired_extremum, beta):
        """
        Args:
            model: the surrogate model fitted to the past trials.
            desired_extremum: `'max' =>` higher cost is better, `'min' =>` lower
                cost is better.
            beta: parameter which controls the trade-off between exploration and
                exploitation. Larger values favour exploration more. (geometrically,
                the uncertainty is scaled more so is more likely to look better than
                known good locations).
                'often beta=2 is used' (Bijl et al., 2016).

                :math:`\beta=0` `=>` 'Expected Value' (EV) acquisition function

                .. math::
                    EV(\mathbf x)=\mu(\mathbf x)

                (pure exploitation, not very useful).

                :math:`\beta=\infty` `=>` pure exploration, taking only the variance
                into account (also not very useful)

        """
        super().__init__(model, desired_extremum)
        self.beta = beta
        self.scale_factor = 1 if desired_extremum == 'max' else -1

    def get_name(self):
        return 'UCB' if self.desired_extremum == 'max' else '-LCB'

    def __call__(self, X):
        """
        Args:
            X: the array of points to evaluate at. `shape=(num_points, num_attribs)`
        """
        mus, sigmas = self.model.predict(X, return_std_dev=True)
        if isinf(self.beta):
            return sigmas # pure exploration
        else:
            # in this form it is clearer that the value is the negative LCB when minimising
            # sf * (mus + sf * beta * sigmas)
            return self.scale_factor * mus + self.beta * sigmas

    class Factory(AcquisitionFunction.Factory):
        def __init__(self, beta):
            """
            Args:
                beta: either a constant float or a function which takes the
                    trial number and returns a float
            """
            self.beta = beta

        def get_type(self):
            return 'optimism'

        def __call__(self, trial_num, model, desired_extremum):
            # beta can be a function or a constant
            beta = self.beta(trial_num) if callable(self.beta) else self.beta
            acq_info = {'beta': beta}
            return UCB(model, desired_extremum, beta), acq_info

        def plot_parameter(self, start_trial, end_trial):
            beta = self.beta if callable(self.beta) else lambda trial_num: self.beta
            from turbo.plotting.overview import _plot_acquisition_parameter
            _plot_acquisition_parameter(beta, start_trial, end_trial)


class PI(AcquisitionFunction):
    r"""Probability of Improvement Acquisition Function

    When maximising:

    .. math::
        PI(\mathbf x)\quad=\quad\mathrm P\Big(f(\mathbf x)\ge f(\mathbf x^+)+\xi\;|\;\mathcal D_{1:t}\Big) \quad=\quad\Phi\left(\frac{\mu(\mathbf x)-f(\mathbf x^+)-\xi}{\sigma(\mathbf x)}\right)

    In general:

    .. math::
        PI(\mathbf x)\quad=\quad\Phi\left(\frac{(-1)^{MAX}\Big(f(\mathbf x^+)-\mu(\mathbf x)+\xi\Big)}{\sigma(\mathbf x)}\right)


    Where:

    - :math:`MAX=1` when maximising and 0 when minimising
    - when maximising :math:`\mathbf x^+=\arg\max_{\mathbf x\in\mathbf x_{1:t}}\mathbb f(\mathbf x)` (ie the best trial so far) where :math`\mathbb f` is the true objective function.

    Note:
        This acquisition function is similar to EI in that they are both
        improvement based acquisition functions, however EI typically gives
        better results because EI takes the amount of improvement into account,
        whereas PI treats all amounts of improvement the same.
    """

    def __init__(self, model, desired_extremum, incumbent_cost, xi):
        """
        Args:
            model: the surrogate model fitted to the past trials.
            desired_extremum: `'max' =>` higher cost is better, `'min' =>` lower
                cost is better.
            incumbent_cost: the current best cost/objective function value for a finished trial
            xi: a parameter >0 (sometimes called 'jitter') for exploration/exploitation trade-off. Larger =>
                more exploration.
        """
        super().__init__(model, desired_extremum)
        self.incumbent_cost = incumbent_cost
        self.xi = xi
        self.scale_factor = 1 if desired_extremum == 'max' else -1

    def get_name(self):
        return 'PI'

    def __call__(self, X):
        """
        Args:
            X: the array of points to evaluate at. `shape=(num_points, num_attribs)`
        """
        mus, sigmas = self.model.predict(X, return_std_dev=True)

        # using masks is slightly faster than ignoring the division by zero errors
        # then fixing later. It also seems like a 'cleaner' approach.
        mask = sigmas != 0 # sigma_x = 0  =>  PI(x) = 0
        sigmas = sigmas[mask]

        # maximisation:
        #     mu(x) - (f(x+) + xi)     =  mu(x) - f(x+) - xi
        # minimisation:
        #     -(mu(x) - (f(x+) - xi))  =  f(x+) - mu(x) - xi
        diff = self.scale_factor * (mus[mask] - self.incumbent_cost) - self.xi
        # Zs is only defined where sigmas != 0 so may have shorter length
        Zs = diff / sigmas

        PIs = np.zeros_like(mus)
        PIs[mask] = norm.cdf(Zs)
        return PIs

    class Factory(AcquisitionFunction.Factory):
        def __init__(self, xi):
            """
            Args:
                xi: either a constant float or a function which takes the trial
                    number and returns a float
            """
            self.xi = xi

        def get_type(self):
            return 'improvement'

        def __call__(self, trial_num, model, desired_extremum, incumbent_cost):
            # xi can be a function or a constant
            xi = self.xi(trial_num) if callable(self.xi) else self.xi
            acq_info = {'xi': xi}
            return PI(model, desired_extremum, incumbent_cost, xi), acq_info

        def plot_parameter(self, start_trial, end_trial):
            xi = self.xi if callable(self.xi) else lambda trial_num: self.xi
            from turbo.plotting.overview import _plot_acquisition_parameter
            _plot_acquisition_parameter(xi, start_trial, end_trial)


class EI(AcquisitionFunction):
    r"""Expected Improvement Acquisition function

    Expected Improvement is related to probability of improvement in that they
    both compare points with the incumbent (best) trial, making them both
    improvement based acquisition functions. However EI is vastly superior to PI
    (which unlike EI is prone to over exploitation and not enough exploration)
    because EI takes the value/amount of improvement over the incumbent into
    account in addition to the probability of that improvement, whereas PI
    treats all amounts of improvement the same.

    Note:
        the formulas shown here are for the case of maximisation.

    .. math::
        EI(\mathbf x)=\mathbb E\left[max(0,\; f(\mathbf x)-f(\mathbf x^+))\right]

    Where :math:`f` is the *surrogate* objective function and :math:`\mathbf x^+` is the
    parameter values for the best known (incumbent) trial.

    If :math:`f` is a Gaussian Process then EI can be calculated analytically:

    .. math::
        EI(\mathbf x)=\begin{cases}
        \left(\mu(\mathbf x)-f(\mathbf x^+)\right)\mathbf\Phi(Z) \;+\; \sigma(\mathbf x)\phi(Z)  &  \text{if } \sigma(\mathbf x)>0\\
        0 & \text{if } \sigma(\mathbf x) = 0
        \end{cases}

    .. math::
        Z=\frac{\mu(\mathbf x)-f(\mathbf x^+)}{\sigma(\mathbf x)}

    Where

    - :math:`\phi(\cdot)=` standard multivariate normal distribution PDF (ie :math:`\boldsymbol\mu=\mathbf 0,\;\Sigma=I`)
    - :math:`\Phi(\cdot)=` standard multivariate normal distribution CDF

    a parameter :math:`\xi` (sometimes called 'jitter') can be introduced to
    control the exploitation-exploration trade-off (:math:`\xi=0.01` works well
    in 'almost all cases' (Lizotte, 2008))

    .. math::
        EI(\mathbf x)=\begin{cases}
        \left(\mu(\mathbf x)-f(\mathbf x^+)-\xi\right)\mathbf\Phi(Z) \;+\; \sigma(\mathbf x)\phi(Z)  &  \text{if } \sigma(\mathbf x)>0\\
        0 & \text{if } \sigma(\mathbf x) = 0
        \end{cases}

    .. math::
        Z=\frac{\mu(\mathbf x)-f(\mathbf x^+)-\xi}{\sigma(\mathbf x)}

    """
    def __init__(self, model, desired_extremum, incumbent_cost, xi):
        """
        Args:
            model: the surrogate model fitted to the past trials.
            desired_extremum: `'max' =>` higher cost is better, `'min' =>` lower
                cost is better.
            incumbent_cost: the current best cost/objective function value for a finished trial
            xi: a parameter >0 (sometimes called 'jitter') for exploration/exploitation trade-off. Larger =>
                more exploration.
        """
        super().__init__(model, desired_extremum)
        self.incumbent_cost = incumbent_cost
        self.xi = xi
        self.scale_factor = 1 if desired_extremum == 'max' else -1

    def get_name(self):
        return 'EI'

    def __call__(self, X):
        """
        Args:
            X: the array of points to evaluate at. `shape=(num_points, num_attribs)`
        """
        mus, sigmas = self.model.predict(X, return_std_dev=True)

        # using masks is slightly faster than ignoring the division by zero errors
        # then fixing later. It also seems like a 'cleaner' approach.
        mask = sigmas != 0 # sigma_x = 0  =>  PI(x) = 0
        sigmas = sigmas[mask]

        # maximisation:
        #     mu(x) - (f(x+) + xi)     =  mu(x) - f(x+) - xi
        # minimisation:
        #     -(mu(x) - (f(x+) - xi))  =  f(x+) - mu(x) - xi
        diff = self.scale_factor * (mus[mask] - self.incumbent_cost) - self.xi
        # Zs is only defined where sigmas != 0 so may have shorter length
        Zs = diff / sigmas

        EIs = np.zeros_like(mus)
        EIs[mask] = (diff * norm.cdf(Zs)) + (sigmas * norm.pdf(Zs))
        return EIs

    class Factory(AcquisitionFunction.Factory):
        def __init__(self, xi):
            """
            Args:
                xi: either a constant float or a function which takes the trial
                    number and returns a float
            """
            self.xi = xi

        def get_type(self):
            return 'improvement'

        def __call__(self, trial_num, model, desired_extremum, incumbent_cost):
            # xi can be a function or a constant
            xi = self.xi(trial_num) if callable(self.xi) else self.xi
            acq_info = {'xi': xi}
            return EI(model, desired_extremum, incumbent_cost, xi), acq_info

        def plot_parameter(self, start_trial, end_trial):
            xi = self.xi if callable(self.xi) else lambda trial_num: self.xi
            from turbo.plotting.overview import _plot_acquisition_parameter
            _plot_acquisition_parameter(xi, start_trial, end_trial)

