#!/usr/bin/env python3
'''
A collection of acquisition functions for use with Bayesian optimisation
'''
# python 2 compatibility
from __future__ import (absolute_import, division, print_function, unicode_literals)
from .py2 import *

import numpy as np
from math import isinf

import sklearn.gaussian_process as gp
from scipy.stats import norm # Gaussian/normal distribution

# local modules
from .utils import *

#TODO: test that negating the function and minimising is the negative of the acquisition
#TODO: gradients for acquisition functions, incorporate in the maximisation process, get_acq_fun can return the value and the derivative, or None if not supported

def thompson_sample(xs, gp_model):
    raise NotImplementedError()

def old_PI(xs, gp_model, maximise_cost, best_cost, xi):
    mus, sigmas = gp_model.predict(xs, return_std=True)
    if len(sigmas) > 100:
        sigmas[40:60] = 0
    sigmas = make2D(sigmas)

    sf = 1 if maximise_cost else -1     # scaling factor
    # maximisation: mu(x) - f(x+) - xi
    # minimisation: f(x+) - mu(x) - xi
    diff = sf * (mus - best_cost) - xi

    with np.errstate(divide='ignore'):
        Zs = diff / sigmas # produces Zs[i]=inf for all i where sigmas[i]=0.0
    Zs[sigmas == 0.0] = 0.0 # replace the infs with 0s

    return norm.cdf(Zs)

def probability_of_improvement(xs, gp_model, maximise_cost, best_cost, xi):
    r'''
    This acquisition function is similar to EI

    best_cost
    xi: a parameter >0 for exploration/exploitation trade-off. Larger =>
        more exploration. The default value of 0.01 is recommended.#TODO: citation needed

    when maximising:
        $$PI(\mathbf x)\quad=\quad\mathrm P\Big(f(\mathbf x)\ge f(\mathbf x^+)+\xi\;|\;\mathcal D_{1:t}\Big)
    \quad=\quad\Phi\left(\frac{\mu(\mathbf x)-f(\mathbf x^+)-\xi}{\sigma(\mathbf x)}\right)$$

    In general:
    $$PI(\mathbf x)\quad=\quad\Phi\left(\frac{(-1)^{MAX}\Big(f(\mathbf x^+)-\mu(\mathbf x)+\xi\Big)}{\sigma(\mathbf x)}\right)$$
    where
    - $MAX=1$ when maximising and $0$ when minimising
    - $$\mathbf x^+=\arg\max_{\mathbf x\in\mathbf x_{1:t}}\mathbb E[f(\mathbf x)]$$ (ie the best concrete sample so far)

    general form based on: https://apsis.readthedocs.io/en/latest/tutorials/usage/optimization.html

    '''
    mus, sigmas = gp_model.predict(xs, return_std=True)
    if len(sigmas) > 100:
        sigmas[40:60] = 0
    # using masks is slightly faster than ignoring the division by zero errors
    # then fixing later. It also seems like a 'cleaner' approach.
    mask = sigmas != 0 # sigma_x = 0  =>  PI(x) = 0
    sigmas = make2D(sigmas[mask])

    sf = 1 if maximise_cost else -1     # scaling factor
    # maximisation: mu(x) - f(x+) - xi
    # minimisation: f(x+) - mu(x) - xi
    diff = sf * (mus[mask] - best_cost) - xi

    Zs = np.zeros_like(mus)
    Zs[mask] = diff / sigmas
    Zs = norm.cdf(Zs)

    assert np.all(Zs == old_PI(xs, gp_model, maximise_cost, best_cost, xi))
    return Zs

def old_EI(xs, gp_model, maximise_cost, best_cost, xi):
    mus, sigmas = gp_model.predict(xs, return_std=True)
    if len(sigmas) > 100:
        sigmas[40:60] = 0
    sigmas = make2D(sigmas)

    sf = 1 if maximise_cost else -1     # scaling factor
    # maximisation: mu(x) - f(x+) - xi
    # minimisation: f(x+) - mu(x) - xi
    diff = sf * (mus - best_cost) - xi

    with np.errstate(divide='ignore'):
        Zs = diff / sigmas # produces Zs[i]=inf for all i where sigmas[i]=0.0

    EIs = diff * norm.cdf(Zs)  +  sigmas * norm.pdf(Zs)
    EIs[sigmas == 0.0] = 0.0 # replace the infs with 0s
    return EIs

def expected_improvement(xs, gp_model, maximise_cost, best_cost, xi):
    r''' expected improvement acquisition function
    xs: array of points to evaluate the GP at. shape=(num_points, num_attribs)
    gp_model: the GP fitted to the past configurations
    maximise_cost: True => higher cost is better, False => lower cost is better
    best_cost: the true cost of the best known concrete sample (either smallest
        or largest depending on maximise_cost)
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
    if len(sigmas) > 100:
        sigmas[40:60] = 0

    # using masks is slightly faster than ignoring the division by zero errors
    # then fixing later. It also seems like a 'cleaner' approach.
    mask = sigmas != 0 # sigma_x = 0  =>  EI(x) = 0
    sigmas = make2D(sigmas[mask])

    sf = 1 if maximise_cost else -1     # scaling factor
    # maximisation: mu(x) - f(x+) - xi
    # minimisation: f(x+) - mu(x) - xi
    diff = sf * (mus[mask] - best_cost) - xi

    Zs = diff / sigmas
    EIs = np.zeros_like(mus)
    EIs[mask] = diff * norm.cdf(Zs)  +  sigmas * norm.pdf(Zs)

    assert np.all(EIs == old_EI(xs, gp_model, maximise_cost, best_cost, xi))
    return EIs

# has been called lower confidence bound when minimising: https://scikit-optimize.github.io/notebooks/bayesian-optimization.html
# upper-confidence bound is also used even when minimising because 'UCB is ingrained in the literature as a standard form' (https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf)
# whether LCB has a negative around the whole thing: https://github.com/automl/RoBO/blob/master/robo/acquisition_functions/lcb.py
# I think that technically LCB should not be negated, however in order to make it into an acquisition function (ie to be maximised) it should be negated
def confidence_bound(xs, gp_model, maximise_cost, kappa):
    r'''
    upper confidence bound when maximising, (negative) lower confidence bound when minimising
    $$\begin{align*}
    UCB(\mathbf x)&=\mu(\mathbf x)+\kappa\sigma(\mathbf x)\\
    LCB(\mathbf x)&=-(\mu(\mathbf x)-\kappa\sigma(\mathbf x))
    \end{align*}$$

    xs: array of points to evaluate the GP at. shape=(num_points, num_attribs)
    gp_model: the GP fitted to the past configurations
    maximise_cost: True => higher cost is better, False => lower cost is better
    kappa: parameter which controls the trade-off between exploration and
        exploitation. Larger values favour exploration more. (geometrically,
        the uncertainty is scaled more so is more likely to look better than
        known good locations). 'often kappa=2 is used' (Bijl et al., 2016)
        kappa=0 => 'Expected Value' (EV) acquisition function
        $$EV(\mathbf x)=\mu(\mathbf x)$$ (pure exploitation, not very useful)
        kappa=inf => pure exploration, taking only the variance into account
        (also not very useful)
    '''
    mus, sigmas = gp_model.predict(xs, return_std=True)
    sigmas = make2D(sigmas)
    if isinf(kappa):
        return sigmas # pure exploration
    else:
        sf = 1 if maximise_cost else -1   # scaling factor
        # in this form it is clearer that the value is the negative LCB when minimising
        # sf * (mus + sf * kappa * sigmas)
        return sf * mus + kappa * sigmas

