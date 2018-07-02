# turbo #
turbo is a modular Bayesian optimisation framework which focuses on gathering and storing the intermediate optimisation steps to give insight into the decision making process.
turbo is capable of producing a wide variety of plots and supports many variations of the basic Bayesian optimisation algorithm.

# Algorithm Features #
- Acquisition Functions
    - Probability of Improvement (PI)
    - Expected Improvement (EI)
    - Upper/Lower Confidence Bound (UCB/LCB)
- Pre-Phase 'naive' selectors
    - Random
    - Latin Hypercube Sampling (LHS)
    - Manual
- Surrogate Models
    - Scikit-Learn Gaussian Process
    - GPy Gaussian Process
- Latent Space
    - Fixed warping (e.g. log-transformed or linear map to `[0,1]` etc)
- Fallback
    - Scheduled random samples ("Harmless" Bayesian Optimisation)
    - de-duplication
- Misc
    - able to use the same storage and plotting functionality with random search or any of the available 'naive' samplers

# Dependencies #
all dependencies can be installed with pip, see `requirements.txt`


# Links
- Black Box Optimisation Benchmarking Procedure: <http://coco.lri.fr/COCOdoc/bbo_experiment.html>
- python implementations of many benchmarking functions <https://github.com/andyfaff/ampgo/blob/master/%20ampgo%20--username%20andrea.gavana%40gmail.com/go_benchmark.py>

## Some other Bayesian optimisation libraries
- https://github.com/fmfn/BayesianOptimization
- https://github.com/thuijskens/bayesian-optimization
- https://github.com/befelix/SafeOpt
- https://github.com/scikit-optimize/scikit-optimize
- https://hyperopt.github.io/hyperopt/
- https://github.com/automl/RoBO
- https://github.com/resibots/limbo

