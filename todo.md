1. Bayesian optimisation 1D
2. Bayesian optimisation 2D (synthetic)
-- make sure results are good
3. Bayesian optimisation 2D (simulated data)
4. dropout MDN
    - https://github.com/RobotsLab/AML/blob/mppi/aml_dl/src/aml_dl/mdn/model/tf_mdn_model.py
    - branch from 'mppi' first
5. maybe combine MDN with dropout?
6. maybe combine MDN with dropout and ensemble?
7. other stuff


hyperparameter cost function: just penalise bad uncertainty and hope that the model fits the dta reasonably


### idea:
1. introduce high variance synthetic data where there are no real data points
2. overfit a neural network (no weight decay regularisation)
3. remove the synthetic data and re-train with regularisation

theory: the neurons that overfit to change the output where there is synthetic data will simply be ignored rather than reset to being 'flat' which might result in better uncertainty estimation?

Jeremy says that this has no theoretical backing.
