to get ground truth with GP:
- make several green lines (functions to fit) with several different data sets
  with chunks removed
- fit the GP to the given data points, choose hyperparameters by how the GP
  fits the underlying generator function
    - make sure that the generator function fits within the uncertainty bounds.
      Want the uncertainty to grow massively outside the range of data

discrete parameters with Bayesian optimisation
- treat the parameters differently, use a GP to fit the continuous ones and use some other method and acquisition function with the discrete data

Parallelise Bayes:
self.ongoing_jobs = [dict(job_num, config, predicted_cost)]
add to the list in next_configuration()
self.get_predicted_samples()
    '''
    get sx,sy points for the predicted results of ongoing jobs.
    automatically removes any jobs from ongoing_jobs that have since finished.
    ''
- create a new variable for the concatenation of the real and predicted samples
and use that for the Bayesian prediction. Keep them separate
- save sx, sy and psx, psy where p means 'predicted' in the step_log
- plot the predicted samples with a different style
- include a flag to enable the parallelisation
    - if the flag is enabled: ready_for_next_configuration should always return true


- Read original thesis for tips on tuning params
- Write a description of what the slice graph does. Eg for 2d, a horizontal or vertical line is cut through the surface passing through the chosen point
- for parameters which are log-uniform, feed log(param) to bayesian optimisation so that it can fit the curve better. convert back once you have the results
- switch over to the 'unique columns' methodology for not breaking the GP instead of checking that new samples won't break it (but keep that too)
- Plot the surrogate function with log axes to confirm that the function wiggles enough
- Write description of module and give licence
- try larger models and try not having every layer as dropout
- Boxplot with log x axis if necessary also width param
- Use linter and other error catching
- find out how to temporarily disable the git filter
- describe each of the notebooks for future reference

## Ideas
- Model selection cost = weighted sum of mse + difference between predicted uncertainty and ground truth uncertainty
- Try different random initialisations and pick best
- Could estimate uncertainty by training multiple times and measuring the variance (cross validation)
- Use GP as ground truth for model selection
- Maybe adding a small amount of random noise will help the optimiser not abandon the minimisation search (has to be >= 0 still always)
- Evaluators shouldn't assume that no jobs => done
- Instead of 'cost' job holds 'results' which can be a dictionary of quantities to retroactively modify the cost function
- early stopping
- scatter plot for the lowest points only (like surface)
- use numpy documentation format
- figure out how to fix the plotly import


## Done or abandoned
- Function to determine if a range is linear/logarithmic/other
- Config strings as tooltips in scatter plot
- Range scale='log' or 'linear'
- Todo1: EI
- Todo2: plotting
- Include cost in the tooltips to ensure the surface doesn't screw up and shoe the wrong costs for the configurations
- Get rid of trainlogger
- Use posted jobs and processed jobs. Get rid of outstanding jobs and n. Might be able to refactor _run
- Vim automatic ctags
- Write a note about how the acquisition function is only locally optimised and so may get stuck on local minima, but this is fine
- Robotics control and planning
- Todo3: optimisation refactor
- Todo4: Bayesian optimisation
- Another range type: if len == 1: return 'constant', allowed with Bayes but not included in the search space
- Test config can return either cost, or a list of [config, cost] for trying variations of the given config
- Random search
- Allow repeat configurations for random
- In synthetic data: assert that the result from orth has d columns (meaning full rank?)
- Save and resume (serialise samples to json)
- Maximising as well as minimising
- Set random search going on desktop
- Bayesian optimisation
- Resume(samples) call next() Len(samples) times and set self samples = samples
- mark l as 'cheap' to recompute and integrate special handling in random search
- along with config, pass a dictionary of constants?
- separate logger for the evaluator that gets thrown away?
- have the optimiser add configurations to a queue, have another class Evaluator which takes an optimiser in its constructor and listens to the queue and processes it how it likes, it can use a thread pool or have the jobs sent over a network















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
