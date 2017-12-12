# Major Refactor #

## Motivation ##
The library was initially written when I did not have a full understanding of the requirements of the different components of the algorithm. This has lead to many sub-optimal design decisions including:
- treating the input space as arrays of possible values
    - this was caused by implementing grid and random search algorithms first, then trying to fit the Bayesian optimiser to that interface
- a clunky interface which is configurable, but in order to facilitate this the internal logic is complex and probably fragile
- there is a bad separation of concerns because the potential interface boundaries did not present themselves until further into development.
- the optimiser and evaluator have servers built into them!!! This should obviously not be so tightly coupled.
- the implementation of client/server is very messy
- the idea of returning multiple cost values for a single input was a very bad idea and complicated the logic a lot. The use case of having a cheap parameter which you would like to evaluate at many places without, for example, re-training the model does not exist, because if the parameter is so cheap, just run a search over the cheap parameter and return the cost of the best value (could then keep a record of the chosen cheap parameter value in the extra field)
- again separation of concerns, but the step log probably shouldn't be kept inside the optimiser. Also, by keeping it outside and adding to it with a single 'transaction'/message, the quiescence condition becomes irrelevant, allowing checkpoints to be taken at arbitrary points. Also makes the saving/loading logic much simpler.
    - should have some separate mechanism for restoring an optimiser or evaluator
- the attempt to not crash when an exception occurs is misguided. In reality this just turns a crash into an infinite loop either of continuous crashes, or restarting the evaluation. It would probably be better to allow for a restarting mechanism outside of the optimiser. This would simplify the optimiser/evaluator logic and make unit testing easier because the exceptions can be properly caught, rather than resorting to reading the log.
- the GUI and plotting components should be further separated from the running of the algorithm
- the source code is not laid out logically and could be better split into modules
- the default values for the library are important and currently they are tied together with the initialisation logic.
- providing multiple ways to initialise a quantity was a misguided effort which introduced unnecessary complexity
- some techniques require storage of data which is not needed by other techniques. In a monolithic design, a variable must be created but left unused in most cases. This feels quite unclean.
- the current system of loading configuration is extremely fragile and a nightmare to debug, even for me who wrote the code to do it.

## Still Unsure ##
- how to store concrete GP
- whether to couple tightly and give references to the optimiser
- whether to include the step log inside the optimiser. Maybe as a compromise store only the bare minimum for the algorithm inside optimiser, and only store the more rich information in the logger.

## Proposal ##
an objective _function_ should be created which takes a configuration dictionary and returns either `cost:Float` or `(cost:Float, extra:dict)`.
    - For the use case where data needs to be stored between runs, global variables can be used, or a class can be created and the method to do the evaluation can be passed
    - think about how to incorporate logging, since `self.log()` is used currently but won't be available in the function version. Could either pass a logging function in, or let the user worry about keeping their own logs.

A ranges list of `[(name:str, min:Float, max:Float), (name:str, min:Int, max:Int, type='integral'), (name:str, [val1, val2], type='categorical')]`
    - assume real valued continuous unless type='integral'|'categorical'
    - maybe use a data holder? or just a tuple? (I think data holder would be better eg `op.ContinuousParam('learning_rate', 1, 10)`
    - categorical parameters mapped to `{0,1}^K` where K is the number of categories (one-hot)
    - integral parameters are rounded by the optimiser and passed as ints to the evaluator

An 'LatentSpace' module can be created, the user will have control over the following properties of the latent space:
- whether to map to a unit hypercube ('prevents hyperparameters with greater dynamic range from dominating' https://arxiv.org/pdf/1706.01566.pdf)
- which (if any) parameters to map to a logarithmic scale
- `to_latent() from_latent()`
- learned warping for non-stationary regression will be handled by the surrogate (since the parameters to be learned belong on that side of the interface boundary)

A 'Sampler' module can be created which provides methods for choosing 'random' configurations for the initialisation phase and for random fallback.
    - methods for generating many random points at once
    - methods for avoiding the existing data set by some margin

A 'Plan' module can be created which guides the overall behaviour of the search, for example choosing when the pre-phase ends and when to fall back to random and when to stop etc.
- keeps relevant counts of how many pre-phase, Bayes and random steps have been executed
- pass the number of pre-phase steps, number of total steps, close_tolerance  and proportion of steps to be random
    - can either pass these values, or overload the methods yourself, eg to terminate the search when a number of configurations are chosen to within some tolerance rather than falling back to random etc.
    - `re_train_interval` to decide how often to train new model hyperparameters. The surrogate may implement a method for adding data cheaply without recomputing hyperparameters

An 'Surrogate' module can be created which can construct new models or restore saved models. The models can be updated with new data, saved and queried
- input warping dealt with transparently from within this module
- MCMC hyperparameter treatment dealt with transparently from within this module
- @TODO: may have to be careful which side of the boundary to put the averaging over fantasies (because MCMC can re-use the current state of the parameters as a starting point). I.e. might want to give a data set and a list of fantasies and it returns
    - pybo uses `InitModel` then after that `AddData` is used to add more data rather than starting from scratch
    - follow pybo for the model methods
    - add sampling methods
    - methods for adding data without re-training hyperparameters
    - methods to duplicate the model (for training fantasies parallel)

An 'Acquisition' module can be created which can instantiate/partially apply an acquisition function given the data it needs. This results in a function X->R ready for optimising.
    - store information specific to this particular instantiation of the acquisition function. For example the sample drawn for Thompson sampling, or the parameter values (if the parameters are functions). Can store as part of the trial being constructed.
    - can choose whether the acquisition function returns the gradient in addition to the value. Also provide a method `gradient_available()` or similar.

An 'AcquisitionMaximiser' module can be created which takes a partially applied acquisition function and globally optimises it
    - eg use CMA-ES
    - in the literature this has been called the 'auxiliary optimiser'

A 'ParallelStrategy' module can be created @TODO
- this might be a slightly redundant module and just a data holder because the logic will still probably be performed in the optimiser.

!! think about networking only after most of the refactor is complete
!! rethink the evaluator thing. Use messagepack or JSON to create a simple RPC client/server. Clients can join and leave at any time. The clients run optimisers identical to the leader. The server knows about the computation budget and keeps track of the relative performance of each client. This framework would make it possible to spread the load when doing L-BFGS restarts etc.
- to serialise acquisition functions, use function object instead of partially applied function
- note can't have the server know where the clients are because the clients may be behind NAT and they may come and go, however unlike the first iteration I think a constant communication channel would be beneficial.


An 'Evaluator' module can be created which can either be a sequential evaluator or a client/server evaluator
- has methods for saying whether the evaluator is ready for more jobs and to poll for any finished jobs
- the sequential evaluator blocks when passed a new job and stores the result which is a fast operation to poll, so incurs little overhead. However this polling architecture fits the client/server model well also.

A 'Logger' module can be created @TODO
- `set_status(str)` where 'status' means 'stage of the algorithm'. can be used for profiling, building Gannt charts and providing information from the GUI
- can set logging level eg INFO, WARN, ERROR
- null logger should be available


@TODO: client/server operation

@TODO: multi-task

After running, the optimiser shouldn't be involved in the plotting, rather, plotting functions should take the step log provided by a logger


## Simplest Possible Example
```python
import turbo as t

def f(config):
    if config.z == 'something' or config.w:
        return 1
    else:
        return config.x ** 2 + config.y

op = t.Optimiser(objective=f, optimal='min', bounds=[
    ('x', -5, 5),   # could also write ('x', -5, 5, 'continuous')
    ('y', 0, 10, 'integral'),
    ('z', ['something', 'else'], 'categorical'),
    ('w', [True, False], 'categorical'),
    ('a', 'mystring', 'constant'),
])

op.run(max_trials=10)
```

## Simplest plotting example
```python
import turbo as t
import turbo.modules as tm
import turbo.plotting as tp

def f(config):
    if config.z == 'something' or config.w:
        return 1
    else:
        return config.x ** 2 + config.y

op = t.Optimiser(objective=f, optimal='min', bounds=[
    ('x', -5, 5),   # could also write ('x', -5, 5, 'continuous')
    ('y', 0, 10, 'integral'),
    ('z', ['something', 'else'], 'categorical'),
    ('w', [True, False], 'categorical'),
    ('a', 'mystring', 'constant'),
])

log = tm.Logger()

op.run(max_trials=10, logger=log)

tp.plot_error_over_time(log.trials, true_best=0)
```


## More customized example
```python
import turbo as t
import turbo.modules as tm
import turbo.plotting as tp

def f(config):
    if config.z == 'something' or config.w:
        return 1
    else:
        return config.x ** 2 + config.y

bounds = [
    ('x', -5, 5),   # could also write ('x', -5, 5, 'continuous')
    ('y', 0, 10, 'integral'),
    ('z', ['something', 'else'], 'categorical'),
    ('w', [True, False], 'categorical'),
    ('a', 'mystring', 'constant'),
]

# constructor creates default modules which can be over-ridden later
op = t.Optimiser(objective=f, optimal='min', bounds)

# this structure would be very easy to read from JSON
# These objects are just one possible implementation of the modules, Plan and
# LatentSpace are fairly general, but note that SciKitSurrogate and UCB
op.use_modules(
    tm.Plan(pre_phase_trials=10, max_trials=20,
            random_proportion=0.1, close_tolerance=1e-5, re_train_interval=1),
    tm.LatentSpace(normalise=True, log_params=['x']),
    tm.Sampler(pre_phase_method='random', fallback_method='random'),
    tm.Surrogate.SciKit(gp_params={}),
    #tm.Surrogate.Gpy_MCMC(gp_params={}),
    tm.AcquisitionMaximiser.QuasiNewton(num_random=100, num_gradient=10),
    #tm.AcquisitionMaximiser.CMA_ES(),
    tm.ParallelStrategy.Fantasies(num_sims=10, num_threads=5),
    #tm.Acquisition.UCB(beta=lambda t: e**-t),
)

ev = tm.ServerEvaluator('localhost', 9000)
op.use_module(ev)

log = tm.Logger(max_trials=-1) # keep all trails
op.use_module(log)

# helper function instead of setting the module directly, since this is a very
# common setting to change
op.set_acquisition('UCB', beta=4)

# the run method should provide helper parameters such as max_trials which
overwrites the relevenet settings in the modules. This isn't needed here
because al the settings are set directly.
op.run()

# plotting is not involved with the optimiser at all
tp.plot_error_over_time(log.trials, true_best=0)

log.save_trials('trials.json')

```



## Ideas ##
- where possible try to keep the modules decoupled from the optimiser. Eg for choosing a random configuration, could use `self.optimiser.data`, but would probably be better to just pass the data set as an argument when needed
    - may not even want to give them a reference to the optimiser object?
- could implement random search (or some other coverage strategy) using a custom Plan module (with pre_phase_trials=max_trials)
    - could implement grid search using a custom Sampler and Plan
- Job == Trial
- modules probably need access to the optimiser internals, so there should be a common register(optimiser) method which stores a reference to the optimiser, then assert registered() at the beginning of each function
- being fully modular makes it easy to swap things out, but could be cumbersome for the user to customize. Probably want to provide helper functions in the optimiser to construct the modules on the user's behalf then use them (eg call optimise(pre_phase_steps=10) and the strategy module will be created and used)
- like a c struct passed by reference, when creating a new evaluation 'job', pass a Job object then fill out its attributes and pass it down the call tree when constructing it, then sending a copy to the 'step log' is trivial.
- embrace the fact that evaluators are asynchronous. Let them start jobs then poll them for finished jobs. The server evaluator should be a single evaluator. When a client asks for a job the connection is held open while the evaluator advertises that it is ready to evaluate a job
- should be a mechanism for manually saying that a job got lost so that the surrogate doesn't always have that estimated value lingering around.
- during the refactor, fix random search. Currently it is just a grid search with a random order, not a truly random search


## Use Cases ##
- when benchmarking, I know the best possible value and so I would like the optimiser to stop early. Would be best if arbitrary stopping conditions are possible.





# Refactor Idea 2 #
- don't go over the top OOP, instead have 'function pointers' (not methods) for doing certain tasks. These can obviously be assigned to different functions easily.
- take the approach that if different implementations need wildly different parameters then there is a problem with the abstraction
- if data is required, then use an object with a `__call__` method and store the data as attributes
    - most of the interfaces would be a single function call anyway, since each module basically does one thing
