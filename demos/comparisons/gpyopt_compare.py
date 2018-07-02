#!/usr/bin/env python3

import os
import time
import numpy as np
import pandas as pd
import sklearn.gaussian_process as sk_gp
import GPy
import GPyOpt

# local modules
import sys
sys.path.append('..')
import turbo as tb
import turbo.gui.cli as tg
import turbo.modules as tm


xmin, xmax = -6, 6
ymin, ymax = -5, 5
f = lambda x, y: 1.5 * (np.sin(0.5*x)**2 * np.cos(y) + 0.1*x + 0.2*y) + \
    np.random.normal(0, 0.2, size=None if isinstance(x, float) else x.shape)
num_trials = 30


def run_random():
    bounds = [('x', xmin, xmax), ('y', ymin, ymax)]

    op = tb.Optimiser(f, 'min', bounds, pre_phase_trials=1, settings_preset='random_search')
    rec = tb.Recorder(op)
    tg.OptimiserProgressBar(op)
    op.run(max_trials=num_trials)
    return [t[1].y for t in rec.get_sorted_trials()]


def run_turbo():
    """ From examining the source of GPyOpt, the default GPyOpt behaviour has been recreated using turbo """
    bounds = [('x', xmin, xmax), ('y', ymin, ymax)]

    op = tb.Optimiser(f, 'min', bounds, pre_phase_trials=5, settings_preset='default')

    op.fallback = tm.Fallback(close_tolerance=-1)
    op.pre_phase_select = tm.random_selector()
    op.acquisition = tm.EI(xi=0.01)

    kernel = GPy.kern.Matern52(input_dim=2)
    op.surrogate = tm.GPySurrogate(model_params={'normalizer': True, 'kernel': kernel}, training_iterations=5)
    rec = tb.Recorder(op)
    tg.OptimiserProgressBar(op)
    op.run(max_trials=num_trials)
    return [t[1].y for t in rec.get_sorted_trials()]


def run_gpyopt():
    domain = [
        {'name': 'x', 'type': 'continuous', 'domain': (xmin, xmax)},
        {'name': 'y', 'type': 'continuous', 'domain': (ymin, ymax)},
    ]

    bo = GPyOpt.methods.BayesianOptimization(f=lambda X: f(X[0, 0], X[0, 1]), domain=domain, maximise=False)
    bo.run_optimization(max_iter=num_trials-5, eps=-1, verbosity=False)
    return bo.Y


class CostRecord:
    def __init__(self, path):
        self.path = path
        self.data = pd.DataFrame(columns=['seed', 'duration'] + ['cost{}'.format(i+1) for i in range(num_trials)])
        if os.path.exists(path):
            print('loading preveously saved data at "{}"'.format(path))
            self.data = pd.read_csv(self.path, index_col=0)

    def add_data(self, seed, duration, results):
        results = np.array(results).flatten()
        results = pd.DataFrame([np.hstack([seed, duration, results])], columns=self.data.columns)
        self.data = self.data.append(results, ignore_index=True)
        self.save()

    def save(self):
        self.data.to_csv(self.path)


class Experiment:
    def __init__(self, name, path, run_test):
        self.name = name
        self.costs = CostRecord(path)
        self.run_test = run_test
        self.seeds_tested = set(np.array(self.costs.data.get('seed').values, dtype=int))

    def run(self, seed):
        if seed in self.seeds_tested:
            print('skipping {} seed {}'.format(self.name, seed))
        else:
            print('running {}'.format(self.name))
        np.random.seed(seed)
        start = time.time()
        results = self.run_test()
        duration = time.time() - start
        self.costs.add_data(seed, duration, results)
        self.seeds_tested.add(seed)


def run_compare():
    random = Experiment('random', 'costs_random.csv', run_random)
    turbo = Experiment('turbo', 'costs_turbo.csv', run_random)
    gpyopt = Experiment('GPyOpt', 'costs_gpyopt.csv', run_random)
    for seed in range(0, 100):
        print('running test with seed: {}'.format(seed))

        random.run(seed)
        turbo.run(seed)
        gpyopt.run(seed)


if __name__ == '__main__':
    run_compare()

