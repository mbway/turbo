import math
import time
import json
import os

import copy
from collections import defaultdict
from itertools import groupby
# dummy => not multiprocessing but fake threading, which allows access to the
# same memory and variables
from multiprocessing.dummy import Process

import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output

# local modules
import plot3D


class dotdict(dict):
    '''
        provide syntactic sugar for accessing dict elements with a dot eg
        mydict = {'somekey': 1}
        d = dotdict(mydict)
        d.somekey   # 1
        d.somekey = 2
        d.somekey   # 2
    '''
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, val):
        self[name] = val

def logspace(from_, to, num_per_mag=1):
    '''
    num_per_mag: number of samples per order of magnitude
    '''
    from_exp = np.log10(from_)
    to_exp = np.log10(to)
    num = abs(to_exp-from_exp)*num_per_mag + 1
    return np.logspace(from_exp, to_exp, num=num, base=10)

def unpack(l):
    ''' convert from a list of tuples to a tuple of lists '''
    return map(list, zip(*l))

def time_string(seconds):
    mins, secs  = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    hours, mins = int(mins), int(hours)

    # if the number of seconds would round to an integer: display it as one
    if math.isclose(round(secs), secs, abs_tol=1e-1):
        secs = '{:02d}'.format(int(secs))
    else:
        # 04 => pad with leading zeros up to a total length of 4 characters (including the decimal point)
        # .1f => display 1 digits after the decimal point
        secs = '{:04.1f}'.format(secs)

    if hours > 0:
        return '{:02d}:{:02d}:{}'.format(hours, mins, secs)
    else:
        return '{:02d}:{}'.format(mins, secs)

def config_string(config, compact=False, order=None):
    '''
        similar to the string representation of a dictionary
    '''
    string = '{'
    order = sorted(list(config.keys())) if order is None else order
    for p in order:
        if type(config[p]) == str or type(config[p]) == np.str_:
            string += '{}="{}", '.format(p, config[p])
        else: # assuming numeric
            string += '{}={:.2g}, '.format(p, config[p])
    string = string[:-2] # remove trailing comma and space
    string += '}'
    return string

class Logger:
    '''
        a class who's objects can be passed to an Optimiser. The 'log' method is
        called with various messages from the optimiser during a run.
    '''
    def __init__(self):
        self.log_record = ''
    def log(self, string):
        self.log_record += string + '\n'
    def __str__(self):
        return self.log_record
    def __repr__(self):
        return str(self)


class Sample:
    def __init__(self, config, cost):
        self.config = config
        self.cost = cost
    def __repr__(self):
        return '(config={}, cost={})'.format(config_string(self.config), self.cost)
    def __iter__(self):
        ''' so you can write my_config, my_cost = my_sample '''
        yield self.config
        yield self.cost

class Optimiser:
    '''
    given a search space and a function to call in order to evaluate the cost at
    a given location, find the minimum of the function in the search space.

    Importantly: an expression for the cost function is not required
    '''
    def __init__(self, ranges, test_config, logger=None):
        '''
        ranges: dictionary of parameter names and their ranges (np.linspace or np.logspace)
        test_config: function for evaluating a configuration (returning its cost)
            configuration: dictionary of parameter names to values
        logger: a function which takes a string to be logged and does as it wishes with it
        '''
        self.ranges = dotdict(ranges)
        self.order = list(ranges.keys())
        self.test_config = test_config # function to evaluate a configuration
        # the configurations that have been tested (list of `Sample` objects)
        self.samples = []

        self.logger = logger if logger is not None else Logger()
        self.log = self.logger.log

        self.running = False
        self.proc = None # handle for the process used for asynchronous execution
        self.done = False # flag to gracefully stop asynchronous run
        self.run_start = None # the time at which the last run was started
        self.duration = 0 # total time spent (persists across runs)


# Running Interrupting and Monitoring

    def run(self, run_async=True):
        '''
        run the optimisation procedure, saving each sample along the way and
        keeping track of the current best
        '''
        assert self.proc is None
        if run_async:
            self.proc = Process(target=self._run, args=[])
            self.proc.start()
        else:
            self._run()

    def interrupt(self):
        '''
        gracefully stop the currently running optimisation process if there is one
        '''
        if not self.running:
            print('already stopped')
        else:
            print('stopping...')
            self.done = True
            if self.proc: # running asynchronously
                self.proc.join()
            print('stopped.')

    def wait_for(self, watch_log=True, stop_if_interrupted=True):
        '''
        wait for the optimisation run to finish. (useful for monitoring an asynchronous run)
        watch_log: whether to print the log contents while waiting
        stop_if_interrupted: whether to interrupt the run if a KeyboardInterrupt is caught
        '''
        try:
            if not self.running:
                print('not running')
            else:
                ticks = 0
                while not self.done:
                    clear_output(wait=True)
                    time.sleep(1)

                    self.report()
                    print('-'*25)

                    if watch_log:
                        print(str(self.logger))
                    else:
                        print('still running' + ('.'*(ticks%3)))
                        ticks += 1

                clear_output(wait=True)
                print('optimiser finished.')
                print('-'*25)

                self.report()
                if watch_log:
                    print('-'*25)
                    print(str(self.logger))
        except KeyboardInterrupt:
            clear_output(wait=True)
            if stop_if_interrupted:
                print('interrupt caught: stopping optimisation run')
                self.interrupt()
            else:
                print('interrupt caught, optimiser will continue in the background')

    def _next_configuration(self):
        '''
        implemented by different optimisation methods
        return the next configuration to try, or None if finished
        '''
        raise NotImplemented

    def total_samples(self):
        '''
        return the total number of samples to be tested
        '''
        total = 1
        for param_range in self.ranges.values():
            total *= len(param_range)
        return total

    def _run(self):
        assert not self.running
        self.done = False
        self.running = True
        self.run_start = time.time()
        self.log('starting optimisation...')
        best = self.best_known_sample()
        current_best = best.cost if best is not None else math.inf # current best cost
        n = len(self.samples)+1 # sample number

        config = self._next_configuration()
        while config is not None:
            config = dotdict(config)
            sample_start_time = time.time()
            try:
                cost = self.test_config(config)
            except Exception as e:
                self.log('exception raised during evaluation: {}({})'.format(str(type(e)), str(e)))
                self.done = True
                break
            dur = time.time()-sample_start_time

            self.log('sample={:03}, time={}, config={}, cost={:.2g} {}'.format(
                n, time_string(dur),
                config_string(config, self.order), cost,
                ('(current best)' if cost < current_best else '')
            ))

            if cost < current_best:
                current_best = cost
            self.samples.append(Sample(config, cost))
            if self.done:
                break
            else:
                config = self._next_configuration()
            n += 1

        if self.done:
            self.log('optimisation interrupted and shut down gracefully')
        else:
            self.log('optimisation finished.')
        self.done = True
        dur = time.time()-self.run_start
        self.duration += dur
        self.log('total time taken: {} ({} this run)'.format(time_string(self.duration), time_string(dur)))
        self.proc = None
        self.running = False


# Extracting Results

    def best_known_sample(self):
        ''' returns the best known (config, cost) or None if there is none '''
        if len(self.samples) > 0:
            return min(self.samples, key=lambda s: s.cost)
        else:
            return None

    def report(self):
        # duration of the current run
        current_dur = time.time() - self.run_start if self.running else 0
        dur = self.duration + current_dur
        if self.running:
            print('currently running (has been for {}).'.format(time_string(current_dur)))
        samples_tested = len(self.samples)
        samples_total = self.total_samples()
        percent_progress = samples_tested/float(samples_total)*100.0
        print('{} of {} samples ({:.1f}%) taken in {}.'.format(
            samples_tested, samples_total, percent_progress, time_string(dur)))
        best = self.best_known_sample()
        if best is None:
            print('no best configuration known')
        else:
            print('best known configuration:\n{}'.format(config_string(best.config, self.order)))
            print('cost of the best known configuration:\n{}'.format(best.cost))

# Plotting and Plotting Utilities

    def group_by_param(self, param_name):
        '''
        return [value, [sample]] for each unique value of the given parameter
        '''
        param_key = lambda sample: sample.config[param_name]
        data = []
        # must be sorted before grouping
        for val, samples in groupby(sorted(self.samples, key=param_key), param_key):
            data.append((val, list(samples)))
        return data

    def group_by_params(self, param_a, param_b):
        '''
        return [(value_a, value_b), [sample]] for each unique pair of values of the given parameters
        '''
        params_key = lambda sample: (sample.config[param_a], sample.config[param_b])
        data = []
        # must be sorted before grouping
        for val, samples in groupby(sorted(self.samples, key=params_key), params_key):
            data.append((val, list(samples)))
        return data

    def plot_param(self, param_name, plot_boxplot=True, plot_samples=True, plot_means=True):
        '''
        plot a boxplot of parameter values against cost
        '''
        values = []
        costs = []
        means = []
        for val, samples in self.group_by_param(param_name):
            values.append(val)
            c = [s.cost for s in samples]
            costs.append(c)
            means.append(np.mean(c))
        labels = ['{:.2g}'.format(v) for v in values]

        plt.figure(figsize=(16,8))

        if plot_means:
            plt.plot(values, means, 'r-', linewidth=1, alpha=0.5)
        if plot_samples:
            xs, ys = zip(*[(s.config[param_name], s.cost) for s in self.samples])
            plt.plot(xs, ys, 'go', markersize=5, alpha=0.6)
        if plot_boxplot:
            plt.boxplot(costs, positions=values, labels=labels)

        plt.margins(0.1, 0.1)
        plt.xlabel('parameter: ' + param_name)
        plt.ylabel('cost')
        plt.yscale('log')
        plt.show()

    def scatter_plot(self, param_a, param_b, interactive=True, color_by='cost'):
        '''
            color_by: either 'cost' or 'age'
        '''
        assert color_by in ['cost', 'age']

        xs, ys, costs, texts = [], [], [], []
        for i, s in enumerate(self.samples):
            xs.append(s.config[param_a])
            ys.append(s.config[param_b])
            costs.append(s.cost)
            texts.append('sample {:03}, config: {}, cost: {}'.format(i+1, config_string(s.config, compact=True), s.cost))

        xs, ys, costs, texts = map(np.array, (xs, ys, costs, texts))
        color = 'z' if color_by == 'cost' else 'age'
        axes_names = ['param: ' + param_a, 'param: ' + param_b, 'cost']

        plot3D.scatter3D(xs, ys, costs, interactive=interactive, color_by=color,
                        markersize=8, tooltips=texts, axes_names=axes_names)

    def surface_plot(self, param_a, param_b):
        '''
        plot the surface of different values of param_a and param_b and how they
        affect the cost (z-axis). If there are multiple configurations with the
        same combination of param_a,param_b values then the minimum is taken for
        the z/cost value.

        This method does not require that in self.samples there is complete
        coverage of all param_a and param_b values, or that the samples have a
        particular ordering.

        If there are gaps where a param_a,param_b combination has not yet been
        evaluated, the cost for that point will be 0.
        '''
        # get all the x and y values found in any of the samples (may not equal self.ranges[...])
        xs = np.array(sorted(set([val for val, samples in self.group_by_param(param_a)])))
        ys = np.array(sorted(set([val for val, samples in self.group_by_param(param_b)])))
        costs = defaultdict(float) # if not all combinations of x and y are available: cost = 0
        texts = defaultdict(lambda: 'no data')
        for val, samples_for_val in self.group_by_params(param_a, param_b):
            sample = min(samples_for_val, key=lambda s: s.cost)
            costs[val] = sample.cost
            texts[val] = 'config: {}, cost: {}'.format(config_string(sample.config, compact=True), sample.cost)
        xs, ys = np.meshgrid(xs, ys)
        costs = np.vectorize(lambda x,y: costs[(x,y)])(xs, ys)
        texts = np.vectorize(lambda x,y: texts[(x,y)])(xs, ys)
        axes_names = ['param: ' + param_a, 'param: ' + param_b, 'cost']
        plot3D.surface3D(xs, ys, costs, tooltips=texts, axes_names=axes_names)

# Saving and Loading Progress

    def save_progress(self, filename):
        '''
        save the progress of the optimisation to a JSON string which can be
        re-loaded and continued.

        Note: this does not save all the state of the optimiser, only the samples
        '''
        if os.path.isfile(filename):
            raise Exception('File "{}" already exists!'.format(filename))
        else:
            with open(filename, 'w') as f:
                f.write(json.dumps(self._save_dict(), indent=4))

    def load_progress(self, filename):
        '''
        restore the progress of an optimisation run (note: the optimiser must be
        initialised identically to when the samples were saved)
        '''
        with open(filename, 'r') as f:
            self._load_dict(json.loads(f.read()))


    def _save_dict(self):
        '''
        generate the dictionary to be JSON serialised and saved
        (designed to be overridden by derived classes in order to save specialised data)
        '''
        best = self.best_known_sample()
        if best is None:
            best = Sample({}, math.inf)
        return {
            'samples' : [(s.config, s.cost) for s in self.samples],
            'duration' : self.duration,
            'log' : self.logger.log_record,
            'best_sample' : {'config' : best.config, 'cost' : best.cost}
        }

    def _load_dict(self, save):
        '''
        load progress from a dictionary
        (designed to be overridden by derived classes in order to load specialised data)
        '''
        self.samples = [Sample(dotdict(config), cost) for config, cost in save['samples']]
        self.duration = save['duration']
        self.logger.log_record = save['log']


class GridSearchOptimiser(Optimiser):
    '''
        Grid search optimisation strategy: search with some step size over each
        dimension to find the best configuration
    '''
    def __init__(self, ranges, test_config, logger=None, order=None):
        '''
        order: the precedence/importance of the parameters (or None for default)
            appearing earlier => more 'primary' (changes more often)
            default could be any order
        '''
        super(GridSearchOptimiser, self).__init__(ranges, test_config, logger)
        self.order = list(ranges.keys()) if order is None else order
        assert set(self.order) == set(ranges.keys())
        # start at the lower boundary for each parameter
        # progress counts from 0 to len(range)-1 for each parameter
        self.progress = {param : 0 for param in ranges.keys()}
        self.progress_overflow = False

    def _current_config(self):
        return {param : self.ranges[param][i] for param, i in self.progress.items()}

    def _increment_progress(self):
        '''
            increment self.progress to the next progress
        '''
        # basically an algorithm for adding 1 to a number, but with each 'digit'
        # being of a different base. (note: 'little endian')
        carry = False # carry flag
        for p in self.order:
            i = self.progress[p] # current value of this 'digit'
            if i+1 >= len(self.ranges[p]): # this digit overflowed
                carry = True
                self.progress[p] = 0
            else:
                carry = False
                self.progress[p] = i + 1
                break
        # if the carry flag is true then the whole 'number' has overflowed => finished
        self.progress_overflow = carry

    def _next_configuration(self):
        if self.progress_overflow:
            return None # done
        else:
            cur = self._current_config()
            self._increment_progress()
            return cur

    def _save_dict(self):
        save = super(GridSearchOptimiser, self)._save_dict()
        save['progress'] = self.progress
        save['progress_overflow'] = self.progress_overflow
        return save

    def _load_dict(self, save):
        super(GridSearchOptimiser, self)._load_dict(save)
        self.progress = save['progress']
        self.progress_overflow = save['progress_overflow']

class RandomSearchOptimiser(Optimiser):
    '''
        Random search optimisation strategy: choose random combinations of
        parameters until either a certain number of samples are taken or all
        combinations have been tested.
    '''
    def __init__(self, ranges, test_config, logger=None, allow_re_tests=False, max_samples=math.inf, max_retries=10000):
        '''
        allow_re_tests: whether a configuration should be tested again if it has
            already been tested. This might be desirable if the cost for a
            configuration is not deterministic. However allowing retests removes
            the option of stopping the optimisation process when max_retries is
            exceeded, another method (eg max_samples) should be used in its place.
        max_samples: stop taking samples after this number has been reached. By
            default there is no cutoff (so the search will never finish)
        max_retries: (only needed if allow_re_tests=False) the number of times
            to try generating a configuration that hasn't been tested already,
            before giving up (to exhaustively explore the parameter space,
            perhaps finish off with a grid search?)
        '''
        super(RandomSearchOptimiser, self).__init__(ranges, test_config, logger)
        self.allow_re_tests = allow_re_tests
        self.tested_configurations = set()
        self.max_samples = max_samples
        self.max_retries = max_retries

    def _random_config(self):
        return {param : np.random.choice(param_range) for param, param_range in self.ranges.items()}

    def _hash_config(self, config):
        '''
        need some way of quickly testing whether configurations are in the set
        of already tested ones. `dict` is not hashable. This is slightly hacky
        but should work so long as the parameters have __str__ methods
        '''
        return '|'.join([str(config[param]) for param in sorted(config.keys())])

    def _next_configuration(self):
        if len(self.samples) >= self.max_samples:
            return None # done
        else:
            c = self._random_config()
            if not self.allow_re_tests:
                attempts = 1
                while self._hash_config(c) in self.tested_configurations and attempts < self.max_retries:
                    c = self._random_config()
                    attempts += 1
                if attempts >= self.max_retries:
                    self.log('max number of retries ({}) exceeded, most of the parameter space must have been explored. Quitting...'.format(self.max_retries))
                    return None # done
                self.tested_configurations.add(self._hash_config(c))
            return c

    def _load_dict(self, save):
        super(RandomSearchOptimiser, self)._load_dict(save)
        if not self.allow_re_tests:
            self.tested_configurations = set([self._hash_config(s.config) for s in self.samples])

