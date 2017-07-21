# fix some of the python2 ugliness
from __future__ import print_function
from __future__ import division
import sys
if sys.version_info[0] == 3: # python 3
    from queue import Empty
    from math import isclose, inf
elif sys.version_info[0] == 2: # python 2
    from Queue import Empty
    inf = float('inf')
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
else:
    print('unsupported python version')

import time
import json
import os

from collections import defaultdict
from itertools import groupby
# dummy => not multiprocessing but fake threading, which allows access to the
# same memory and variables
import multiprocessing.dummy as dummy

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
    def copy(self):
        ''' copy.copy() does not work with dotdict '''
        return dotdict(dict.copy(self))

class NumpyJSONEncoder(json.JSONEncoder):
    '''
    unfortunately numpy primitives are not JSON serialisable
    '''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJSONEncoder, self).default(obj)

def logspace(from_, to, num_per_mag=1):
    '''
    num_per_mag: number of samples per order of magnitude
    '''
    from_exp = np.log10(from_)
    to_exp = np.log10(to)
    num = abs(to_exp-from_exp)*num_per_mag + 1
    return np.logspace(from_exp, to_exp, num=num, base=10)

def time_string(seconds):
    mins, secs  = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    hours, mins = int(hours), int(mins)

    # if the number of seconds would round to an integer: display it as one
    if isclose(round(secs), secs, abs_tol=1e-1):
        secs = '{:02d}'.format(int(secs))
    else:
        # 04 => pad with leading zeros up to a total length of 4 characters (including the decimal point)
        # .1f => display 1 digits after the decimal point
        secs = '{:04.1f}'.format(secs)

    if hours > 0:
        return '{:02d}:{:02d}:{}'.format(hours, mins, secs)
    else:
        return '{:02d}:{}'.format(mins, secs)

def config_string(config, order=None):
    '''
        similar to the string representation of a dictionary
    '''
    assert order is None or (set(order) == config.keys())
    string = '{'
    order = sorted(list(config.keys())) if order is None else order
    for p in order:
        if type(config[p]) == str or type(config[p]) == np.str_:
            string += '{}="{}", '.format(p, config[p])
        else: # assuming numeric
            # 2 significant figures
            string += '{}={:.2g}, '.format(p, config[p])
    string = string[:-2] # remove trailing comma and space
    string += '}'
    return string

def exception_string():
    '''
    get a string formatted with the last exception
    '''
    import traceback
    return ''.join(traceback.format_exception(*sys.exc_info()))

class StoreLogger(object):
    '''
    A logger for Optimisers which stores the output to a string rather than printing
    '''
    def __init__(self):
        self.log_record = ''
    def log(self, string, newline=True):
        self.log_record += string
        if newline:
            self.log_record += '\n'
    def __str__(self):
        return self.log_record
    def __repr__(self):
        return str(self)

class Sample(object):
    def __init__(self, config, cost):
        self.config = config
        self.cost = cost
    def __repr__(self):
        return '(config={}, cost={})'.format(config_string(self.config), self.cost)
    def __iter__(self):
        ''' so you can write my_config, my_cost = my_sample '''
        yield self.config
        yield self.cost
    def __eq__(self, other):
        return self.config == other.config and self.cost == other.cost

class Job(object):
    '''
    a job is a single configuration to be tested, it may result in one or
    multiple samples being evaluated.
    '''
    def __init__(self, config, n):
        self.config = config
        self.n = n # job number
        self.samples = None
        self.exception = None
        self.exception_string = None # traceback for the exception
        self.duration = None
    def set_results(self, samples, duration=None):
        self.samples = samples
        self.duration = duration

class LocalEvaluator(object):
    '''
    an evaluator listens to the job queue of an optimiser and evaluates
    configurations for it, responding with the cost for that configuration.

    note: if an evaluator takes a job, it is obliged to process it before
    shutting down, however if it obeys this then it may start and stop at will.

    a LocalEvaluator runs in the same python process as the optimiser and
    listens to the queue directly, either from a background thread or on the
    thread it is called from.
    '''
    def __init__(self, optimiser):
        '''
        optimiser: an Optimiser object to poll jobs from
        '''
        self.optimiser = optimiser
        self.proc = None
        self.start_time = None
        self.done = False # flag to quit
        self.log_record = ''

    def start(self, run_async=True):
        '''
        note: must start the optimiser before starting the evaluator
        '''
        assert self.proc is None
        assert self.optimiser.running # must start optimiser first
        self.done = False
        self.log('started')
        self.start_time = time.time()
        if run_async:
            self.proc = dummy.Process(target=self._poll_jobs, args=[])
            self.proc.start()
        else:
            self._poll_jobs()

    def stop(self):
        print('stopping evaluator')
        self.done = True
        if self.proc is not None:
            self.proc.join()
            self.proc = None
        self.start_time = None
        self.log('stopped')
        print('stopped')

    def wait_for(self):
        if self.proc is None:
            print('evaluator already finished')
        else:
            self.proc.join()
            self.proc = None
            print('evaluator finished')

    def log(self, string, newline=True):
        self.log_record += string
        if newline:
            self.log_record += '\n'

    def monitor(self):
        while not self.done:
            print('running for {}'.format(time_string(time.time()-self.start_time)))
            print('-'*25)
            print(self.log_record)
            time.sleep(1)
            clear_output(wait=True)
        clear_output(wait=True)
        print('not running')
        print('-'*25)
        print(self.log_record)

    def _poll_jobs(self):
        while not self.done and self.optimiser.running:
            try:
                job = self.optimiser.job_queue.get_nowait()
                self.log('received job {}'.format(job.n))
                start_time = time.time()
                try:
                    results = self.test_config(job.config)
                except Exception as e:
                    job.exception = e
                    job.exception_string = exception_string()
                    self.log('Exception Raised: {}'.format(job.exception_string))
                    self.optimiser.result_queue.put(job)
                    continue
                # evaluator can return either a list of samples, or just a cost
                samples = results if isinstance(results, list) else [Sample(job.config, results)]
                job.set_results(samples, duration=time.time()-start_time)
                self.log('returning results of job {}: samples={}'.format(job.n, job.samples))
                self.optimiser.result_queue.put(job)
            except Empty: # job_queue empty
                time.sleep(0.1)
                continue
        if self.done:
            self.log('stopped because of an interruption')
        elif not self.optimiser.running:
            self.log('stopped because the optimiser is no longer running')
        else:
            self.log('stopped for an unknown reason')
        self.done = True

    def test_config(self, config):
        '''
        given a configuration, evaluate and return its cost. Can also test
        multiple configurations based on the given config and return a list of
        Sample objects (eg could run multiple times if there is some randomness,
        or could introduce another parameter which is cheap to test)

        config: dictionary of parameter names to values
        '''
        raise NotImplemented

class Optimiser(object):
    '''
    given a search space and a function to call in order to evaluate the cost at
    a given location, find the minimum of the function in the search space.

    Importantly: an expression for the cost function is not required
    '''
    def __init__(self, ranges, logger=None):
        '''
        ranges: dictionary of parameter names and their ranges (numpy arrays, can be created by np.linspace or np.logspace)
        logger: a function which takes a string to be logged and does as it wishes with it
        '''
        self.ranges = dotdict(ranges)

        # both queues hold Job objects, with the result queue having jobs with 'cost' filled out
        #TODO max_size as an argument
        self.job_queue = dummy.Queue(maxsize=10)
        self.result_queue = dummy.Queue()

        # the configurations that have been tested (list of `Sample` objects)
        self.samples = []
        # may not be equal to len(self.samples) because the evaluator can evaluate multiple configurations at once
        self.num_processed_jobs = 0

        self.logger = logger if logger is not None else StoreLogger()
        self._log = self.logger.log

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
            self.proc = dummy.Process(target=self._safe_run, args=[])
            self.proc.start()
            while not self.running:
                time.sleep(0.01)
        else:
            self._run()

    def interrupt(self):
        '''
        gracefully stop the currently running optimisation process if there is one
        '''
        if not self.running:
            print('already stopped')
        else:
            print('stopping optimiser...')
            self.done = True
            if self.proc: # running asynchronously
                self.proc.join()
            print('stopped.')

    def wait_for(self):
        if self.proc is None:
            print('optimiser already finished')
        else:
            self.proc.join()
            self.proc = None
            print('optimiser finished')

    def monitor(self, watch_log=True, stop_if_interrupted=True):
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
                    self.report()
                    print('-'*25)

                    if watch_log:
                        print(str(self.logger))
                    else:
                        print('still running' + ('.'*(ticks%3)))
                        ticks += 1

                    time.sleep(1)
                    clear_output(wait=True)


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

    def _safe_run(self):
        try:
            self._run()
        except Exception as e:
            self._log('Exception raised during run: {}'.format(exception_string()))
            self.done = True
            self.running = False

    def _run(self):
        assert not self.running
        self.running = True
        self.done = False
        self.run_start = time.time()
        self._log('starting optimisation...')
        best = self.best_known_sample()
        current_best = best.cost if best is not None else inf # current best cost
        n = self.num_processed_jobs+1 # sample number, 1-based

        outstanding_jobs = 0 # the number of jobs added to the queue that have not yet been processed
        out_of_configs = False # whether there are no more configurations available from _next_configuration()

        while True:
            # add new jobs
            while not self.job_queue.full() and not out_of_configs:
                config = self._next_configuration()
                if config is None:
                    self._log('out of configurations')
                    out_of_configs = True
                    break # finish processing the outstanding jobs
                job = Job(dotdict(config), n)
                n += 1
                self.job_queue.put(job)
                outstanding_jobs += 1
                self._log('job {} added to queue'.format(job.n))

            # process results
            while not self.result_queue.empty():
                job = self.result_queue.get()

                if job.exception is not None:
                    self._log('exception raised during evaluation: {}({})\n{}'.format(
                        str(type(job.exception)), str(job.exception), job.exception_string))
                    self.done = True
                    break # stop running

                for i, s in enumerate(job.samples):
                    self._log('job={:03}, job_time={}, sample {:02}: config={}, cost={:.2g}{}'.format(
                        job.n, time_string(job.duration), i,
                        config_string(s.config), s.cost,
                        (' (current best)' if s.cost < current_best else '')
                    ))

                    if s.cost < current_best:
                        current_best = s.cost
                self.samples.extend(job.samples)
                self.num_processed_jobs += 1 # regardless of how many samples
                outstanding_jobs -= 1

            if self.done or (out_of_configs and outstanding_jobs == 0):
                break

            time.sleep(0.05)

        if self.done:
            self._log('optimisation interrupted and shut down gracefully')
        elif out_of_configs and outstanding_jobs == 0:
            self._log('optimisation finished.')
        else:
            self._log('stopped for an unknown reason.')

        dur = time.time()-self.run_start
        self.duration += dur
        self._log('total time taken: {} ({} this run)'.format(time_string(self.duration), time_string(dur)))
        self.done = True
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
        samples_total = self.total_samples()
        percent_progress = self.num_processed_jobs/float(samples_total)*100.0
        print('{} of {} samples ({:.1f}%) taken in {}.'.format(
            self.num_processed_jobs, samples_total, percent_progress, time_string(dur)))
        best = self.best_known_sample()
        if best is None:
            print('no best configuration known')
        else:
            print('best known configuration:\n{}'.format(config_string(best.config)))
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

    def plot_param(self, param_name, plot_boxplot=True, plot_samples=True, plot_means=True, log_axes=(False,False), widths=None):
        '''
        plot a boxplot of parameter values against cost
        plot_boxplot: whether to plot boxplots
        plot_samples: whether to plot each sample as a point
        plot_means: whether to plot a line through the mean costs
        log_axes: (xaxis,yaxis) whether to display the axes with a logarithmic scale
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
        if log_axes[0]:
            plt.xscale('log')
        if log_axes[1]:
            plt.yscale('log')
        plt.autoscale(True)
        plt.show()

    def scatter_plot(self, param_a, param_b, interactive=True, color_by='cost', log_axes=(False,False,False)):
        '''
            interactive: whether to display a slider for changing the number of samples to display
            color_by: either 'cost' or 'age'
            log_axes: whether to display the x,y,z axes with a logarithmic scale
        '''
        assert color_by in ['cost', 'age']

        xs, ys, costs, texts = [], [], [], []
        for i, s in enumerate(self.samples):
            xs.append(s.config[param_a])
            ys.append(s.config[param_b])
            costs.append(s.cost)
            texts.append('sample {:03}, config: {}, cost: {}'.format(i+1, config_string(s.config), s.cost))

        xs, ys, costs, texts = map(np.array, (xs, ys, costs, texts))
        color = 'z' if color_by == 'cost' else 'age'
        axes_names = ['param: ' + param_a, 'param: ' + param_b, 'cost']

        plot3D.scatter3D(xs, ys, costs, interactive=interactive, color_by=color,
                        markersize=4, tooltips=texts, axes_names=axes_names, log_axes=log_axes)

    def surface_plot(self, param_a, param_b, log_axes=(False,False,False)):
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

        log_axes: whether to display the x,y,z axes with a logarithmic scale
        '''
        # get all the x and y values found in any of the samples (may not equal self.ranges[...])
        xs = np.array(sorted(set([val for val, samples in self.group_by_param(param_a)])))
        ys = np.array(sorted(set([val for val, samples in self.group_by_param(param_b)])))
        costs = defaultdict(float) # if not all combinations of x and y are available: cost = 0
        texts = defaultdict(lambda: 'no data')
        for val, samples_for_val in self.group_by_params(param_a, param_b):
            sample = min(samples_for_val, key=lambda s: s.cost)
            costs[val] = sample.cost
            texts[val] = 'config: {}, cost: {}'.format(config_string(sample.config), sample.cost)
        xs, ys = np.meshgrid(xs, ys)
        costs = np.vectorize(lambda x,y: costs[(x,y)])(xs, ys)
        texts = np.vectorize(lambda x,y: texts[(x,y)])(xs, ys)
        axes_names = ['param: ' + param_a, 'param: ' + param_b, 'cost']
        plot3D.surface3D(xs, ys, costs, tooltips=texts, axes_names=axes_names, log_axes=log_axes)

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
                f.write(json.dumps(self._save_dict(), indent=4, cls=NumpyJSONEncoder))

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
            best = Sample({}, inf)
        return {
            'samples' : [(s.config, s.cost) for s in self.samples],
            'num_processed_jobs' : self.num_processed_jobs,
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
        self.num_processed_jobs = save['num_processed_jobs']
        self.duration = save['duration']
        self.logger.log_record = save['log']


class GridSearchOptimiser(Optimiser):
    '''
        Grid search optimisation strategy: search with some step size over each
        dimension to find the best configuration
    '''
    def __init__(self, ranges, logger=None, order=None):
        '''
        order: the precedence/importance of the parameters (or None for default)
            appearing earlier => more 'primary' (changes more often)
            default could be any order
        '''
        super(self.__class__, self).__init__(ranges, logger)
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
        save = super(self.__class__, self)._save_dict()
        save['progress'] = self.progress
        save['progress_overflow'] = self.progress_overflow
        return save

    def _load_dict(self, save):
        super(self.__class__, self)._load_dict(save)
        self.progress = save['progress']
        self.progress_overflow = save['progress_overflow']

class RandomSearchOptimiser(Optimiser):
    '''
        Random search optimisation strategy: choose random combinations of
        parameters until either a certain number of samples are taken or all
        combinations have been tested.
    '''
    def __init__(self, ranges, logger=None, allow_re_tests=False, max_samples=inf, max_retries=10000):
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
        super(self.__class__, self).__init__(ranges, logger)
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
        # only use parameters relevant to the optimiser, ie the ones from ranges.keys()
        # (evaluators may introduce new parameters to a configuration)
        return '|'.join([str(config[param]) for param in sorted(self.ranges.keys())])

    def _next_configuration(self):
        if self.num_processed_jobs >= self.max_samples:
            return None # done
        else:
            c = self._random_config()
            if not self.allow_re_tests:
                attempts = 1
                while self._hash_config(c) in self.tested_configurations and attempts < self.max_retries:
                    c = self._random_config()
                    attempts += 1
                if attempts >= self.max_retries:
                    self._log('max number of retries ({}) exceeded, most of the parameter space must have been explored. Quitting...'.format(self.max_retries))
                    return None # done
                self.tested_configurations.add(self._hash_config(c))
            return c

    def _load_dict(self, save):
        super(self.__class__, self)._load_dict(save)
        if not self.allow_re_tests:
            self.tested_configurations = set([self._hash_config(s.config) for s in self.samples])



def monitor_to_file(optimiser, evaluator):
    '''
    A helper function to monitor the logs of the optimiser and LocalEvaluator
    and redirect them to files in /tmp
    '''
    op_file = '/tmp/optimiser.log'
    ev_file = '/tmp/evaluator.log'
    print('logging to:')
    print(op_file)
    print(ev_file)
    with open(op_file, 'w') as o, open(ev_file, 'w') as e:
        try:
            while optimiser.running or not evaluator.done:
                # offset=0, whence=0 => absolute from beginning
                o.seek(0, 0)
                o.write(str(optimiser.logger))
                o.flush()

                e.seek(0, 0)
                e.write(evaluator.log_record)
                e.flush()

                time.sleep(1)
            print('both the optimiser and evaluator have stopped')
        except KeyboardInterrupt:
            print('interrupt caught, monitoring has stopped')
        finally:
            message = '\n\n--NO LONGER MONITORING--'
            o.write(message)
            e.write(message)

