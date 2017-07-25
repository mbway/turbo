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

from IPython.display import clear_output

import numpy as np
import matplotlib.pyplot as plt

# for Bayesian Optimisation
import sklearn.gaussian_process as gp
import scipy.optimize
from scipy.stats import norm # Gaussian/normal distribution


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

def make2D(arr):
    ''' convert a numpy array with shape (l,) into an array with shape (l,1)
        (np.atleast_2d behaves similarly but would give shape (1,l) instead)
    '''
    return arr.reshape(-1, 1)

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
    assert order is None or set(order) == set(config.keys())
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
    def __init__(self, config, job_num):
        self.config = config
        self.num = job_num
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
        assert self.proc is None, 'already running'
        assert self.optimiser.running, 'must start optimiser first'
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

    def wait_for(self, quiet=False):
        if self.proc is None:
            if not quiet:
                print('evaluator already finished')
        else:
            self.proc.join()
            self.proc = None
            if not quiet:
                print('evaluator finished')

    def log(self, string, newline=True):
        if not isinstance(string, str):
            string = str(string)
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
                self.log('received job {}: {}'.format(job.num, config_string(job.config)))
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
                self.log('returning results of job {}: samples={}'.format(job.num, job.samples))
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
    def __init__(self, ranges, queue_size=1):
        '''
        ranges: dictionary of parameter names and their ranges (numpy arrays, can be created by np.linspace or np.logspace)
        queue_size: the maximum number of jobs that can be posted at once
        '''
        self.ranges = dotdict(ranges)

        # both queues hold Job objects, with the result queue having jobs with 'cost' filled out
        self.job_queue = dummy.Queue(maxsize=queue_size)
        self.result_queue = dummy.Queue() # unlimited size

        # note: number of samples may diverge from the number of jobs since a
        # job can result in any number of samples (including 0).
        # the configurations that have been tested (list of `Sample` objects)
        self.samples = []
        self.num_posted_jobs = 0 # number of jobs added to the job queue
        self.num_processed_jobs = 0
        self.processed_job_ids = set() # job.num is added after it is finished

        # written to by _log()
        self.log_record = ''

        self.running = False
        self.proc = None # handle for the process used for asynchronous execution
        self.done = False # flag to gracefully stop asynchronous run
        self.run_start = None # the time at which the last run was started
        self.duration = 0 # total time spent (persists across runs)
        self.poll_interval = 0.05 # number of seconds between each check of the job/results queues during run


    def _log(self, string, newline=True):
        if not isinstance(string, str):
            string = str(string)
        self.log_record += string
        if newline:
            self.log_record += '\n'

# Running Interrupting and Monitoring

    def run(self, run_async=True):
        '''
        run the optimisation procedure, saving each sample along the way and
        keeping track of the current best
        '''
        assert self.proc is None, 'already running'
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

    def wait_for(self, quiet=False):
        if self.proc is None:
            if not quiet:
                print('optimiser already finished')
        else:
            self.proc.join()
            self.proc = None
            if not quiet:
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
                        print(self.log_record)
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
                    print(self.log_record)
        except KeyboardInterrupt:
            clear_output(wait=True)
            if stop_if_interrupted:
                print('interrupt caught: stopping optimisation run')
                self.interrupt()
            else:
                print('interrupt caught, optimiser will continue in the background')

    def _ready_for_next_configuration(self):
        '''
        query the optimiser for whether it is ready to produce another _next_configuration.
        This is useful for when the queue is not full, however the optimiser
        wishes to wait for some of the jobs to finish to inform future samples.

        note: False => _next_configuration will not be called
        '''
        return True

    def _next_configuration(self, job_num):
        '''
        implemented by different optimisation methods
        return the next configuration to try, or None if finished
        job_num: the ID of the job that the configuration will be assigned to
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

    def _add_jobs(self):
        '''
        add as many jobs as possible (limited by the queue size and _ready_for_next_configuration)
        return whether the optimiser is out of configurations
        '''
        while not self.job_queue.full() and self._ready_for_next_configuration():
            job_num = self.num_posted_jobs+1 # job number is 1-based
            config = self._next_configuration(job_num)
            if config is None:
                self._log('out of configurations')
                return True # out of configs: finish processing the outstanding jobs
            job = Job(dotdict(config), job_num)
            self.job_queue.put(job)
            self.num_posted_jobs += 1
            self._log('job {} added to queue'.format(job.num))
        return False # not out of configs

    def _process_jobs(self):
        ''' pop each finished job off of the results queue and store their results '''
        while not self.result_queue.empty():
            job = self.result_queue.get()

            if job.exception is not None:
                self._log('exception raised during evaluation: {}({})\n{}'.format(
                    str(type(job.exception)), str(job.exception), job.exception_string))
                # stop running
                self.done = True
                return

            for i, s in enumerate(job.samples):
                self._log('job={:03}, job_time={}, sample {:02}: config={}, cost={:.2g}{}'.format(
                    job.num, time_string(job.duration), i,
                    config_string(s.config), s.cost,
                    (' (current best)' if self.is_best(s) else '')
                ))

            self.samples.extend(job.samples)
            self.num_processed_jobs += 1 # regardless of how many samples
            self.processed_job_ids.add(job.num)

    def _run(self):
        assert not self.running, 'already running'
        self.running = True
        self.done = False
        self.run_start = time.time()
        self._log('starting optimisation...')
        out_of_configs = False # whether there are no more configurations available from _next_configuration()

        while True:
            # add new jobs
            if not out_of_configs:
                out_of_configs = self._add_jobs()
            # process results of finished jobs
            self._process_jobs()

            outstanding_jobs = self.num_posted_jobs - self.num_processed_jobs
            if self.done or (out_of_configs and outstanding_jobs == 0):
                break

            time.sleep(self.poll_interval)

        outstanding_jobs = self.num_posted_jobs - self.num_processed_jobs
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

    def best_sample(self):
        ''' returns the best known (config, cost) or None if there is none '''
        if len(self.samples) > 0:
            return min(self.samples, key=lambda s: s.cost)
        else:
            return None

    def is_best(self, sample):
        ''' returns whether the given sample is as good as or better than those in self.samples '''
        best = self.best_sample()
        return best is None or sample.cost <= best.cost

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
        best = self.best_sample()
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

        plt.figure(figsize=(16, 8))

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
        re-loaded and continued. The optimiser must not be running for the state
        to be saved.

        Note: this does not save all the state of the optimiser, only the samples
        '''
        # make sure the optimiser stopped is in a good state before saving
        assert self.proc is None and not self.optimiser.running
        assert self.job_queue.empty() and self.result_queue.empty()

        if os.path.isfile(filename):
            raise Exception('File "{}" already exists!'.format(filename))
        else:
            with open(filename, 'w') as f:
                f.write(json.dumps(self._save_dict(), indent=4, cls=NumpyJSONEncoder))

    def load_progress(self, filename):
        '''
        restore the progress of an optimisation run (note: the optimiser must be
        initialised identically to when the samples were saved). The optimiser
        must not be running for the state to be loaded.
        '''
        # make sure the optimiser stopped is in a good state before loading
        assert self.proc is None and not self.optimiser.running
        assert self.job_queue.empty() and self.result_queue.empty()

        with open(filename, 'r') as f:
            self._load_dict(json.loads(f.read()))


    def _save_dict(self):
        '''
        generate the dictionary to be JSON serialised and saved
        (designed to be overridden by derived classes in order to save specialised data)
        '''
        best = self.best_sample()
        if best is None:
            best = Sample({}, inf)
        return {
            'samples' : [(s.config, s.cost) for s in self.samples],
            'num_posted_jobs' : self.num_posted_jobs,
            'num_processed_jobs' : self.num_processed_jobs,
            'processed_job_ids' : self.processed_job_ids,
            'duration' : self.duration,
            'log' : self.log_record,
            'best_sample' : {'config' : best.config, 'cost' : best.cost}
        }

    def _load_dict(self, save):
        '''
        load progress from a dictionary
        (designed to be overridden by derived classes in order to load specialised data)
        '''
        self.samples = [Sample(dotdict(config), cost) for config, cost in save['samples']]
        self.num_posted_jobs = save['num_posted_jobs']
        self.num_processed_jobs = save['num_processed_jobs']
        self.processed_job_ids = save['processed_job_ids']
        self.duration = save['duration']
        self.log_record = save['log']


class GridSearchOptimiser(Optimiser):
    '''
        Grid search optimisation strategy: search with some step size over each
        dimension to find the best configuration
    '''
    def __init__(self, ranges, queue_size=1, order=None):
        '''
        order: the precedence/importance of the parameters (or None for default)
            appearing earlier => more 'primary' (changes more often)
            default could be any order
        '''
        super(self.__class__, self).__init__(ranges, queue_size)
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

    def _next_configuration(self, job_num):
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
    def __init__(self, ranges, queue_size=1, allow_re_tests=False, max_jobs=inf, max_retries=10000):
        '''
        allow_re_tests: whether a configuration should be tested again if it has
            already been tested. This might be desirable if the cost for a
            configuration is not deterministic. However allowing retests removes
            the option of stopping the optimisation process when max_retries is
            exceeded, another method (eg max_jobs) should be used in its place.
        max_jobs: stop taking samples after this number has been reached. By
            default there is no cutoff (so the search will never finish)
        max_retries: (only needed if allow_re_tests=False) the number of times
            to try generating a configuration that hasn't been tested already,
            before giving up (to exhaustively explore the parameter space,
            perhaps finish off with a grid search?)
        '''
        super(self.__class__, self).__init__(ranges, queue_size)
        self.allow_re_tests = allow_re_tests
        self.tested_configurations = set()
        self.max_jobs = max_jobs
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

    def _next_configuration(self, job_num):
        if self.num_processed_jobs >= self.max_jobs:
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



def range_type(range_):
    ''' determine whether the range is 'linear', 'logarithmic', 'arbitrary' or 'constant'
    range_: must be numpy array

    note: range_ must be sorted either ascending or descending to be detected as
        linear or logarithmic

    range types:
        - linear: >2 elements, constant difference
        - logarithmic: >2 elements, constant difference between log(elements)
        - arbitrary: >1 element, not linear or logarithmic (perhaps not numeric)
        - constant: 1 element (perhaps not numeric)
    '''
    if len(range_) == 1:
        return 'constant'
    # 'i' => integer, 'u' => unsigned integer, 'f' => floating point
    elif len(range_) < 2 or range_.dtype.kind not in 'iuf':
        return 'arbitrary'
    else:
        tmp = range_[1:] - range_[:-1] # differences between element i and element i+1
        is_lin = np.all(np.isclose(tmp[0], tmp)) # same difference between each element
        if is_lin:
            return 'linear'
        else:
            tmp = np.log(range_)
            tmp = tmp[1:] - tmp[:-1]
            is_log = np.all(np.isclose(tmp[0], tmp))
            if is_log:
                return 'logarithmic'
            else:
                return 'arbitrary'

def log_uniform(low, high):
    ''' sample a random number in the interval [low, high] distributed logarithmically within that space '''
    return np.exp(np.random.uniform(np.log(low), np.log(high)))


class BayesianOptimisationOptimiser(Optimiser):
    def __init__(self, ranges,
                 acquisition_function='EI',
                 maximising_cost=False, gp_params=None, max_steps=inf,
                 pre_samples=4, ac_num_restarts=10):
        '''
        acquisition_function: the function to determine where to sample next
            either a function or a string with the name of the function (eg 'EI')
        maximising_cost: True => higher cost better. False => lower cost better
        gp_params: parameter dictionary for the Gaussian Process surrogate
            function, None will choose some sensible defaults. (See "sklearn
            gaussian process regressor")
        max_steps: stop taking samples after this number has been reached. By
            default there is no cutoff (so the search will never finish)
        pre_samples: the _minimum_ number of samples to be taken randomly before
            starting Bayesian optimisation (may be more)
        ac_num_restarts: number of restarts during the acquisition function
            optimisation. Higher => more likely to find the optimum of the
            acquisition function.
        '''
        ranges = {param:np.array(range_) for param,range_ in ranges.items()} # numpy arrays are required
        super(self.__class__, self).__init__(ranges, queue_size=1)

        if acquisition_function == 'EI':
            self.acquisition_function_name = 'EI'
            self.acquisition_function = self.expected_improvement
        else:
            self.acquisition_function_name = 'custom acquisition function'
            self.acquisition_function = acquisition_function

        self.maximising_cost = maximising_cost

        if gp_params is None:
            self.gp_params = dict(
                alpha = 1e-5, # larger => more noise. Default = 1e-10
                # the default kernel
                kernel = gp.kernels.RBF(length_scale=1.0, length_scale_bounds="fixed"),
                n_restarts_optimizer = 10,
                # make the mean 0 (theoretically a bad thing, see docs, but can help)
                normalize_y = True,
                copy_X_train = True # make a copy of the training data
            )
        else:
            self.gp_params = gp_params

        self.max_steps = max_steps
        self.pre_samples = pre_samples
        self.ac_num_restarts = ac_num_restarts

        self.params = sorted(self.ranges.keys())
        self.range_types = {param : range_type(range_) for param,range_ in self.ranges.items()}

        if 'arbitrary' in self.range_types.values():
            raise ValueError('arbitrary ranges are not allowed with Bayesian optimisation'.format(param))

        # record the bounds only for the linear and logarithmic ranges
        self.range_bounds = {param: (min(self.ranges[param]), max(self.ranges[param])) for param in self.params}

        for param in self.params:
            low, high = self.range_bounds[param]
            self._log('param "{}": detected type: {}, bounds: [{}, {}]'.format(
                param, self.range_types[param], low, high))

        # not ready for a next configuration until the job with id ==
        # self.wait_until has been processed
        self.wait_for_job = None

        # a log of the Bayesian optimisation steps
        # dict of job number to dict with keys: sx, sy, best_sample, next_x, next_ac
        self.step_log = {}
        self.step_log_keep = 10 # max number of steps to keep

    def _ready_for_next_configuration(self):
        in_pre_phase = self.num_posted_jobs < self.pre_samples
        # all jobs from the pre-phase are finished, need the first Bayesian
        # optimisation sample
        pre_phase_finished = (self.wait_for_job == None and
                              self.num_processed_jobs >= self.pre_samples)
        # finished waiting for the last Bayesian optimisation job to finish
        bayes_job_finished = self.wait_for_job in self.processed_job_ids

        return in_pre_phase or pre_phase_finished or bayes_job_finished

    def trim_step_log(self, keep=-1):
        '''
        remove old steps from the step_log to save space. Keep the N steps with
        largest job numbers
        keep: the number of steps to keep (-1 => keep = self.step_log_keep)
        '''
        if keep == -1:
            keep = self.step_log_keep

        steps = self.step_log.keys()
        if len(steps) > keep:
            removing = sorted(steps, reverse=True)[keep:]
            for step in removing:
                del self.step_log[step]

    def _maximise_acquisition(self, gp_model, best_sample):
        '''
        maximise the acquisition function to obtain the next configuration to test
        returns: config (as a point/numpy array), acquisition value (not negative)

        Important note: This is a _local_ optimisation. This means that _any_
        local optimum is acceptable. There may be some slight variations in the
        function even if it looks flat when plotted, and the 'maximum' sometimes
        rests there, and not at the obvious global maximum. This is fine.
        '''
        # scipy has no maximise function, so instead minimise the negation of the acquisition function
        # reshape(1,-1) => 1 sample (row) with N attributes (cols). Needed because x is passed as shape (N,)
        neg_acquisition_function = lambda x: -self.acquisition_function(
            x.reshape(1,-1), gp_model, best_sample.cost, self.maximising_cost)

        # self.params is ordered. Only provide bounds for the parameters that are
        # included in self.config_to_point
        bounds = [self.range_bounds[param] for param in self.params
                    if self.range_types[param] in ['linear', 'logarithmic']]

        # minimise the negative acquisition function
        best_next_x = None
        best_neg_ac = 0 # negative acquisition function value for best_next_x
        for j in range(self.ac_num_restarts):
            starting_point = self.config_to_point(self._random_config())

            # result is an OptimizeResult object
            result = scipy.optimize.minimize(
                fun=neg_acquisition_function,
                x0=starting_point,
                bounds=bounds,
                method='L-BFGS-B', # Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Bounded
                options=dict(maxiter=15000) # maxiter=15000 is default
            )
            if not result.success:
                self._log('negative acquisition minimisation failed, restarting')
                continue
            #assert result.success, 'minimisation of negative acquisition function failed'

            # result.fun == negative acquisition function evaluated at result.x
            if result.fun < best_neg_ac:
                best_next_x = result.x
                best_neg_ac = result.fun

        # acquisition function optimisation finished:
        # best_next_x = argmax(acquisition_function)

        assert best_next_x is not None, 'all the restarts of the maximisation have failed'

        # reshape to make shape=(1,num_attribs) and negate best_neg_ac to make
        # it the positive acquisition function value
        return best_next_x.reshape(1,-1), -best_neg_ac

    def _next_configuration(self, job_num):
        if self.num_posted_jobs > self.max_steps:
            return None # finished, no more jobs
        if self.num_posted_jobs < self.pre_samples:
            # still in the pre-phase where samples are chosen at random
            # self.pre_samples is the _minimum_ number of samples to take before starting
            config = self._random_config()
            self._log('in pre-phase: choosing random configuration {}/{}'.format(job_num, self.pre_samples))
            return config
        else:
            # Bayesian optimisation
            self.wait_for_job = job_num # do not add a new job until this job has been processed

            # samples converted to points which can be used in calculations
            # shape=(num_samples, num_attribs)
            sx = np.vstack([self.config_to_point(s.config) for s in self.samples])
            # shape=(num_samples, 1)
            sy = np.array([[s.cost] for s in self.samples])

            # best known configuration and the corresponding cost of that configuration
            chooser = max if self.maximising_cost else min
            best_sample = chooser(self.samples, key=lambda s: s.cost)


            gp_model = gp.GaussianProcessRegressor(**self.gp_params)
            gp_model.fit(sx, sy)

            next_x, next_ac = self._maximise_acquisition(gp_model, best_sample)

            if next_x is None:
                self._log('choosing random sample because no sample was found with acquisition value >0')
                next_x = self._random_config()
                next_ac = -1
            # having two samples too close together will 'break' the GP
            # will assume that randomly chosen samples and the pre-samples are unlikely to ever be too close
            elif np.any(np.linalg.norm(next_x - sx) <= 1e-7):
                self._log('choosing random sample to avoid samples being too close')
                next_x = self._random_config()
                next_ac = -1
            else:
                next_x = self.point_to_config(next_x)

            self.step_log[job_num] = dict(
                sx=sx, sy=sy,
                best_sample=best_sample,
                next_x=next_x, next_ac=next_ac
            )
            self.trim_step_log()

            return next_x



    @staticmethod
    def expected_improvement(xs, gp_model, best_cost, maximising_cost=False, xi=0.01):
        ''' expected improvement acquisition function
        xs: array of configurations to evaluate the GP at. shape=(num_samples,num_attribs)
        gp_model: the GP fitted to the past configurations
        best_cost: the (actual) cost of the best known configuration (either smallest or largest depending on maximising_cost)
        maximising_cost: True => higher cost is better, False => lower cost is better
        xi: a parameter >0 for exploration/exploitation trade-off. Larger => more exploration. default of 0.01 is recommended
        '''
        mus, sigmas = gp_model.predict(xs, return_std=True)
        sigmas = make2D(sigmas)

        sf = 1 if maximising_cost else -1   # scaling factor
        diff = sf * (mus - best_cost - xi)  # mu(x) - f(x+) - xi

        with np.errstate(divide='ignore'):
            Zs = diff / sigmas # produces inf where sigmas[i] == 0.0

        EIs = diff * norm.cdf(Zs)  +  sigmas * norm.pdf(Zs)
        EIs[sigmas == 0.0] = 0.0 # replace the infs with 0s

        return EIs


    def _random_config(self):
        '''
        generate a random configuration, sampling each parameter appropriately
        based on its type (uniformly or log-uniformly)
        '''
        config = {}
        for param in self.params:
            type_ = self.range_types[param]
            if type_ == 'linear':
                low, high = self.range_bounds[param]
                config[param] = np.random.uniform(low, high)
            elif type_ == 'logarithmic':
                low, high = self.range_bounds[param]
                config[param] = log_uniform(low, high)
            elif type_ == 'constant':
                config[param] = self.ranges[param][0] # only 1 choice
            else:
                raise ValueError('invalid range type: {}'.format(type_))
        return dotdict(config)

    def config_to_point(self, config):
        '''
        convert a configuration (dictionary of param:val) to a point (numpy
        array) in the parameter space that the Gaussian process uses.

        config: a dictionary of parameter names to values

        note: as a point, constant parameters are ignored

        returns: numpy array with shape=(1,num_attribs)
        '''
        assert set(config.keys()) == set(self.ranges.keys())
        # self.params is sorted
        return np.array([[config[param] for param in self.params
                if self.range_types[param] in ['linear', 'logarithmic']]])

    def point_to_config(self, point):
        '''
        convert a point (numpy array) used by the Gaussian process into a
        configuration (dictionary of param:val).

        note: as a point, constant parameters are ignored

        returns: a configuration dict with all parameters included
        '''
        assert len(point.shape) == 2, 'must be a 2D point'
        assert point.shape[0] == 1, 'only 1 point can be converted at a time'
        config = {}
        pi = 0 # current point index
        for param in self.params: # self.params is sorted
            type_ = self.range_types[param]
            if type_ == 'constant':
                config[param] = self.ranges[param][0] # only 1 choice
            else:
                # type_ = linear or logarithmic
                if pi >= point.shape[1]: raise ValueError('point has too few attributes')
                config[param] = point[0,pi]
                pi += 1

        if pi != point.shape[1]: raise ValueError('point has too many attributes')

        return dotdict(config)



    def _save_dict(self):
        raise NotImplemented # TODO

    def _load_dict(self, save):
        raise NotImplemented # TODO


    def plot_step_slice(self, param, step, true_cost=None, log_ac=False):
        '''
        plot a Bayesian optimisation step, perturbed along a single parameter.

        the 1D case is trivial: the plot is simply the parameter value and the
        corresponding cost and acquisition values.
        in 2D, imagine the surface plot of the two parameters against cost (as
        the height). This plot takes a cross section of that surface along the
        specified axis and passing through the point of the next configuration
        to test to show how the acquisition function varies along that dimension.
        The same holds for higher dimensions but is harder to visualise.

        param: the name of the parameter to perturb to obtain the graph
        bayes_step: the job ID to plot (must be in self.step_log)
        true_cost: true cost function corresponding to self.ranges[param] (None to omit)
        log_ac: whether to display the negative log acquisition function instead
        '''
        assert step in self.step_log.keys(), 'step not recorded in the log'

        s = dotdict(self.step_log[step])
        xs = self.ranges[param]


        gp_model = gp.GaussianProcessRegressor(**self.gp_params)
        gp_model.fit(s.sx, s.sy)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Bayesian Optimisation step {}'.format(step-self.pre_samples), fontsize=14)
        ax1.margins(0.01, 0.1)
        ax2.margins(0.01, 0.1)
        plt.subplots_adjust(hspace=0.3)

        #plt.subplot(2, 1, 1) # nrows, ncols, plot_number
        ax1.set_xlabel('parameter: ' + param)
        ax1.set_ylabel('cost')
        ax1.set_title('Surrogate objective function')

        if true_cost is not None:
            ax1.plot(xs, true_cost, 'k--', label='true cost')

        # plot samples projected onto the `param` axis
        # reshape needed because using x in sx reduces each row to a 1D array
        sample_xs = [self.point_to_config(x.reshape(1,-1))[param] for x in s.sx]
        ax1.plot(sample_xs, s.sy, 'bo', label='samples') #TODO: plot best specially

        # take the next_x configuration and perturb the parameter `param` while leaving the others intact
        # this essentially produces a line through the parameter space to predict uncertainty along
        def perturb(x):
            c = s.next_x.copy()
            c[param] = x
            return self.config_to_point(c)
        points = np.vstack([perturb(x) for x in xs])

        mu, sigma = gp_model.predict(points, return_std=True)
        mu = mu.flatten()

        ax1.plot(xs, mu, 'm-', label='surrogate cost')
        #plt.fill_between(xs, mu - sigma, mu + sigma, alpha=0.3, color='y')
        ax1.fill_between(xs, mu - 3*sigma, mu + 3*sigma, alpha=0.3, color='mediumpurple', label="uncertainty $3\sigma$")
        ax1.axvline(x=s.next_x[param])
        ax1.legend()

        #plt.subplot(2, 1, 2) # nrows, ncols, plot_number
        ax2.set_xlabel('parameter: ' + param)
        ax2.set_ylabel(self.acquisition_function_name)
        ax2.set_title('acquisition function')

        ac = self.acquisition_function(points, gp_model, s.best_sample.cost, self.maximising_cost)
        if log_ac:
            ac[ac == 0.0] = 1e-10
            ac = -np.log(ac)
            label = '-log(acquisition function)'
        else:
            label = 'acquisition function'

        # show close-up on the next sample
        #ax2.set_xlim(s.next_x[param]-1, s.next_x[param]+1)
        #ax2.set_ylim((0.0, s.next_ac))

        ax2.plot(xs, ac, '-', color='g', linewidth=1.0, label=label)
        ax2.fill_between(xs, np.zeros_like(xs), ac.flatten(), alpha=0.3, color='palegreen')

        ax2.axvline(x=s.next_x[param])
        ax2.plot(s.next_x[param], s.next_ac, 'b*', markersize=15, alpha=0.8, label='next sample')
        ax2.legend()

        plt.show()
        return fig



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
                o.write(optimiser.log_record)
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

