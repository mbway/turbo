'''
TODO: module docstring
'''

# fix some of the python2 ugliness
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import time
import json
import os

import threading
# dummy => uses Threads rather than processes
from multiprocessing.dummy import Pool as ThreadPool

from collections import defaultdict
from itertools import groupby

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# for Bayesian Optimisation
import sklearn.gaussian_process as gp
import scipy.optimize
from scipy.stats import norm # Gaussian/normal distribution


# local modules
import optimisation_net as op_net
from optimisation_utils import *
import plot3D


# constants gathered here so that the defaults can be changed easily (eg for testing)
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 9187

CLIENT_TIMEOUT = 1.0 # seconds for the client to wait for a connection before retrying
SERVER_TIMEOUT = 1.0 # seconds for the server to wait for a connection before retrying
NON_CRITICAL_WAIT = 1.0 # seconds to wait after a non-critical network error before retrying


def ON_EXCEPTION(e):
    '''
    Called when a critical exception is raised.
    Can be used to ignore, re-raise, drop into debugger etc
    '''
    print(exception_string(), flush=True)
    #print(exception_string())
    #raise e
    import pdb
    pdb.set_trace()



class Job(object):
    '''
    a job is a single configuration to be tested, it may result in one or
    multiple samples being evaluated.
    '''
    def __init__(self, config, num, setup_duration=0, evaluation_duration=0):
        '''
        config: the configuration to test
        job_num: a unique ID for the job
        setup_duration: the time taken for _next_configuration to generate 'config'
        '''
        self.config = config
        self.num = num
        self.setup_duration = setup_duration
        self.evaluation_duration = evaluation_duration
    def total_time(self):
        return self.setup_duration + self.evaluation_duration

class Sample(object):
    '''
    a sample is a configuration and its corresponding cost as determined by an
    evaluator. Potentially also stores information of interest in 'extra'.
    '''
    def __init__(self, config, cost, extra=None, job_num=None):
        '''
        extra: miscellaneous information about the sample. Not used by the
            optimiser but can be used to store useful information for later. The
            extra information will be saved along with the rest of the sample data.
        job_num: the job number which the sample was evaluated for
        '''
        self.config = config
        self.cost = cost
        self.extra = {} if extra is None else extra
        self.job_num = job_num
    def __repr__(self):
        ''' used in unit tests '''
        if self.extra: # not empty
            return '(config={}, cost={}, extra={})'.format(
                config_string(self.config), self.cost, self.extra)
        else:
            return '(config={}, cost={})'.format(config_string(self.config), self.cost)
    def __eq__(self, other):
        ''' used in unit tests '''
        return (self.config == other.config and
                self.cost == other.cost and
                self.extra == other.extra)


# Details of the network protocol between the optimiser and the evaluator:
# Optimiser sets up a server and listens for clients.
# Each interaction takes the form of
# client -> server: request
# server -> client: response
# (with 4 bytes for the length followed by a JSON string)
# the request may be asking for a new job, or giving back the results of a
# finished job. In the case of a new job: the response is a job or empty if
# there is no job available. In the case of results: the response is a small
# acknowledgement to verify that the results were received.
# If an error occurs then the entire interaction restarts (after a short delay).
# If the job request fails, the optimiser keeps a copy of the job and is able to
# re-send the data until it succeeds. If returning the results fails, the
# evaluator can keep re-sending them until it succeeds, the optimiser will
# discard any duplicates.
#
# If an evaluator accepts a job then they are obliged to complete it. If the
# evaluator crashes or otherwise fails to produce results, the job remains in the
# queue and will never finish processing (which makes the state of the optimiser
# inconsistent and unable to checkpoint).

class Evaluator(object):
    '''
    an evaluator takes configurations and evaluates the cost function on them.
    An evaluator can either be passed as an argument to
    Optimiser.run_sequential, or by running the method: Evaluator.run_client,
    which connects to an optimiser server to receive jobs.
    '''
    def __init__(self):
        self.log_record = ''
        # whether to print to stdout from log()
        self.noisy = False
        self._stop_flag = threading.Event() # is_set() => finish gracefully and stop

    def run_client(self, host=DEFAULT_HOST, port=DEFAULT_PORT, max_jobs=inf,
                   timeout=CLIENT_TIMEOUT):
        self._stop_flag.clear()
        self.log('evaluator client starting...')
        num_jobs = 0

        def should_stop():
            return self.wants_to_stop() or num_jobs >= max_jobs

        def on_error(e):
            time.sleep(NON_CRITICAL_WAIT)

        def message_client(request, should_stop):
            ''' most of the arguments are the same '''
            return op_net.message_client((host, port), timeout, request,
                                         should_stop, on_error)


        try:
            while not should_stop():
                ### Request a job
                self.log('requesting job from {}:{}'.format(host, port))
                job_request = op_net.encode_JSON({'type' : 'job_request'})
                job = message_client(job_request, should_stop)
                if job == None:
                    # stopped attempting to connect because should_stop became True
                    break
                elif job == op_net.empty_msg():
                    self.log('no job available')
                    time.sleep(NON_CRITICAL_WAIT) # not an error, but wait a while
                    continue

                job = op_net.decode_JSON(job)
                self.log('received job: {}'.format(job))

                assert set(job.keys()) == set(['config', 'num', 'setup_duration']), \
                        'malformed job request: {}'.format(job)

                ### Evaluate the job
                results_msg = self._evaluate_job(job)

                ### Return the results
                # don't ever stop trying to connect even if self.wants_to_stop()
                # because after a job is requested, the client must return the
                # results
                ack = message_client(results_msg, op_net.never_stop)
                assert ack is not None and ack == op_net.empty_msg()

                num_jobs += 1
                self.log('results sent successfully')

            if self.wants_to_stop():
                self.log('stopping because of manual shut down')
            elif num_jobs >= max_jobs:
                self.log('stopping because max_jobs reached')
        except Exception as e:
            self.log('Exception raised during client run:\n{}'.format(exception_string()))
            ON_EXCEPTION(e)
        finally:
            self.log('evaluator shut down')

    def _evaluate_job(self, job):
        start_time = time.time()
        job_num = job['num']
        config = dotdict(job['config'])

        self.log('evaluating job {}: config: {}'.format(
            job_num, config_string(config, precise=True)))

        results = self.test_config(config)
        samples = Evaluator.samples_from_test_results(results, config)
        # for JSON serialisation
        samples = [(s.config, s.cost, s.extra) for s in samples]


        self.log('returning results: {}'.format(
            [(config, cost, list(extra.keys()))
                for config, cost, extra in samples]))

        job['evaluation_duration'] = time.time()-start_time # add this field
        results_msg = {
            'type'    : 'job_results',
            'job'     : job,
            'samples' : samples
        }
        results_msg = op_net.encode_JSON(results_msg, encoder=NumpyJSONEncoder)
        return results_msg


    def stop(self):
        ''' signal to the evaluator client to stop and shutdown gracefully '''
        self._stop_flag.set()
    def wants_to_stop(self):
        return self._stop_flag.is_set()

    def log(self, msg, newline=True):
        '''
        write to the evaluator's log (self.log_record), if self.noisy is set
        then the output is also printed to stdout.
        msg: the message to append to the log. If The message is not a string it
            will be converted to one using the `str` constructor.
        newline: whether to append a newline character automatically
        '''
        if not isinstance(msg, str):
            msg = str(msg)
        if newline:
            msg += '\n'
        self.log_record += msg
        if self.noisy:
            print(msg, end='')

    @staticmethod
    def samples_from_test_results(results, config):
        '''
        An evaluator may return several different things from test_config. This
        function converts those results into a standard form: a list of Sample
        objects.
        '''
        # evaluator can return either a cost value, a sample object or a list of sample objects
        if is_numeric(results):
            # evaluator returned cost
            samples = [Sample(config, results)]
        elif (isinstance(results, tuple) and len(results) == 2 and
              is_numeric(results[0]) and isinstance(results[1], dict)):
            # evaluator returned cost and extra as a tuple
            samples = [Sample(config, results[0], results[1])]
        elif isinstance(results, Sample):
            # evaluator returned a single sample
            samples = [results]
        elif isinstance(results, list) and all(isinstance(r, Sample) for r in results):
            # evaluator returned a list of samples
            samples = results
        else:
            raise ValueError('invalid results type from evaluator')
        return samples

    def test_config(self, config):
        '''
        given a configuration, evaluate it and return one of the following:
            - number corresponding to the cost of the configuration
            - a tuple of (cost, 'extra' dict)
                'extra' data can be anything (not used directly by the optimiser
                but useful for storage).
            - Sample object
            - list of Sample objects (of any length including 0)
                returning multiple samples may be advantageous in certain
                situations (such as nested optimisation or iterating over a
                'cheap' parameter range and returning all results at once).

        config: dictionary of parameter names to values
        '''
        raise NotImplementedError


class Optimiser(object):
    '''
    given a search space and a function to call in order to evaluate the cost at
    a given location, find the minimum of the function in the search space.

    Importantly: an expression for the cost function is not required
    '''
    def __init__(self, ranges, maximise_cost=False):
        '''
        ranges: dictionary of parameter names and their ranges (numpy arrays, can be created by np.linspace or np.logspace)
        maximise_cost: True => higher cost is better. False => lower cost is better
        '''
        self.ranges = dotdict(ranges)
        self.maximise_cost = maximise_cost

        # note: number of samples may diverge from the number of jobs since a
        # job can result in any number of samples (including 0).
        # the configurations that have been tested (list of `Sample` objects)
        self.samples = []
        self.num_started_jobs = 0 # number of started jobs
        self.num_finished_jobs = 0 # number of finished jobs
        self.finished_job_ids = set() # job.num is added after it is finished

        self.duration = 0.0 # total time (seconds) spent running (persists across runs)

        # written to by _log()
        self.log_record = ''
        # whether to print to stdout from _log()
        self.noisy = False

        # state useful only when the optimiser is running
        self.run_state = None

        # is_set() => finish gracefully and stop
        self._stop_flag = threading.Event()

        # used with save_when_ready() to signal to a running optimisation to
        # save a checkpoint and where to save it.
        self.checkpoint_filename = './checkpoint.json'
        self._checkpoint_flag = threading.Event()



    def _log(self, msg, newline=True):
        '''
        write to the optimiser's log (self.log_record), if self.noisy is set
        then the output is also printed to stdout.
        msg: the message to append to the log. If The message is not a string it
            will be converted to one using the `str` constructor.
        newline: whether to append a newline character automatically
        '''
        if not isinstance(msg, str):
            msg = str(msg)
        if newline:
            msg += '\n'

        # used to make parsing of separate log entries easier
        #sep = u'\u2063' # 'INVISIBLE SEPARATOR' (U+2063)
        #self.log_record += sep + msg + sep

        self.log_record += msg
        if self.noisy:
            print(msg, end='')

# Running Optimisation

    def _next_job(self):
        '''
        get the next configuration and return a job object for it (or return
        None if there are no more configurations)
        '''
        assert self._ready_for_next_configuration()
        job_num = self.num_started_jobs+1 # job number is 1-based

        # for Bayesian optimisation this may take a little time
        start_time = time.time()
        config = self._next_configuration(job_num)
        setup_duration = time.time()-start_time

        if config is None:
            self._log('out of configurations')
            return None # out of configurations
        else:
            config = dotdict(config)
            self._log('started job {}: config={}'.format(job_num, config_string(config)))
            self.num_started_jobs += 1
            return Job(config, job_num, setup_duration)

    def _process_job_results(self, job, results):
        '''
        check the results of the job are valid, record them and post to the log
        '''
        samples = Evaluator.samples_from_test_results(results, job.config)
        for s in samples:
            s.job_num = job.num
        self.samples.extend(samples)
        self.num_finished_jobs += 1
        self.finished_job_ids.add(job.num)

        self._log('finished job {} in {} (setup: {} evaluation: {}):'.format(
            job.num, time_string(job.total_time()),
            time_string(job.setup_duration), time_string(job.evaluation_duration)
        ))
        for i, s in enumerate(samples):
            self._log('\tsample {:02}: config={}, cost={:.2g}{}'.format(
                i, config_string(s.config, precise=True), s.cost,
                (' (current best)' if self.sample_is_best(s) else '')
            ))

    def _shutdown_message(self, state):
        '''
        post to the log about the run that just finished, including reason for
        the optimiser stopping.
        old_duration: the duration of all runs except for this one
        this_duration: the duration of this run
        '''
        self.duration = state.old_duration + state.run_time()
        self._log('total time taken: {} ({} this run)'.format(
            time_string(self.duration), time_string(state.run_time())))

        if state.exception_caught:
            self._log('optimisation stopped because an exception was thrown.')
        elif self.num_started_jobs - self.num_finished_jobs == 0: # no outstanding jobs
            if state.finished_this_run >= state.max_jobs:
                self._log('optimisation stopped because the maximum number of jobs was exceeded')
            elif self._stop_flag.is_set():
                self._log('optimisation manually shut down gracefully')
            elif state.out_of_configs:
                self._log('optimisation finished (out of configurations).')
        else:
            self._log('error: stopped for an unknown reason with {} outstanding jobs.\n' +
                      '\tMay be in an inconsistent state. (details: stop_flag={}, state={})'.format(
                self.num_started_jobs - self.num_finished_jobs, self._stop_flag.is_set(), state.__dict__))


    def _handle_client(self, msg, state):
        '''
        handle a single interaction with a client
        msg: the message from the client. Assumed not to be None or empty.
        state: the optimiser run_state
        return: the data for the reply message to the client
        '''
        assert 'type' in msg.keys(), 'malformed request: {}'.format(msg)

        def job_msg(job):
            job_dict = {
                'config' : job.config,
                'num' : job.num,
                'setup_duration' : job.setup_duration
            }
            return op_net.encode_JSON(job_dict, encoder=NumpyJSONEncoder)

        if msg['type'] == 'job_request':
            # allowed to re-send the failed job even if generating new ones is
            # not allowed at this time.
            if state.requested_job is not None:
                # re-send the last requested job because the last request did not succeed
                self._log('re-sending the last job since there was an error')
                return job_msg(state.requested_job)

            elif self._wants_to_stop(state):
                self._log('not allowing new jobs (stopping)')
                return op_net.empty_msg()

            elif self._checkpoint_flag.is_set():
                self._log('not allowing new jobs (taking checkpoint)')
                return op_net.empty_msg()

            elif not self._ready_for_next_configuration():
                self._log('not allowing new jobs (not ready)')
                return op_net.empty_msg()

            else:
                job = self._next_job()
                # store in-case it has to be re-sent
                state.requested_job = job
                if job is None:
                    state.out_of_configs = True
                    return op_net.empty_msg() # send empty to signal no more jobs available
                else:
                    state.started_job_ids.add(job.num)
                    state.started_this_run += 1
                    return job_msg(job)

        elif msg['type'] == 'job_results':

            if msg['job']['num'] not in state.started_job_ids:
                self._log('Non-critical error: received results from a job that was not started this run: {}'.format(msg))

            elif msg['job']['num'] in self.finished_job_ids:
                self._log('Non-critical error: received results from a job that is already finished: {}'.format(msg))

            else:
                # during transmission: serialised to tuples
                results = [Sample(dotdict(config), cost, extra)
                        for config, cost, extra in msg['samples']]
                job = Job(**msg['job'])
                self._process_job_results(job, results)
                state.finished_this_run += 1

                # if the results are for the job that the optimiser wasn't sure
                # whether the evaluator received successfully, then evidently it did.
                if state.requested_job is not None and job.num == state.requested_job.num:
                    state.requested_job = None

            # send acknowledgement that the results were received and not
            # malformed (even if they were duplicates and were discarded)
            return op_net.empty_msg()

        else:
            raise Exception('invalid request: {}'.format(msg))

    def _wants_to_stop(self, state):
        '''
        whether the optimiser wants to stop, regardless of whether it can
        (ignores outstanding jobs). In this state the optimiser should not
        accept new job requests.
        '''
        return (self._stop_flag.is_set() or
                state.finished_this_run >= state.max_jobs or
                state.out_of_configs)

    def run_server(self, host=DEFAULT_HOST, port=DEFAULT_PORT, max_jobs=inf,
                   timeout=SERVER_TIMEOUT):
        self._log('starting optimisation server at {}:{}'.format(host, port))
        self._stop_flag.clear()

        class ServerRunState:
            def __init__(self, optimiser, max_jobs):
                self.start_time = time.time()
                self.old_duration = optimiser.duration # duration as-of the start of this run
                self.max_jobs = max_jobs

                self.checkpoint_flag_was_set = False # to only log once

                # count number of jobs started and finished this run
                self.started_this_run = 0
                self.finished_this_run = 0
                self.started_job_ids = set()

                # used to re-send jobs that failed to reach the evaluator. Set
                # back to None after a request-response succeeds.
                self.requested_job = None

                # flags to diagnose the stopping conditions
                self.out_of_configs = False
                self.exception_caught = False

            def run_time(self):
                return time.time() - self.start_time
            def num_outstanding_jobs(self):
                return self.started_this_run - self.finished_this_run
        state = ServerRunState(self, max_jobs)
        self.run_state = state

        def should_stop():
            # deal with checkpoints
            if self._checkpoint_flag.is_set():
                outstanding_count = state.num_outstanding_jobs()
                if outstanding_count > 0 and not state.checkpoint_flag_was_set:
                    self._log('waiting for {} outstanding jobs {} to finish before taking a snapshot'.format(
                        outstanding_count, set_str(state.started_job_ids-self.finished_job_ids)))
                    state.checkpoint_flag_was_set = True
                elif outstanding_count == 0:
                    self._handle_checkpoint()
                    state.checkpoint_flag_was_set = False
                elif outstanding_count < 0:
                    raise ValueError(outstanding_count)

            # return whether the server should stop
            can_stop = state.num_outstanding_jobs() == 0
            return self._wants_to_stop(state) and can_stop

        def handle_request(request):
            request = op_net.decode_JSON(request)
            return self._handle_client(request, state)

        def on_success(request, response):
            # update the timer
            self.duration = state.old_duration + state.run_time()
            request = op_net.decode_JSON(request)
            if request['type'] == 'job_request':
                # the interaction that succeeded was a job request
                state.requested_job = None # job reached evaluator

        try:
            op_net.message_server((host, port), timeout,
                                  should_stop, handle_request, on_success)
        except Exception as e:
            self._log('Exception raised during run:\n{}'.format(exception_string()))
            state.exception_caught = True
            ON_EXCEPTION(e)

        self._handle_checkpoint() # save a checkpoint if signalled to
        self._shutdown_message(state)
        self.run_state = None


    def run_sequential(self, evaluator, max_jobs=inf):
        '''
        run the optimiser with the given evaluator one job after another in the
        current thread

        evaluator: the Evaluator object to use to evaluate the configurations
        max_jobs: the maximum number of jobs to allow for this run (not in total)
        '''
        try:
            self._stop_flag.clear()
            self._log('starting sequential optimisation...')

            class SequentialRunState:
                def __init__(self, optimiser, max_jobs):
                    self.start_time = time.time()
                    self.old_duration = optimiser.duration # duration as-of the start of this run
                    self.max_jobs = max_jobs

                    self.finished_this_run = 0

                    # flags to diagnose the stopping conditions
                    self.out_of_configs = False
                    self.exception_caught = False

                def run_time(self):
                    return time.time() - self.start_time
            state = SequentialRunState(self, max_jobs)
            self.run_state = state


            while not self._wants_to_stop(state):
                self._handle_checkpoint() # save a checkpoint if signalled to
                assert self._ready_for_next_configuration(), \
                    'not ready for the next configuration in sequential optimisation'
                job = self._next_job()
                if job is None:
                    state.out_of_configs = True
                    break

                evaluation_start = time.time()
                results = evaluator.test_config(job.config)
                job.evaluation_duration = time.time() - evaluation_start

                self._process_job_results(job, results)

                state.finished_this_run += 1
                self.duration = state.old_duration + state.run_time()

        except Exception as e:
            self._log('Exception raised during run:\n{}'.format(exception_string()))
            state.exception_caught = True
            ON_EXCEPTION(e)

        self._handle_checkpoint() # save a checkpoint if signalled to
        self._shutdown_message(state)
        self.run_state = None

    def run_multithreaded(self, evaluators, max_jobs=inf):
        '''
        run the optimiser server and run each evaluator in a separate thread

        evaluators: a list of Evaluator objects to run
        max_jobs: the maximum number of jobs to allow for this run (not in total)
        '''
        num_clients = len(evaluators)
        pool = ThreadPool(num_clients)
        try:
            for e in evaluators:
                pool.apply_async(e.run_client)
            self.run_server(max_jobs=max_jobs)
        finally:
            # the pool is the optimiser's responsibility, so make sure it gets
            # shut down gracefully
            for e in evaluators:
                e.stop()
            pool.close()
            pool.join()

    def stop(self):
        ''' signal to the optimiser to stop and shutdown gracefully '''
        self._stop_flag.set()

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
        raise NotImplementedError

    def configuration_space_size(self):
        '''
        return the total number of configurations in the configuration space:
            can be finite or infinite, depending on the type of optimiser.
        '''
        total = 1
        for param_range in self.ranges.values():
            total *= len(param_range)
        return total


# Extracting Results

    def best_sample(self):
        ''' returns the best known (config, cost) or None if there is none '''
        if len(self.samples) > 0:
            chooser = max if self.maximise_cost else min
            return chooser(self.samples, key=lambda s: s.cost)
        else:
            return None

    def sample_is_best(self, sample):
        ''' returns whether the given sample is as good as or better than those in self.samples '''
        best = self.best_sample()
        return (best is None or
                (self.maximise_cost and sample.cost >= best.cost) or
                (not self.maximise_cost and sample.cost <= best.cost))

    def report(self):
        '''
        print a short message about the progress of the optimisation and the
        best configuration so far
        '''
        size = self.configuration_space_size()
        percent_progress = self.num_finished_jobs/float(size)*100.0
        print('{} of {} possible configurations tested ({:.1f}%) taken in {}.'.format(
            self.num_finished_jobs, size, percent_progress, time_string(self.duration)))
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

    def plot_cost_over_time(self, plot_each=True, plot_best=True, true_best=None):
        '''
        plot a line graph showing the progress that the optimiser makes towards
        the optimum as the number of samples increases.
        plot_each: plot the cost of each sample
        plot_best: plot the running-best cost
        true_best: if available: plot a horizontal line for the best possible cost
        '''
        fig, ax = plt.subplots(figsize=(16, 10)) # inches

        xs = range(1, len(self.samples)+1)
        costs = [s.cost for s in self.samples]

        if true_best is not None:
            ax.axhline(true_best, color='black', label='true best')

        if plot_best:
            chooser = max if self.maximise_cost else min
            best_cost = [chooser(costs[:x]) for x in xs]
            ax.plot(xs, best_cost, color='#55a868', label='best cost')

        if plot_each:
            ax.plot(xs, costs, color='#4c72b0', label='cost')

        ax.set_title('Cost Over Time', fontsize=14)
        ax.set_xlabel('samples')
        ax.set_ylabel('cost')
        ax.margins(0.0, 0.15)
        if len(self.samples) < 50:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
        elif len(self.samples) < 100:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
        ax.legend()

        return fig

    def plot_param(self, param_name, plot_boxplot=True, plot_samples=True,
                   plot_means=True, log_axes=(False, False)):
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

    #TODO: instead of 'interactive' pass an argument of how many points to show, then deal with the slider business outside of optimisation.py and plot3D.py
    def scatter_plot(self, param_a, param_b, interactive=True, color_by='cost',
                     log_axes=(False, False, False)):
        '''
            interactive: whether to display a slider for changing the number of
                samples to display
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
                         markersize=4, tooltips=texts, axes_names=axes_names,
                         log_axes=log_axes)

    def surface_plot(self, param_a, param_b, log_axes=(False, False, False)):
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
        costs = np.vectorize(lambda x, y: costs[(x, y)])(xs, ys)
        texts = np.vectorize(lambda x, y: texts[(x, y)])(xs, ys)
        axes_names = ['param: ' + param_a, 'param: ' + param_b, 'cost']
        plot3D.surface3D(xs, ys, costs, tooltips=texts, axes_names=axes_names, log_axes=log_axes)

# Saving and Loading Checkpoints

    def _consistent_and_quiescent(self):
        '''
        Whether the optimiser is in a consistent state (no problems) and
        quiescent (not currently evaluating any jobs).
        '''
        no_outstanding_jobs = (
            self.num_started_jobs == self.num_finished_jobs and
            len(self.finished_job_ids) == self.num_finished_jobs
        )
        return no_outstanding_jobs

    def save_when_ready(self, filename=None):
        '''
        while running, save the progress of the optimiser to a checkpoint at the
        next available opportunity to ensure that the optimiser is quiescent and
        in a consistent state.

        filename: filename to set self.checkpoint_filename to use for the next
                  save and from now on. None to leave unchanged.

        note: will not save until the optimiser starts if the optimiser is not
              currently running
        note: the checkpoint is ready once self._checkpoint_flag is cleared
        '''
        if filename is not None:
            self.checkpoint_filename = filename
        self._checkpoint_flag.set()

    def _handle_checkpoint(self):
        if self._checkpoint_flag.is_set():
            self.save_now()
            self._checkpoint_flag.clear()

    def save_now(self, filename=None):
        '''
        save the progress of the optimiser to a JSON file which can be
        re-loaded and continued.

        This method assumes that the optimiser is quiescent for the duration of
        the save and in a consistent state!

        filename: the location to save the checkpoint to, or None to use
                  self.checkpoint_filename. This filename will be appended with
                  a counter (ascending number) if a file already exists.

        Note: this does not save all the state of the optimiser. Only an
              optimiser initialised identically to this one should load the
              checkpoint.
        '''
        assert self._consistent_and_quiescent()

        # either use the default value or set the default value
        if filename is None:
            filename = self.checkpoint_filename
        else:
            self.checkpoint_filename = filename
        self._log('saving checkpoint to "{}"'.format(filename))

        # keep counting until a filename is available
        if os.path.isfile(filename):
            self._log('While saving a checkpoint: File "{}" already exists!'.format(filename))
            name, ext = os.path.splitext(filename)
            count = 1
            while os.path.isfile(name + str(count) + ext):
                count += 1
            filename = name + str(count) + ext
            self._log('writing to "{}" instead'.format(filename))

        save = json.dumps(self._save_dict(), indent=2, sort_keys=True, cls=NumpyJSONEncoder)
        with open(filename, 'w') as f:
            f.write(save)

        assert self._consistent_and_quiescent()

    def load_checkpoint(self, filename=None):
        '''
        restore the progress of an optimisation run from a saved checkpoint.
        (note: the optimiser must be initialised identically to when the
        checkpoint was saved).

        This method assumes that the optimiser is _not_running_

        filename: the location to load the checkpoint from, or None to use
                  self.checkpoint_filename
        '''
        filename = filename if filename is not None else self.checkpoint_filename

        with open(filename, 'r') as f:
            try:
                save = json.loads(f.read())
            except json.JSONDecodeError as e:
                self._log('exception: invalid checkpoint! Cannot load JSON: {}'.format(exception_string()))
                ON_EXCEPTION(e)
                return
            self._load_dict(save)

        assert self._consistent_and_quiescent()


    def _save_dict(self):
        '''
        generate the dictionary to be JSON serialised and saved
        (designed to be overridden by derived classes in order to save specialised data)
        '''
        best = self.best_sample()
        if best is None:
            best = Sample({}, inf)
        return {
            'samples' : [(s.config, s.cost, JSON_encode_binary(s.extra)) for s in self.samples],
            'num_started_jobs' : self.num_started_jobs, # must be equal to num_finished_jobs
            'num_finished_jobs' : self.num_finished_jobs,
            'finished_job_ids' : list(self.finished_job_ids),
            'duration' : self.duration,
            'log_record' : self.log_record,
            'checkpoint_filename' : self.checkpoint_filename,
            # convenience for viewing the save, but will not be loaded
            'best_sample' : {'config' : best.config, 'cost' : best.cost, 'extra' : best.extra}
        }

    def _load_dict(self, save):
        '''
        load progress from a dictionary
        (designed to be overridden by derived classes in order to load specialised data)
        '''
        self.samples = [Sample(dotdict(config), cost, JSON_decode_binary(extra))
                        for config, cost, extra in save['samples']]
        self.num_started_jobs = save['num_started_jobs']
        self.num_finished_jobs = save['num_finished_jobs']
        self.finished_job_ids = set(save['finished_job_ids'])
        self.duration = save['duration']
        self.log_record = save['log_record']
        self.checkpoint_filename = save['checkpoint_filename']

    def __getstate__(self):
        '''
        used instead of __dict__ for pickling and copying.
        Excludes non-pickleable attributes.
        '''
        return {k:v for k, v in self.__dict__.items() if k not in
                ['_stop_flag', '_checkpoint_flag']}

    def __setstate__(self, state):
        ''' restore from a pickled state (with default values for non-pickleable attributes '''
        self.__dict__ = state
        self._stop_flag = threading.Event()
        self._checkpoint_flag = threading.Event()



class GridSearchOptimiser(Optimiser):
    '''
        Grid search optimisation strategy: search with some step size over each
        dimension to find the best configuration
    '''
    def __init__(self, ranges, maximise_cost=False, order=None):
        '''
        order: the precedence/importance of the parameters (or None for default)
            appearing earlier => more 'primary' (changes more often)
            default could be any order
        '''
        super(GridSearchOptimiser, self).__init__(ranges, maximise_cost)
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
        carry = True # carry flag. (Start True to account for ranges={})
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
    def __init__(self, ranges, maximise_cost=False, allow_re_tests=False, max_retries=10000):
        '''
        allow_re_tests: whether a configuration should be tested again if it has
            already been tested. This might be desirable if the cost for a
            configuration is not deterministic. However allowing retests removes
            the option of stopping the optimisation process when max_retries is
            exceeded, another method (eg max_jobs) should be used in its place.
        max_retries: (only needed if allow_re_tests=False) the number of times
            to try generating a configuration that hasn't been tested already,
            before giving up (to exhaustively explore the parameter space,
            perhaps finish off with a grid search?)
        '''
        super(RandomSearchOptimiser, self).__init__(ranges, maximise_cost)
        self.allow_re_tests = allow_re_tests
        self.tested_configurations = set()
        self.max_retries = max_retries
        self.params = sorted(self.ranges.keys())

    def configuration_space_size(self):
        if self.allow_re_tests:
            return inf
        else:
            return super(RandomSearchOptimiser, self).configuration_space_size()

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
        return '|'.join([str(config[param]) for param in self.params]) # self.params is sorted
        #TODO: try this. convert numpy arrays and lists to strings
        return hash(frozenset(config.items()))

    def _next_configuration(self, job_num):
        c = self._random_config()
        if not self.allow_re_tests:
            attempts = 1
            while self._hash_config(c) in self.tested_configurations and attempts < self.max_retries:
                c = self._random_config()
                attempts += 1
            if attempts >= self.max_retries:
                self._log('max number of retries ({}) exceeded, most of the parameter space must have been explored. Quitting...'.format(self.max_retries))
                return None # done
            #TODO: won't work if evaluator changed the config
            self.tested_configurations.add(self._hash_config(c))
        return c

    def _load_dict(self, save):
        super(RandomSearchOptimiser, self)._load_dict(save)
        if not self.allow_re_tests:
            self.tested_configurations = set([self._hash_config(s.config) for s in self.samples])


class BayesianOptimisationOptimiser(Optimiser):
    '''
    Bayesian Optimisation Strategy:
    1. Sample some random initial points
    2. Using a surrogate function to stand in for the cost function (which is
       unknown) define an acquisition function which estimates how good sampling at
       each point would be. Then maximise this value to obtain the next point to
       sample.
    3. With each obtained sample, the accuracy of the surrogate function with
       respect to the true cost function increases, which in turn allows better
       choices of next samples to test.

    Technicalities:
    - If the next suggested sample is too close to an existing sample, then this
      would not give much new information, so instead the next point is chosen
      randomly. This results in a better picture of the cost function which in
      turn may make new points look more desirable, allowing the algorithm to
      progress and not get stuck in local optima
    - Bayesian optimisation is inherently a serial process. To parallelise the
      algorithm, the results for ongoing jobs are estimated by trusting the
      expected value of the surrogate function. After the job has finished the
      correct value is used, but in the meantime it allows more jobs to be
      started based on the estimated results.
    - Logarithmically spaced parameters (where the order of magnitude is more
      important than the absolute value) must be sampled log-uniformly rather than
      uniformly.
    - Discrete parameters are not compatible with Bayesian optimisation since
      the chosen surrogate function is a Gaussian Process, which fits real-valued
      data, and the acquisition function also relies on real-number calculations.
      Modifications to the algorithm may allow for discrete valued parameters however.
    '''
    def __init__(self, ranges, maximise_cost=False,
                 acquisition_function='UCB', acquisition_function_params=None,
                 gp_params=None, pre_samples=4, ac_max_params=None,
                 close_tolerance=1e-5, allow_parallel=True):
        '''
        acquisition_function: the function to determine where to sample next
            either a function or a string with the name of the function (eg 'EI')
        acquisition_function_params: a dictionary of parameter names and values
            to be passed to the acquisition function. (see specific acquisition
            function for details on what parameters it takes)
        gp_params: parameter dictionary for the Gaussian Process surrogate
            function, None will choose some sensible defaults. (See "sklearn
            gaussian process regressor")
        pre_samples: the number of jobs (not samples, despite the name) to run
            before starting Bayesian optimisation
        ac_max_params: parameters for maximising the acquisition function. None
            to use default values, or a dictionary  with integer values for:
                'num_random': number of random samples to take when maximising
                    the acquisition function
                'num_restarts': number of restarts to use for the gradient-based
                    maximisation of the acquisition function
                larger values for each of these parameters means that the
                optimisation is more likely to find the global maximum of the
                acquisition function, however the optimisation becomes more
                costly (however this will probably be insignificant in
                comparison to the time to evaluate a configuration).
                0 may be passed for one of the two parameters to ignore that
                step of the optimisation.
        close_tolerance: in some situations Bayesian optimisation may get stuck
            on local optima and will continue to sample points roughly in the
            same location. When this happens the GP can break (as input values
            must be unique within some tolerance). It is also a waste of
            resources to sample lots of times in a very small neighbourhood.
            Instead, when the next sample is to be 'close' to any of the points
            sampled before (ie squared Euclidean distance <= close_tolerance),
            sample a random point instead.
        allow_parallel: whether to hypothesise about the results of ongoing jobs
            in order to start another job in parallel. (useful when running a
            server with multiple client evaluators).
        '''
        ranges = {param:np.array(range_) for param, range_ in ranges.items()} # numpy arrays are required
        super(BayesianOptimisationOptimiser, self).__init__(ranges, maximise_cost)

        self.acquisition_function_params = ({} if acquisition_function_params is None
                                            else acquisition_function_params)
        ac_param_keys = set(self.acquisition_function_params.keys())

        if acquisition_function == 'PI':
            self.acquisition_function_name = 'PI'
            self.acquisition_function = self.probability_of_improvement
            # <= is subset. Not all params must be provided, but those given must be valid
            assert ac_param_keys <= set(['xi']), 'invalid acquisition function parameters'

        elif acquisition_function == 'EI':
            self.acquisition_function_name = 'EI'
            self.acquisition_function = self.expected_improvement
            # <= is subset. Not all params must be provided, but those given must be valid
            assert ac_param_keys <= set(['xi']), 'invalid acquisition function parameters'

        elif acquisition_function == 'UCB':
            self.acquisition_function_name = 'UCB'
            self.acquisition_function = self.upper_confidence_bound
            # <= is subset. Not all params must be provided, but those given must be valid
            assert ac_param_keys <= set(['kappa']), 'invalid acquisition function parameters'

        elif callable(acquisition_function):
            self.acquisition_function_name = 'custom acquisition function'
            self.acquisition_function = acquisition_function
        else:
            raise ValueError('invalid acquisition_function')

        if gp_params is None:
            self.gp_params = dict(
                alpha = 1e-10, # larger => more noise. Default = 1e-10
                # nu=1.5 assumes the target function is once-differentiable
                kernel = 1.0 * gp.kernels.Matern(nu=1.5) + gp.kernels.WhiteKernel(),
                #kernel = 1.0 * gp.kernels.RBF(),
                n_restarts_optimizer = 5,
                # make the mean 0 (theoretically a bad thing, see docs, but can help)
                # with the constant offset in the kernel this shouldn't be required
                #normalize_y = True,
                copy_X_train = True # whether to make a copy of the training data (in-case it is modified)
            )
        else:
            self.gp_params = gp_params

        if ac_max_params is None:
            self.ac_max_params = dotdict({'num_random' : 10000, 'num_restarts' : 10})
        else:
            assert set(ac_max_params.keys()) <= set(['num_random', 'num_restarts'])
            # convert each parameter to an integer
            self.ac_max_params = dotdict({k:int(v) for k, v in ac_max_params.items()})
            # at least one of the methods has to be used (non-zero)
            assert self.ac_max_params.num_random > 0 or self.ac_max_params.num_restarts > 0

        assert pre_samples > 1, 'not enough pre-samples'
        self.pre_samples = pre_samples
        self.close_tolerance = close_tolerance

        self.params = sorted(self.ranges.keys())
        self.range_types = {param : range_type(range_) for param, range_ in self.ranges.items()}

        if RangeType.Arbitrary in self.range_types.values():
            bad_ranges = [param for param, type_ in self.range_types.items()
                          if type_ == RangeType.Arbitrary]
            raise ValueError('arbitrary ranges: {} are not allowed with Bayesian optimisation'.format(bad_ranges))
        elif not self.ranges:
            raise ValueError('empty ranges not allowed with Bayesian optimisation')

        # record the bounds only for the linear and logarithmic ranges
        self.range_bounds = {param: (min(self.ranges[param]), max(self.ranges[param])) for param in self.params}

        for param in self.params:
            low, high = self.range_bounds[param]
            self._log('param "{}": detected type: {}, bounds: [{}, {}]'.format(
                param, self.range_types[param], low, high))

        # Only provide bounds for the parameters that are included in
        # self.config_to_point. Provide the log(lower), log(upper) bounds for
        # logarithmically spaced ranges.
        # IMPORTANT: use range_bounds when dealing with configs and point_bounds
        # when dealing with points
        self.point_bounds = []
        for param in self.params: # self.params is ordered
            type_ = self.range_types[param]
            low, high = self.range_bounds[param]
            if type_ == RangeType.Linear:
                self.point_bounds.append((low, high))
            elif type_ == RangeType.Logarithmic:
                self.point_bounds.append((np.log(low), np.log(high)))


        self.allow_parallel = allow_parallel
        # not ready for a next configuration until the job with id ==
        # self.wait_until has been processed. Not used when allow_parallel.
        self.wait_for_job = None
        # estimated samples for ongoing jobs. list of (job_num, Sample). Not
        # used when (not allow_parallel)
        self.hypothesised_samples = []

        # a log of the Bayesian optimisation steps
        # dict of job number to dict with values:
        #
        # gp: trained Gaussian process
        # sx, sy: numpy arrays corresponding to points of samples taken thus far
        # hx, hy: numpy arrays corresponding to _hypothesised_ points of ongoing
        #   jobs while the step was being calculated
        # best_sample: best sample so far
        # next_x: next config to test
        # next_ac: value of acquisition function evaluated at next_x (if not
        #    chosen_at_random)
        # chosen_at_random: whether next_x was chosen randomly rather than by
        #   maximising the acquisition function
        # argmax_acquisition: the next config to test (as a point) as determined
        #   by maximising the acquisition function (different to next_x if
        #   chosen_at_random)
        self.step_log = {}
        self.step_log_keep = 100 # max number of steps to keep

    def configuration_space_size(self):
        return inf

    def _ready_for_next_configuration(self):
        if self.allow_parallel:
            # wait for all of the pre-phase samples to be taken before starting
            # the Bayesian optimisation steps. Otherwise always ready.
            waiting_for_pre_phase = (self.num_started_jobs >= self.pre_samples and
                                     self.num_finished_jobs < self.pre_samples)
            return not waiting_for_pre_phase
        else:
            in_pre_phase = self.num_started_jobs < self.pre_samples
            # all jobs from the pre-phase are finished, need the first Bayesian
            # optimisation sample
            pre_phase_finished = (self.wait_for_job is None and
                                  self.num_finished_jobs >= self.pre_samples)
            # finished waiting for the last Bayesian optimisation job to finish
            bayes_job_finished = self.wait_for_job in self.finished_job_ids

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

    def _maximise_acquisition(self, gp_model, best_cost):
        '''
        maximise the acquisition function to obtain the next configuration to test
        gp_model: a Gaussian process trained on the samples taken so far
        best_cost: the cost function value of the best known sample so far
        returns: config (as a point/numpy array), acquisition value (not negative)
            or None,0 if all attempts to optimise the acquisition_function fail

        Important note: This is a _local_ optimisation. This means that _any_
        local optimum is acceptable. There may be some slight variations in the
        function even if it looks flat when plotted, and the 'maximum' sometimes
        rests there, and not at the obvious global maximum. This is fine.
        '''
        # Maximise the acquisition function by random sampling
        if self.ac_max_params.num_random > 0:
            random_points = self._random_config_points(self.ac_max_params.num_random)
            random_ac = self.acquisition_function(random_points, gp_model,
                            self.maximise_cost, best_cost, **self.acquisition_function_params)
            best_random_i = random_ac.argmax()

            # keep track of the current best
            best_next_x = make2D_row(random_points[best_random_i])
            best_neg_ac = -random_ac[best_random_i] # negative acquisition function value for best_next_x
        else:
            best_next_x = None
            best_neg_ac = inf

        # Maximise the acquisition function by minimising the negative acquisition function

        # scipy has no maximise function, so instead minimise the negation of the acquisition function
        # reshape(1,-1) => 1 sample (row) with N attributes (cols). Needed because x is passed as shape (N,)
        # unpacking the params dict is harmless if the dict is empty
        neg_acquisition_function = lambda x: -self.acquisition_function(
            make2D_row(x), gp_model, self.maximise_cost, best_cost,
            **self.acquisition_function_params)

        if self.ac_max_params.num_restarts > 0:
            # it doesn't matter if these points are close to any existing samples
            starting_points = self._random_config_points(self.ac_max_params.num_restarts)
            if self.ac_max_params.num_random > 0:
                # see if gradient-based optimisation can improve upon the best
                # randomly chosen sample.
                starting_points = np.vstack([best_next_x, starting_points])

        # num_restarts may be 0 in which case this step is skipped
        for j in range(starting_points.shape[0]):
            starting_point = make2D_row(starting_points[j])

            # result is an OptimizeResult object
            # note: nested WarningCatchers work as expected
            log_warning = lambda warn: self._log('warning when maximising the acquisition function: {}'.format(warn))
            with WarningCatcher(log_warning):
                result = scipy.optimize.minimize(
                    fun=neg_acquisition_function,
                    x0=starting_point,
                    bounds=self.point_bounds,
                    method='L-BFGS-B', # Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Bounded
                    options=dict(maxiter=15000) # maxiter=15000 is default
                )
            if not result.success:
                self._log('restart {}/{} of negative acquisition minimisation failed'.format(
                    j, starting_points.shape[0]))
                continue

            # result.fun == negative acquisition function evaluated at result.x
            if result.fun < best_neg_ac:
                best_next_x = result.x # shape=(num_attribs,)
                best_neg_ac = result.fun # shape=(1,1)

        # acquisition function optimisation finished:
        # best_next_x = argmax(acquisition_function)

        if best_next_x is None:
            self._log('all attempts at acquisition function maximisation failed')
            return None, 0
        else:
            # reshape to make shape=(1,num_attribs) and negate best_neg_ac to make
            # it the positive acquisition function value
            best_next_x = make2D_row(best_next_x)
            # ensure that the chosen value lies within the bounds (which may not
            # be the case due to floating point error)
            best_next_x = np.clip(best_next_x, [lo for lo, hi in self.point_bounds], [hi for lo, hi in self.point_bounds])
            return best_next_x, -np.asscalar(best_neg_ac)

    def _bayes_step(self, job_num):
        '''
        generate the next configuration to test using Bayesian optimisation
        '''
        # samples converted to points which can be used in calculations
        # shape=(num_samples, num_attribs)
        sx = np.vstack([self.config_to_point(s.config) for s in self.samples])
        # shape=(num_samples, 1)
        sy = np.array([[s.cost] for s in self.samples])

        # if running parallel jobs: add hypothesised samples to the data set to
        # fit the surrogate cost function to. If running serial: mark this job
        # as having to finish before proceeding
        if self.allow_parallel:
            # remove samples whose jobs have since finished
            self.hypothesised_samples = [(job_num, s) for job_num, s in self.hypothesised_samples
                                         if job_num not in self.finished_job_ids]

            if len(self.hypothesised_samples) > 0:
                hx = np.vstack([self.config_to_point(s.config)
                                for job_num, s in self.hypothesised_samples])
                hy = np.array([[s.cost] for job_num, s in self.hypothesised_samples])
            else:
                hx = np.empty(shape=(0, sx.shape[1]))
                hy = np.empty(shape=(0, 1))
        else:
            self.wait_for_job = job_num # do not add a new job until this job has been processed

            hx = np.empty(shape=(0, sx.shape[1]))
            hy = np.empty(shape=(0, 1))

        # combination of the true samples (sx, sy) and the hypothesised samples
        # (hx, hy) if there are any
        xs = np.vstack([sx, hx])
        ys = np.vstack([sy, hy])

        # setting up a new model each time shouldn't be too wasteful and it
        # has the benefit of being easily reproducible (eg for plotting)
        # because the model is definitely 'clean' each time. In my tests,
        # there was no perceptible difference in timing.
        gp_model = gp.GaussianProcessRegressor(**self.gp_params)
        # the optimiser may fail for various reasons, one being that 'the
        # function is dominated by noise'. In one example I looked at the GP was
        # still sensible even with the warning, so ignoring it should be fine.
        # Worst case scenario is that a few bad samples are taken before the GP
        # sorts itself out again.
        # warnings may be triggered for fitting or predicting
        log_warning = lambda warn: self._log('warning with the gp: {}'.format(warn))
        #log_warning = lambda warn: print(warn)
        with WarningCatcher(log_warning):
            # NOTE: fitting only optimises _certain_ kernel parameters with given
            # bounds, see gp_model.kernel_.theta for the optimised kernel
            # parameters.
            # NOTE: RBF(...) has NO parameters to optimise, however 1.0 * RBF(...) does!
            gp_model.fit(xs, ys)

            # gp_model.kernel_ is a copy of gp_model.kernel with the parameters optimised
            #self._log('GP params={}'.format(gp_model.kernel_.theta))

            # best known configuration and the corresponding cost of that configuration
            best_sample = self.best_sample()

            next_x, next_ac = self._maximise_acquisition(gp_model, best_sample.cost)

            # next_x as chosen by the acquisition function maximisation (for the step log)
            argmax_acquisition = next_x


        # maximising the acquisition function failed
        if next_x is None:
            self._log('choosing random sample because maximising acquisition function failed')
            next_x = self._unique_random_config(different_from=xs, num_attempts=1000)
            next_ac = 0
            chosen_at_random = True
        # acquisition function successfully maximised, but the resulting configuration would break the GP.
        # having two samples too close together will 'break' the GP
        elif close_to_any(next_x, xs, self.close_tolerance):
            self._log('argmax(acquisition function) too close to an existing sample: choosing randomly instead')
            #TODO: what about if the evaluator changes the config and happens to create a duplicate point?
            next_x = self._unique_random_config(different_from=xs, num_attempts=1000)
            next_ac = 0
            chosen_at_random = True
        else:
            next_x = self.point_to_config(next_x)
            chosen_at_random = False

        if self.allow_parallel:
            # use the GP to estimate the cost of the configuration, later jobs
            # can use this guess to determine where to sample next
            with WarningCatcher(log_warning):
                est_cost = gp_model.predict(self.config_to_point(next_x))
            est_cost = np.asscalar(est_cost) # ndarray of shape=(1,1) is returned from predict()
            self.hypothesised_samples.append((job_num, Sample(next_x, est_cost)))


        self.step_log[job_num] = dict(
            gp=gp_model,
            sx=sx, sy=sy,
            hx=hx, hy=hy,
            best_sample=best_sample,
            next_x=next_x, next_ac=next_ac, # chosen_at_random => next_ac=0
            chosen_at_random=chosen_at_random,
            argmax_acquisition=argmax_acquisition # different to next_x when chosen_at_random
        )
        self.trim_step_log()

        return next_x


    def _next_configuration(self, job_num):
        if self.num_started_jobs < self.pre_samples:
            # still in the pre-phase where samples are chosen at random
            # make sure that each configuration is sufficiently different from all previous samples
            if len(self.samples) == 0:
                config = self._random_config()
            else:
                sx = np.vstack([self.config_to_point(s.config) for s in self.samples])
                config = self._unique_random_config(different_from=sx, num_attempts=1000)

            if config is None: # could not find a unique configuration
                return None # finished
            else:
                self._log('in pre-phase: choosing random configuration {}/{}'.format(job_num, self.pre_samples))
                return config
        else:
            # Bayesian optimisation
            return self._bayes_step(job_num)


    @staticmethod
    def probability_of_improvement(xs, gp_model, maximise_cost, best_cost, xi=0.01):
        r'''
        This acquisition function is similar to EI
        $$PI(\mathbf x)\quad=\quad\mathrm P\Big(f(\mathbf x)\ge f(\mathbf x^+)+\xi\Big)\quad=\quad\Phi\left(\frac{\mu(\mathbf x)-f(\mathbf x^+)-\xi}{\sigma(\mathbf x)}\right)$$
        '''
        mus, sigmas = gp_model.predict(xs, return_std=True)
        sigmas = make2D(sigmas)

        sf = 1 if maximise_cost else -1   # scaling factor
        diff = sf * (mus - best_cost - xi)  # mu(x) - f(x+) - xi

        with np.errstate(divide='ignore'):
            Zs = diff / sigmas # produces Zs[i]=inf for all i where sigmas[i]=0.0
        Zs[sigmas == 0.0] = 0.0 # replace the infs with 0s

        return norm.cdf(Zs)

    @staticmethod
    def expected_improvement(xs, gp_model, maximise_cost, best_cost, xi=0.01):
        r''' expected improvement acquisition function
        xs: array of points to evaluate the GP at. shape=(num_points, num_attribs)
        gp_model: the GP fitted to the past configurations
        maximise_cost: True => higher cost is better, False => lower cost is better
        best_cost: the (actual) cost of the best known configuration (either
            smallest or largest depending on maximise_cost)
        xi: a parameter >0 for exploration/exploitation trade-off. Larger =>
            more exploration. The default value of 0.01 is recommended.

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
        sigmas = make2D(sigmas)

        sf = 1 if maximise_cost else -1   # scaling factor
        diff = sf * (mus - best_cost - xi)  # mu(x) - f(x+) - xi

        with np.errstate(divide='ignore'):
            Zs = diff / sigmas # produces Zs[i]=inf for all i where sigmas[i]=0.0

        EIs = diff * norm.cdf(Zs)  +  sigmas * norm.pdf(Zs)
        EIs[sigmas == 0.0] = 0.0 # replace the infs with 0s

        return EIs

    #TODO: should rename? make CB/UCB/LCB all refer to this function
    @staticmethod
    def upper_confidence_bound(xs, gp_model, maximise_cost, best_cost, kappa=1.0):
        r'''
        upper confidence bound when maximising, lower confidence bound when minimising
        $$\begin{align*}
        UCB(\mathbf x)&=\mu(\mathbf x)+\kappa\sigma(\mathbf x)\\
        LCB(\mathbf x)&=\mu(\mathbf x)-\kappa\sigma(\mathbf x)
        \end{align*}$$

        xs: array of points to evaluate the GP at. shape=(num_points, num_attribs)
        gp_model: the GP fitted to the past configurations
        maximise_cost: True => higher cost is better, False => lower cost is better
        best_cost: not used in this acquisition function
        kappa: parameter which controls the trade-off between exploration and
            exploitation. Larger values favour exploration more. (geometrically,
            the uncertainty is scaled more so is more likely to look better than
            known good locations)
        '''
        mus, sigmas = gp_model.predict(xs, return_std=True)
        sigmas = make2D(sigmas)
        sf = 1 if maximise_cost else -1   # scaling factor
        return sf * (mus + sf * kappa * sigmas)

    def _unique_random_config(self, different_from, num_attempts=1000):
        ''' generate a random config which is different from any configurations tested in the past
        different_from: numpy array of points shape=(num_points, num_attribs)
            which the resulting configuration must not be identical to (within a
            very small tolerance). This is because identical configurations
            would break the GP (it models a function, so each 'x' corresponds to
            exactly one 'y')
        num_attempts: number of re-tries before giving up
        returns: a random configuration, or None if num_attempts is exceeded
        '''
        for _ in range(num_attempts):
            config = self._random_config()
            if not close_to_any(self.config_to_point(config), different_from, tol=self.close_tolerance):
                return config
        self._log('could not find a random configuration sufficiently different from previous samples, parameter space must be (almost) fully explored.')
        return None

    def _random_config(self):
        '''
        generate a random configuration, sampling each parameter appropriately
        based on its type (uniformly or log-uniformly)
        '''
        config = {}
        for param in self.params:
            type_ = self.range_types[param]

            if type_ == RangeType.Linear:
                low, high = self.range_bounds[param]
                config[param] = np.random.uniform(low, high)

            elif type_ == RangeType.Logarithmic:
                low, high = self.range_bounds[param]
                # not exponent, but a value in the original space
                config[param] = log_uniform(low, high)

            elif type_ == RangeType.Constant:
                config[param] = self.ranges[param][0] # only 1 choice

            else:
                raise ValueError('invalid range type: {}'.format(type_))

        return dotdict(config)

    def _random_config_points(self, num_points):
        '''
        generate an array of vertically stacked configuration points equivalent to self.config_to_point(self._random_config())
        num_points: number of points to generate (height of output)
        returns: numpy array with shape=(num_points,num_attribs)
        '''
        cols = [] # generate points column/parameter-wise
        for param in self.params: # self.params is sorted
            type_ = self.range_types[param]
            low, high = self.range_bounds[param]

            if type_ == RangeType.Linear:
                cols.append(np.random.uniform(low, high, size=(num_points, 1)))
            elif type_ == RangeType.Logarithmic:
                # note: NOT log_uniform because that computes a value in the
                # original space but distributed logarithmically. We are looking
                # for just the exponent here, not the value.
                cols.append(np.random.uniform(np.log(low), np.log(high), size=(num_points, 1)))
        return np.hstack(cols)


    def config_to_point(self, config):
        '''
        convert a configuration (dictionary of param:val) to a point (numpy
        array) in the parameter space that the Gaussian process uses.

        As a point, constant parameters are ignored, and values from logarithmic
        ranges are the exponents of the values. ie a value of 'n' as a point
        corresponds to a value of e^n as a configuration.

        config: a dictionary of parameter names to values
        returns: numpy array with shape=(1,number of linear or logarithmic parameters)
        '''
        assert set(config.keys()) == set(self.ranges.keys())
        elements = []
        for param in self.params: # self.params is sorted
            type_ = self.range_types[param]
            if type_ == RangeType.Linear:
                elements.append(config[param])
            elif type_ == RangeType.Logarithmic:
                elements.append(np.log(config[param]))
        return np.array([elements])

    def _index_for_param(self, param):
        '''
        return the index of the given parameter name in a point created by
        config_to_point(). None if the parameter is not present.
        '''
        assert param in self.params
        i = 0
        for p in self.params: # self.params is sorted
            if param == p:
                return i
            if self.range_types[p] in [RangeType.Linear, RangeType.Logarithmic]:
                i += 1
        return None

    def point_to_config(self, point):
        '''
        convert a point (numpy array) used by the Gaussian process into a
        configuration (dictionary of param:val).

        As a point, constant parameters are ignored, and values from logarithmic
        ranges are the exponents of the values. ie a value of 'n' as a point
        corresponds to a value of e^n as a configuration.

        returns: a configuration dict with all parameters included
        '''
        assert len(point.shape) == 2, 'must be a 2D point'
        assert point.shape[0] == 1, 'only 1 point can be converted at a time'
        config = {}
        pi = 0 # current point index
        for param in self.params: # self.params is sorted
            type_ = self.range_types[param]

            if type_ == RangeType.Constant:
                config[param] = self.ranges[param][0] # only 1 choice
            else:
                if pi >= point.shape[1]:
                    raise ValueError('point has too few attributes')
                val = point[0, pi]
                pi += 1

                if type_ == RangeType.Linear:
                    config[param] = val
                elif type_ == RangeType.Logarithmic:
                    config[param] = np.exp(val)

        if pi != point.shape[1]:
            raise ValueError('point has too many attributes')

        return dotdict(config)

    def _consistent_and_quiescent(self):
        # super class checks general properties
        sup = super(BayesianOptimisationOptimiser, self)._consistent_and_quiescent()
        # either not waiting for a job, or waiting for a job which has finished
        not_waiting = (
            self.wait_for_job is None or
            self.wait_for_job in self.finished_job_ids
        )
        # either there are no hypothesised samples, or they are for jobs which
        # have already finished and just haven't been removed yet.
        no_hypotheses = (
            len(self.hypothesised_samples) == 0 or
            all(job_id in self.finished_job_ids for job_id, sample in
                 self.hypothesised_samples)
        )
        # 1. make sure that the arrays are exactly 2 dimensional
        # 2. make sure that there are equal numbers of rows in the samples (and
        # hypothesised samples) xs and ys (ie every x must have a y).
        # 3. make sure that there is only 1 output attribute for each y
        eq_rows = lambda a, b: a.shape[0] == b.shape[0]
        is_2d = lambda x: len(x.shape) == 2
        try:
            step_log_valid = (
                all(all([
                    is_2d(step['sx']), is_2d(step['sy']),
                    is_2d(step['hx']), is_2d(step['hy']),
                    np.isscalar(step['next_ac']), is_2d(step['argmax_acquisition']),

                    eq_rows(step['sx'], step['sy']),
                    eq_rows(step['hx'], step['hy']),

                    step['hy'].shape[1] == step['sy'].shape[1] == 1,

                ]) for job_id, step in self.step_log.items())
            )
        except IndexError:
            # one of the shapes has <2 elements
            step_log_valid = False

        return sup and not_waiting and no_hypotheses and step_log_valid

    def _save_dict(self):
        save = super(BayesianOptimisationOptimiser, self)._save_dict()

        # hypothesised samples and wait_for_job are not needed to be saved since
        # the optimiser should be quiescent

        # save GP models compressed and together since they contain a lot of
        # redundant information and are not human readable anyway.
        # for a test run with ~40 optimisation steps, naive storage (as part of
        # step log): 1MB, separate with compression: 200KB
        step_log = {}
        gps = {}
        for n, s in self.step_log.items():
            step_log[n] = {k:v for k, v in s.items() if k != 'gp'}
            gps[n] = s['gp']

        # convert the step log (dict) to an array of tuples sorted by job_num
        step_log_copy = [(k, v.copy()) for k, v in step_log.items()]
        save['step_log'] = sorted(step_log_copy, key=lambda s: s[0])
        for job_num, step in save['step_log']:
            bs = step['best_sample']
            step['best_sample'] = (bs.config, bs.cost, JSON_encode_binary(bs.extra))

        save['gps'] = JSON_encode_binary(gps)
        return save

    def _load_dict(self, save):
        super(BayesianOptimisationOptimiser, self)._load_dict(save)
        self.step_log = save['step_log']

        gps = JSON_decode_binary(save['gps'])

        for n, s in self.step_log:
            s['gp'] = gps[n]
            config, cost, extra = s['best_sample']
            s['best_sample'] = Sample(dotdict(config), cost, JSON_decode_binary(extra))

            # convert lists back to numpy arrays
            for key in ['sx', 'sy', 'hx', 'hy', 'argmax_acquisition']:
                s[key] = np.array(s[key])
            # ensure the shapes are correct
            # sx,sy will never be empty since there will always be pre-samples
            # hx,hy may be empty
            if s['hx'].size == 0:
                s['hx'] = np.empty(shape=(0, s['sx'].shape[1]))
                s['hy'] = np.empty(shape=(0, 1))

        # convert list of tuples to dictionary
        self.step_log = dict(self.step_log)

        # reset any progress attributes
        self.wait_for_job = None
        self.hypothesised_samples = []


    def _points_vary_one(self, config, param, xs):
        '''
        points generated by fixing all but one parameter to match the given
        config, and varying the remaining parameter.
        config: the configuration to base the points on
        param: the name of the parameter to vary
        xs: the values to provide in place of config[param]
        '''
        assert len(xs.shape) == 1 # not 2D
        # create many duplicates of the point version of the given configuration
        points = np.repeat(self.config_to_point(config), len(xs), axis=0)
        param_index = self._index_for_param(param) # column to swap out for xs
        points[:,param_index] = xs
        return points

    #TODO: rename to plot1D
    def plot_step_slice(self, param, step, true_cost=None, log_ac=False,
                        n_sigma=2, gp_through_all=True):
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
        true_cost: true cost function (or array of pre-computed cost values corresponding to self.ranges[param]) (None to omit)
        log_ac: whether to display the negative log acquisition function instead
        n_sigma: the number of standard deviations from the mean to plot the
            uncertainty confidence interval.
            Note 1=>68%, 2=>95%, 3=>99% (for a normal distribution, which this is)
        gp_through_all: whether to plot a gp prediction through every sample or
            just through the location of the next point to be chosen
        '''
        assert step in self.step_log.keys(), 'step not recorded in the log'

        type_ = self.range_types[param]
        assert type_ in [RangeType.Linear, RangeType.Logarithmic]
        # whether the range of the focused parameter is logarithmic
        is_log = type_ == RangeType.Logarithmic

        s = dotdict(self.step_log[step])
        all_xs = self.ranges[param]

        # combination of the true samples (sx, sy) and the hypothesised samples
        # (hx, hy) if there are any
        xs = np.vstack([s.sx, s.hx])
        ys = np.vstack([s.sy, s.hy])

        # training the GP is nondeterministic if there are any parameters to
        # tune so may give a different result here to during optimisation
        # gp_model = gp.GaussianProcessRegressor(**self.gp_params)
        # gp_model.fit(xs, ys)

        fig = plt.figure(figsize=(16, 10)) # inches
        grid = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])
        ax1, ax2 = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])

        fig.suptitle('Bayesian Optimisation step {}{}'.format(
            step-self.pre_samples,
            (' (chosen at random)' if s.chosen_at_random else '')), fontsize=14)
        ax1.margins(0.01, 0.1)
        ax2.margins(0.01, 0.1)
        if is_log:
            ax1.set_xscale('log')
            ax2.set_xscale('log')
        fig.subplots_adjust(hspace=0.3)

        ax1.set_ylabel('cost')
        ax1.set_title('Surrogate objective function')

        ### Plot True Cost
        if true_cost is not None:
            # true cost is either the cost function, or pre-computed costs as an array
            ys = true_cost(all_xs) if callable(true_cost) else true_cost
            ax1.plot(all_xs, ys, 'k--', label='true cost')

        ### Plot Samples
        param_index = self._index_for_param(param)
        # if logarithmic: value stored as exponent
        get_param = lambda p: np.exp(p[param_index]) if is_log else p[param_index]
        # plot samples projected onto the `param` axis
        sample_xs = [get_param(x) for x in s.sx]
        ax1.plot(sample_xs, s.sy, 'bo', label='samples')

        if len(s.hx) > 0:
            # there are some hypothesised samples
            hypothesised_xs = [get_param(x) for x in s.hx]
            ax1.plot(hypothesised_xs, s.hy, 'o', color='tomato', label='hypothesised samples')

        # index of the best current real sample
        best_i = np.argmax(s.sy) if self.maximise_cost else np.argmin(s.sy)
        ax1.plot(sample_xs[best_i], s.sy[best_i], '*', markersize=15,
                 color='deepskyblue', zorder=10, label='best sample')


        ### Plot Surrogate Function
        def plot_gp_prediction_through(config, mu_label, sigma_label, mu_alpha,
                                       sigma_alpha):
            # if logarithmic: the GP is trained on the exponents of the values
            gp_xs = np.log(all_xs) if is_log else all_xs
            # points with all but the chosen parameter fixed to match the given
            # config, but the chosen parameter varies
            perturbed = self._points_vary_one(config, param, gp_xs)
            mus, sigmas = s.gp.predict(perturbed, return_std=True)
            mus = mus.flatten()
            ax1.plot(all_xs, mus, 'm-', label=mu_label, alpha=mu_alpha)
            ax1.fill_between(all_xs, mus - n_sigma*sigmas, mus + n_sigma*sigmas, alpha=sigma_alpha,
                            color='mediumpurple', label=sigma_label)

        #TODO: fit the view to the cost function, don't expand to fit in the uncertainty

        plot_gp_prediction_through(s.next_x,
            mu_label='surrogate cost', sigma_label='uncertainty ${}\\sigma$'.format(n_sigma),
            mu_alpha=1, sigma_alpha=0.3)

        # plot the predictions through each sample
        def predictions_through_all_samples():
            configs_to_use = []

            # avoid drawing predictions of the same place more than once, so
            # avoid duplicate configurations which are identical to another
            # except for the value of 'param', since the plot varies this
            # parameter: the resulting plot will be the same in both cases.
            param_index = self._index_for_param(param)
            # a copy of the current samples with the focused parameter zeroed
            # start with s.next_x since that is a point which is guaranteed to
            # have a prediction plotted through it
            param_zeroed = self.config_to_point(s.next_x)
            param_zeroed[0,param_index] = 0
            for x in xs:
                x_zeroed = make2D_row(np.array(x))
                x_zeroed[0,param_index] = 0
                if not close_to_any(x_zeroed, param_zeroed, tol=1e-3):
                    configs_to_use.append(self.point_to_config(make2D_row(x)))
                    param_zeroed = np.append(param_zeroed, x_zeroed, axis=0)

            if len(configs_to_use) > 0:
                # cap to make sure they don't become invisible
                alpha = max(0.4/len(configs_to_use), 0.015)
                for cfg in configs_to_use:
                    plot_gp_prediction_through(cfg,
                        mu_label=None, sigma_label=None,
                        mu_alpha=alpha, sigma_alpha=alpha)

        if gp_through_all:
            predictions_through_all_samples()


        ### Plot Vertical Bars
        ax1.axvline(x=s.next_x[param])

        if s.chosen_at_random and s.argmax_acquisition is not None:
            ax1.axvline(x=get_param(s.argmax_acquisition[0]), color='y')

        ax1.legend()

        ax2.set_xlabel('parameter {}'.format(param))
        ax2.set_ylabel(self.acquisition_function_name)
        ax2.set_title('acquisition function')

        ### Plot Acquisition Function
        # it makes sense to plot the acquisition function through the slice
        # corresponding to the next point to be chosen.
        # if logarithmic: the GP is trained on the exponents of the values
        gp_xs = np.log(all_xs) if is_log else all_xs
        perturbed = self._points_vary_one(s.next_x, param, gp_xs)
        ac = self.acquisition_function(perturbed, s.gp, self.maximise_cost,
                                       s.best_sample.cost,
                                       **self.acquisition_function_params)
        #TODO: remove log_ac option
        if log_ac:
            # only useful for EI where ac >= 0 always
            ac[ac == 0.0] = 1e-10
            ac = -np.log(ac)
            label = '-log(acquisition function)'
        else:
            label = 'acquisition function'

        # show close-up on the next sample
        #ax2.set_xlim(s.next_x[param]-1, s.next_x[param]+1)
        #ax2.set_ylim((0.0, s.next_ac))

        ax2.plot(all_xs, ac, '-', color='g', linewidth=1.0, label=label)
        ax2.fill_between(all_xs, np.zeros_like(all_xs), ac.flatten(), alpha=0.3, color='palegreen')

        ax2.axvline(x=s.next_x[param])
        # may not want to plot if chosen_at_random because next_ac will be incorrect (ie 0)
        ax2.plot(s.next_x[param], s.next_ac, 'b^', markersize=10, alpha=0.8, label='next sample')

        # when chosen at random, next_x is different from what the maximisation
        # of the acquisition function suggested as the next configuration to
        # test. So plot both.
        if s.chosen_at_random and s.argmax_acquisition is not None:
            ac_x = get_param(s.argmax_acquisition[0])
            label='$\\mathrm{{argmax}}\\; {}$'.format(self.acquisition_function_name)
            ax2.axvline(x=ac_x, color='y', label=label)

        ax2.legend()

        return fig

    def plot_step_2D(self, x_param, y_param, step, true_cost=None,
                     plot_through='next', force_view_linear=False):
        '''
        x_param: the name of the parameter to place along the x axis
        y_param: the name of the parameter to place along the y axis
        step: the step number to plot
        true_cost: a function (that takes x and y arguments) or meshgrid
            containing the true cost values
        plot_through: unlike the 1D step plotting, in 2D the heatmaps cannot be
            easily overlaid to get a better understanding of the whole space.
            Instead, a Sample object can be provided to vary x_pram and y_param
            but leave the others constant to produce the graphs.
            Pass 'next' to signify the next sample (the one chosen by the
            current step) or 'best' to signify the current best sample as-of the
            current step.
        force_view_linear: force the images to be displayed with linear axes
            even if the parameters are logarithmic
        '''
        assert step in self.step_log.keys(), 'step not recorded in the log'

        x_type, y_type = self.range_types[x_param], self.range_types[y_param]
        assert all(type_ in [RangeType.Linear, RangeType.Logarithmic]
                   for type_ in [x_type, y_type])
        x_is_log = x_type == RangeType.Logarithmic
        y_is_log = y_type == RangeType.Logarithmic

        s = dotdict(self.step_log[step])

        if plot_through == 'next':
            plot_through = s.next_x
        elif plot_through == 'best':
            plot_through = self.config_to_point(s.best_sample.config)
        elif isinstance(plot_through, Sample):
            plot_through = self.config_to_point(plot_through.config)
        else:
            raise ValueError(plot_through)


        all_xs, all_ys = self.ranges[x_param], self.ranges[y_param]
        # the GP is trained on the exponents of the values if the parameter is logarithmic
        gp_xs = np.log(all_xs) if x_is_log else all_xs
        gp_ys = np.log(all_ys) if y_is_log else all_ys
        gp_X, gp_Y = np.meshgrid(gp_xs, gp_ys)
        # all combinations of x and y values, each point as a row
        gp_all_combos = np.vstack([gp_X.ravel(), gp_Y.ravel()]).T # ravel squashes to 1D

        grid_size = gp_X.shape # both X and Y have shape=(len xs, len ys)

        # combination of the true samples (sx, sy) and the hypothesised samples
        # (hx, hy) if there are any
        xs = np.vstack([s.sx, s.hx])
        ys = np.vstack([s.sy, s.hy])

        fig = plt.figure(figsize=(16, 16)) # inches
        grid = gridspec.GridSpec(nrows=2, ncols=2)
        # layout:
        # ax1 ax2
        # ax3 ax4
        ax1      = fig.add_subplot(grid[0])
        ax3, ax4 = fig.add_subplot(grid[2]), fig.add_subplot(grid[3])
        ax2 = fig.add_subplot(grid[1]) if true_cost is not None else None
        axes = [ax1, ax2, ax3, ax4] if true_cost is not None else [ax1, ax3, ax4]

        for ax in axes:
            ax.set_xlim(self.range_bounds[x_param])
            if x_is_log and not force_view_linear:
                ax.set_xscale('log')
            ax.set_ylim(self.range_bounds[y_param])
            if y_is_log and not force_view_linear:
                ax.set_yscale('log')
            ax.grid(False)

        # need to specify rect so that the suptitle isn't cut off
        fig.tight_layout(h_pad=4, w_pad=8, rect=[0, 0, 1, 0.96]) # [left, bottom, right, top] 0-1

        fig.suptitle('Bayesian Optimisation step {}{}'.format(
            step-self.pre_samples,
            (' (chosen at random)' if s.chosen_at_random else '')), fontsize=20)

        config = s.next_x # plot the GP through this config
        # points with all but the chosen parameter fixed to match the given
        # config, but the focused parameters vary
        perturbed = self._points_vary_one(config, x_param, gp_all_combos[:,0])
        y_index = self._index_for_param(y_param)
        perturbed[:,y_index] = gp_all_combos[:,1]

        mus, sigmas = s.gp.predict(perturbed, return_std=True)
        mus = mus.flatten()
        ac = self.acquisition_function(perturbed, s.gp, self.maximise_cost,
                                       s.best_sample.cost,
                                       **self.acquisition_function_params)

        x_param_index = self._index_for_param(x_param)
        y_param_index = self._index_for_param(y_param)
        # if logarithmic: value stored as exponent
        get_x_param = lambda p: np.exp(p[x_param_index]) if x_is_log else p[x_param_index]
        get_y_param = lambda p: np.exp(p[y_param_index]) if y_is_log else p[y_param_index]

        # plot samples projected onto the `x_param` and `y_param' axes
        sample_xs = [get_x_param(x) for x in s.sx]
        sample_ys = [get_y_param(x) for x in s.sx]

        if len(s.hx) > 0:
            # there are some hypothesised samples
            hypothesised_xs = [get_x_param(x) for x in s.hx]
            hypothesised_ys = [get_y_param(x) for x in s.hx]

        next_x, next_y = s.next_x[x_param], s.next_x[y_param]

        best_i = np.argmax(s.sy) if self.maximise_cost else np.argmin(s.sy)
        best_x, best_y = sample_xs[best_i], sample_ys[best_i]

        def plot_heatmap(ax, data, colorbar):
            # pcolormesh is better than imshow because: no need to fiddle around
            # with extents and aspect ratios because the x and y values can be
            # fed in and so just works. This also prevents the problem of the
            # origin being in the wrong place. It is compatible with log scaled
            # axes unlike imshow. There is no interpolation by default unlike
            # imshow.
            im = ax.pcolormesh(all_xs, all_ys, data, cmap='viridis')
            if colorbar:
                c = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.051)
                c.set_label('cost')
            ax.set_xlabel('parameter {}'.format(x_param))
            ax.set_ylabel('parameter {}'.format(y_param))
            ax.plot(best_x, best_y, '*', markersize=15,
                 color='deepskyblue', zorder=10, linestyle='None', label='best sample')
            ax.plot(sample_xs, sample_ys, 'ro', markersize=4, linestyle='None', label='samples')
            if len(s.hx) > 0:
                ax1.plot(hypothesised_xs, hypothesised_ys, 'o', color='tomato',
                         linestyle='None', label='hypothesised samples')

            ax.plot(next_x, next_y, marker='d', color='orangered',
                    markeredgecolor='black', markeredgewidth=1.5, markersize=10,
                    linestyle='None', label='next sample')

        title_size = 16

        ax1.set_title('Surrogate $\\mu$', fontsize=title_size)
        mus = mus.reshape(*grid_size)
        im = plot_heatmap(ax1, mus, colorbar=True)

        ax3.set_title('Surrogate $\\sigma$', fontsize=title_size)
        sigmas = sigmas.reshape(*grid_size)
        plot_heatmap(ax3, sigmas, colorbar=True)

        if true_cost is not None:
            ax2.set_title('True Cost', fontsize=title_size)
            plot_heatmap(ax2, true_cost, colorbar=True)

        ax4.set_title('Acquisition Function', fontsize=title_size)
        ac = ac.reshape(*grid_size)
        plot_heatmap(ax4, ac, colorbar=True)

        if s.chosen_at_random and s.argmax_acquisition is not None:
            label='$\\mathrm{{argmax}}\\; {}$'.format(self.acquisition_function_name)
            ax4.axhline(y=get_y_param(s.argmax_acquisition[0]), color='y', label=label)
            ax4.axvline(x=get_x_param(s.argmax_acquisition[0]), color='y')

        ax4.legend(bbox_to_anchor=(0, 1.01), loc='lower left', borderaxespad=0.0)

        return fig

    def num_randomly_chosen(self):
        count = 0
        for s in self.samples:
            is_pre_sample = s.job_num <= self.pre_samples
            is_random = s.job_num in self.step_log.keys() and self.step_log[s.job_num]['chosen_at_random']
            if is_pre_sample or is_random:
                count += 1
        return count

    def plot_cost_over_time(self, plot_each=True, plot_best=True,
                            true_best=None, plot_random=True):
        '''
        plot a line graph showing the progress that the optimiser makes towards
        the optimum as the number of samples increases.
        plot_each: plot the cost of each sample
        plot_best: plot the running-best cost
        true_best: if available: plot a horizontal line for the best possible cost
        plot_random: whether to plot markers over the samples which were chosen randomly
        '''
        fig = super(BayesianOptimisationOptimiser, self).plot_cost_over_time(plot_each, plot_best, true_best)
        ax = fig.axes[0]

        if plot_random:
            random_sample_nums = []
            random_sample_costs = []
            for i, s in enumerate(self.samples):
                is_pre_sample = s.job_num <= self.pre_samples
                is_random = s.job_num in self.step_log.keys() and self.step_log[s.job_num]['chosen_at_random']
                if is_pre_sample or is_random:
                    random_sample_nums.append(i+1)
                    random_sample_costs.append(s.cost)
            ax.plot(random_sample_nums, random_sample_costs, 'ro', markersize=5, label='randomly chosen')
            ax.margins(0.0, 0.18)
            ax.legend()

        def sample_num_to_bayes_step(s_num):
            i = int(s_num)-1
            if i >= 0 and i < len(self.samples):
                s = self.samples[i]
                if s.job_num in self.step_log.keys():
                    return s.job_num - self.pre_samples
                else:
                    return ''
            else:
                return ''
        labels = [sample_num_to_bayes_step(s_num) for s_num in ax.get_xticks()]

        ax2 = ax.twiny() # 'twin y'
        ax2.grid(False)
        ax2.set_xlim(ax.get_xlim())
        ax2.xaxis.set_major_locator(ax.xaxis.get_major_locator())
        # convert the labels marked on ax into new labels for the top
        ax2.set_xticklabels(labels)
        ax2.set_xlabel('Bayesian Step')

        # raise the title to get out of the way of ax2
        ax.title.set_position([0.5, 1.08])

        return fig

