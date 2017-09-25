#!/usr/bin/env python3
'''
The core components of the optimisation library:
- Job
- Sample
- Evaluator
- Optimiser (Base class)
'''
# python 2 compatibility
from __future__ import (absolute_import, division, print_function, unicode_literals)
from .py2 import *

import time
import json
import os

import threading
# dummy => uses Threads rather than processes
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np

# local modules
from . import net as op_net
from .utils import *
from .plot import OptimiserPlotting


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
    #raise e
    import pdb
    pdb.set_trace()



class Job(DataHolder):
    '''
    A unit of work consisting of a single configuration to be evaluated. A
    single job may result in multiple samples because the evaluator may choose
    to evaluate other configurations as well as the suggested one.

    ID: a numeric ID to uniquely identify the job (assigned with a contiguous increasing counter)
    config: the configuration to test for this job
    setup_duration: number of seconds taken to choose the config
    evaluation_duration: number of seconds taken to evaluate the config
    '''
    __slots__ = ('ID', 'config', 'setup_duration', 'evaluation_duration')
    __defaults__ = {
        'setup_duration'      : None,
        'evaluation_duration' : None
    }
    def total_time(self):
        return self.setup_duration + self.evaluation_duration

class Sample(DataHolder):
    '''
    The results of evaluating a single configuration.

    config: the configuration that was evaluated
    cost: the cost of 'config'
    extra: problem-specific data (optional, not used by the optimiser)
    job_ID: the ID of the job that this sample was taken in
    '''
    __slots__ = ('config', 'cost', 'extra', 'job_ID')
    __defaults__ = {
        'extra'  : {},
        'job_ID' : None
    }
    def to_encoded_tuple(self):
        ''' for storage or sending as part of a JSON message '''
        # for storage it may be slightly smaller than a raw JSON encoding, for
        # transferring over the network, pickling ensures numpy arrays or other
        # data is preserved and not converted to lists for example.
        extra = JSON_encode_binary(self.extra) if self.extra else None
        return (self.config, self.cost, extra, self.job_ID)
    @staticmethod
    def from_encoded_tuple(data):
        ''' read from a tuple previously created with to_encoded_tuple '''
        extra = JSON_decode_binary(data[2]) if data[2] is not None else {}
        return Sample(dotdict(data[0]), data[1], extra, data[3])

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
        # number of finished evaluations this run
        num_finished_jobs = 0
        # number of times there have been no jobs available in a row
        num_rejections = 0

        should_stop = lambda: self.wants_to_stop() or num_finished_jobs >= max_jobs
        on_error = lambda e: time.sleep(NON_CRITICAL_WAIT)

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
                    num_rejections += 1
                    # not an error, but wait a while
                    time.sleep(min(2 ** num_rejections, 16))
                    continue
                else:
                    num_rejections = 0

                job = Job(**op_net.decode_JSON(job))
                job.config = dotdict(job.config)
                self.log('received job: {}'.format(job))

                ### Evaluate the job
                results_msg = self._evaluate_job(job)

                ### Return the results
                # don't ever stop trying to connect even if self.wants_to_stop()
                # because after a job is requested, the client must return the
                # results
                ack = message_client(results_msg, op_net.never_stop)
                assert ack is not None and ack == op_net.empty_msg()

                num_finished_jobs += 1
                self.log('results sent successfully')

            if self.wants_to_stop():
                self.log('stopping because of manual shut down')
            elif num_finished_jobs >= max_jobs:
                self.log('stopping because max_jobs reached')
        except Exception as e:
            self.log('Exception raised during client run:\n{}'.format(exception_string()))
            ON_EXCEPTION(e)
        finally:
            self.log('evaluator shut down')

    def _convert_results(self, results, config):
        '''
        test_config may return several different things. This function converts
        those results into a standard form: a list of Sample objects.
        '''
        # permit 'scalar' ndarrays, but convert them to true scalars
        if isinstance(results, np.ndarray) and results.size == 1:
            results = np.asscalar(results)
        # test_config can return either a cost value, a sample object or a list
        # of sample objects
        if is_numeric(results):
            # evaluator returned cost
            results = [Sample(config, results)]
        elif (isinstance(results, tuple) and len(results) == 2 and
              is_numeric(results[0]) and isinstance(results[1], dict)):
            # evaluator returned cost and extra as a tuple
            results = [Sample(config, results[0], results[1])]
        elif isinstance(results, Sample):
            # evaluator returned a single sample
            results = [results]
        elif isinstance(results, list) and all(isinstance(r, Sample) for r in results):
            # evaluator returned a list of samples
            results = results
        else:
            raise ValueError('invalid results type from evaluator: {} {}'.format(results, type(results)))

        if any(set(res.config.keys()) != set(config.keys()) for res in results):
            raise ValueError('incorrect entries in config dict: {}'.format(results))

        # cost values cannot be NaN or infinity because this breaks the surrogate model
        if any(np.isnan(res.cost) or np.isinf(res.cost) for res in results):
            raise ValueError('invalid cost (infinity or NaN): {}'.format(results))

        return results

    def _evaluate_config(self, config):
        '''
        evaluate a configuration and return the results in a standard form (list
        of Sample objects.

        In the case of an error: instead of crashing, test_config will be run
        again with the same configuration in the hopes that it won't happen again
        '''
        while True:
            try:
                # pass a copy to keep 'config' intact for reference later
                results = self.test_config(config.copy())
            except Exception:
                self.log('exception when testing a configuration!\n' +
                         exception_string())
                # prevent too much spamming if the evaluator is broken completely
                time.sleep(NON_CRITICAL_WAIT)
                continue

            try:
                samples = self._convert_results(results, config)
            except ValueError:
                self.log('exception when converting results\n' + exception_string())
                time.sleep(NON_CRITICAL_WAIT)
                continue

            return samples

    def _evaluate_job(self, job):
        start_time = time.time()

        self.log('evaluating job {}: config: {}'.format(
            job.ID, config_string(job.config, precise=True)))

        samples = self._evaluate_config(job.config)

        self.log('returning results: {}'.format(
            [(s.config, s.cost, {k:('[...]' if isinstance(v, np.ndarray) else v) for k, v in s.extra.items()}) for s in samples]))

        samples = [s.to_encoded_tuple() for s in samples] # for JSON serialisation
        job.evaluation_duration = time.time()-start_time
        results_msg = {
            'type'    : 'job_results',
            'job'     : job.to_dict(),
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


class Optimiser(OptimiserPlotting, object):
    '''
    given a search space and a function to call in order to evaluate the cost at
    a given location, find the minimum of the function in the search space.

    Importantly: an expression for the cost function is not required
    '''
    def __init__(self, ranges, maximise_cost):#TODO: replace maximise_cost with optimal_value='max'|'min'
        '''
        ranges: dictionary of parameter names and their ranges (numpy arrays, can be created by np.linspace or np.logspace)
        maximise_cost: True => higher cost is better. False => lower cost is better
        '''
        self.ranges = dotdict(ranges)
        self.params = sorted(self.ranges.keys())
        self.maximise_cost = maximise_cost

        # note: number of samples may diverge from the number of jobs since a
        # job can result in any number of samples (including 0).
        # the configurations that have been tested (list of `Sample` objects)
        self.samples = []
        self.num_started_jobs = 0
        self.num_finished_jobs = 0
        #TODO: rename to finished_job_IDs
        self.finished_job_ids = set() # job.ID is added after it is finished

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

    def clean_log(self):
        '''
        remove adjacent and identical lines from the log since they only add noise.
        '''
        len_ = len(self.log_record)
        lines = self.log_record.split('\n')
        remove_adjacent_duplicates(lines)

        log_record = '\n'.join(lines)
        if len(self.log_record) == len_:
            self.log_record = log_record
        else:
            # some more content was added to the log during this operation and
            # so the operation failed. Note: This isn't completely fool proof
            # because the check and assignment are not atomic.
            self._log('log cleaning failed because new content was written')

# Running Optimisation

    def _next_job(self):
        '''
        get the next configuration and return a job object for it (or return
        None if there are no more configurations)
        '''
        assert self._ready_for_next_configuration()
        job_ID = self.num_started_jobs+1 # 1-based

        # for Bayesian optimisation this may take a little time
        start_time = time.time()
        config = self._next_configuration(job_ID)
        setup_duration = time.time()-start_time

        if config is None:
            self._log('out of configurations')
            return None # out of configurations
        else:
            config = dotdict(config)
            self._log('started job {}: config={}'.format(job_ID, config_string(config)))
            self.num_started_jobs += 1
            return Job(job_ID, config, setup_duration)

    def _process_job_results(self, job, samples):
        '''
        check the results of the job are valid, record them and post to the log
        samples: must be in the 'standard format': a list of Sample objects. The
            job_ID attribute need not be set
        '''
        for s in samples:
            s.job_ID = job.ID
        self.samples.extend(samples)
        self.num_finished_jobs += 1
        self.finished_job_ids.add(job.ID)

        self._log('finished job {} in {} (setup: {} evaluation: {}):'.format(
            job.ID, time_string(job.total_time()),
            time_string(job.setup_duration), time_string(job.evaluation_duration)
        ))
        if len(samples) > 1:
            fmt = ('\tsample {sample_num:02}:\n'
                   '\t\tcost   = {}\n'
                   '\t\tconfig = {}\n'
                   '\t\textra  = {}')
        else:
            fmt = ('\tcost   = {}\n'
                   '\tconfig = {}\n'
                   '\textra  = {}')
        for i, s in enumerate(samples):
            if self.sample_is_best(s):
                self._log('\tcurrent best')
            self._log(fmt.format(
                          s.cost,
                          config_string(s.config, order=self.params, precise=True),
                          {k:('...' if isinstance(v, np.ndarray) else v) for k, v in s.extra.items()},
                          sample_num=i))

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
        assert 'type' in msg, 'malformed request: {}'.format(msg)

        def job_msg(job):
            return op_net.encode_JSON(job.to_dict(), encoder=NumpyJSONEncoder)

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
                self._log('obtaining next config to evaluate')
                job = self._next_job()
                # store in-case it has to be re-sent
                state.requested_job = job
                if job is None:
                    state.out_of_configs = True
                    return op_net.empty_msg() # send empty to signal no more jobs available
                else:
                    state.started_job_ids.add(job.ID)
                    state.started_this_run += 1
                    return job_msg(job)

        elif msg['type'] == 'job_results':
            # during transmission: serialised to tuples
            samples = [Sample.from_encoded_tuple(s) for s in msg['samples']]
            job = Job(**msg['job'])
            job.config = dotdict(job.config)

            if job.ID not in state.started_job_ids:
                self._log('Non-critical error: received results from a job that was not started this run: {}'.format(msg))

            elif job.ID in self.finished_job_ids:
                self._log('Non-critical error: received results from a job that is already finished: {}'.format(msg))

            else:
                self._process_job_results(job, samples)
                state.finished_this_run += 1

                # if the results are for the job that the optimiser wasn't sure
                # whether the evaluator received successfully, then evidently it did.
                if state.requested_job is not None and job.ID == state.requested_job.ID:
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

    class _ServerRunState(DataHolder):
        __slots__ = ('start_time', 'old_duration', 'max_jobs',
                        'checkpoint_flag_was_set', 'started_this_run',
                        'finished_this_run', 'started_job_ids',
                        'requested_job', 'out_of_configs', 'exception_caught')
        def run_time(self):
            return time.time() - self.start_time
        def num_outstanding_jobs(self):
            return self.started_this_run - self.finished_this_run

    def run_server(self, host=DEFAULT_HOST, port=DEFAULT_PORT, max_jobs=inf,
                   timeout=SERVER_TIMEOUT):
        assert self.run_state is None
        self._log('starting optimisation server at {}:{}'.format(host, port))
        self._stop_flag.clear()

        state = Optimiser._ServerRunState(
            start_time = time.time(),
            old_duration = self.duration, # duration as-of the start of this run
            max_jobs = max_jobs,
            checkpoint_flag_was_set = False, # to only log once

            # count number of jobs started and finished this run
            started_this_run = 0,
            finished_this_run = 0,
            started_job_ids = set(),

            # used to re-send jobs that failed to reach the evaluator. Set
            # back to None after a request-response succeeds.
            requested_job = None,

            # flags to diagnose the stopping conditions
            out_of_configs = False,
            exception_caught = False
        )
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
                if response != op_net.empty_msg():
                    self._log('job sent successfully')

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


    class _SequentialRunState(DataHolder):
        __slots__ = ('start_time', 'old_duration', 'max_jobs',
                        'finished_this_run', 'out_of_configs',
                        'exception_caught')
        def run_time(self):
            return time.time() - self.start_time

    def run_sequential(self, evaluator, max_jobs=inf):
        '''
        run the optimiser with the given evaluator one job after another in the
        current thread

        evaluator: the Evaluator object to use to evaluate the configurations
        max_jobs: the maximum number of jobs to allow for this run (not in total)
        '''
        assert self.run_state is None
        try:
            self._stop_flag.clear()
            self._log('starting sequential optimisation...')

            state = Optimiser._SequentialRunState(
                start_time = time.time(),
                old_duration = self.duration, # duration as-of the start of this run
                max_jobs = max_jobs,

                finished_this_run = 0,

                # flags to diagnose the stopping conditions
                out_of_configs = False,
                exception_caught = False
            )
            self.run_state = state


            while not self._wants_to_stop(state):
                self._handle_checkpoint() # save a checkpoint if signalled to
                assert self._ready_for_next_configuration(), \
                    'not ready for the next configuration in sequential optimisation'
                self._log('obtaining next config to evaluate')
                job = self._next_job()
                if job is None:
                    state.out_of_configs = True
                    break

                evaluation_start = time.time()
                samples = evaluator._evaluate_config(job.config)
                job.evaluation_duration = time.time() - evaluation_start

                self._process_job_results(job, samples)

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

    def _next_configuration(self, job_ID):
        '''
        implemented by different optimisation methods
        return the next configuration to try, or None if finished
        job_ID: the ID of the job that the configuration will be assigned to
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

    #TODO: put in optimisation_gui and make it report more things
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

    def cancel_save(self):
        '''
        if save_when_ready has been called, but you wish to abort the checkpoint
        so that new jobs can be started again, this function will instruct the
        optimiser to stop waiting for quiescence and continue without saving.
        '''
        if self._checkpoint_flag.is_set():
            self._log('cancelling checkpoint. Continuing as normal.')
            self._checkpoint_flag.clear()

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
        if not self._consistent_and_quiescent():
            self._log('warning: optimiser is not quiescent and or consistent when '
                      'taking a checkpoint, the checkpoint may be corrupted')

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

        save = json.dumps(self._save_dict(), sort_keys=True, cls=NumpyJSONEncoder).encode('utf-8')
        with open(filename, 'wb') as f:
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

        with open(filename, 'rb') as f:
            try:
                save = json.loads(f.read().decode('utf-8'))
            except json.JSONDecodeError as e:
                self._log('exception: invalid checkpoint! Cannot load JSON: {}'.format(exception_string()))
                ON_EXCEPTION(e)
                return
            self._load_dict(save)

        if not self._consistent_and_quiescent():
            self._log('warning: loaded checkpoint is not consistent or the '
                      'optimiser was not quiescent when loading')


    def _save_dict(self):
        '''
        generate the dictionary to be JSON serialised and saved
        (designed to be overridden by derived classes in order to save specialised data)
        '''
        return {
            'samples' : [s.to_encoded_tuple() for s in self.samples],
            'num_started_jobs' : self.num_started_jobs, # must be equal to num_finished_jobs
            'num_finished_jobs' : self.num_finished_jobs,
            'finished_job_ids' : list(self.finished_job_ids),
            'duration' : self.duration,
            'log_record' : self.log_record,
            'checkpoint_filename' : self.checkpoint_filename,
            # convenience for viewing the save, but will not be loaded
            'best_sample' : (self.best_sample() or Sample({}, inf)).to_dict()
        }

    def _load_dict(self, save):
        '''
        load progress from a dictionary
        (designed to be overridden by derived classes in order to load specialised data)
        '''
        self.samples = [Sample.from_encoded_tuple(s) for s in save['samples']]
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



