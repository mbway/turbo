# fix some of the python2 ugliness
from __future__ import print_function
from __future__ import division
import sys
if sys.version_info[0] == 3: # python 3
    from math import isclose, inf
elif sys.version_info[0] == 2: # python 2
    inf = float('inf')
    # implementation from the python3 documentation
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
else:
    print('unsupported python version')

import time
import json
import struct
import os

import socket
import threading
# dummy => uses Threads rather than processes
from multiprocessing.dummy import Pool as ThreadPool

from collections import defaultdict
from itertools import groupby

import numpy as np
import matplotlib.pyplot as plt

# for Bayesian Optimisation
import sklearn.gaussian_process as gp
import scipy.optimize
from scipy.stats import norm # Gaussian/normal distribution


# local modules
import plot3D


PORT = 9187


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

def set_str(set_):
    '''
    format a set as a string. The results of str.format are undesirable and inconsistent:

    python2:
    >>> '{}'.format(set([]))
    'set([])'
    >>> '{}'.format(set([1,2,3]))
    'set([1, 2, 3])'

    python3:
    >>> '{}'.format(set([]))
    'set()'
    >>> '{}'.format(set([1,2,3]))
    '{1, 2, 3}'

    set_str():
    >>> set_str(set([]))
    '{}'
    >>> set_str(set([1,2,3]))
    '{1, 2, 3}'

    '''
    return '{' + ', '.join([str(e) for e in set_]) + '}'

class NumpyJSONEncoder(json.JSONEncoder):
    ''' unfortunately numpy primitives are not JSON serialisable '''
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

def config_string(config, order=None, precise=False):
    '''
    similar to the string representation of a dictionary
    order: order of dictionary keys, None => alphabetical order
    precise: True => truncate numbers to a certain length. False => display all precision
    '''
    assert order is None or set(order) == set(config.keys())
    order = sorted(list(config.keys())) if order is None else order
    if len(order) == 0:
        return '{}'
    else:
        string = '{'
        for p in order:
            if type(config[p]) == str or type(config[p]) == np.str_:
                string += '{}="{}", '.format(p, config[p])
            else: # assuming numeric
                if precise:
                    string += '{}={}, '.format(p, config[p])
                else:
                    # 2 significant figures
                    string += '{}={:.2g}, '.format(p, config[p])
        string = string[:-2] # remove trailing comma and space
        string += '}'
        return string

def exception_string():
    ''' get a string formatted with the last exception '''
    import traceback
    return ''.join(traceback.format_exception(*sys.exc_info()))

def is_numeric(obj):
    '''
    whether 'obj' is a numeric quantity, not including types which may be
    converted to a numeric quantity such as strings. Also numpy arrays are
    specifically excluded, however they do support +-*/ etc.

    Modified from: https://stackoverflow.com/a/500908
    '''
    if isinstance(obj, np.ndarray):
        return False

    if sys.version_info[0] == 3: # python 3
        attrs = ['__add__', '__sub__', '__mul__', '__truediv__', '__pow__']
    elif sys.version_info[0] == 2: # python 2
        attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']

    return all(hasattr(obj, attr) for attr in attrs)


class Job(object):
    '''
    a job is a single configuration to be tested, it may result in one or
    multiple samples being evaluated.
    '''
    def __init__(self, config, job_num, setup_duration):
        '''
        config: the configuration to test
        job_num: a unique ID for the job
        setup_duration: the time taken for _next_configuration to generate 'config'
        '''
        self.config = config
        self.num = job_num
        self.start_time = time.time()
        self.setup_duration = setup_duration

class Sample(object):
    '''
    a sample is a configuration and its corresponding cost as determined by an evaluator.
    '''
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


def send_json(conn, obj, encoder=None):
    '''
    send the given object through the given connection by first serialising the
    object to JSON.
    conn: the connection to send through
    obj: the object to send (must be JSON serialisable)
    encoder: a JSONEncoder to use
    '''
    data = json.dumps(obj, cls=encoder).encode('utf-8')
    # ! => network byte order (Big Endian)
    # I => unsigned integer (4 bytes)
    length = struct.pack('!I', len(data))
    # don't wait for length to fully send with sendall because we can start
    # sending the payload along with it.
    conn.send(length)
    conn.sendall(data)

def send_empty(conn):
    ''' send a 4 byte length of 0 to signify 'no data' '''
    conn.sendall(struct.pack('!I', 0))

def read_exactly(conn, num_bytes):
    ''' read exactly the given number of bytes from the connection '''
    data = bytes()
    while len(data) < num_bytes: # until the length is fully read
        left = num_bytes - len(data)
        # documentation recommends a small power of 2 to give best results
        data += conn.recv(min(4096, left))
    assert len(data) == num_bytes
    return data

def recv_json(conn):
    '''
    receive a JSON object from the given connection
    '''
    # read the length
    data = read_exactly(conn, 4)
    length, = struct.unpack('!I', data) # see send_json for the protocol
    if length == 0:
        return None # indicates 'no data'
    else:
        data = read_exactly(conn, length)
        obj = json.loads(data.decode('utf-8'))
        return obj


''' Details of the network protocol between the optimiser and the evaluator
Optimiser sets up a server and listens for clients. Every time a client
(evaluator) connects the optimiser spawns a thread to handle the client,
allowing it to accept more clients. In the thread for the connected client, it
is sent a configuration (serialised as JSON) to evaluate. The message is a
length followed by the JSON content. The thread waits for the evaluator to reply
with the results (also JSON serialised). Once the reply has been received, the
thread is free to handle more clients (as part of a thread pool).

If a client connection is open and the optimiser wishes to shutdown, it can send
a length of 0 to indicate 'no data' the evaluator can then resume trying to
connect in-case the server comes back. Alternatively, when the server shuts
down, the connection breaks (but not always). Both are used to detect a
disconnection.

When connecting to the server, there are several different ways the call to
connect could fail. In any of these situations: simply wait a while and try
again (without reporting an error)

If an evaluator accepts a job then they are obliged to complete it. If the
evaluator crashes or otherwise fails to produce results, the job remains in the
queue and will never finish processing.
'''

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
        self.stop_flag = threading.Event() # is_set() => finish gracefully and stop

    def run_client(self, host='0.0.0.0', port=PORT):
        '''
        receive jobs from an Optimiser server and evaluate them until the server
        shuts down.
        '''
        sock = None
        try:
            self.log('evaluator client starting...')
            self.stop_flag.clear()

            while not self.stop_flag.is_set():
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP
                self.log('attempting to connect to {}:{}'.format(host, port))
                sock.settimeout(1.0) # only while trying to connect

                # connect to the server
                while not self.stop_flag.is_set():
                    try:
                        sock.connect((host, port))
                        break
                    except (socket.timeout, ConnectionRefusedError, ConnectionAbortedError):
                        time.sleep(1.0)
                if self.stop_flag.is_set():
                    break

                self.log('connection established')
                sock.settimeout(None) # wait forever (blocking mode)

                try:
                    job = recv_json(sock)
                except ConnectionResetError:
                    self.log('FAILED to receive job from optimiser (ConnectionResetError)')
                    continue

                # optimiser finished and sent 0 length to inform the evaluator
                if job is None:
                    self.log('FAILED to receive job from optimiser (0 length)')
                    continue

                job_num = job['num']
                config = dotdict(job['config'])

                self.log('evaluating job {}: config: {}'.format(job_num, config_string(config, precise=True)))
                results = self.test_config(config)

                # evaluator can return either a list of samples, or just a cost
                samples = results if isinstance(results, list) else [Sample(config, results)]
                samples = [{'config':s.config,'cost':s.cost} for s in samples] # for JSON serialisation

                self.log('returning results: {}'.format(results))
                send_json(sock, { 'samples' : samples }, encoder=NumpyJSONEncoder)
                sock.close()
        finally:
            if sock is not None:
                sock.close()

    def log(self, string, newline=True):
        if not isinstance(string, str):
            string = str(string)
        self.log_record += string
        if newline:
            self.log_record += '\n'
        if self.noisy:
            print(string, end=('\n' if newline else ''))

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

        # written to by _log()
        self.log_record = ''
        # whether to print to stdout from _log()
        self.noisy = False

        self.stop_flag = threading.Event() # is_set() => finish gracefully and stop
        self.duration = 0 # total time spent (persists across runs)


    def _log(self, string, newline=True):
        if not isinstance(string, str):
            string = str(string)
        self.log_record += string
        if newline:
            self.log_record += '\n'
        if self.noisy:
            print(string, end=('\n' if newline else ''))

# Running Optimisation

    def _next_job(self):
        '''
        get the next configuration and return a job object for it (or return
        None if there are no more configurations)
        '''
        if not self._ready_for_next_configuration():
            self._log('not ready for the next configuration yet')
            while not self._ready_for_next_configuration():
                time.sleep(0.1)
            self._log('now ready for next configuration')

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
        duration = time.time()-job.start_time

        assert is_numeric(results) or isinstance(results, list), 'invalid results type from evaluator'
        # evaluator can return either a list of samples, or just a cost
        samples = results if isinstance(results, list) else [Sample(job.config, results)]
        self.samples.extend(samples)
        self.num_finished_jobs += 1
        self.finished_job_ids.add(job.num)

        self._log('finished job {} in {} (setup: {} evaluation: {}):'.format(
            job.num, time_string(duration+job.setup_duration),
            time_string(job.setup_duration), time_string(duration)
        ))
        for i,s in enumerate(samples):
            self._log('\tsample {:02}: config={}, cost={:.2g}{}'.format(
                i, config_string(s.config, precise=True), s.cost,
                (' (current best)' if self.sample_is_best(s) else '')
            ))

    def _shutdown_message(self, old_duration, this_duration,
                          max_jobs_exceeded, out_of_configs, exception_caught):
        '''
        post to the log about the run that just finished, including reason for
        the optimiser stopping.
        old_duration: the duration of all runs except for this one
        this_duration: the duration of this run
        '''
        self.duration = old_duration + this_duration
        self._log('total time taken: {} ({} this run)'.format(
            time_string(self.duration), time_string(this_duration)))

        outstanding_jobs = self.num_started_jobs - self.num_finished_jobs
        if exception_caught:
            self._log('optimisation stopped because an exception was thrown.')
        elif max_jobs_exceeded:
            self._log('optimisation stopped because the maximum number of jobs was exceeded')
        elif self.stop_flag.is_set():
            self._log('optimisation manually shut down gracefully')
        elif out_of_configs and outstanding_jobs == 0:
            self._log('optimisation finished (out of configurations).')
        else:
            self._log('stopped for an unknown reason. May be in an inconsistent state. (details: {} / {} / {})'.format(
                self.stop_flag.is_set(), out_of_configs, outstanding_jobs))

    def _handle_client(self, conn, lock, exception_caught, job):
        '''
        run as part of a thread pool, interact with a remote evaluator and
        process the results
        conn: the socket to communicate with the client (already connected).
            Closed after the job has finished
        lock: a lock which must be obtained before modifying anything in self
        exception_caught: an event to set if an exception is thrown
        job: the job to have the client evaluate
        '''
        try:
            msg = {'num': job.num, 'config': job.config}
            send_json(conn, msg, encoder=NumpyJSONEncoder)
            results = recv_json(conn) # keep waiting until the client is done
            # during transmission: serialised to dictionaries
            results = [Sample(dotdict(s['config']), s['cost'])
                       for s in results['samples']]

            with lock:
                self._process_job_results(job, results)
        except Exception as e:
            exception_caught.set()
            with lock:
                self._log('Exception raised while processing job {} (in a worker thread):\n{}'.format(
                    job.num, exception_string()))
        finally:
            conn.close()

    def run_server(self, host='0.0.0.0', port=PORT, max_clients=4, max_jobs=inf):
        '''
        run a server which serves jobs to any listening evaluators

        max_clients: the maximum number of clients to expect to connect. If
            another connects then their job will not be served until there is a
            free thread to deal with it.
        evaluator: the Evaluator object to use to evaluate the configurations
        max_jobs: the maximum number of jobs to allow for this run (not in total)
        '''
        pool = None
        sock = None
        conn = None
        try:
            self._log('starting optimisation server...')

            self.stop_flag.clear()
            run_start_time = time.time()
            old_duration = self.duration # duration as-of the start of this run
            num_jobs = 0 # count number of jobs this run
            started_job_ids = set()

            # flags to diagnose the stopping conditions
            out_of_configs = False
            exception_caught = False
            # set when an exception happens in one of the client threads
            exception_in_pool = threading.Event()

            # make sure the client handlers don't update self at the same time
            lock = threading.RLock() # re-entrant lock
            pool = ThreadPool(processes=max_clients)

            # server socket for the clients to connect to
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # able to re-use host/port combo even if in use
            sock.bind((host, port))
            sock.listen(max_clients) # enable listening on the socket. backlog=max_clients
            sock.settimeout(1.0) # timeout for accept, not inherited by the client connections

            while not self.stop_flag.is_set() and num_jobs < max_jobs:
                with lock:
                    self._log('outstanding job IDs: {}'.format(
                        set_str(started_job_ids-self.finished_job_ids)))
                    self._log('waiting for a connection')

                # wait for a client to connect
                while not self.stop_flag.is_set():
                    try:
                        # conn is another socket object
                        conn, addr = sock.accept()
                        break
                    except socket.timeout:
                        conn = None
                if self.stop_flag.is_set():
                    break

                with lock: self._log('connection accepted from {}:{}'.format(*addr))

                job = self._next_job()
                if job is None:
                    out_of_configs = True
                    break
                started_job_ids.add(job.num)

                pool.apply_async(self._handle_client, (conn, lock, exception_in_pool, job))
                #self._handle_client(conn, lock, exception_in_pool, job)

                conn = None
                num_jobs += 1
                with lock: self.duration = old_duration + (time.time()-run_start_time)

        except Exception as e:
            self._log('Exception raised during run:\n{}'.format(exception_string()))
            exception_caught = True
        finally:
            # stop accepting new connections
            if sock is not None:
                sock.close()

            # finish up active jobs
            if pool is not None:
                with lock: self._log('waiting for active jobs to finish')
                pool.close() # worker threads will exit once done
                pool.join()
                self._log('active jobs finished')

            # clean up lingering client connection if there is one (connection
            # was accepted before optimiser stopped)
            if conn is not None:
                self._log('notifying client which was waiting for a job')
                send_empty(conn)
                conn.close()

        this_duration = time.time() - run_start_time
        max_jobs_exceeded = num_jobs >= max_jobs
        exception_caught |= exception_in_pool.is_set()
        self._shutdown_message(old_duration, this_duration,
                               max_jobs_exceeded, out_of_configs, exception_caught)


    def run_sequential(self, evaluator, max_jobs=inf):
        '''
        run the optimiser with the given evaluator one job after another in the
        current thread

        evaluator: the Evaluator object to use to evaluate the configurations
        max_jobs: the maximum number of jobs to allow for this run (not in total)
        '''
        try:
            self._log('starting sequential optimisation...')

            self.stop_flag.clear()
            run_start_time = time.time()
            old_duration = self.duration # duration as-of the start of this run
            num_jobs = 0 # count number of jobs this run

            # flags to diagnose the stopping conditions
            out_of_configs = False
            exception_caught = False

            while not self.stop_flag.is_set() and num_jobs < max_jobs:
                job = self._next_job()
                if job is None:
                    out_of_configs = True
                    break

                results = evaluator.test_config(job.config)

                self._process_job_results(job, results)

                num_jobs += 1
                self.duration = old_duration + (time.time()-run_start_time)

        except Exception as e:
            self._log('Exception raised during run:\n{}'.format(exception_string()))
            exception_caught = True

        this_duration = time.time() - run_start_time
        max_jobs_exceeded = num_jobs >= max_jobs
        self._shutdown_message(old_duration, this_duration,
                               max_jobs_exceeded, out_of_configs, exception_caught)


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

    #TODO: change this. maybe set grid search max_jobs and use that instead
    def total_jobs(self):
        '''
        return the total number of configurations to be tested
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
        total_jobs = self.total_jobs()
        percent_progress = self.num_finished_jobs/float(total_jobs)*100.0
        print('{} of {} samples ({:.1f}%) taken in {}.'.format(
            self.num_finished_jobs, total_jobs, percent_progress, time_string(self.duration)))
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
        save the progress of the optimisation to a JSON file which can be
        re-loaded and continued. The optimiser must not be running for the state
        to be saved.

        Note: this does not save all the state of the optimiser, only the samples
        '''
        #TODO make sure stopped in a good state before saving

        # keep counting until a filename is available
        if os.path.isfile(filename):
            print('File "{}" already exists!'.format(filename))
            name, ext = os.path.splitext(filename)
            count = 1
            while os.path.isfile(name + str(count) + ext):
                count += 1
            filename = name + str(count) + ext
            print('writing to "{}" instead'.format(filename))

        with open(filename, 'w') as f:
            f.write(json.dumps(self._save_dict(), indent=4, cls=NumpyJSONEncoder))

    def load_progress(self, filename):
        '''
        restore the progress of an optimisation run (note: the optimiser must be
        initialised identically to when the samples were saved). The optimiser
        must not be running for the state to be loaded.
        '''
        #TODO make sure stopped in a good state before saving
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
            'num_started_jobs' : self.num_started_jobs,
            'num_finished_jobs' : self.num_finished_jobs,
            'finished_job_ids' : list(self.finished_job_ids),
            'duration' : self.duration,
            'log' : self.log_record,
            # convenience for viewing the save, but will not be loaded
            'best_sample' : {'config' : best.config, 'cost' : best.cost}
        }

    def _load_dict(self, save):
        '''
        load progress from a dictionary
        (designed to be overridden by derived classes in order to load specialised data)
        '''
        self.samples = [Sample(dotdict(config), cost) for config, cost in save['samples']]
        self.num_started_jobs = save['num_started_jobs']
        self.num_finished_jobs = save['num_finished_jobs']
        self.finished_job_ids = set(save['finished_job_ids'])
        self.duration = save['duration']
        self.log_record = save['log']


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
        super(self.__class__, self).__init__(ranges, maximise_cost)
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
        super(self.__class__, self).__init__(ranges, maximise_cost)
        self.allow_re_tests = allow_re_tests
        self.tested_configurations = set()
        self.max_retries = max_retries
        self.params = sorted(self.ranges.keys())

    def total_jobs(self):
        if self.allow_re_tests:
            return inf
        else:
            return super(self.__class__, self).total_jobs()

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

def close_to_any(x, xs, tol=1e-5):
    ''' whether the point x is close to any of the points in xs
    x: the point to test. shape=(1, num_attribs)
    xs: the points to compare with. shape=(num_points, num_attribs)
    tol: maximum size of the squared Euclidean distance to be considered 'close'
    '''
    assert x.shape[1] == xs.shape[1], 'different number of attributes'
    assert x.shape[0] == 1, 'x must be a single point'
    assert xs.shape[0] > 0, 'xs must not be empty'
    assert len(x.shape) == len(xs.shape) == 2, 'must be 2D arrays'

    #return np.any(np.linalg.norm(xs - x, axis=1) <= tol)  # l2 norm (Euclidean distance)
    # x is subtracted from each row of xs, each element is squared, each row is
    # summed to leave a 1D array and each sum is checked with the tolerance
    return np.any(np.sum((xs - x)**2, axis=1) <= tol) # squared Euclidean distance


class BayesianOptimisationOptimiser(Optimiser):
    def __init__(self, ranges, maximise_cost=False,
                 acquisition_function='EI', acquisition_function_params=dict(),
                 gp_params=None, pre_samples=4, ac_num_restarts=10,
                 close_tolerance=1e-5):
        '''
        acquisition_function: the function to determine where to sample next
            either a function or a string with the name of the function (eg 'EI')
        acquisition_function_params: a dictionary of parameter names and values
            to be passed to the acquisition function. (see specific acquisition
            function for details on what parameters it takes)
        gp_params: parameter dictionary for the Gaussian Process surrogate
            function, None will choose some sensible defaults. (See "sklearn
            gaussian process regressor")
        pre_samples: the number of samples to be taken randomly before starting
            Bayesian optimisation
        ac_num_restarts: number of restarts during the acquisition function
            optimisation. Higher => more likely to find the optimum of the
            acquisition function.
        close_tolerance: in some situations Bayesian optimisation may get stuck
            on local optima and will continue to sample points roughly in the
            same location. When this happens the GP can break (as input values
            must be unique within some tolerance). It is also a waste of
            resources to sample lots of times in a very small neighbourhood.
            Instead, when the next sample is to be 'close' to any of the points
            sampled before (ie squared Euclidean distance <= close_tolerance),
            sample a random point instead.
        '''
        ranges = {param:np.array(range_) for param,range_ in ranges.items()} # numpy arrays are required
        super(self.__class__, self).__init__(ranges, maximise_cost)

        self.acquisition_function_params = acquisition_function_params
        ac_param_keys = set(self.acquisition_function_params.keys())
        if acquisition_function == 'EI':
            self.acquisition_function_name = 'EI'
            self.acquisition_function = self.expected_improvement
            # <= is subset. Not all params must be provided, but those given must be valid
            assert ac_param_keys <= set(['xi']), 'invalid acquisition function parameters'

        elif acquisition_function == 'UCB':
            self.acquisition_function_name = 'UCB'
            self.acquisition_function = self.upper_confidence_bound
            # <= is subset. Not all params must be provided, but those given must be valid
            assert ac_param_keys <= set(['kappa']), 'invalid acquisition function parameters'

        else:
            self.acquisition_function_name = 'custom acquisition function'
            self.acquisition_function = acquisition_function

        if gp_params is None:
            self.gp_params = dict(
                alpha = 1e-5, # larger => more noise. Default = 1e-10
                # the default kernel
                kernel = gp.kernels.RBF(length_scale=1.0, length_scale_bounds="fixed"),
                n_restarts_optimizer = 10,
                # make the mean 0 (theoretically a bad thing, see docs, but can help)
                normalize_y = True,
                copy_X_train = True # whether to make a copy of the training data (in-case it is modified)
            )
        else:
            self.gp_params = gp_params

        self.pre_samples = pre_samples
        self.ac_num_restarts = ac_num_restarts
        self.close_tolerance = close_tolerance

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

        # Only provide bounds for the parameters that are included in
        # self.config_to_point. Provide the log(lower), log(upper) bounds for
        # logarithmically spaced ranges.
        self.point_bounds = []
        for param in self.params: # self.params is ordered
            type_ = self.range_types[param]
            low, high = self.range_bounds[param]
            if type_ == 'linear':
                self.point_bounds.append((low, high))
            elif type_ == 'logarithmic':
                self.point_bounds.append((np.log(low), np.log(high)))

        # not ready for a next configuration until the job with id ==
        # self.wait_until has been processed
        self.wait_for_job = None

        # a log of the Bayesian optimisation steps
        # dict of job number to dict with values:
        #
        # sx, sy: numpy arrays corresponding to points of samples taken thus far
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

    def total_jobs(self):
        return inf

    def _ready_for_next_configuration(self):
        in_pre_phase = self.num_started_jobs < self.pre_samples
        # all jobs from the pre-phase are finished, need the first Bayesian
        # optimisation sample
        pre_phase_finished = (self.wait_for_job == None and
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
        # scipy has no maximise function, so instead minimise the negation of the acquisition function
        # reshape(1,-1) => 1 sample (row) with N attributes (cols). Needed because x is passed as shape (N,)
        # unpacking the params dict is harmless if the dict is empty
        neg_acquisition_function = lambda x: -self.acquisition_function(
            x.reshape(1,-1), gp_model, self.maximise_cost, best_cost, **self.acquisition_function_params)

        # minimise the negative acquisition function
        best_next_x = None
        best_neg_ac = inf # negative acquisition function value for best_next_x
        for j in range(self.ac_num_restarts):
            # this random configuration can be anywhere, it doesn't matter if it
            # is close to an existing sample.
            starting_point = self.config_to_point(self._random_config())

            # result is an OptimizeResult object
            result = scipy.optimize.minimize(
                fun=neg_acquisition_function,
                x0=starting_point,
                bounds=self.point_bounds,
                method='L-BFGS-B', # Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Bounded
                options=dict(maxiter=15000) # maxiter=15000 is default
            )
            if not result.success:
                self._log('negative acquisition minimisation failed, restarting')
                continue

            # result.fun == negative acquisition function evaluated at result.x
            if result.fun < best_neg_ac:
                best_next_x = result.x
                best_neg_ac = result.fun

        # acquisition function optimisation finished:
        # best_next_x = argmax(acquisition_function)

        if best_next_x is None:
            self._log('all the restarts of the maximisation failed')
            return None, 0
        else:
            # reshape to make shape=(1,num_attribs) and negate best_neg_ac to make
            # it the positive acquisition function value
            return best_next_x.reshape(1,-1), -best_neg_ac

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
            self.wait_for_job = job_num # do not add a new job until this job has been processed

            # samples converted to points which can be used in calculations
            # shape=(num_samples, num_attribs)
            sx = np.vstack([self.config_to_point(s.config) for s in self.samples])
            # shape=(num_samples, 1)
            sy = np.array([[s.cost] for s in self.samples])

            # setting up a new model each time shouldn't be too wasteful and it
            # has the benefit of being easily reproducible (eg for plotting)
            # because the model is definitely 'clean' each time. In my tests,
            # there was no perceptible difference in timing.
            gp_model = gp.GaussianProcessRegressor(**self.gp_params)
            gp_model.fit(sx, sy)

            # best known configuration and the corresponding cost of that configuration
            best_sample = self.best_sample()

            next_x, next_ac = self._maximise_acquisition(gp_model, best_sample.cost)

            # next_x as chosen by the acquisition function maximisation (for the step log)
            argmax_acquisition = next_x

            # maximising the acquisition function failed
            if next_x is None:
                self._log('choosing random sample because maximising acquisition function failed')
                next_x = self._unique_random_config(different_from=sx, num_attempts=1000)
                next_ac = 0
                chosen_at_random = True
            # acquisition function successfully maximised, but the resulting configuration would break the GP.
            # having two samples too close together will 'break' the GP
            elif close_to_any(next_x, sx, self.close_tolerance):
                self._log('choosing random sample to avoid samples being too close')
                next_x = self._unique_random_config(different_from=sx, num_attempts=1000)
                next_ac = 0
                chosen_at_random = True
            else:
                next_x = self.point_to_config(next_x)
                chosen_at_random = False

            self.step_log[job_num] = dict(
                sx=sx, sy=sy,
                best_sample=best_sample,
                next_x=next_x, next_ac=next_ac, # chosen_at_random => next_ac=0
                chosen_at_random=chosen_at_random,
                argmax_acquisition=argmax_acquisition # different to next_x when chosen_at_random
            )
            self.trim_step_log()

            return next_x



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

    @staticmethod
    def upper_confidence_bound(xs, gp_model, maximise_cost, best_cost, kappa=1.0):
        '''
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

        As a point, constant parameters are ignored, and values from logarithmic
        ranges are the exponents of the values. ie a value of 'n' as a point
        corresponds to a value of e^n as a configuration.

        config: a dictionary of parameter names to values
        returns: numpy array with shape=(1,num_attribs)
        '''
        assert set(config.keys()) == set(self.ranges.keys())
        elements = []
        for param in self.params: # self.params is sorted
            type_ = self.range_types[param]
            if type_ == 'linear':
                elements.append(config[param])
            elif type_ == 'logarithmic':
                elements.append(np.log(config[param]))
        return np.array([elements])

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

            if type_ == 'constant':
                config[param] = self.ranges[param][0] # only 1 choice
            else:
                if pi >= point.shape[1]:
                    raise ValueError('point has too few attributes')
                val = point[0,pi]
                pi += 1

                if type_ == 'linear':
                    config[param] = val
                elif type_ == 'logarithmic':
                    config[param] = np.exp(val)

        if pi != point.shape[1]: raise ValueError('point has too many attributes')

        return dotdict(config)



    def _save_dict(self):
        save = super(self.__class__, self)._save_dict()
        save['step_log'] = self.step_log
        return save

    def _load_dict(self, save):
        super(self.__class__, self)._load_dict(save)
        self.step_log = save['step_log']



    def plot_step_slice(self, param, step, true_cost=None, log_ac=False, n_sigma=2):
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
        n_sigma: the number of standard deviations from the mean to plot the
            uncertainty confidence inerval.
            Note 1=>68%, 2=>95%, 3=>99% (for a normal distribution, which this is)
        '''
        assert step in self.step_log.keys(), 'step not recorded in the log'

        type_ = self.range_types[param]
        assert type_ in ['linear', 'logarithmic']
        is_log = type_ == 'logarithmic' # whether the range of the chosen parameter is logarithmic

        s = dotdict(self.step_log[step])
        xs = self.ranges[param]


        gp_model = gp.GaussianProcessRegressor(**self.gp_params)
        gp_model.fit(s.sx, s.sy)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Bayesian Optimisation step {}{}'.format(
            step-self.pre_samples,
            (' (chosen at random)' if s.chosen_at_random else '')), fontsize=14)
        ax1.margins(0.01, 0.1)
        ax2.margins(0.01, 0.1)
        if is_log:
            ax1.set_xscale('log')
            ax2.set_xscale('log')
        plt.subplots_adjust(hspace=0.3)

        #plt.subplot(2, 1, 1) # nrows, ncols, plot_number
        #ax1.set_xlabel('parameter: ' + param) # don't need both plots to display the axis
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

        mus, sigmas = gp_model.predict(points, return_std=True)
        mus = mus.flatten()

        #TODO: fit the view to the cost function, don't expand to fit in the uncertainty
        ax1.plot(xs, mus, 'm-', label='surrogate cost')
        ax1.fill_between(xs, mus - n_sigma*sigmas, mus + n_sigma*sigmas, alpha=0.3,
                         color='mediumpurple', label='uncertainty ${}\\sigma$'.format(n_sigma))
        ax1.axvline(x=s.next_x[param])

        if s.chosen_at_random and s.argmax_acquisition is not None:
            ax1.axvline(x=self.point_to_config(s.argmax_acquisition)[param], color='y')

        ax1.legend()

        #plt.subplot(2, 1, 2) # nrows, ncols, plot_number
        ax2.set_xlabel('parameter {}'.format(param))
        ax2.set_ylabel(self.acquisition_function_name)
        ax2.set_title('acquisition function')

        ac = self.acquisition_function(points, gp_model, self.maximise_cost, s.best_sample.cost, **self.acquisition_function_params)
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

        ax2.plot(xs, ac, '-', color='g', linewidth=1.0, label=label)
        ax2.fill_between(xs, np.zeros_like(xs), ac.flatten(), alpha=0.3, color='palegreen')

        ax2.axvline(x=s.next_x[param])
        # may not want to plot if chosen_at_random because next_ac will be incorrect (ie 0)
        ax2.plot(s.next_x[param], s.next_ac, 'b^', markersize=10, alpha=0.8, label='next sample')

        # when chosen at random, next_x is different from what the maximisation
        # of the acquisition function suggested as the next configuration to
        # test. So plot both.
        if s.chosen_at_random and s.argmax_acquisition is not None:
            ax2.axvline(x=self.point_to_config(s.argmax_acquisition)[param], color='y', label='$\\mathrm{{argmax}}\; {}$'.format(self.acquisition_function_name))

        ax2.legend()

        plt.show()
        return fig


def interactive(loggable, run_task, log_filename=None, poll_interval=0.5):
    '''
    run a task related to a loggable object (eg Optimiser server or Evaluator
    client) in the background while printing its output, catching any
    KeyboardInterrupts and shutting down the loggable object gracefully when one
    occurs.

    Evaluator example:
    >>> evaluator = MyEvaluator()
    >>> task = lambda: evaluator.run_client(host, port)
    >>> op.interactive(evaluator, task, '/tmp/evaluator.log')

    Optimiser example
    >>> optimiser = op.GridSearchOptimiser(ranges)
    >>> task = lambda: optimiser.run_server(host, port, max_jobs=20)
    >>> op.interactive(optimiser, task, '/tmp/optimiser.log')

    loggable: an object with a log_record and stop_flag attributes
    run_task: a function to run (related to the loggable object),
        eg lambda: optimiser.run_sequential(my_evaluator)
    log_filename: filename to write the log to (recommend somewhere in /tmp/) or
        None to not write.
    poll_interval: time to sleep between checking if the log has been updated
    '''
    thread = threading.Thread(target=run_task)
    thread.start()
    f = None
    amount_printed = 0

    def print_more():
        ''' print everything that has been added to the log since amount_printed '''
        length = len(loggable.log_record)
        if length > amount_printed:
            more = loggable.log_record[amount_printed:length]
            print(more, end='')
            if f is not None:
                f.write(more)
                f.flush()
        return length

    try:
        f = open(log_filename, 'w') if log_filename is not None else None
        try:
            while thread.is_alive():
                amount_printed = print_more()
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            print('-- KeyboardInterrupt caught, stopping gracefully --')
            loggable.stop_flag.set()
            thread.join()

        # finish off anything left not printed
        print_more()
        print('-- interactive task finished -- ')
    finally:
        if f is not None:
            f.close()

