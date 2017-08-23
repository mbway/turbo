'''
TODO: module docstring
'''

# fix some of the python2 ugliness
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
if sys.version_info[0] == 3: # python 3
    from math import isclose, inf
elif sys.version_info[0] == 2: # python 2
    inf = float('inf')
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        ''' implementation from the python3 documentation '''
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
else:
    print('unsupported python version')

import time
import json
import struct
import os
import warnings

# for serialising Gaussian Process models for saving to disk
import pickle
import base64
import zlib

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


# constants gathered here so that the defaults can be changed easily (eg for testing)
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 9187
CLIENT_TIMEOUT = 1.0 # seconds for the client to wait for a connection before retrying
SERVER_TIMEOUT = 1.0 # seconds for the server to wait for a connection before retrying
#TODO only needed for sequential running. sequential should probably just crash if not ready. might want to make a note of the change in behaviour in _ready_for_next_configuration
CONFIG_POLL = 0.1 # seconds to wait for the optimiser to be ready for another configuration before retrying
#TODO: no longer need i think. check for other instances of time.sleep
CHECKPOINT_POLL = 1.0 # seconds to wait for outstanding jobs to finish
#TODO: make sure there is something logged on a last resort timeout
# This is to prevent deadlock but if a socket times out then it may be fatal. At
# least this way the problem is visible and you can restore from a checkpoint
# rather than have it sit in deadlock forever.
LAST_RESORT_TIMEOUT = 20.0 # seconds
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
        try:
            return self[name]
        except KeyError:
            raise AttributeError # have to convert to the correct exception
    def __setattr__(self, name, val):
        self[name] = val
    def copy(self):
        ''' copy.copy() does not work with dotdict '''
        return dotdict(dict.copy(self))

class no_op_context():
    '''
    example usage:
    with my_lock or null_context(): # if my_lock is sometimes None
    '''
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        pass

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

def JSON_encode_binary(data):
    ''' encode the given data in a compressed binary format, then encode that in base64 so that it can be stored compactly in a JSON string '''
    return base64.b64encode(zlib.compress(pickle.dumps(data))).decode('utf-8')
def JSON_decode_binary(data):
    ''' decode data encoded with JSON_encode_binary '''
    return pickle.loads(zlib.decompress(base64.b64decode(data)))

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
    '''
    given a duration in the number of seconds (not necessarily an integer),
    format a string of the form: 'HH:MM:SS.X' where HH and X are present only
    when required.
    - HH is displayed if the duration exceeds 1 hour
    - X is displayed if the time does not round to an integer when truncated to
      1dp. eg durations ending in [.1, .9)
    '''
    mins, secs  = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    hours, mins = int(hours), int(mins)

    dps = 1 # decimal places to display
    # if the number of seconds would round to an integer: display it as one
    if isclose(round(secs), secs, abs_tol=10**(-dps)): # like abs_tol=1e-dps
        secs = '{:02d}'.format(int(round(secs)))
    else:
        # 0N => pad with leading zeros up to a total length of N characters
        #       (including the decimal point)
        # .Df => display D digits after the decimal point
        # eg for 2 digits before the decimal point and 1dp: '{:04.1f}'
        chars = 2+1+dps # (before decimal point)+1+(after decimal point)
        secs = ('{:0' + str(chars) +'.' + str(dps) + 'f}').format(secs)

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
            if isinstance(config[p], str) or isinstance(config[p], np.str_):
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
    def __init__(self, config, cost, extra=None):
        '''
        extra: miscellaneous information about the sample. Not used by the
            optimiser but can be used to store useful information for later. The
            extra information will be saved along with the rest of the sample data.
        '''
        self.config = config
        self.cost = cost
        self.extra = {} if extra is None else extra
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
    '''
    read exactly the given number of bytes from the connection
    returns None if the connection closes before the number of bytes is reached
    '''
    data = bytes()
    while len(data) < num_bytes: # until the length is fully read
        left = num_bytes - len(data)
        # documentation recommends a small power of 2 to give best results
        chunk = conn.recv(min(4096, left))
        if len(chunk) == 0: # connection broken: will never receive any data over conn again
            return None
        else:
            data += chunk
    assert len(data) == num_bytes
    return data

def recv_json(conn):
    '''
    receive a JSON object from the given connection
    '''
    # read the length. Be lenient with the connection here since once the length
    # is received the peer obviously wants to communicate, but until then we are
    # not sure. If the connection breaks or times out before the length is read,
    # treat that as if a length of 0 was transmitted.
    try:
        data = read_exactly(conn, 4)
    except socket.timeout: # last resort timeout to prevent deadlock
        return None
    except socket.error as e:
        if e.errno == 104: # connection reset by peer
            return None
        else:
            raise e
    if data is None: # connection closed before a length was sent
        return None # indicates 'no data'
    length, = struct.unpack('!I', data) # see send_json for the protocol
    if length == 0:
        return None # indicates 'no data'
    else:
        data = read_exactly(conn, length)
        assert data is not None, 'json data is None because the connection closed'
        obj = json.loads(data.decode('utf-8'))
        return obj


# Details of the network protocol between the optimiser and the evaluator
# Optimiser sets up a server and listens for clients. Every time a client
# (evaluator) connects the optimiser spawns a thread to handle the client,
# allowing it to accept more clients. In the thread for the connected client, it
# is sent a configuration (serialised as JSON) to evaluate. The message is a
# length followed by the JSON content. The thread waits for the evaluator to reply
# with the results (also JSON serialised). Once the reply has been received, the
# thread is free to handle more clients (as part of a thread pool).
#
# If a client connection is open and the optimiser wishes to shutdown, it can send
# a length of 0 to indicate 'no data' the evaluator can then resume trying to
# connect in-case the server comes back. Alternatively, when the server shuts
# down, the connection breaks (but not always). Both are used to detect a
# disconnection. The server does not keep a record of individual clients and so
# does not notify each client that the server has shut down (only the currently
# connected ones) because it cannot know when it has notified every client.
# After the server shuts down the clients return to attempting connections.
#
# When connecting to the server, there are several different ways the call to
# connect could fail. In any of these situations: simply wait a while and try
# again (without reporting an error)
#
# If an evaluator accepts a job then they are obliged to complete it. If the
# evaluator crashes or otherwise fails to produce results, the job remains in the
# queue and will never finish processing.

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

    def _make_connection(self, host, port, timeout, ignore_stop_flag):
        connected = False
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP
        self.log('attempting to connect to {}:{}'.format(host, port))
        sock.settimeout(timeout) # only while trying to connect

        # connect to the server
        while ignore_stop_flag or not self.wants_to_stop():
            try:
                sock.connect((host, port))
                self.log('connection established')
                # should never time out, but don't want deadlock
                sock.settimeout(LAST_RESORT_TIMEOUT)
                connected = True
                break
            except socket.timeout:
                continue # this is normal and intentional to keep checking _stop_flag
            except socket.error as e:
                # socket.error is a catch-all instead of manually
                # dealing with individual exceptions. The ones I have
                # encountered are: ConnectionRefusedError,
                # ConnectionAbortedError, ConnectionResetError,
                # BlockingIOError, SIGPIPE

                # don't add the full traceback because it will break the unit tests and it isn't necessary
                self.log('Non-critical error trying to connect to optimiser (will retry): {}: {}'.format(type(e), e))
                #print(e, flush=True)
                time.sleep(NON_CRITICAL_WAIT) # good idea to wait before trying again
                continue

        if not ignore_stop_flag and self.wants_to_stop():
            if connected:
                self.log('shutting down connection')
                send_empty(sock)
            sock.close() # close (free resources) regardless of whether connection was made
            return None
        else:
            return sock

    def _request_job(self, connect_args):
        sock = self._make_connection(*connect_args, ignore_stop_flag=False)
        if sock is None: # evaluator wants to shut down
            return None
        try:
            try:
                send_json(sock, {'type' : 'job_request'})
            except socket.error as e:
                if e.errno == 104: # connection reset by peer
                    return None
                elif e.errno == 32: # broken pipe
                    return None
                else:
                    raise e

            job = recv_json(sock)

            if job is None: # optimiser signalled no job available
                self.log('no job available')
                return None
            else:
                # job has 'config', 'num' and 'setup_duration' fields
                assert set(job.keys()) == set(['config', 'num', 'setup_duration']), 'malformed job request: {}'.format(job)
                #TODO if the job is malformed, don't ACK, instead log the error and discard the job
                return job
        finally:
            sock.close()

    def _process_job(self, job, connect_args):
        start_time = time.time()
        job_num = job['num']
        config = dotdict(job['config'])

        self.log('evaluating job {}: config: {}'.format(
            job_num, config_string(config, precise=True)))
        results = self.test_config(config)
        samples = Evaluator.samples_from_test_results(results, config)
        # for JSON serialisation
        samples = [(s.config, s.cost, s.extra) for s in samples]

        # will make a connection regardless of whether stop_flag is set
        sock = self._make_connection(*connect_args, ignore_stop_flag=True)
        assert sock is not None
        # don't print extra values because it could be large
        self.log('returning results: {}'.format([(config, cost, list(extra.keys())) for config, cost, extra in samples]))
        try:
            # add this field
            job['evaluation_duration'] = time.time()-start_time
            msg_dict = {
                'type' : 'job_results',
                'job' : job,
                'samples' : samples
            }
            send_json(sock, msg_dict, encoder=NumpyJSONEncoder)
        finally:
            sock.close()

    def run_client(self, host=DEFAULT_HOST, port=DEFAULT_PORT, max_jobs=inf,
                   timeout=CLIENT_TIMEOUT):
        self._stop_flag.clear()
        num_jobs = 0
        connect_args = (host, port, timeout)
        try:
            while not self.wants_to_stop() and num_jobs < max_jobs:
                job = self._request_job(connect_args)
                if job is None:
                    # either no job available, in which case try again, or the
                    # evaluator wants to shut down, in which case the loop will stop.
                    if self.wants_to_stop():
                        break
                    else:
                        time.sleep(NON_CRITICAL_WAIT)
                        continue
                else:
                    # will return the results regardless of whether the stop
                    # flag is set because the contract with the server is that
                    # the evaluator _must_ return the results of a job it
                    # started.
                    self._process_job(job, connect_args)
                    num_jobs += 1

            if self.wants_to_stop():
                self.log('stopping because of manual shut down')
            elif num_jobs >= max_jobs:
                self.log('stopping because max_jobs reached')
        except Exception as e:
            self.log('Exception raised during client run:\n{}'.format(exception_string()))
            ON_EXCEPTION(e)
        finally:
            self.log('evaluator shut down')


    def run_client_old(self, host=DEFAULT_HOST, port=DEFAULT_PORT, max_jobs=inf,
                   poll_interval=CLIENT_TIMEOUT):
        '''
        receive jobs from an Optimiser server and evaluate them until the server
        shuts down.

        host: the hostname/IP where the optimiser server is running
        port: the port number that the optimiser server is listening on
        max_jobs: the maximum number of jobs to allow for this run (not in total)
        poll_interval: seconds for the client to wait for a connection before retrying
        '''
        sock = None
        try:
            self.log('evaluator client starting...')
            self._stop_flag.clear()
            num_jobs = 0

            while not self._stop_flag.is_set() and num_jobs < max_jobs:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP
                self.log('attempting to connect to {}:{}'.format(host, port))
                sock.settimeout(poll_interval) # only while trying to connect

                # connect to the server
                while not self._stop_flag.is_set():
                    try:
                        sock.connect((host, port))
                        break
                    except socket.timeout:
                        continue # this is normal and intentional to keep checking _stop_flag
                    except socket.error as e:
                        # socket.error is a catch-all instead of manually
                        # dealing with individual exceptions. The ones I have
                        # encountered are: ConnectionRefusedError,
                        # ConnectionAbortedError, ConnectionResetError,
                        # BlockingIOError

                        #TODO: handle SIGPIPE
                        # don't add the full traceback because it will break the unit tests and it isn't necessary
                        self.log('Non-critical error trying to connect to optimiser (will retry): {}: {}'.format(type(e), e))
                        time.sleep(1) # good idea to wait before trying again
                        continue
                if self._stop_flag.is_set():
                    break

                self.log('connection established')
                sock.settimeout(None) # wait forever (blocking mode)

                try:
                    job = recv_json(sock)
                except socket.error as e:
                    self.log('FAILED to receive job from optimiser ({}: {})'.format(type(e), e))
                    continue

                # optimiser finished and sent 0 length to inform the evaluator
                if job is None:
                    self.log('FAILED to receive job from optimiser (0 length)')
                    continue

                job_num = job['num']
                config = dotdict(job['config'])

                self.log('evaluating job {}: config: {}'.format(
                    job_num, config_string(config, precise=True)))
                results = self.test_config(config)
                samples = Evaluator.samples_from_test_results(results, config)

                # for JSON serialisation
                samples = [(s.config, s.cost, s.extra) for s in samples]

                self.log('returning results: {}'.format(results))
                send_json(sock, {'samples' : samples}, encoder=NumpyJSONEncoder)
                sock.close()

                num_jobs += 1

            if self._stop_flag.is_set():
                self.log('stopping because of manual shut down')
            elif num_jobs >= max_jobs:
                self.log('stopping because max_jobs reached')
        except Exception as e:
            self.log('Exception raised during client run:\n{}'.format(exception_string()))
            ON_EXCEPTION(e)
        finally:
            self.log('evaluator shutting down')
            if sock is not None:
                sock.close()

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

    def _wait_for_ready(self):
        '''
        wait for the optimiser to be ready to produce another configuration to
        test. Has no effect if already ready.
        '''
        if not self._ready_for_next_configuration():
            self._log('not ready for the next configuration yet')
            while not self._ready_for_next_configuration():
                time.sleep(CONFIG_POLL)
            self._log('now ready for next configuration')

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
        elif self._stop_flag.is_set():
            self._log('optimisation manually shut down gracefully')
        elif out_of_configs and outstanding_jobs == 0:
            self._log('optimisation finished (out of configurations).')
        else:
            self._log('stopped for an unknown reason. May be in an inconsistent state. (details: {} / {} / {})'.format(
                self._stop_flag.is_set(), out_of_configs, outstanding_jobs))

    def _handle_checkpoint(self, lock=None):
        ''' handle saving a checkpoint during a run if signalled to '''
        if self._checkpoint_flag.is_set():
            if self.num_started_jobs > self.num_finished_jobs:
                self._log('waiting for {} outstanding jobs to finish before taking a snapshot'.format(
                    self.num_started_jobs-self.num_finished_jobs))
                while self.num_started_jobs > self.num_finished_jobs:
                    time.sleep(CHECKPOINT_POLL)

            with lock or no_op_context(): # use no-op if lock is None
                self._log('saving checkpoint to "{}"'.format(self.checkpoint_filename))
                self.save_now()
                self._checkpoint_flag.clear()

    def _handle_client_old(self, conn, lock, exception_caught, job):
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
            # during transmission: serialised to tuples
            results = [Sample(dotdict(config), cost, extra)
                       for config, cost, extra in results['samples']]

            with lock:
                self._process_job_results(job, results)
        except Exception as e:
            exception_caught.set()
            with lock:
                self._log('Exception raised while processing job {} (in a worker thread):\n{}'.format(
                    job.num, exception_string()))
            ON_EXCEPTION(e)
        finally:
            conn.close()

    def _accept_connection(self, sock):
        '''
        wait for a client to connect
        note: may throw exceptions
        note: may return a connection even if stop flag is set
        '''
        while not self._stop_flag.is_set():
            try:
                # conn is another socket object
                conn, addr = sock.accept()
                self._log('connection from {}:{}'.format(*addr))
                return conn, addr
            except socket.timeout:
                # if the checkpoint flag is set, give clients a chance to
                # connect (unlike with the stop flag) but if there are none
                # waiting, return so the optimiser can check if it is ready to
                # make a checkpoint.
                if self._checkpoint_flag.is_set():
                    return None, None
                else:
                    continue
        return None, None

    def _handle_client(self, conn, state, max_jobs):
        '''
        handle a single interaction with a client
        '''
        try:
            msg = recv_json(conn)
        except socket.timeout:
            self._log('evaluator connection timed out')
            conn.shutdown(socket.SHUT_RDWR)
            time.sleep(NON_CRITICAL_WAIT)
            return

        #TODO: if msg is not None: check that 'type' in keys
        if msg is None: # evaluator wishes to shutdown
            self._log('evaluator requested nothing')
            conn.shutdown(socket.SHUT_RDWR)
            time.sleep(NON_CRITICAL_WAIT)

        elif msg['type'] == 'job_request':
            if state.started_this_run >= max_jobs:
                self._log('not allowing new jobs (stopping)')
                send_empty(conn)
            elif self._checkpoint_flag.is_set():
                self._log('not allowing new jobs (taking checkpoint)')
                send_empty(conn)
            elif not self._ready_for_next_configuration():
                self._log('not allowing new jobs (not ready)')
                send_empty(conn)
            else:
                job = self._next_job()
                if job is None:
                    state.out_of_configs = True
                    send_empty(conn) # signal no more jobs available
                else:
                    state.started_job_ids.add(job.num)
                    state.started_this_run += 1
                    job_dict = {
                        'config' : job.config,
                        'num' : job.num,
                        'setup_duration' : job.setup_duration
                    }
                    send_json(conn, job_dict, encoder=NumpyJSONEncoder)

        elif msg['type'] == 'job_results':
            assert set(msg.keys()) == set(['type', 'job', 'samples']), 'malformed message: {}'.format(msg)
            assert set(msg['job'].keys()) == set(['config', 'num', 'setup_duration', 'evaluation_duration']), 'malformed job message: {}'.format(msg['job'])

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

        else:
            raise Exception('invalid request: {}'.format(msg))

    def run_server(self, host=DEFAULT_HOST, port=DEFAULT_PORT, max_jobs=inf,
                   timeout=SERVER_TIMEOUT):
        self._log('starting optimisation server at {}:{}'.format(host, port))

        self._stop_flag.clear()
        old_duration = self.duration # duration as-of the start of this run
        checkpoint_flag_was_set = False # to only log once

        class RunState:
            def __init__(self):
                self.start_time = time.time()

                # count number of jobs started and finished this run
                self.started_this_run = 0
                self.finished_this_run = 0
                self.started_job_ids = set()

                # flags to diagnose the stopping conditions
                self.out_of_configs = False
                self.exception_caught = False

            def run_time(self):
                return time.time() - self.start_time
            def num_outstanding_jobs(self):
                return self.started_this_run - self.finished_this_run
        state = RunState()

        sock = None
        try:
            # server socket for the clients to connect to
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # able to re-use host/port combo even if in use
            sock.bind((host, port))
        except Exception as e:
            self._log('failed to set up optimiser server with exception: {}'.format(exception_string()))
            return

        #TODO
        sock.listen(16) # maximum number of connections allowed to queue
        sock.settimeout(timeout) # timeout for accept, not inherited by the client connections

        conn = None
        try:
            ready_to_stop = lambda: self._stop_flag.is_set() and state.num_outstanding_jobs() == 0
            while (not ready_to_stop() and
                   state.finished_this_run < max_jobs and
                   not state.out_of_configs):

                #self._log('waiting for a connection')
                #outstanding = started_job_ids - self.finished_job_ids
                #self._log('outstanding job IDs: {}'.format(set_str(outstanding)))

                # deal with checkpoints
                if self._checkpoint_flag.is_set():
                    outstanding_count = state.num_outstanding_jobs()
                    if outstanding_count > 0 and not checkpoint_flag_was_set:
                        self._log('waiting for {} outstanding jobs to finish before taking a snapshot'.format(
                            outstanding_count))
                        checkpoint_flag_was_set = True
                    elif outstanding_count == 0:
                        self.save_now()
                        self._checkpoint_flag.clear()
                        checkpoint_flag_was_set = False
                    elif outstanding_count < 0:
                        raise ValueError(outstanding_count)


                conn, addr = self._accept_connection(sock)
                if conn is None:
                    # stop flag was set _before_ a client connected
                    # or checkpoint flag was set _and_ no client connected
                    continue
                # stop flag may be set after the client connected, in which case we
                # have to serve that client before stopping


                try:
                    conn.settimeout(LAST_RESORT_TIMEOUT)
                    self._handle_client(conn, state, max_jobs)
                finally:
                    conn.close()
                    conn = None

                self.duration = old_duration + state.run_time()

        except Exception as e:
            self._log('Exception raised during run:\n{}'.format(exception_string()))
            state.exception_caught = True
            ON_EXCEPTION(e)
        finally:
            if sock is not None:
                sock.shutdown(socket.SHUT_RDWR)
                #TODO: how much waiting is required?
                #time.sleep(1)
                sock.close()
            if conn is not None:
                conn.close()

        if self._checkpoint_flag.is_set():
            self._log('saving checkpoint to "{}"'.format(self.checkpoint_filename))
            self.save_now()
            self._checkpoint_flag.clear()

        max_jobs_exceeded = state.finished_this_run >= max_jobs
        #TODO refactor to just pass state
        self._shutdown_message(old_duration, state.run_time(),
                               max_jobs_exceeded, state.out_of_configs, state.exception_caught)



    def run_server_old(self, host=DEFAULT_HOST, port=DEFAULT_PORT, max_clients=4,
                   max_jobs=inf, poll_interval=SERVER_TIMEOUT):
        '''
        run a server which serves jobs to any listening evaluators. Evaluator
        clients are assumed to be interchangeable, performing the same
        calculations, excluding some random variation. Evaluators may process
        jobs at different rates, faster evaluators will be assigned more jobs
        and so need not sit idle.

        host: the hostname/IP for the optimiser to listen on
        port: the port number for the optimiser to listen on
        max_clients: the maximum number of clients to expect to connect. If
            another connects then their job will not be served until there is a
            free thread to deal with it.
        max_jobs: the maximum number of jobs to allow for this run (not in total)
        poll_interval: seconds for the server to wait for a connection before retrying
        '''
        pool = None
        sock = None
        conn = None
        try:
            self._log('starting optimisation server...')

            self._stop_flag.clear()
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

            # for some reason even with REUSEADDR the socket may fail to bind.
            # it may be because it takes a moment for the socket to reset after closing
            while True:
                try:
                    sock.bind((host, port))
                    break
                except OSError as e:
                    if e.errno == 98: # address already in use
                        self._log('address already in use, try again')
                        time.sleep(1)
                        continue
                    else:
                        raise e

            sock.listen(max_clients) # enable listening on the socket. backlog=max_clients
            sock.settimeout(poll_interval) # timeout for accept, not inherited by the client connections

            while (not self._stop_flag.is_set() and
                   not exception_in_pool.is_set() and
                   num_jobs < max_jobs):

                with lock:
                    self._log('outstanding job IDs: {}'.format(
                        set_str(started_job_ids-self.finished_job_ids)))
                    self._log('waiting for a connection')

                # wait for a client to connect
                while not self._stop_flag.is_set() and not exception_in_pool.is_set():
                    self._handle_checkpoint(lock) # save a checkpoint if signalled to
                    try:
                        # conn is another socket object
                        conn, addr = sock.accept()
                        break
                    except socket.timeout:
                        conn = None
                        continue
                if self._stop_flag.is_set() or exception_in_pool.is_set():
                    break

                with lock:
                    self._log('connection accepted from {}:{}'.format(*addr))

                # don't want to wait for the next configuration while holding the lock
                # since this may result in deadlock
                self._wait_for_ready()

                # don't want finished samples to be added during the Bayes step
                with lock:
                    # will wait for ready again, but should have no effect since
                    # the optimiser is now ready
                    job = self._next_job()
                    if job is None:
                        out_of_configs = True
                        break
                    started_job_ids.add(job.num)

                pool.apply_async(self._handle_client_old, (conn, lock, exception_in_pool, job))
                #self._handle_client(conn, lock, exception_in_pool, job)

                conn = None
                num_jobs += 1
                with lock:
                    self.duration = old_duration + (time.time()-run_start_time)

        except Exception as e:
            self._log('Exception raised during run:\n{}'.format(exception_string()))
            exception_caught = True
            ON_EXCEPTION(e)
        finally:
            # stop accepting new connections
            if sock is not None:
                sock.close()

            # finish up active jobs
            if pool is not None:
                with lock:
                    self._log('waiting for active jobs to finish')
                pool.close() # worker threads will exit once done
                pool.join()
                self._log('active jobs finished')

            # clean up lingering client connection if there is one (connection
            # was accepted before optimiser stopped)
            if conn is not None:
                self._log('notifying client which was waiting for a job')
                send_empty(conn)
                conn.close()

        self._handle_checkpoint() # save a checkpoint if signalled to

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

            self._stop_flag.clear()
            run_start_time = time.time()
            old_duration = self.duration # duration as-of the start of this run
            num_jobs = 0 # count number of jobs this run

            # flags to diagnose the stopping conditions
            out_of_configs = False
            exception_caught = False

            while not self._stop_flag.is_set() and num_jobs < max_jobs:
                self._handle_checkpoint() # save a checkpoint if signalled to
                self._wait_for_ready()
                job = self._next_job()
                if job is None:
                    out_of_configs = True
                    break

                evaluation_start = time.time()
                results = evaluator.test_config(job.config)
                job.evaluation_duration = time.time() - evaluation_start

                self._process_job_results(job, results)

                num_jobs += 1
                self.duration = old_duration + (time.time()-run_start_time)

        except Exception as e:
            self._log('Exception raised during run:\n{}'.format(exception_string()))
            exception_caught = True
            ON_EXCEPTION(e)

        self._handle_checkpoint() # save a checkpoint if signalled to

        this_duration = time.time() - run_start_time
        max_jobs_exceeded = num_jobs >= max_jobs
        self._shutdown_message(old_duration, this_duration,
                               max_jobs_exceeded, out_of_configs, exception_caught)

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



class RangeType:
    ''' The possible types of parameter ranges (see range_type() for details) '''
    Arbitrary   = 'arbitrary'
    Constant    = 'constant'
    Linear      = 'linear'
    Logarithmic = 'logarithmic'

def range_type(range_):
    ''' determine whether the range is arbitrary, constant, linear or logarithmic
    range_: must be numpy array

    Note: range_ must be sorted either ascending or descending to be detected as
        linear or logarithmic

    Range types:
        - Arbitrary: 0 or >1 element, not linear or logarithmic (perhaps not numeric)
                     Note: arrays of identical or nearly identical elements are
                     Arbitrary, not Constant
        - Constant: 1 element (perhaps not numeric)
        - Linear: >2 elements, constant non-zero difference between adjacent elements
        - Logarithmic: >2 elements, constant non-zero difference between adjacent log(elements)
    '''
    if len(range_) == 1:
        return RangeType.Constant
    # 'i' => integer, 'u' => unsigned integer, 'f' => floating point
    elif len(range_) < 2 or range_.dtype.kind not in 'iuf':
        return RangeType.Arbitrary
    else:
        # guaranteed: >2 elements, numeric

        # if every element is identical then it is Arbitrary. Not Constant
        # because constant ranges must have a single element.
        if np.all(np.isclose(range_[0], range_)):
            return RangeType.Arbitrary

        tmp = range_[1:] - range_[:-1] # differences between element i and element i+1
        # same non-zero difference between each element
        is_lin = np.all(np.isclose(tmp[0], tmp))
        if is_lin:
            return RangeType.Linear
        else:
            tmp = np.log(range_)
            tmp = tmp[1:] - tmp[:-1]
            is_log = np.all(np.isclose(tmp[0], tmp))
            if is_log:
                return RangeType.Logarithmic
            else:
                return RangeType.Arbitrary

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
                 acquisition_function='EI', acquisition_function_params=None,
                 gp_params=None, pre_samples=4, ac_num_restarts=10,
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
        allow_parallel: whether to hypothesise about the results of ongoing jobs
            in order to start another job in parallel. (useful when running a
            server with multiple client evaluators).
        '''
        ranges = {param:np.array(range_) for param, range_ in ranges.items()} # numpy arrays are required
        super(BayesianOptimisationOptimiser, self).__init__(ranges, maximise_cost)

        self.acquisition_function_params = ({} if acquisition_function_params is None
                                            else acquisition_function_params)
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

        elif callable(acquisition_function):
            self.acquisition_function_name = 'custom acquisition function'
            self.acquisition_function = acquisition_function
        else:
            raise ValueError('invalid acquisition_function')

        if gp_params is None:
            self.gp_params = dict(
                alpha = 1e-5, # larger => more noise. Default = 1e-10
                # the default kernel
                kernel = 1.0 * gp.kernels.RBF(length_scale=1.0, length_scale_bounds="fixed"),
                n_restarts_optimizer = 10,
                # make the mean 0 (theoretically a bad thing, see docs, but can help)
                normalize_y = True,
                copy_X_train = True # whether to make a copy of the training data (in-case it is modified)
            )
        else:
            self.gp_params = gp_params

        assert pre_samples > 1, 'not enough pre-samples'
        self.pre_samples = pre_samples
        self.ac_num_restarts = ac_num_restarts
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
        # scipy has no maximise function, so instead minimise the negation of the acquisition function
        # reshape(1,-1) => 1 sample (row) with N attributes (cols). Needed because x is passed as shape (N,)
        # unpacking the params dict is harmless if the dict is empty
        neg_acquisition_function = lambda x: -self.acquisition_function(
            x.reshape(1, -1), gp_model, self.maximise_cost, best_cost,
            **self.acquisition_function_params)

        # minimise the negative acquisition function
        best_next_x = None
        best_neg_ac = inf # negative acquisition function value for best_next_x
        for j in range(self.ac_num_restarts):
            # this random configuration can be anywhere, it doesn't matter if it
            # is close to an existing sample.
            starting_point = self.config_to_point(self._random_config())

            # result is an OptimizeResult object
            # if something goes wrong, scikit will write a warning to stderr by
            # default. Instead capture the warnings and log them
            with warnings.catch_warnings(record=True) as w:
                result = scipy.optimize.minimize(
                    fun=neg_acquisition_function,
                    x0=starting_point,
                    bounds=self.point_bounds,
                    method='L-BFGS-B', # Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Bounded
                    options=dict(maxiter=15000) # maxiter=15000 is default
                )
                if len(w) > 0:
                    for warn in w:
                        self._log('warning when maximising the acquisition function: {}'.format(warn))
            if not result.success:
                self._log('restart {}/{} of negative acquisition minimisation failed'.format(
                    j, self.ac_num_restarts))
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
            return best_next_x.reshape(1, -1), -best_neg_ac

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
        with warnings.catch_warnings(record=True) as w:
            # NOTE: fitting only optimises _certain_ kernel parameters with given
            # bounds, see gp_model.kernel_.theta for the optimised kernel
            # parameters.
            # NOTE: RBF(...) has NO parameters to optimise, however 1.0 * RBF(...) does!
            gp_model.fit(xs, ys)

            if len(w) > 0:
                for warn in w:
                    self._log('warning when fitting the gp: {}'.format(warn))

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

            if type_ == RangeType.Linear:
                low, high = self.range_bounds[param]
                config[param] = np.random.uniform(low, high)

            elif type_ == RangeType.Logarithmic:
                low, high = self.range_bounds[param]
                config[param] = log_uniform(low, high)

            elif type_ == RangeType.Constant:
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
            if type_ == RangeType.Linear:
                elements.append(config[param])
            elif type_ == RangeType.Logarithmic:
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
                    is_2d(step['next_ac']), is_2d(step['argmax_acquisition']),

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
            for key in ['sx', 'sy', 'hx', 'hy', 'next_ac', 'argmax_acquisition']:
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
        assert type_ in [RangeType.Linear, RangeType.Logarithmic]
        is_log = type_ == RangeType.Logarithmic # whether the range of the chosen parameter is logarithmic

        s = dotdict(self.step_log[step])
        all_xs = self.ranges[param]

        # combination of the true samples (sx, sy) and the hypothesised samples
        # (hx, hy) if there are any
        xs = np.vstack([s.sx, s.hx])
        ys = np.vstack([s.sy, s.hy])

        gp_model = s.gp
        # training the GP is nondeterministic if there are any parameters to
        # tune so may give a different result here to during optimisation
        #gp_model = gp.GaussianProcessRegressor(**self.gp_params)
        #gp_model.fit(xs, ys)

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
            ax1.plot(all_xs, true_cost, 'k--', label='true cost')

        # get the value for the parameter 'param' from the given point
        param_from_point = lambda p: self.point_to_config(p.reshape(1, -1))[param]
        # plot samples projected onto the `param` axis
        # reshape needed because using x in sx reduces each row to a 1D array
        sample_xs = [param_from_point(x) for x in s.sx]
        ax1.plot(sample_xs, s.sy, 'bo', label='samples')

        if len(s.hx) > 0:
            # there are some hypothesised samples
            hypothesised_xs = [param_from_point(x) for x in s.hx]
            ax1.plot(hypothesised_xs, s.hy, 'o', color='tomato', label='hypothesised samples')

        # index of the best current real sample
        best_i = np.argmax(s.sy) if self.maximise_cost else np.argmin(s.sy)
        ax1.plot(sample_xs[best_i], s.sy[best_i], '*', markersize=15,
                 color='deepskyblue', zorder=10, label='best sample')


        def perturb(x):
            '''
            take the next_x configuration and perturb the parameter `param`
            while leaving the others intact this essentially produces a line
            through the parameter space to predict uncertainty along.
            '''
            c = s.next_x.copy()
            c[param] = x
            return self.config_to_point(c)
        points = np.vstack([perturb(x) for x in all_xs])

        mus, sigmas = gp_model.predict(points, return_std=True)
        mus = mus.flatten()

        #TODO: fit the view to the cost function, don't expand to fit in the uncertainty
        ax1.plot(all_xs, mus, 'm-', label='surrogate cost')
        ax1.fill_between(all_xs, mus - n_sigma*sigmas, mus + n_sigma*sigmas, alpha=0.3,
                         color='mediumpurple', label='uncertainty ${}\\sigma$'.format(n_sigma))
        ax1.axvline(x=s.next_x[param])

        if s.chosen_at_random and s.argmax_acquisition is not None:
            ax1.axvline(x=self.point_to_config(s.argmax_acquisition)[param], color='y')

        ax1.legend()

        #plt.subplot(2, 1, 2) # nrows, ncols, plot_number
        ax2.set_xlabel('parameter {}'.format(param))
        ax2.set_ylabel(self.acquisition_function_name)
        ax2.set_title('acquisition function')

        ac = self.acquisition_function(points, gp_model, self.maximise_cost,
                                       s.best_sample.cost,
                                       **self.acquisition_function_params)
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
            ac_x = self.point_to_config(s.argmax_acquisition)[param]
            label='$\\mathrm{{argmax}}\\; {}$'.format(self.acquisition_function_name)
            ax2.axvline(x=ac_x, color='y', label=label)

        ax2.legend()

        plt.show()
        return fig

#TODO: move to gui library
class LogMonitor:
    '''
    asynchronously monitor the log of the given loggable object and output it to
    the given file.
    return a flag which will stop the monitor when set
    '''
    def __init__(self, loggable, f):
        '''
        f: a filename or file object (eg open(..., 'w') or sys.stdout)
        '''
        self.stop_flag = threading.Event()
        self.loggable = loggable
        self.f = open(f, 'w') if isinstance(f, str) else f
    def listen_async(self):
        t = threading.Thread(target=self.listen, name='LogMonitor')
        t.start()
    def listen(self):
        self.stop_flag.clear()
        amount_written = 0
        while not self.stop_flag.is_set():
            length = len(self.loggable.log_record)
            if length > amount_written:
                more = self.loggable.log_record[amount_written:length]
                self.f.write(more)
                self.f.flush()
                amount_written = length
        self.f.close()

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

    loggable: an object with a log_record attribute and stop() method
    run_task: a function to run (related to the loggable object),
        eg lambda: optimiser.run_sequential(my_evaluator)
    log_filename: filename to write the log to (recommend somewhere in /tmp/) or
        None to not write.
    poll_interval: time to sleep between checking if the log has been updated
    '''
    thread = threading.Thread(target=run_task, name='interactive')
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
            loggable.stop()
            thread.join()

        # finish off anything left not printed
        print_more()
        print('-- interactive task finished -- ')
    finally:
        if f is not None:
            f.close()

