#!/usr/bin/env python
from __future__ import print_function
'''
Notes:
- the tests assume that the network is relatively well behaved. When using a
  script to sabotage the connection with delays, packet loss, duplication and
  re-ordering, there is a possibility that the optimisers won't match exactly
  because the response for a job is overtaken by the response to the next job.
  I have ignored this case because the optimiser still works as expected but the
  test fails.
- the tests assume that evaluating the configurations is instant (because it
  assumes that the logs will be character identical and the logs contain the
  durations of the computations)


for debugging the following tools are useful:

- can insert the following code to mirror the logs to files:
```
op_gui.LogMonitor(optimiser, '/tmp/optimiser.log').listen_async()
op_gui.LogMonitor(evaluator, '/tmp/evaluator.log').listen_async()
```

- can use `killall -SIGUSR1 python3` to signal to this process to break into pdb at any time

- can use `less /tmp/frames.txt` to monitor the stack frames of every running thread, updated every second

- can use the following GUI library to monitor optimisers and evaluators as tests run
```
dg = op_gui.DebugGUIs(optimiser, evaluator) # simple case
dg = op_gui.DebugGUIs([optimiser1, optimiser2], [evaluator1, evaluator2]) # or more complex

try:
    run some tests which may fail
finally:
    dg.wait() # wait for user to close windows
    dg.stop() # or force closed
```

'''

import unittest

import os
import sys
import time
import numpy as np
import random
import threading
import statistics
import psutil
import copy
import re
from scipy.stats import uniform
import sklearn.gaussian_process as gp

# local modules
import optimisation as op
import optimisation_gui as op_gui
import optimisation_net as op_net

# whether to skip tests which are slow (useful when writing a new test)
NO_SLOW_TESTS = False

# don't truncate diffs between objects which are not equal
unittest.TestCase.maxDiff = None
unittest.TestCase.longMessage = True

# speed up polling to make tests go faster. For normal usage this just wastes
# CPU since the evaluation should be massively expensive, so there is no reason
# to poll so quickly, however when testing most of the evaluators are trivial
# and so finish instantly.
op.DEFAULT_HOST = '127.0.0.1' # internal only
op.CLIENT_TIMEOUT = 0.2
op.SERVER_TIMEOUT = 0.4
op.NON_CRITICAL_WAIT = 0.5



# use these parameters for the Gaussian processes of the Bayesian optimisers
# used in testing. These defaults somewhat reflect the defaults of scikit. There
# are no parameters to train for this kernel and so trains quickly.
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/gaussian_process/gpr.py
#
# NOTE: THESE ARE NOT GOOD VALUES for real world situations, they are chosen to
# train quickly. See one of the Jupyter notebooks for realistic parameters.
TEST_GP_PARAMS = {
    'alpha':1e-5, # more noise than the default
    'kernel': gp.kernels.ConstantKernel(1.0, constant_value_bounds='fixed') * \
              gp.kernels.RBF(length_scale=1.0, length_scale_bounds='fixed'),
    'n_restarts_optimizer':0, # nothing to train, parameters are fixed
    'normalize_y':True,
    'copy_X_train':True
}

def finish_off_evaluators(timeout=0.4):
    '''
    There is a possibility that the ACK from server->client was lost in which
    case the client will keep re-sending the last results indefinitely since the
    optimiser has shut down. In a real-world scenario they are doing the right
    thing so there is nothing to fix, however since the threads cannot be
    aborted, they must be dealt with some other way.

    Call this function every time that the optimiser shuts down before the evaluator.
    '''
    start_time = time.time()
    def should_stop():
        return time.time() - start_time > timeout
    def handle_request(msg):
        return op_net.empty_msg()
    def on_success(request, response):
        pass
    op_net.message_server((op.DEFAULT_HOST, op.DEFAULT_PORT), 0.1,
                          should_stop, handle_request, on_success)


class NumpyCompatableTestCase(unittest.TestCase):
    '''
    unfortunately when using assertEqual on numpy arrays (especially painful if
    a numpy is nested deep in some structure you are trying to compare) this
    error is raised:
    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

    A workaround is to use tolist() which works fine when comparing numpy arrays
    directly, but not useful for deeply nested structures. Also just overloading
    assertSequenceEqual is not enough because assertDictEqual does not use it
    when a sequence is the value of an item!
    '''

    # for objects of these classes, iterate over __dict__ and convert any
    # attributes which are numpy arrays into lists
    convertable_classes = [op.Sample]

    def convert_np(self, val):
        ''' recursively change all numpy arrays to lists '''
        if isinstance(val, dict):
            return {k: self.convert_np(v) for k, v in val.items()}
        elif isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, list):
            return [self.convert_np(v) for v in val]
        elif isinstance(val, tuple):
            return tuple([self.convert_np(v) for v in val])
        elif hasattr(val, '__dict__') and any(isinstance(val, c) for c in NumpyCompatableTestCase.convertable_classes):
            val.__dict__ = self.convert_np(val.__dict__)
        else:
            return val
    def convert_types(self, val):
        '''
        convert_np destroys the information that one of the dictionaries may
        have a list and the other a numpy array.
        This function recursively converts to a tree of types for comparison.

        don't care about specific sized types and convert types such as
        np.float64 to float
        '''
        if isinstance(val, dict):
            return {k:self.convert_types(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [self.convert_types(v) for v in val]
        elif isinstance(val, np.floating):
            return float
        elif isinstance(val, np.integer):
            return int
        else:
            return type(val)
    def assertDictEqual(self, d1, d2, msg=None):
        unittest.TestCase.assertDictEqual(self, self.convert_types(d1), self.convert_types(d2), 'types don\'t match')
        try:
            unittest.TestCase.assertDictEqual(self, self.convert_np(d1), self.convert_np(d2), msg)
        except ValueError as e:
            op.ON_EXCEPTION(e)

    def assertSequenceEqual(self, it1, it2, msg=None, seq_type=list):
        if seq_type == np.ndarray:
            self.assertEqual(it1.shape, it2.shape, 'shapes don\'t match: {} vs {}'.format(it1.shape, it2.shape))
            self.assertListEqual(it1.tolist(), it2.tolist(), msg)
        unittest.TestCase.assertSequenceEqual(self, it1, it2, msg, seq_type)


def dstring(d, depth=0):
    ''' dict to string, easier to compare the output '''
    s = '{\n'
    for k, v in sorted(d.items(), key=lambda t: t[0]):
        v = dstring(v, depth+1) if isinstance(v, dict) else repr(v).replace('\n', '\n'+'\t'*(depth+1))
        s += '{}"{}" : {}\n'.format('\t'*(depth+1), k, v)
    s += '\t'*depth + '}\n'
    return s

def print_dot():
    ''' print a single dot. Useful to show liveness to user. '''
    sys.stdout.write('.')
    sys.stdout.flush() # required since no EOL

def wait_for(event):
    ''' example usage: wait_for(lambda: some_test()) '''
    while not event():
        time.sleep(0.01)
    # give some time after the condition comes true to make sure caches are flushed etc
    time.sleep(0.1)

this_process = psutil.Process(os.getpid())
def no_open_files():
    return len(this_process.open_files()) == 0

def no_exceptions(self, loggable):
    '''
    assert that there were no exceptions raised.
    may pass an object with a log_record attribute (Optimiser or Evaluator)
    loggable: Optimiser or Evaluator

    make sure this function is called everywhere by inspecting:
    grep -E '(def test_|no_exceptions)' optimisation_tests.py
    '''
    if True:
        if 'Traceback' in loggable.log_record or 'Exception' in loggable.log_record:
            print(loggable.log_record)

    self.assertNotIn('Traceback', loggable.log_record)
    self.assertNotIn('Exception', loggable.log_record)
    self.assertNotIn('inconsistent state', loggable.log_record)

    #self.assertNotIn('error', loggable.log_record)
    #self.assertNotIn('warning', loggable.log_record)
    #self.assertNotIn('Non-critical error', loggable.log_record)

def remove_job_nums(samples):
    for s in samples:
        s.job_num = None
    return samples


class TestUtils(NumpyCompatableTestCase):
    def test_dotdict(self):
        d = {'a' : 1}
        dd = op.dotdict(d)
        self.assertEqual(d, dd) # constructing doesn't change contents
        self.assertEqual(dd.a, dd['a']) # values can be accessed both ways
        dd.a = 2 # values can be assigned to
        self.assertEqual(dd.a, dd['a'])
        self.assertNotEqual(d, dd) # changes don't propagate back to the original dict
        # new values can be created both ways
        dd.b = 3
        dd['c'] = 'test'
        self.assertEqual(dd.b, dd['b'])
        self.assertEqual(dd.c, dd['c'])
        dd['123 problematic key'] = 12 # can still assign to keys which cannot be accessed with a dot
        self.assertEqual(dd, {'a':2,'b':3,'c':'test','123 problematic key':12})

        # must copy dotdict using dotdict.copy and not copy.copy (from the copy module)
        dd = op.dotdict(d)
        c = dd.copy()
        c.abc = 123
        self.assertEqual(dd, d)
        self.assertEqual(c, {'a':1,'abc':123})


    def test_logspace(self):
        self.assertEqual(op.logspace(1, 10).tolist(), [1, 10])
        self.assertEqual(op.logspace(1, 100).tolist(), [1, 10, 100])
        self.assertEqual(op.logspace(1e-2, 100).tolist(), [0.01, 0.1, 1, 10, 100])
        self.assertEqual(op.logspace(1, 100, num_per_mag=2).tolist(), [1, 10**0.5, 10, 10**1.5, 100])

    def test_time_string(self):
        self.assertEqual(op.time_string(0), '00:00')
        self.assertEqual(op.time_string(1), '00:01')
        self.assertEqual(op.time_string(1.09), '00:01')
        self.assertEqual(op.time_string(1.1), '00:01.1')
        self.assertEqual(op.time_string(1.5), '00:01.5')
        self.assertEqual(op.time_string(1.9), '00:01.9')
        self.assertEqual(op.time_string(1.90001), '00:02')
        self.assertEqual(op.time_string(60), '01:00')
        self.assertEqual(op.time_string(61.5), '01:01.5')
        self.assertEqual(op.time_string(60*60 + 61.5), '01:01:01.5')
        self.assertEqual(op.time_string(123*60*60 + 61.5), '123:01:01.5')
        self.assertEqual(op.time_string(123*60*60 + 61), '123:01:01')

    def test_config_string(self):
        self.assertEqual(op.config_string({}), '{}')
        self.assertIn(op.config_string({'a':1,'b':3.456}), ['{a=1, b=3.5}', '{b=3.5, a=1}']) # don't have to specify order
        self.assertEqual(op.config_string({'a':1,'b':3.456,'x':'hi'}, order=['a','b','x']), '{a=1, b=3.5, x="hi"}') # can have several data types
        self.assertEqual(op.config_string({'a':1,'x':'hi'}, order=['x','a']), '{x="hi", a=1}') # can give non-alphabetical order
        self.assertRaises(AssertionError, op.config_string, {'a':1,'b':2}, order=['a']) # if order is given, must list all keys

    def test_range_type(self):
        self.assertEqual(op.range_type(np.linspace(1, 10, num=12)), 'linear')
        self.assertEqual(op.range_type(np.linspace(10, 1, num=12)), 'linear') # can be in descending order
        self.assertEqual(op.range_type(np.array([1,2,3])), 'linear')
        self.assertEqual(op.range_type(np.array([0,10])), 'linear') # can be 2 points
        self.assertEqual(op.range_type(np.logspace(1, 10, num=2)), 'linear') # need at least 3 points to determine logarithmic

        self.assertEqual(op.range_type(np.logspace(1, 10, num=12)), 'logarithmic')
        self.assertEqual(op.range_type(np.logspace(10, 1, num=12)), 'logarithmic') # can be in descending order
        self.assertEqual(op.range_type(np.logspace(1, 10, num=12, base=14)), 'logarithmic') # can be any base

        self.assertEqual(op.range_type(np.array([1, 1])), 'arbitrary')
        self.assertEqual(op.range_type(np.array([0, 0])), 'arbitrary')
        self.assertEqual(op.range_type(np.array([1, 1, 1, 1])), 'arbitrary')
        self.assertEqual(op.range_type(np.array([0, 0, 0, 0])), 'arbitrary')
        self.assertEqual(op.range_type(np.array([5, 5, 5, 5])), 'arbitrary')
        self.assertEqual(op.range_type(np.array([12, 1, 10])), 'arbitrary')
        self.assertEqual(op.range_type(np.array([])), 'arbitrary')
        self.assertEqual(op.range_type(np.array(["hi", "there"])), 'arbitrary')

        self.assertEqual(op.range_type(np.array(["hi"])), 'constant')
        self.assertEqual(op.range_type(np.array([1])), 'constant')

    def test_is_numeric(self):
        self.assertTrue(op.is_numeric(1))
        self.assertTrue(op.is_numeric(1.0))

        self.assertFalse(op.is_numeric([1]))
        self.assertFalse(op.is_numeric(np.array([1])))
        self.assertFalse(op.is_numeric(np.array([[1]])))
        self.assertFalse(op.is_numeric('1'))
        self.assertFalse(op.is_numeric('1.0'))

    @unittest.skipIf(NO_SLOW_TESTS, 'slow test')
    def test_random_config_points(self):
        '''
        test that randomly chosen configurations (through the various methods)
        are distributed correctly (according to the range types).

        note: when dealing with _configs_, the values for a logarithmically
        distributed parameter should span from min to max but be distributed
        logarithmically, that is, more samples closer to min. However when
        dealing with _points_, the values should be uniformly distributed over
        the range log(min) to log(max)
        '''
        amin, amax = -10, -5
        cmin, cmax = 10, 10000
        # uniform between loc and loc+scale
        a_dist = uniform(loc=amin, scale=amax-amin)
        c_point_dist = uniform(loc=np.log(cmin), scale=np.log(cmax)-np.log(cmin))
        b = 4
        ranges = {'a':np.linspace(amin, amax), 'b':[b], 'c':op.logspace(cmin, cmax)}
        opt = op.BayesianOptimisationOptimiser(ranges)

        num_points = 100000
        random_cfgs = [opt._random_config() for _ in range(num_points)]
        random_points = opt._random_config_points(num_points)
        self.assertEqual(random_points.shape, (num_points, 2)) # exclude b constant parameter
        print_dot()

        # make sure that each parameter has a legal value (in range)
        for c in random_cfgs:
            self.assertTrue(amin <= c.a <= amax)
            self.assertEqual(c.b, b)
            self.assertTrue(cmin <= c.c <= cmax)
        for row in random_points:
            self.assertTrue(amin <= row[0] <= amax)
            # note: the values for the points should be
            self.assertTrue(np.log(cmin) <= row[1] <= np.log(cmax))
        print_dot()

        # make sure that converting between points and configurations works correctly
        converted = [opt.point_to_config(opt.config_to_point(c)) for c in random_cfgs]
        self.assertEqual(random_cfgs, converted)
        print_dot()

        # make sure that the mean values of each parameter is similar
        def check(samples, mu, sigma, tol=0.01):
            #print('mean[sample: {}, true: {}]'.format(statistics.mean(samples), mu))
            #print('stddev[sample: {}, true: {}]'.format(statistics.stdev(samples), sigma))
            self.assertTrue(abs(statistics.mean(samples)-mu) <= tol)
            self.assertTrue(abs(statistics.stdev(samples)-sigma) <= tol)

        check([c.a for c in random_cfgs], a_dist.mean(), a_dist.std())
        print_dot()
        check([row[0] for row in random_points], a_dist.mean(), a_dist.std())
        print_dot()
        check([opt.point_to_config(op.make2D_row(row)).a for row in random_points], a_dist.mean(), a_dist.std())
        print_dot()

        r'''
        Take a continuous random variable $X$ and define $Y=g(X)$ for some 1-1 mapping: $y$. For the case of log-uniform: $X\sim\mathcal U(\log b, \log a)$ and $g(x)=e^x$ so $g^{-1}(y)=\log y$.

        to transform a pdf:
        $f_Y(y)=f_X(g^{-1}(y))\left|\frac{\mathrm d}{\mathrm dy}g^{-1}(y)\right|$

        for the log-uniform case: $f_Y(y)=\frac{1}{\log b-\log a}\left|\frac{1}{y}\right|$

        so $E[Y]=\int_a^b y f_Y(y)\;\mathrm d y=\int_a^b y \frac{1}{\log b-\log a}\left|\frac{1}{y}\right|\;\mathrm d y=\frac{1}{\log b-\log a}\int_a^b \frac{y}{y}\;\mathrm d y=\frac{b-a}{\log b-\log a}$
        treat $y$ as always positive to make integrating $|1/y|$ simple.

        $E[Y^2]=\frac{b^2-a^2}{2(\log b-\log a)}$

        which makes the variance easy to find: $Var[Y]=E[Y^2]-E[Y]^2$
        '''
        a, b, la, lb = cmin, cmax, np.log(cmin), np.log(cmax)
        c_mu = (b-a)/(lb-la)
        c_E_xsq = (b**2-a**2)/(2*(lb-la))
        c_stddev = np.sqrt(c_E_xsq - c_mu)
        check([c.c for c in random_cfgs], c_mu, c_stddev, tol=0.05*cmax)
        print_dot()
        check([row[1] for row in random_points], c_point_dist.mean(), c_point_dist.std())
        print_dot()
        check([opt.point_to_config(op.make2D_row(row)).c for row in random_points], c_mu, c_stddev, tol=0.05*cmax)
        print_dot()


class TestOptimiser(NumpyCompatableTestCase):
    def test_simple_grid(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()

        self.assertEqual(optimiser.best_sample(), None)

        optimiser.run_sequential(evaluator)

        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        def to_samples(l):
            return [op.Sample({'a':a, 'b':b}, cost, job_num=job_num)
                    for a, b, cost, job_num in l]

        samples = to_samples([(1,3,1,1), (2,3,2,2), (1,4,1,3), (2,4,2,4)])

        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_sample(), to_samples([(1,3,1,1), (1,4,1,3)])) # either would be acceptable

    def test_grid_maximise(self):
        # same as _simple_grid but maximising instead
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.GridSearchOptimiser(ranges, maximise_cost=True, order=['a','b'])
        evaluator = TestEvaluator()

        self.assertEqual(optimiser.best_sample(), None)

        optimiser.run_sequential(evaluator)

        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        def to_samples(l):
            return [op.Sample({'a':a, 'b':b}, cost, job_num=job_num)
                    for a, b, cost, job_num in l]
        samples = to_samples([(1,3,1,1), (2,3,2,2), (1,4,1,3), (2,4,2,4)])

        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_sample(), to_samples([(2,3,2,2), (2,4,2,4)])) # either would be acceptable


    def test_grid_specific_order(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.GridSearchOptimiser(ranges, order=['b','a'])
        evaluator = TestEvaluator()

        optimiser.run_sequential(evaluator)

        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        def to_samples(l):
            return [op.Sample({'a':a, 'b':b}, cost, job_num=job_num)
                    for a, b, cost, job_num in l]
        samples = to_samples([(1,3,1,1), (1,4,1,2), (2,3,2,3), (2,4,2,4)])

        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_sample(), to_samples([(1,3,1,1), (1,4,1,2)])) # either would be acceptable

    def test_range_length_1(self):
        # ranges of length 0 are not allowed
        ranges = {'a':[1], 'b':[2]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()

        optimiser.run_sequential(evaluator)

        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        s = op.Sample({'a':1, 'b': 2}, cost=1, job_num=1)

        self.assertEqual(optimiser.samples, [s])
        self.assertEqual(optimiser.best_sample(), s)

    def test_empty_range(self):
        ranges = {}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return 123 # placeholder cost function

        optimiser = op.GridSearchOptimiser(ranges, order=[])
        evaluator = TestEvaluator()

        optimiser.run_sequential(evaluator)

        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        s = op.Sample(config={}, cost=123, job_num=1)
        self.assertEqual(optimiser.samples, [s])
        self.assertEqual(optimiser.best_sample(), s)

    def test_empty_range_Bayes(self):
        ranges = {}
        # not allowed arbitrary ranges
        self.assertRaises(ValueError, op.BayesianOptimisationOptimiser, ranges)

    def test_simple_random(self):
        # there is a very small chance that this will fail because after 100
        # samples, not all the configurations are tested
        for allow_re_tests in [True, False]:
            self._simple_random(allow_re_tests)


    def _simple_random(self, allow_re_tests):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.RandomSearchOptimiser(ranges, allow_re_tests=allow_re_tests)
        evaluator = TestEvaluator()

        optimiser.run_sequential(evaluator, max_jobs=100)

        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        def to_samples(l):
            return [op.Sample({'a':a, 'b':b}, cost) for a, b, cost in l]
        samples = to_samples([(1,3,1), (2,3,2), (1,4,1), (2,4,2)])
        remove_job_nums(optimiser.samples) # order doesn't matter since random
        # samples subset of optimiser.samples
        for s in optimiser.samples:
            self.assertIn(s, samples)
        # optimiser.samples subset of samples
        for s in samples:
            self.assertIn(s, optimiser.samples)

        self.assertIn(optimiser.best_sample(), to_samples([(1,3,1), (1,4,1)])) # either would be acceptable

    def test_random_maximise(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.RandomSearchOptimiser(ranges, maximise_cost=True, allow_re_tests=False)
        evaluator = TestEvaluator()

        optimiser.run_sequential(evaluator)

        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        def to_samples(l):
            return [op.Sample({'a':a, 'b':b}, cost) for a, b, cost in l]
        samples = to_samples([(1,3,1), (2,3,2), (1,4,1), (2,4,2)])
        remove_job_nums(optimiser.samples) # order doesn't matter
        # samples subset of optimiser.samples
        for s in optimiser.samples:
            self.assertIn(s, samples)
        # optimiser.samples subset of samples
        for s in samples:
            self.assertIn(s, optimiser.samples)

        self.assertIn(optimiser.best_sample(), to_samples([(2,3,2), (2,4,2)])) # either would be acceptable

    @unittest.skipIf(NO_SLOW_TESTS, 'slow test')
    def test_simple_server(self):
        ''' have the optimiser run several times with max_jobs set '''
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()

        ev_thread = threading.Thread(target=evaluator.run_client, name='evaluator')
        ev_thread.start()

        optimiser.run_server(max_jobs=4)

        evaluator.stop()
        finish_off_evaluators()
        ev_thread.join()

        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        def to_samples(l):
            return [op.Sample({'a':a, 'b':b}, cost, job_num=job_num)
                    for a, b, cost, job_num in l]
        samples = to_samples([(1,3,1,1), (2,3,2,2), (1,4,1,3), (2,4,2,4)])
        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_sample(), to_samples([(1,3,1,1), (1,4,1,3)])) # either would be acceptable


    def test_optimisation_sequential_stop(self):
        ''' have the optimiser run several times with max_jobs set '''
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()

        optimiser.run_sequential(evaluator, max_jobs=1)
        optimiser.run_sequential(evaluator, max_jobs=2)
        optimiser.run_sequential(evaluator, max_jobs=4)

        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        def to_samples(l):
            return [op.Sample({'a':a, 'b':b}, cost, job_num=job_num)
                    for a, b, cost, job_num in l]
        samples = to_samples([(1,3,1,1), (2,3,2,2), (1,4,1,3), (2,4,2,4)])
        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_sample(), to_samples([(1,3,1,1), (1,4,1,3)])) # either would be acceptable

    @unittest.skipIf(NO_SLOW_TESTS, 'slow test')
    def test_optimisation_server_stop(self):
        ''' have the optimiser run several times with max_jobs set '''
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        def to_samples(l):
            return [op.Sample({'a':a, 'b':b}, cost, job_num=job_num)
                    for a, b, cost, job_num in l]

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()

        ev_thread = threading.Thread(target=evaluator.run_client, name='evaluator')
        ev_thread.start()


        optimiser.run_server(max_jobs=1)
        self.assertEqual(optimiser.samples, to_samples([(1,3,1,1)]))
        optimiser.run_server(max_jobs=2)
        optimiser.run_server(max_jobs=4)

        evaluator.stop()
        finish_off_evaluators()
        ev_thread.join()

        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        samples = to_samples([(1,3,1,1), (2,3,2,2), (1,4,1,3), (2,4,2,4)])
        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_sample(), to_samples([(1,3,1,1), (1,4,1,3)])) # either would be acceptable

    @unittest.skipIf(NO_SLOW_TESTS, 'slow test')
    def test_evaluator_client_stop(self):
        '''
        have the evaluator take a single job, process it and then shutdown,
        requiring it to be started multiple times.
        '''
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        def to_samples(l):
            return [op.Sample({'a':a, 'b':b}, cost, job_num=job_num)
                    for a, b, cost, job_num in l]

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()

        op_thread = threading.Thread(target=optimiser.run_server, name='optimiser')
        op_thread.start()

        evaluator.run_client(max_jobs=1)

        # wait for the optimiser to process the results
        wait_for(lambda: len(optimiser.samples) == 1)
        self.assertEqual(optimiser.samples, to_samples([(1,3,1,1)]))

        evaluator.run_client(max_jobs=2)
        # optimiser does not tell clients to stop, has to be done manually from
        # another thread or max_jobs has to be exact.
        evaluator.run_client(max_jobs=1)

        optimiser.stop()
        op_thread.join()

        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        samples = to_samples([(1,3,1,1), (2,3,2,2), (1,4,1,3), (2,4,2,4)])
        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_sample(), to_samples([(1,3,1,1), (1,4,1,3)])) # either would be acceptable

    @unittest.skipIf(NO_SLOW_TESTS, 'slow test')
    def test_multithreaded(self):
        ''' test that a multithreaded run (with 1 evaluator) matches the sequential one '''
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser1 = op.GridSearchOptimiser(ranges, order=['a','b'])
        optimiser2 = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()

        optimiser1.run_sequential(evaluator)
        optimiser2.run_multithreaded([evaluator])

        no_exceptions(self, optimiser1)
        no_exceptions(self, optimiser2)
        no_exceptions(self, evaluator)

        self.assertEqual(get_dict(optimiser1, different_run=True), get_dict(optimiser2, different_run=True))

    def test_evaluator_modification(self):
        #TODO: have the evaluator change the config before returning
        # should this even be allowed?
        pass

    def test_evaluator_list(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                new_config = config.copy()
                new_config.abc = 123
                return [op.Sample(config, config.a), op.Sample(new_config, 10)]

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()

        optimiser.run_sequential(evaluator)

        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        def to_samples(l):
            samples = [op.Sample({'a':a, 'b':b, 'abc':abc}, cost, job_num=job_num)
                       for a, b, abc, cost, job_num in l]
            for s in samples:
                if s.config['abc'] is None:
                    del s.config['abc']
            return samples
        samples = to_samples([
            (1,3,None,1,1), (1,3,123,10,1),
            (2,3,None,2,2), (2,3,123,10,2),
            (1,4,None,1,3), (1,4,123,10,3),
            (2,4,None,2,4), (2,4,123,10,4)
        ])

        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_sample(), to_samples([(1,3,None,1,1), (1,4,None,1,2)])) # either would be acceptable

    def test_evaluator_extra(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return op.Sample(config, config.a, extra={'test':'abc'})

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()

        optimiser.run_sequential(evaluator)

        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        def to_samples(l):
            return [op.Sample({'a':a, 'b':b}, cost, extra, job_num=job_num)
                    for a, b, cost, extra, job_num in l]
        samples = to_samples([
            (1,3,1,{'test':'abc'},1),
            (2,3,2,{'test':'abc'},2),
            (1,4,1,{'test':'abc'},3),
            (2,4,2,{'test':'abc'},4)])

        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_sample(), to_samples([
            (1,3,1,{'test':'abc'},1), (1,4,1,{'test':'abc'},3)])) # either would be acceptable

    def test_evaluator_empty_list(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return []

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()

        optimiser.run_sequential(evaluator)

        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        self.assertEqual(optimiser.samples, [])
        self.assertEqual(optimiser.best_sample(), None)



# Checkpoint utilities


def rm(filename):
    if os.path.isfile(filename):
        os.remove(filename)

def get_dict(optimiser, different_run=False):
        '''
        get a dictionary of the elements to compare two optimisers,
        excluding the attributes which are allowed to differ

        different_run: whether to exclude some fields that would be expected
                        to differ if the optimiser was run after loading a
                        checkpoint
        '''
        if different_run:
            # allowed to differ. The log and duration _should_ be different for different runs.
            # the stop flag may or may not be set depending on how the optimiser stopped (max_jobs or stop())
            exclude = ['log_record', 'run_state', 'duration', '_stop_flag']
        else:
            exclude = ['run_state'] # doesn't matter if this differs
        d = {k:v for k, v in optimiser.__dict__.items() if k not in exclude}

        if 'duration' in d.keys():
            # allow the duration to differ slightly
            # (note: ndigits is the number of decimal places, not total digits)
            d['duration'] = round(d['duration'], ndigits=2)

        # cannot compare threading.Event objects, but what matters is their
        # set states.
        if '_stop_flag' in d.keys():
            d['_stop_flag'] = d['_stop_flag'].is_set()
        if '_checkpoint_flag' in d.keys():
            d['_checkpoint_flag'] = d['_checkpoint_flag'].is_set()

        if 'hypothesised_samples' in d.keys():
            # ignore hypothesised samples for jobs that have finished since they
            # will be removed at the next possible opportunity anyway
            d['hypothesised_samples'] = [s for s in d['hypothesised_samples'] if s[0] not in optimiser.finished_job_ids]
        if 'step_log' in d.keys():
            # Gaussian Processes cannot be compared
            d['step_log'] = {job_num: {k:v for k, v in step.items() if k != 'gp'} for job_num, step in d['step_log'].items()}
        return d

class reset_random:
    ''' reset random number generators to before the with statement was entered '''
    def __enter__(self, *args):
        self.np_state = np.random.get_state()
        self.r_state = random.getstate()
    def __exit__(self, *args):
        np.random.set_state(self.np_state)
        random.setstate(self.r_state)


class CheckpointManager:
    def __init__(self, test_case, checkpoint_path, optimiser, create_optimiser, num_jobs):
        self.test_case = test_case
        self.checkpoint_path = checkpoint_path
        self.optimiser = optimiser
        self.create_optimiser = create_optimiser
        self.num_jobs = num_jobs
        # list of evaluators created by loading saved checkpoints
        self.saved = []

    def save_optimiser(self, opt):
        np_state = np.random.get_state()
        r_state = random.getstate()
        self.saved.append((opt, np_state, r_state))

    def take_checkpoint(self, save_now, make_copy, compare_after_load=True):
        '''
        take a checkpoint, re-load it and save the result
        save_now: whether to save immediately (only pass True if optimiser not running)
        make_copy: whether to copy the optimiser before the save and compare afterwards
        compare_after_load: whether to compare the optimiser to the new
            optimiser which loads the checkpoint (ie should be False if the
            optimiser is not quiescent)
        '''
        rm(self.checkpoint_path)
        op_before = copy.deepcopy(self.optimiser) if make_copy else None
        if save_now:
            self.optimiser.save_now(self.checkpoint_path)
        else:
            self.optimiser.save_when_ready(self.checkpoint_path)
        wait_for(lambda: not self.optimiser._checkpoint_flag.is_set()) # wait for the save to happen
        o2 = self.load_and_check_checkpoint(op_before, compare_after_load)
        self.save_optimiser(o2)


    def load_and_check_checkpoint(self, op_before, compare_after_load):
        '''
        create a new optimiser (potentially dirty) and load the checkpoint into it.
        Make sure:
            optimiser copy before checkpoint == optimiser now
            optimiser now == checkpoint loaded into clean optimiser
        then return the optimiser with the loaded checkpoint

        op_before: a copy of the optimiser before the snapshot was initiated. None to ignore.
        '''
        if op_before is not None:
            # make sure that taking the snapshot had no side effects on the
            # optimiser other than writing to the log.
            self.test_case.assertTrue(self.optimiser.log_record.startswith(op_before.log_record))
            op_before.log_record = self.optimiser.log_record

            if False:
                print('\n' + '-'*20)
                print('\n\nbefore =')
                print(dstring(get_dict(op_before)))
                print('\n\nafter =')
                print(dstring(get_dict(self.optimiser)))

            self.test_case.assertEqual(get_dict(op_before), get_dict(self.optimiser))


        # create_optimiser may run the optimiser for some steps to make it
        # 'dirty'. This may affect the random number generators
        with reset_random():
            o2 = self.create_optimiser()

        o2.load_checkpoint(self.checkpoint_path)
        no_exceptions(self.test_case, o2)

        if False:
            print('\n' + '-'*20)
            print('\n\noptimiser =')
            print(dstring(get_dict(self.optimiser)))
            print('\n\nre-loaded optimiser =')
            print(dstring(get_dict(o2)))

        if compare_after_load:
            self.test_case.assertEqual(get_dict(self.optimiser), get_dict(o2))
        return o2

class TestCheckpoints(NumpyCompatableTestCase):

    @unittest.skipIf(NO_SLOW_TESTS, 'slow test')
    def test_checkpoints_grid(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                # placeholder cost function
                return op.Sample(config, config.a, extra={'test':np.array([1,2,3])})

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()
        def create_optimiser():
            ''' create a 'dirty' optimiser which is initialised identically but has been run '''
            opt = op.GridSearchOptimiser(ranges, order=['a','b'])
            opt.run_sequential(evaluator, max_jobs=random.randint(0, 4))
            return opt

        self._test_checkpoints(optimiser, evaluator, create_optimiser, num_jobs=4)

    @unittest.skipIf(NO_SLOW_TESTS, 'slow test')
    def test_checkpoints_random(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                # placeholder cost function
                return op.Sample(config, config.a, extra={'test':np.array([1,2,3])})

        optimiser = op.RandomSearchOptimiser(ranges)
        evaluator = TestEvaluator()
        def create_optimiser():
            ''' create a 'dirty' optimiser which is initialised identically but has been run '''
            opt = op.RandomSearchOptimiser(ranges)
            opt.run_sequential(evaluator, max_jobs=random.randint(0, 4))
            return opt

        self._test_checkpoints(optimiser, evaluator, create_optimiser, num_jobs=4)

    @unittest.skipIf(NO_SLOW_TESTS, 'slow test')
    def test_checkpoints_bayes(self):
        ranges = {'a':[5,10,15], 'b':[0,2,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                # placeholder cost function
                return op.Sample(config, config.a+config.b, extra={'test':np.array([1,2,3])})

        optimiser = op.BayesianOptimisationOptimiser(ranges, pre_samples=3, gp_params=TEST_GP_PARAMS)
        evaluator = TestEvaluator()
        def create_optimiser():
            ''' create a 'dirty' optimiser which is initialised identically but has been run '''
            opt = op.BayesianOptimisationOptimiser(ranges, pre_samples=3, gp_params=TEST_GP_PARAMS)
            # have to at least run a few Bayesian steps (ie max_jobs > pre_samples)
            opt.run_sequential(evaluator, max_jobs=random.randint(3, 5))
            return opt

        self._test_checkpoints(optimiser, evaluator, create_optimiser, num_jobs=6)


    def _test_checkpoints(self, optimiser, evaluator, create_optimiser, num_jobs):
        '''
        check that saving and re-loading from a checkpoint before and during
        execution produces identical optimisers.

        Note: some fields such as duration are only updated after a job is
        processed and so will be identical before and after the load.

        Note: this test also verifies that the results of running the server are
        identical to those obtained sequentially

        optimiser: the initial optimiser which will have checkpoints taken
        evaluator: the evaluator to use for every optimiser
        create_optimiser: set up an optimiser to load the checkpoint into. Can
            be 'dirty' ie not freshly constructed, as loading should deal with
            any state.
        num_jobs: the number of jobs to run each optimiser for
        '''
        self.assertTrue(optimiser.configuration_space_size() >= num_jobs)

        cp = '/tmp/checkpoint.json'
        cm = CheckpointManager(self, cp, optimiser, create_optimiser, num_jobs)

        # checkpoint after 0 jobs
        # to make sure that the copy matches, since the modification to this
        # attribute happens after the copy
        optimiser.checkpoint_filename = cp
        cm.take_checkpoint(save_now=True, make_copy=True)

        op_thread = threading.Thread(target=optimiser.run_server, name='optimiser')
        op_thread.start()

        # checkpoint after started but 0 jobs
        cm.take_checkpoint(save_now=False, make_copy=True)

        # checkpoint after some number of jobs have been completed
        choice_counter = 0
        choices = [1, 2, 0] # cycle through the choices
        i = 0
        while i < num_jobs:
            num_to_run = min(choices[choice_counter % len(choices)], num_jobs-1)
            choice_counter += 1
            if num_to_run != 0:
                evaluator.run_client(max_jobs=num_to_run) # blocks until finished
                wait_for(lambda: optimiser.num_finished_jobs == i+num_to_run)
            cm.take_checkpoint(save_now=False, make_copy=True)
            i += num_to_run
            print_dot()

        optimiser.stop()
        op_thread.join()

        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        # just a sanity check
        self.assertTrue(len(cm.saved) > 4)

        # continue from the load until the end, make sure everything went fine
        # and that the end result is the same as the optimiser which was not
        # loaded.
        # Note: log and duration _will_ be different so exclude those fields from being identical
        for opt, np_state, r_state in cm.saved:
            np.random.set_state(np_state)
            random.setstate(r_state)
            jobs_left = num_jobs - opt.num_finished_jobs
            opt.run_sequential(evaluator, max_jobs=jobs_left)
            no_exceptions(self, opt)
            no_exceptions(self, evaluator)
            self.assertEqual(get_dict(optimiser, different_run=True), get_dict(opt, different_run=True))

    @unittest.skipIf(NO_SLOW_TESTS, 'slow test')
    def test_async_checkpoints_grid(self):
        ranges = {'a':list(range(25)), 'b':[3,4,5,6]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                time.sleep(random.choice([0.1, 0.17, 0.32]))
                return config.a # placeholder cost function
        class FastEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        evaluators = [TestEvaluator(), TestEvaluator(), TestEvaluator()]
        def create_optimiser():
            opt = op.GridSearchOptimiser(ranges, order=['a','b'])
            return opt
        sort_samples = lambda samples: sorted(samples, key=lambda s:(s.config.a, s.config.b))

        self._test_async_checkpoints(FastEvaluator(), evaluators,
                                     create_optimiser, sort_samples,
                                     compare_results=True, num_jobs=100)

    @unittest.skipIf(NO_SLOW_TESTS, 'slow test')
    def test_async_checkpoints_random(self):
        ranges = {'a':list(range(25)), 'b':[3,4,5,6]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                time.sleep(random.choice([0.1, 0.17, 0.32]))
                return config.a # placeholder cost function
        class FastEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        create_optimiser = lambda: op.RandomSearchOptimiser(ranges)
        evaluators = [TestEvaluator(), TestEvaluator(), TestEvaluator()]
        sort_samples = lambda samples: sorted(samples, key=lambda s:(s.config.a, s.config.b))

        # cannot compare results because there is no guarantee of evaluating
        # every configuration in the given number of jobs
        self._test_async_checkpoints(FastEvaluator(), evaluators,
                                     create_optimiser, sort_samples,
                                     compare_results=False, num_jobs=100)

    @unittest.skipIf(NO_SLOW_TESTS, 'slow test')
    def test_async_checkpoints_bayes(self):
        ranges = {'a':list(range(25)), 'b':[3,4,5,6]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                time.sleep(random.choice([0.1, 0.17, 0.32]))
                return config.a # placeholder cost function
        class FastEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        create_optimiser = lambda: op.BayesianOptimisationOptimiser(ranges, gp_params=TEST_GP_PARAMS)
        evaluators = [TestEvaluator(), TestEvaluator(), TestEvaluator()]
        sort_samples = lambda samples: sorted(samples, key=lambda s:(s.config.a, s.config.b))

        # cannot compare results because depending on the order that samples are
        # received, the surrogate function may look completely different, giving
        # different results
        self._test_async_checkpoints(FastEvaluator(), evaluators,
                                     create_optimiser, sort_samples,
                                     compare_results=False, num_jobs=20)

    def _test_async_checkpoints(self, fast_evaluator, evaluators, create_optimiser, sort_samples, compare_results, num_jobs):
        '''
        test checkpoints with multiple evaluators processing at different rates
        fast_evaluator: an evaluator which has no delay when testing configurations
        evaluators: a list of evaluator objects to run
        create_optimiser: a function to create a fresh optimiser ready to load a checkpoint
        sort_samples: a function to sort a list of Sample objects into a deterministic order
        compare_results: whether to compare the dictionaries of the optimisers which loaded the checkpoints
        '''
        cp = '/tmp/checkpoint.json'
        optimiser = create_optimiser()
        cm = CheckpointManager(self, cp, optimiser, create_optimiser, num_jobs)

        op_thread = threading.Thread(target=lambda: optimiser.run_server(max_jobs=num_jobs), name='optimiser')
        op_thread.start()

        e_threads = [threading.Thread(target=e.run_client, name='evaluator{}'.format(i)) for i,e in enumerate(evaluators)]
        for t in e_threads:
            t.start()

        # optimiser only runs for specified number of jobs

        # DEBUGGING NOTE: if this test deadlocks, see if the evaluators have crashed

        # being cautious and only starting a checkpoint when there are
        # definitely some more jobs to start. The specification says that
        # nothing happens if save_when_ready() is called but the optimiser is
        # not running, which would break take_checkpoint() because it waits
        # indefinitely for a checkpoint to be created.
        while op_thread.is_alive() and optimiser.num_started_jobs < num_jobs:
            # if a copy is made, might not match since we don't know that the
            # optimiser is currently ready to take a snapshot. Cannot compare
            # the optimiser with the loaded checkpoint with the optimiser, since
            # it does not stop processing.
            cm.take_checkpoint(save_now=False, make_copy=False, compare_after_load=False)
            print_dot()
            if len(cm.saved) > 100:
                time.sleep(1.0)
            elif len(cm.saved) > 50:
                time.sleep(0.5)
            else:
                time.sleep(0.1)
            #for e in evaluators:
                #no_exceptions(self, e)

        # random thing I just learned (the hard way). Checking is_alive in a
        # loop isn't good enough. You still have to use join() to ensure that
        # the thread is completely finished and shut down (could be a caching
        # problem?)
        op_thread.join()


        no_exceptions(self, optimiser)
        for e in evaluators:
            no_exceptions(self, e)

        for e in evaluators:
            e.stop()
        # if the evaluators are still trying to re-send results back, give them
        # a small opportunity to do so
        finish_off_evaluators()
        for t in e_threads:
            t.join()

        print('|', end='')

        if compare_results:
            # order of execution not guaranteed, so sort the samples
            optimiser.samples = sort_samples(optimiser.samples)

        for opt, np_state, r_state in cm.saved:
            print_dot()
            np.random.set_state(np_state)
            random.setstate(r_state)
            jobs_left = num_jobs - opt.num_finished_jobs
            opt.run_sequential(fast_evaluator, max_jobs=jobs_left)
            no_exceptions(self, opt)
            if compare_results:
                opt.samples = sort_samples(opt.samples)
                d1 = get_dict(optimiser, different_run=True)
                d2 = get_dict(opt, different_run=True)
                self.assertEqual(d1, d2)




class TestBayesianOptimisationUtils(NumpyCompatableTestCase):
    def test_config_to_point(self):
        ranges = {
            'a' : [1,2,3], # linear
            'b' : [5], # constant
            'c' : [1,2,3] # linear
        }
        optimiser = op.BayesianOptimisationOptimiser(ranges)

        self.assertEqual(optimiser.config_to_point({'a':2,'b':5,'c':1}).tolist(), [[2,1]]) # ignores the constant parameter
        self.assertEqual(optimiser.config_to_point({'c':1,'b':5,'a':2}).tolist(), [[2,1]]) # alphabetical order
        self.assertEqual(optimiser.config_to_point({'a':9,'b':6,'c':12}).tolist(), [[9,12]]) # fine with values outside 'valid range'

        self.assertEqual(optimiser.config_to_point({'a':9,'b':6,'c':12}).shape, (1,2)) # must be a 2D array

        self.assertRaises(AssertionError, optimiser.config_to_point, {'a':1,'b':5}) # value for 'c' not provided (which is included in the output)
        self.assertRaises(AssertionError, optimiser.config_to_point, {'a':1,'c':2}) # value for 'c' not provided (which is not included in the output)
        self.assertRaises(AssertionError, optimiser.config_to_point, {'a':1,'b':5,'c':3,'z':123}) # extra value

        ranges = {
            'a' : [1,2,3] # linear
        }
        optimiser = op.BayesianOptimisationOptimiser(ranges)
        self.assertEquals(optimiser.config_to_point({'a':2}).tolist(), [[2]])
        ranges = {
            'a' : [1,2,3], # linear
            'b' : [1,2,3]  # linear
        }
        optimiser = op.BayesianOptimisationOptimiser(ranges)
        self.assertEquals(optimiser.config_to_point({'a':2,'b':3}).tolist(), [[2,3]])

        ranges = {
            'a' : [1,10,100] # logarithmic
        }
        optimiser = op.BayesianOptimisationOptimiser(ranges)
        self.assertEquals(optimiser.config_to_point({'a':np.exp(1)}).tolist(), [[1]])

    def test_point_to_config(self):
        ranges = {
            'a' : [1,2,3], # linear
            'b' : [5], # constant
            'c' : [1,2,3] # linear
        }
        optimiser = op.BayesianOptimisationOptimiser(ranges)

        self.assertRaisesRegexp(ValueError, 'too few attributes', optimiser.point_to_config, np.array([[1]])) # point too short
        self.assertEquals(optimiser.point_to_config(np.array([[2,3]])), {'a':2,'b':5,'c':3})
        self.assertRaisesRegexp(ValueError, 'too many attributes', optimiser.point_to_config, np.array([[1,2,3]])) # point too long

        ranges = {
            'a' : [1,2,3] # linear
        }
        optimiser = op.BayesianOptimisationOptimiser(ranges)
        self.assertEquals(optimiser.point_to_config(np.array([[2]])), {'a':2})
        ranges = {
            'a' : [1,2,3], # linear
            'b' : [1,2,3] # linear
        }
        optimiser = op.BayesianOptimisationOptimiser(ranges)
        self.assertEquals(optimiser.point_to_config(np.array([[2,3]])), {'a':2,'b':3})

        # there was a bug (too few attributes) specifically when there are trailing constant ranges
        ranges = {
            'a' : [1,2,3], # linear
            'b' : [3] # linear
        }
        optimiser = op.BayesianOptimisationOptimiser(ranges)
        self.assertEquals(optimiser.point_to_config(np.array([[2]])), {'a':2,'b':3})

        ranges = {
            'a' : [1,10,100] # logarithmic
        }
        optimiser = op.BayesianOptimisationOptimiser(ranges)
        self.assertEquals(optimiser.point_to_config(np.array([[1]])), {'a': np.exp(1)})

    def test_close_to_any(self):
        sx = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = lambda a1, a2: np.array([[a1, a2]])

        self.assertTrue(op.close_to_any(x(3, 4), sx)) # exact matches
        self.assertTrue(op.close_to_any(x(1, 2), sx))

        self.assertFalse(op.close_to_any(x(100, 100), sx, tol=50)) # doesn't account for if the points in xs are close to one another

        self.assertFalse(op.close_to_any(x(10, 10), sx)) # far away from any point
        self.assertFalse(op.close_to_any(x(-10, -10), sx))
        self.assertFalse(op.close_to_any(x(-1, -2), sx))
        self.assertFalse(op.close_to_any(x(2, 3), sx))

        self.assertTrue(op.close_to_any(x(3.0, 4.1), sx, tol=0.01001)) # just inside the tolerance
        self.assertTrue(op.close_to_any(x(3.0, 4.1), sx, tol=0.01)) # exact squared Euclidean distance
        self.assertFalse(op.close_to_any(x(3.0, 4.1), sx, tol=0.00999)) # just further away than the tolerance

class TestBayesianOptimisation(NumpyCompatableTestCase):
    @unittest.skipIf(NO_SLOW_TESTS, 'slow test')
    def test_simple_bayes(self):
        for a in ['EI', 'UCB']:
            self._simple_bayes(a)

    def _simple_bayes(self, acquisition_function):
        ranges = {'a':[5,10,15], 'b':[0,2,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                print_dot()
                return config.a + config.b # placeholder cost function

        optimiser = op.BayesianOptimisationOptimiser(ranges, acquisition_function=acquisition_function, gp_params=TEST_GP_PARAMS)
        evaluator = TestEvaluator()

        self.assertEqual(optimiser.best_sample(), None)

        optimiser.run_sequential(evaluator, max_jobs=30)

        #print(optimiser.log_record)
        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        # make sure the result is close to the global optimum
        self.assertTrue(abs(optimiser.best_sample().cost - 5.0) <= 0.1)

    @unittest.skipIf(NO_SLOW_TESTS, 'slow test')
    def test_maximise_bayes(self):
        ranges = {'a':[5,10,15], 'b':[0,2,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                print_dot()
                return config.a + config.b # placeholder cost function

        optimiser = op.BayesianOptimisationOptimiser(ranges, maximise_cost=True, acquisition_function='UCB', gp_params=TEST_GP_PARAMS)
        evaluator = TestEvaluator()

        self.assertEqual(optimiser.best_sample(), None)

        optimiser.run_sequential(evaluator, max_jobs=30)

        #print(optimiser.log_record)
        no_exceptions(self, optimiser)
        no_exceptions(self, evaluator)

        # there is a warning with the GP optimisation on job 24
        '''
        optimiser.plot_step_slice('a', 24)
        optimiser.plot_step_slice('b', 24)
        '''

        # make sure the result is close to the global optimum
        self.assertTrue(abs(optimiser.best_sample().cost - 19.0) <= 0.1)

# Debugging

if sys.version_info[0] == 3:
    # python 2 is not very good.

    def dump_thread_frames(exclude_current=False):
        '''
        return a string containing the current frames (Tracebacks) of each running thread.
        '''
        import traceback
        s = '\n'
        current = threading.get_ident()
        main = threading.main_thread().ident

        named = []
        for tid, stack in sys._current_frames().items():
            t = threading._active.get(tid)
            name = 'Unknown (thread missing)' if t is None else t.name
            named.append((tid, stack, name))

        custom_named = [x for x in named if not x[2].startswith('Thread-')]
        not_named = [x for x in named if x[2].startswith('Thread-')]
        # sort to make the custom named threads first
        named = custom_named + not_named

        for thread_id, stack, name in named:
            if exclude_current and thread_id == current:
                continue
            # getting a thread name from an ID isn't in the public API so this may break in future
            extra = 'Main' if thread_id == main else name
            s += '#### THREAD: {} {} ####\n'.format(thread_id, extra)
            for filename, lineno, name, line in traceback.extract_stack(stack):
                s += 'File: "{}", line {}, in {}\n'.format(filename, lineno, name)
                if line:
                    s += '  {}\n'.format(line.strip())
            s += '-'*25 + '\n'
        return s

    def continuously_dump_frames(filename='/tmp/frames.txt', interval=1.0):
        '''
        every few seconds dump the current frames (Tracebacks) of each running
        thread to the specified file.
        '''
        print('thread frames will be dumped to "{}" every {} seconds'.format(filename, interval))
        def loop():
            while True:
                frames = dump_thread_frames(exclude_current=True)
                with open(filename, 'w') as f:
                    f.write(frames)
                time.sleep(interval)
        t = threading.Thread(target=loop, name='dump_frames')
        t.setDaemon(True)
        t.start()

def pdb_on_signal():
    '''
    break into pdb by sending a signal. Use one of the following:
    pkill -SIGUSR1 myprocess
    killall -SIGUSR1 python3
    '''
    print('send SIGUSR1 signal to break into pdb')
    import signal
    # http://blog.devork.be/2009/07/how-to-bring-running-python-program.html
    def handle_pdb(sig, frame):
        import pdb
        pdb.Pdb().set_trace(frame)
    signal.signal(signal.SIGUSR1, handle_pdb)

if __name__ == '__main__':
    if NO_SLOW_TESTS:
        print('\n' + '-'*40)
        print('SLOW TESTS DISABLED')
        print('-'*40 + '\n')

    pdb_on_signal()
    if sys.version_info[0] == 3:
        continuously_dump_frames()
    print()

    #unittest.main()
    np.random.seed(42) # make deterministic

    suite = unittest.TestSuite([
        unittest.TestLoader().loadTestsFromTestCase(TestUtils),
        unittest.TestLoader().loadTestsFromTestCase(TestOptimiser),
        unittest.TestLoader().loadTestsFromTestCase(TestBayesianOptimisationUtils),
        unittest.TestLoader().loadTestsFromTestCase(TestCheckpoints),
        unittest.TestLoader().loadTestsFromTestCase(TestBayesianOptimisation)
    ])
    # verbosity: 0 => quiet, 1 => default, 2 => verbose
    unittest.TextTestRunner(verbosity=2, descriptions=False, failfast=True).run(suite)

