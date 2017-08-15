#!/usr/bin/env python
import unittest

import time
import numpy as np

# local modules
import optimisation as op

class TestUtils(unittest.TestCase):
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

class TestOptimiser(unittest.TestCase):
    def test_simple_grid(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()

        self.assertEqual(optimiser.best_sample(), None)

        optimiser.run_sequential(evaluator)

        self.assertTrue('Traceback' not in optimiser.log_record)
        self.assertTrue('Exception' not in optimiser.log_record)

        mks = lambda a,b,cost: op.Sample({'a':a, 'b':b}, cost) # make sample
        samples = [mks(1,3,1), mks(2,3,2), mks(1,4,1), mks(2,4,2)]

        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_sample(), [mks(1,3,1), mks(1,4,1)]) # either would be acceptable

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

        self.assertTrue('Traceback' not in optimiser.log_record)
        self.assertTrue('Exception' not in optimiser.log_record)

        self.assertTrue('Traceback' not in optimiser.log_record)
        self.assertTrue('Exception' not in optimiser.log_record)

        mks = lambda a,b,cost: op.Sample({'a':a, 'b':b}, cost) # make sample
        samples = [mks(1,3,1), mks(2,3,2), mks(1,4,1), mks(2,4,2)]

        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_sample(), [mks(2,3,2), mks(2,4,2)]) # either would be acceptable


    def test_grid_specific_order(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.GridSearchOptimiser(ranges, order=['b','a'])
        evaluator = TestEvaluator()

        optimiser.run_sequential(evaluator)

        self.assertTrue('Traceback' not in optimiser.log_record)
        self.assertTrue('Exception' not in optimiser.log_record)

        mks = lambda a,b,cost: op.Sample({'a':a, 'b':b}, cost) # make sample
        samples = [mks(1,3,1), mks(1,4,1), mks(2,3,2), mks(2,4,2)]

        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_sample(), [mks(1,3,1), mks(1,4,1)]) # either would be acceptable

    def test_range_length_1(self):
        # ranges of length 0 are not allowed
        ranges = {'a':[1], 'b':[2]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()

        optimiser.run_sequential(evaluator)

        self.assertTrue('Traceback' not in optimiser.log_record)
        self.assertTrue('Exception' not in optimiser.log_record)

        mks = lambda a,b,cost: op.Sample({'a':a, 'b':b}, cost) # make sample
        samples = [mks(1,2,1)]

        self.assertEqual(optimiser.samples, samples)
        self.assertEqual(optimiser.best_sample(), mks(1,2,1))

    def test_empty_range(self):
        #TODO: also test with Bayes
        ranges = {}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return 123 # placeholder cost function

        optimiser = op.GridSearchOptimiser(ranges, order=[])
        evaluator = TestEvaluator()

        optimiser.run_sequential(evaluator)

        #print(optimiser.log_record)
        self.assertTrue('Traceback' not in optimiser.log_record)
        self.assertTrue('Exception' not in optimiser.log_record)

        s = op.Sample(config={}, cost=123)
        self.assertEqual(optimiser.samples, [s])
        self.assertEqual(optimiser.best_sample(), s)

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

        self.assertTrue('Traceback' not in optimiser.log_record)
        self.assertTrue('Exception' not in optimiser.log_record)

        mks = lambda a,b,cost: op.Sample({'a':a, 'b':b}, cost) # make sample
        samples = [mks(1,3,1), mks(2,3,2), mks(1,4,1), mks(2,4,2)]
        # samples subset of optimiser.samples
        for s in optimiser.samples:
            self.assertIn(s, samples)
        # optimiser.samples subset of samples
        for s in samples:
            self.assertIn(s, optimiser.samples)

        self.assertIn(optimiser.best_sample(), [mks(1,3,1), mks(1,4,1)]) # either would be acceptable

    def test_random_maximise(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.RandomSearchOptimiser(ranges, maximise_cost=True, allow_re_tests=False)
        evaluator = TestEvaluator()

        optimiser.run_sequential(evaluator)

        self.assertTrue('Traceback' not in optimiser.log_record)
        self.assertTrue('Exception' not in optimiser.log_record)

        mks = lambda a,b,cost: op.Sample({'a':a, 'b':b}, cost) # make sample
        samples = [mks(1,3,1), mks(2,3,2), mks(1,4,1), mks(2,4,2)]
        # samples subset of optimiser.samples
        for s in optimiser.samples:
            self.assertIn(s, samples)
        # optimiser.samples subset of samples
        for s in samples:
            self.assertIn(s, optimiser.samples)

        self.assertIn(optimiser.best_sample(), [mks(2,3,2), mks(2,4,2)]) # either would be acceptable


    def test_evaluator_stop(self):
        '''
        have the evaluator take a single job, process it and then shutdown,
        requiring it to be started multiple times.
        '''
        return #TODO remake with server


        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                self.done = True # stop processing jobs
                return config.a # placeholder cost function
        mks = lambda a,b,cost: op.Sample({'a':a, 'b':b}, cost) # make sample

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()

        optimiser.start(run_async=True)
        self.assertTrue(optimiser.running)

        evaluator.start(run_async=True)
        evaluator.wait_for(quiet=True)
        while len(optimiser.samples) == 0:
            time.sleep(0.05) # wait
        self.assertEqual(optimiser.samples, [mks(1,3,1)])

        for i in range(3):
            evaluator.start(run_async=True)
            evaluator.wait_for(quiet=True)

        optimiser.wait_for(quiet=True)

        # should both have been shut down
        self.assertFalse(optimiser.running)
        self.assertIsNone(optimiser.proc)
        self.assertIsNone(evaluator.proc)

        samples = [mks(1,3,1), mks(2,3,2), mks(1,4,1), mks(2,4,2)]
        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_sample(), [mks(1,3,1), mks(1,4,1)]) # either would be acceptable

    def test_save_progress(self):
        pass
        #TODO
        #TODO need to account for jobs in the queue or maybe not?
        #TODO save and load and make sure that the state doesn't change
        #TODO save, make a new optimiser, load, make sure they match (__dict__ equal, perhaps with some keys removed)
    def test_evaluator_modification(self):
        pass
        #TODO: have the evaluator change the config before returning
    #TODO: ensure that exceptions raised by the evaluator appear in the output log
    #TODO: test with client/server
    #TODO: test with multiple clients with different speeds
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

        self.assertTrue('Traceback' not in optimiser.log_record)
        self.assertTrue('Exception' not in optimiser.log_record)

        mks = lambda a,b,cost: op.Sample({'a':a, 'b':b}, cost) # make sample
        mks_2 = lambda a,b,cost: op.Sample({'a':a, 'b':b, 'abc':123}, cost) # make sample
        samples = [mks(1,3,1), mks_2(1,3,10),
                   mks(2,3,2), mks_2(2,3,10),
                   mks(1,4,1), mks_2(1,4,10),
                   mks(2,4,2), mks_2(2,4,10)]

        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_sample(), [mks(1,3,1), mks(1,4,1)]) # either would be acceptable

    def test_evaluator_extra(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return op.Sample(config, config.a, extra={'test':'abc'})

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()

        optimiser.run_sequential(evaluator)

        self.assertTrue('Traceback' not in optimiser.log_record)
        self.assertTrue('Exception' not in optimiser.log_record)

        mks = lambda a,b,cost: op.Sample({'a':a, 'b':b}, cost, extra={'test':'abc'}) # make sample
        samples = [mks(1,3,1), mks(2,3,2), mks(1,4,1), mks(2,4,2)]

        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_sample(), [mks(1,3,1), mks(1,4,1)]) # either would be acceptable

    def test_evaluator_empty_list(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return []

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator()

        optimiser.run_sequential(evaluator)

        self.assertTrue('Traceback' not in optimiser.log_record)
        self.assertTrue('Exception' not in optimiser.log_record)

        self.assertEqual(optimiser.samples, [])
        self.assertEqual(optimiser.best_sample(), None)

    # TODO: test multiple evaluators, maybe one slower than the other

class TestBayesianOptimisationUtils(unittest.TestCase):
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

class TestBayesianOptimisation(unittest.TestCase):
    def test_simple_bayes(self):
        for a in ['EI', 'UCB']:
            self._simple_bayes(a)

    def _simple_bayes(self, acquisition_function):
        ranges = {'a':[5,10,15], 'b':[0,2,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a + config.b # placeholder cost function

        optimiser = op.BayesianOptimisationOptimiser(ranges, acquisition_function=acquisition_function)
        evaluator = TestEvaluator()

        self.assertEqual(optimiser.best_sample(), None)

        optimiser.run_sequential(evaluator, max_jobs=30)

        #print(optimiser.log_record)
        self.assertTrue('Traceback' not in optimiser.log_record)
        self.assertTrue('Exception' not in optimiser.log_record)

        # make sure the result is close to the global optimum
        self.assertTrue(abs(optimiser.best_sample().cost - 5.0) <= 0.1)

    def test_maximise_bayes(self):
        ranges = {'a':[5,10,15], 'b':[0,2,4]}
        class TestEvaluator(op.Evaluator):
            def test_config(self, config):
                return config.a + config.b # placeholder cost function

        optimiser = op.BayesianOptimisationOptimiser(ranges, maximise_cost=True, acquisition_function='UCB')
        evaluator = TestEvaluator()

        self.assertEqual(optimiser.best_sample(), None)

        optimiser.run_sequential(evaluator, max_jobs=30)

        #print(optimiser.log_record)
        self.assertTrue('Traceback' not in optimiser.log_record)
        self.assertTrue('Exception' not in optimiser.log_record)

        # make sure the result is close to the global optimum
        self.assertTrue(abs(optimiser.best_sample().cost - 19.0) <= 0.1)

if __name__ == '__main__':
    #unittest.main()
    np.random.seed(42) # make deterministic

    suite = unittest.TestSuite([
        unittest.TestLoader().loadTestsFromTestCase(TestUtils),
        unittest.TestLoader().loadTestsFromTestCase(TestOptimiser),
        unittest.TestLoader().loadTestsFromTestCase(TestBayesianOptimisationUtils),
        unittest.TestLoader().loadTestsFromTestCase(TestBayesianOptimisation)
    ])
    # verbosity: 0 => quiet, 1 => default, 2 => verbose
    unittest.TextTestRunner(verbosity=2).run(suite)

