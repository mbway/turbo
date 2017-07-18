#!/usr/bin/env python3
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
        self.assertEqual(op.time_string(1.5), '00:01.5')
        self.assertEqual(op.time_string(60), '01:00')
        self.assertEqual(op.time_string(61.5), '01:01.5')
        self.assertEqual(op.time_string(60*60 + 61.5), '01:01:01.5')
        self.assertEqual(op.time_string(123*60*60 + 61.5), '123:01:01.5')
        self.assertEqual(op.time_string(123*60*60 + 61), '123:01:01')

    def test_config_string(self):
        self.assertIn(op.config_string({'a':1,'b':3.456}), ['{a=1, b=3.5}', '{b=3.5, a=1}']) # don't have to specify order
        self.assertEqual(op.config_string({'a':1,'b':3.456,'x':'hi'}, order=['a','b','x']), '{a=1, b=3.5, x="hi"}') # can have several data types
        self.assertEqual(op.config_string({'a':1,'x':'hi'}, order=['x','a']), '{x="hi", a=1}') # can give non-alphabetical order
        self.assertRaises(AssertionError, op.config_string, {'a':1,'b':2}, order=['a']) # if order is given, must list all keys

class TestOptimiser(unittest.TestCase):
    def test_simple_grid(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.LocalEvaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator(optimiser)

        self.assertEqual(optimiser.best_known_sample(), None)

        optimiser.run(run_async=True)
        evaluator.start(run_async=True)
        optimiser.wait_for()
        evaluator.wait_for()

        # should both have been shut down
        self.assertFalse(optimiser.running)
        self.assertIsNone(optimiser.proc)
        self.assertIsNone(evaluator.proc)

        mks = lambda a,b,cost: op.Sample({'a':a, 'b':b}, cost) # make sample
        samples = [mks(1,3,1), mks(2,3,2), mks(1,4,1), mks(2,4,2)]

        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_known_sample(), [mks(1,3,1), mks(1,4,1)]) # either would be acceptable


    def test_grid_specific_order(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.LocalEvaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.GridSearchOptimiser(ranges, order=['b','a'])
        evaluator = TestEvaluator(optimiser)

        optimiser.run(run_async=True)
        evaluator.start(run_async=True)
        optimiser.wait_for()
        evaluator.wait_for()

        # should both have been shut down
        self.assertFalse(optimiser.running)
        self.assertIsNone(optimiser.proc)
        self.assertIsNone(evaluator.proc)

        mks = lambda a,b,cost: op.Sample({'a':a, 'b':b}, cost) # make sample
        samples = [mks(1,3,1), mks(1,4,1), mks(2,3,2), mks(2,4,2)]

        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_known_sample(), [mks(1,3,1), mks(1,4,1)]) # either would be acceptable

    def test_range_length_1(self):
        # ranges of length 0 are not allowed
        ranges = {'a':[1], 'b':[2]}
        class TestEvaluator(op.LocalEvaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator(optimiser)

        optimiser.run(run_async=True)
        evaluator.start(run_async=True)
        optimiser.wait_for()
        evaluator.wait_for()

        # should both have been shut down
        self.assertFalse(optimiser.running)
        self.assertIsNone(optimiser.proc)
        self.assertIsNone(evaluator.proc)

        mks = lambda a,b,cost: op.Sample({'a':a, 'b':b}, cost) # make sample
        samples = [mks(1,2,1)]

        self.assertEqual(optimiser.samples, samples)
        self.assertEqual(optimiser.best_known_sample(), mks(1,2,1))

    def test_empty_range(self):
        ranges = {}
        class TestEvaluator(op.LocalEvaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.GridSearchOptimiser(ranges, order=[])
        evaluator = TestEvaluator(optimiser)

        optimiser.run(run_async=True)
        evaluator.start(run_async=True)
        optimiser.wait_for()
        evaluator.wait_for()

        # should both have been shut down
        self.assertFalse(optimiser.running)
        self.assertIsNone(optimiser.proc)
        self.assertIsNone(evaluator.proc)

        self.assertEqual(optimiser.samples, [])
        self.assertEqual(optimiser.best_known_sample(), None)

    def test_simple_random(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.LocalEvaluator):
            def test_config(self, config):
                return config.a # placeholder cost function

        optimiser = op.RandomSearchOptimiser(ranges)
        evaluator = TestEvaluator(optimiser)

        optimiser.run(run_async=True)
        evaluator.start(run_async=True)
        optimiser.wait_for()
        evaluator.wait_for()

        # should both have been shut down
        self.assertFalse(optimiser.running)
        self.assertIsNone(optimiser.proc)
        self.assertIsNone(evaluator.proc)

        mks = lambda a,b,cost: op.Sample({'a':a, 'b':b}, cost) # make sample
        samples = [mks(1,3,1), mks(2,3,2), mks(1,4,1), mks(2,4,2)]
        # samples subset of optimiser.samples
        for s in optimiser.samples:
            self.assertIn(s, samples)
        # optimiser.samples subset of samples
        for s in samples:
            self.assertIn(s, optimiser.samples)

        self.assertIn(optimiser.best_known_sample(), [mks(1,3,1), mks(1,4,1)]) # either would be acceptable

    def test_evaluator_stop(self):
        '''
        have the evaluator take a single job, process it and then shutdown,
        requiring it to be started multiple times.
        '''
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.LocalEvaluator):
            def test_config(self, config):
                self.done = True # stop processing jobs
                return config.a # placeholder cost function
        mks = lambda a,b,cost: op.Sample({'a':a, 'b':b}, cost) # make sample

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator(optimiser)

        optimiser.run(run_async=True)
        self.assertTrue(optimiser.running)

        evaluator.start(run_async=True)
        evaluator.wait_for()
        while len(optimiser.samples) == 0:
            time.sleep(0.05) # wait
        self.assertEqual(optimiser.samples, [mks(1,3,1)])

        for i in range(3):
            evaluator.start(run_async=True)
            evaluator.wait_for()

        optimiser.wait_for()

        # should both have been shut down
        self.assertFalse(optimiser.running)
        self.assertIsNone(optimiser.proc)
        self.assertIsNone(evaluator.proc)

        samples = [mks(1,3,1), mks(2,3,2), mks(1,4,1), mks(2,4,2)]
        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_known_sample(), [mks(1,3,1), mks(1,4,1)]) # either would be acceptable

    def test_save_progress(self):
        pass
        #TODO
        #TODO need to account for jobs in the queue or maybe not?
    def test_evaluator_modification(self):
        pass
        #TODO: have the evaluator change the config before returning
    def test_evaluator_list(self):
        ranges = {'a':[1,2], 'b':[3,4]}
        class TestEvaluator(op.LocalEvaluator):
            def test_config(self, config):
                new_config = config.copy()
                new_config.abc = 123
                return [op.Sample(config, config.a), op.Sample(new_config, 10)]

        optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
        evaluator = TestEvaluator(optimiser)

        optimiser.run(run_async=True)
        evaluator.start(run_async=True)
        optimiser.wait_for()
        evaluator.wait_for()

        # should both have been shut down
        self.assertFalse(optimiser.running)
        self.assertIsNone(optimiser.proc)
        self.assertIsNone(evaluator.proc)

        mks = lambda a,b,cost: op.Sample({'a':a, 'b':b}, cost) # make sample
        mks_2 = lambda a,b,cost: op.Sample({'a':a, 'b':b, 'abc':123}, cost) # make sample
        samples = [mks(1,3,1), mks_2(1,3,10),
                   mks(2,3,2), mks_2(2,3,10),
                   mks(1,4,1), mks_2(1,4,10),
                   mks(2,4,2), mks_2(2,4,10)]

        self.assertEqual(optimiser.samples, samples)
        self.assertIn(optimiser.best_known_sample(), [mks(1,3,1), mks(1,4,1)]) # either would be acceptable




if __name__ == '__main__':
    #unittest.main()

    suite = unittest.TestSuite([
        unittest.TestLoader().loadTestsFromTestCase(TestUtils),
        unittest.TestLoader().loadTestsFromTestCase(TestOptimiser)
    ])
    # verbosity: 0 => quiet, 1 => default, 2 => verbose
    unittest.TextTestRunner(verbosity=2).run(suite)

