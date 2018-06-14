#!/usr/bin/env python3

import unittest
import random
import numpy as np

import utils_tests

unittest.TestCase.maxDiff = None
unittest.TestCase.longMessage = True


def pdb_on_signal():
    """
    break into pdb by sending a signal. Use one of the following:
    pkill -SIGUSR1 myprocess
    killall -SIGUSR1 python3
    """
    print('send SIGUSR1 signal to break into pdb')
    import signal
    # http://blog.devork.be/2009/07/how-to-bring-running-python-program.html

    def handle_pdb(sig, frame):
        import pdb
        pdb.Pdb().set_trace(frame)
    signal.signal(signal.SIGUSR1, handle_pdb)


def run_tests():
    pdb_on_signal()
    np.random.seed(42)
    random.seed(42)

    suite = unittest.TestSuite([
        unittest.TestLoader().loadTestsFromModule(utils_tests)
    ])

    unittest.TextTestRunner(verbosity=2, descriptions=False, failfast=True).run(suite)


if __name__ == '__main__':
    run_tests()
