#!/usr/bin/env python3
# python 2 compatibility
from __future__ import (absolute_import, division, print_function, unicode_literals)
import sys
sys.path.append('../')
from optimisation.py2 import *

if PY_VERSION == 3:
    from PyQt5.QtWidgets import QApplication
elif PY_VERSION == 2:
    from PyQt4.QtGui import QApplication

import signal

import optimisation as op
from optimisation.basic_optimisers import GridSearchOptimiser

def main():
    '''
    example of the module usage
    '''

    ranges = {'a':[1,2], 'b':[3,4]}
    class TestEvaluator(op.Evaluator):
        def test_config(self, config):
            return config.a # placeholder cost function
    optimiser = GridSearchOptimiser(ranges, maximise_cost=True, order=['a','b'])
    evaluator = TestEvaluator()

    app = QApplication(sys.argv)

    op_gui = op.gui.OptimiserGUI(optimiser)
    ev_gui = op.gui.EvaluatorGUI(evaluator)

    def handle_ctrl_c(*args):
        op_gui.close()
        ev_gui.close()
        app.quit()
    signal.signal(signal.SIGINT, handle_ctrl_c)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()


