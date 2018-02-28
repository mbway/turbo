#!/usr/bin/env python3

import time
import sys
import numpy as np
import PyQt5.QtWidgets as qt
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg

# local modules
import turbo as tb
import turbo.modules as tm
import turbo.plotting as tp
import turbo.gui.qt as tg

def objective(x):
    noise = np.random.normal(0, 6, size=None if isinstance(x, float) else x.shape)
    time.sleep(np.random.normal(2, 1)) # simulate an expensive objective function
    return 100 * np.sin(x**2/5) * np.cos(x*1.5) + 100 + noise

def main():
    # make deterministic
    np.random.seed(42)

    app = qt.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    op = tb.Optimiser(objective, 'min',
                      bounds=[('x', 0, 12)],
                      pre_phase_trials=6,
                      settings_preset='default')

    desc = 'testing the Qt GUI module as well as the recorder auto-save and external plotting'
    rec = tp.PlottingRecorder(op, description=desc, autosave_filename='/tmp/turbo_autosave')
    gui = tg.OverviewWindow(op)

    # need to do optimisation on another thread so that Qt messages can be processed
    t = tg.Thread(target=lambda: op.run(max_trials=30))
    t.start()

    tg.setup_ctrl_c() # Ctrl+C to close the application

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
