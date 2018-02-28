#!/usr/bin/env python3
'''
A Qt powered GUI for interacting with turbo
'''
import sys
import signal

try:
    import PyQt5.QtWidgets as qt
    import PyQt5.QtCore as qtc
    import PyQt5.QtGui as qtg
except ImportError:
    # Qt not installed
    qt = None
    qtc = None
    qtg = None

# local imports
import turbo.modules as tm


def qt_set_trace():
    qtc.pyqtRemoveInputHook()
    import pdb
    pdb.set_trace()

class Thread(qtc.QThread):
    ''' construct a QThread using a similar API to the threading module '''
    def __init__(self, target, args=None):
        super().__init__()
        self.args = [] if args is None else args
        self.target = target
    def run(self):
        self.target(*self.args)

class OverviewWindow(qt.QWidget, tm.Listener):
    ''' A small widget which shows a progress bar for the run and the state (selecting or evaluating) for the current trial
    '''

    state_changed = qtc.pyqtSignal()

    def __init__(self, optimiser, close_on_run_finish=True):
        super().__init__()
        grid = qt.QGridLayout()
        grid.setSpacing(2)

        self.run_progress_label = qt.QLabel('run progress')
        grid.addWidget(self.run_progress_label, 0, 0)
        self.run_progress = qt.QProgressBar(self)
        grid.addWidget(self.run_progress, 1, 0)

        self.trial_progress = qt.QLabel('trial progress')
        grid.addWidget(self.trial_progress, 2, 0)

        self.incumbent_label = qt.QLabel('incumbent')
        grid.addWidget(self.incumbent_label, 3, 0)

        self.optimiser = optimiser
        self.finished_trials = 0
        self.trials_this_run = 0
        self.trial_state = 0
        self.incumbent = (None, None, None)

        self.state_changed.connect(self.update_state)

        self.setLayout(grid)
        self.resize(650, 250)
        self.setWindowTitle('Turbo Optimiser')
        self._center()

        self.show()

        self.close_on_run_finish = close_on_run_finish
        optimiser.register_listener(self)

    def _center(self):
        frame = self.frameGeometry()
        frame.moveCenter(qt.QDesktopWidget().availableGeometry().center())
        self.move(frame.topLeft())

    def update_state(self):
        self.run_progress_label.setText('run progress: {}/{}'.format(self.finished_trials, self.trials_this_run))
        self.run_progress.setValue(self.finished_trials)

        self.trial_progress.setText('trial progress: {}'.format(self.trial_state))
        self.incumbent_label.setText('incumbent:\n  trial_num: {}\n  input: {}\n  cost: {}'.format(*self.incumbent))
        self.update() # redraw

    def run_started(self, finished_trials, max_trials):
        self.trials_this_run = max_trials-finished_trials
        self.finished_trials = 0
        self.run_progress.setRange(0, self.trials_this_run)
        self.state_changed.emit()

    def selection_started(self, trial_num):
        self.trial_state = 'selecting'
        self.state_changed.emit()
    def eval_started(self, trial_num):
        self.trial_state = 'evaluating'
        self.state_changed.emit()
    def eval_finished(self, trial_num, y, eval_info):
        self.trial_state = 'finished'
        self.finished_trials += 1
        self.incumbent = self.optimiser.get_incumbent(as_dict=True)
        self.state_changed.emit()
    def run_finished(self):
        if self.close_on_run_finish:
            self.close()


def setup_ctrl_c():
    '''handle Ctrl+C using the default signal handler, allowing the Qt application to be closed with Ctrl+C'''
    # DFL = 'default signal handler'
    signal.signal(signal.SIGINT, signal.SIG_DFL)
