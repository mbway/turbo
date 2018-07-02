
import math
import time
import sys

import turbo as tb
import turbo.modules as tm


class OptimiserProgressBar(tm.Listener):
    def __init__(self, optimiser, single_run=True):
        """
        display a progress bar in the command line as the optimiser runs, once the maximum
        number of iterations has been reached, stop watching.

        Args:
            optimiser (turbo.Optimiser): the optimiser to listen to
        """
        self.single_run = single_run
        self.opt = optimiser
        self.opt.register_listener(self)
        self.bar = None
        self.trials_this_run = 0
        self.progress = 0  # incremented after selecting each trial and evaluating each trial
        self.start_time = 0

    def run_started(self, finished_trials, max_trials):
        self.trials_this_run = max_trials - finished_trials
        self.bar = ProgressBar(2 * self.trials_this_run)
        self.start_time = time.time()

    def selection_started(self, trial_num):
        self.bar.print(self.progress, suffix=self._get_suffix('selecting'))

    def selection_finished(self, trial_num, x, selection_info):
        self.progress += 1
        self.bar.print(self.progress)

    def evaluation_started(self, trial_num):
        self.bar.print(self.progress, suffix=self._get_suffix('evaluating'))

    def evaluation_finished(self, trial_num, y, eval_info):
        self.progress += 1
        self.bar.print(self.progress)

    def _get_suffix(self, stage):
        elapsed = tb.utils.duration_string(time.time()-self.start_time)
        return '{}/{} {} elapsed, stage: {}'.format(self.progress/2, self.trials_this_run, elapsed, stage)

    def run_finished(self):
        self.bar.finish()
        if self.single_run:
            self.opt.unregister_listener(self)


class ProgressBar:
    def __init__(self, max_value, bar_length=40):
        self.max_value = max_value
        self.bar_length = bar_length
        self._last_length = 0

    def print(self, value, suffix=''):
        value = min(value, self.max_value)  # cap
        fraction = value/self.max_value
        filled = math.floor(self.bar_length*fraction)
        bar = ('#' * filled) + (' ' * (self.bar_length-filled))
        line = '\r[{}] {}'.format(bar, suffix)

        this_length = len(line)
        # if this line is shorter, need to clear the characters past the end
        if this_length < self._last_length:
            line += ' ' * (self._last_length-this_length)
        self._last_length = this_length

        sys.stdout.write(line)
        sys.stdout.flush()

    def finish(self):
        sys.stdout.write('\n')
        sys.stdout.flush()
