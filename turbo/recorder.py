#!/usr/bin/env python3

import os
import time
import datetime
import numpy as np

import turbo as tb
import turbo.modules as tm

#TODO: I have decided that pickling is a totally inappropriate serialisation method for this data for the following reasons (note them down in the documentation)
# - from Gpy: Pickling is meant to serialize models within the same environment, and not to store models on disk to be used later on.
# - if the code which created the optimiser changes (eg file deleted) then the pickled data CANNOT BE LOADED!
#   as a fix, can use sys.modules['old_module'] = new_module
#   if the module has moved and hasn't changed much
# - the data isn't human readable
# - the data isn't readable by any other tool
# - the data isn't guaranteed to last between python versions
# - it introduces dill as a dependency
# - the data isn't readable if turbo changes particular classes (like Trial for example)
# - the save cannot be loaded easily from another directory because the modules for the original source code will not be present!
#
# Potential fix:
# write a way for the optimiser to be initialised from a static configuration file / dictionary. That way only static data has to be stored
# then it will be trivial to store using JSON or a binary serialisation method
# can use inspect to get the source code for functions like `beta = lambda trial_num: np.log(trial_num)` and save as strings
# could use dis to get the bytecode instead perhaps? Either way, save as something which can be re-loaded. Or perhaps pickle just the function stuff and store it as a string inside the JSON file or whatever


class Recorder(tm.Listener):
    """ A listener which records data about the trials of an optimiser for plotting later

    Note: design justification: recorders are required because the optimiser
        only keeps track of the data it needs to perform the optimisation (for
        the sake of simplicity). Different data (and different formats and
        arrangements of the same data) is required for plotting, and so the
        cleanest solution is to completely separate this recording of data for
        plotting and the optimisation process itself.

    Attributes:
        trials: a list of Trial objects recorded from the optimiser
    """
    class Trial:
        def __init__(self):
            self.trial_num = None
            self.x = None
            self.y = None
            self.selection_info = None
            self.selection_time = None
            self.eval_info = None
            self.eval_time = None

        def __repr__(self):
            attrs = ('trial_num', 'x', 'y', 'selection_info', 'selection_time', 'eval_info', 'eval_time')
            return 'Trial({})'.format(', '.join('{}={}'.format(k, getattr(self, k)) for k in attrs))

        def is_pre_phase(self):
            return self.selection_info['type'] == 'pre_phase'

        def is_bayes(self):
            return self.selection_info['type'] == 'bayes'

        def is_fallback(self):
            return self.selection_info['type'] == 'fallback'

    class Run:
        def __init__(self, previously_finished, max_trials):
            self.start_date = datetime.datetime.now()
            self.finish_date = None
            self.previously_finished = previously_finished
            self.max_trials = max_trials

        def finish(self):
            self.finish_date = datetime.datetime.now()

        def is_finished(self):
            return self.finish_date is not None

        @property
        def duration(self):
            return None if self.finish_date is None else (self.finish_date-self.start_date).total_seconds()

        @property
        def num_trials(self):
            return self.max_trials-self.previously_finished

    def __init__(self, optimiser=None, description=None, autosave_filename=None):
        """
        Args:
            optimiser: (optional) the optimiser to register with, otherwise
                `Optimiser.register_listener()` should be called with this
                object as an argument in order to receive messages.
            description (str): a long-form explanation of what was being done
                during this run (eg the details of the experiment set up) to
                give more information than the filename alone can provide.
            autosave_filename (str): when provided, save the recorder to this
                file every time a trial is finished. This allows the user to do
                plotting before the optimisation process has finished. The file
                must not already exist.
        """
        self.runs = []
        self.trials = {}
        self.description = description
        self.unfinished_trial_nums = set()
        self.optimiser = optimiser
        if optimiser is not None:
            optimiser.register_listener(self)
        if autosave_filename is not None and os.path.exists(autosave_filename):
            base_filename = autosave_filename
            suffix = 1
            while os.path.exists(autosave_filename):
                autosave_filename = '{}_{}'.format(base_filename, suffix)
                suffix += 1
            print('the file "{}" already exists, using "{}" instead'.format(base_filename, autosave_filename))
        self.autosave_filename = autosave_filename

    def save_compressed(self, path, overwrite=False):
        """ save the recorder to a file which can be loaded later and used for plotting

        Args:
            path: the path where the recording will be saved to
            overwrite (bool): whether to overwrite the file if it already exists

        Note:
            this method is necessary as `utils.save_compressed()` will crash
            otherwise due to the circular reference between the recorder and the
            optimiser
        """
        opt = self.optimiser
        assert opt is not None

        listeners = opt._listeners
        objective = opt.objective
        try:
            # since self is a listener of the optimiser, if the listeners are saved
            # then there is a circular reference!
            opt._listeners = []
            # The objective function could be arbitrarily complex and so may not pickle
            opt.objective = None

            problems = tb.utils.detect_pickle_problems(self, quiet=True)
            assert not problems, 'problems detected: {}'.format(problems)

            while True:
                try:
                    tb.utils.save_compressed(self, path, overwrite)
                    break
                except Exception as e:
                    print('in save_compressed:')
                    print(e)
                    input('press enter to try again')

        finally:
            opt._listeners = listeners
            opt.objective = objective

    @staticmethod
    def load_compressed(filename, quiet=False):
        rec = tb.utils.load_compressed(filename)
        if not quiet:
            print('Recorder loaded:')
            print(rec.get_summary())
        return rec

    def get_summary(self):
        s = '{} trials over {} run{}\n'.format(len(self.trials), len(self.runs), 's' if len(self.runs) > 1 else '')
        for i, r in enumerate(self.runs):
            if r.is_finished():
                s += 'run {}: {} trials in {}, started {}\n'.format(i, r.num_trials, tb.utils.duration_string(r.duration), r.start_date)
            else:
                s += 'run {}: unfinished\n'.format(i)
        if self.unfinished_trial_nums:
            s += 'unfinished trials: {}\n'.format(self.unfinished_trial_nums)
        s += 'description:\n{}\n'.format(self.description)
        return s

    def registered(self, optimiser):
        assert self.optimiser is None or self.optimiser == optimiser, \
            'cannot use the same Recorder with multiple optimisers'
        self.optimiser = optimiser

    def run_started(self, finished_trials, max_trials):
        r = Recorder.Run(finished_trials, max_trials)
        self.runs.append(r)

    def selection_started(self, trial_num):
        assert trial_num not in self.trials.keys()
        t = Recorder.Trial()
        t.trial_num = trial_num
        t.selection_time = time.time() # use as storage for the start time until selection has finished
        self.trials[trial_num] = t
        self.unfinished_trial_nums.add(trial_num)

    def selection_finished(self, trial_num, x, selection_info):
        t = self.trials[trial_num]
        t.selection_time = time.time() - t.selection_time
        t.x = x
        t.selection_info = selection_info

    def evaluation_started(self, trial_num):
        t = self.trials[trial_num]
        t.eval_time = time.time() # use as storage for the start time until evaluation has finished

    def evaluation_finished(self, trial_num, y, eval_info):
        t = self.trials[trial_num]
        t.y = y
        t.eval_info = eval_info
        t.eval_time = time.time() - t.eval_time
        self.unfinished_trial_nums.remove(trial_num)
        if self.autosave_filename is not None:
            self.save_compressed(self.autosave_filename, overwrite=True)

    def run_finished(self):
        r = self.runs[-1]
        r.finish()
        if self.autosave_filename is not None:
            self.save_compressed(self.autosave_filename, overwrite=True)

    # Utility functions

    def get_sorted_trials(self):
        """ return a list of (trial_num, Trial) sorted by trial_num (and so sorted by start time) """
        return sorted(self.trials.items())

    def get_ranked_trials(self):
        """ return a list of (trial_num, Trial) sorted by cost (best first)  """
        maximising = self.optimiser.is_maximising()
        return sorted(self.trials.items(), key=lambda item: -item[1].y if maximising else item[1].y)

    def get_incumbent(self, up_to=None):
        """
        Args:
            up_to (int): the incumbent for the trials up to and including this trial number. Pass None to include all trials.
        """
        trials = self.get_sorted_trials()
        if up_to is not None:
            trials = trials[:up_to+1]
        assert trials, 'no trials'
        costs = [t.y for n, t in trials]
        i = int(np.argmax(costs)) if self.optimiser.is_maximising() else int(np.argmin(costs))
        return trials[i]  # (trial_num, trial)

    def get_data_for_trial(self, trial_num):
        #TODO: for async cannot assume that finished == all trials before trial_num
        finished = [self.trials[n] for n in range(trial_num)]
        trial = self.trials[trial_num]
        return finished, trial

    def get_acquisition_function(self, trial_num):
        """ see `Optimiser._get_acquisition_function()` """
        opt = self.optimiser
        acq_type = opt.acquisition.get_type()
        finished, t = self.get_data_for_trial(trial_num)
        assert 'model' in t.selection_info, 'the trial doesn\'t have a model'
        acq_args = [trial_num, t.selection_info['model'], opt.desired_extremum]
        if acq_type == 'optimism':
            pass # no extra arguments needed
        elif acq_type == 'improvement':
            ys = [f.y for f in finished]
            incumbent_cost = max(ys) if opt.is_maximising() else min(ys)
            acq_args.append(incumbent_cost)
        else:
            raise NotImplementedError('unsupported acquisition function type: {}'.format(acq_type))
        acq_fun, acq_info = opt.acquisition.construct_function(*acq_args)
        return acq_fun

    def remove_unfinished(self):
        """ remove any unfinished trials. This is useful for still being able to
        plot after interrupting an Optimiser before it finished
        """
        for trial_num in self.unfinished_trial_nums:
            del self.trials[trial_num]
        self.unfinished_trial_nums = set()

    def has_unfinished_trials(self):
        return len(self.unfinished_trial_nums) > 0
