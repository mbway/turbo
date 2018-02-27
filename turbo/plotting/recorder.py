#!/usr/bin/env python3

import time
import numpy as np

import turbo as tb
import turbo.modules as tm

#TODO: probably should move out of plotting and have it at the root called something else. Recorder or TrialRecorder or something?
class PlottingRecorder(tm.Listener):
    ''' A listener which records data about the trials of an optimiser for plotting later

    Note: design justification: recorders are required because the optimiser
        only keeps track of the data it needs to perform the optimisation (for
        the sake of simplicity). Different data (and different formats and
        arrangements of the same data) is required for plotting, and so the
        cleanest solution is to completely separate this recording of data for
        plotting and the optimisation process itself.

    Attributes:
        trials: a list of Trial objects recorded from the optimiser
    '''
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

    def __init__(self, optimiser=None):
        '''
        Args:
            optimiser: (optional) the optimiser to register with, otherwise
                `Optimiser.register_listener()` should be called with this
                object as an argument in order to receive messages.
        '''
        self.trials = {}
        self.description = None
        self.unfinished_trial_nums = set()
        self.optimiser = optimiser
        if optimiser is not None:
            optimiser.register_listener(self)

    def save_compressed(self, filename, description=None, overwrite=False):
        ''' save the recorder to a file which can be loaded later and used for plotting

        Args:
            description (str): a long-form explanation of what was being done
                during this run (eg the details of the experiment set up) to
                give more information than the filename alone can provide.
            overwrite (bool): whether to overwrite the file if it already exists

        Note:
            this method is necessary as `utils.save_compressed()` will crash
            otherwise due to the circular reference between the recorder and the
            optimiser
        '''
        opt = self.optimiser
        assert opt is not None
        assert not opt.rt.running

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

            if description is not None:
                self.description = description
            tb.utils.save_compressed(self, filename, overwrite)
        finally:
            opt._listeners = listeners
            opt.objective = objective

    @staticmethod
    def load_compressed(filename, quiet=False):
        rec = tb.utils.load_compressed(filename)
        if not quiet and rec.description is not None:
            print('Recorder loaded with description:')
            print(rec.description)
        return rec

    def registered(self, optimiser):
        assert self.optimiser is None or self.optimiser == optimiser, \
            'cannot use the same PlottingRecorder with multiple optimisers'
        self.optimiser = optimiser

    def selection_started(self, trial_num):
        assert trial_num not in self.trials.keys()
        t = PlottingRecorder.Trial()
        t.trial_num = trial_num
        t.selection_time = time.time() # use as storage for the start time until selection has finished
        self.trials[trial_num] = t
        self.unfinished_trial_nums.add(trial_num)

    def selection_finished(self, trial_num, x, selection_info):
        t = self.trials[trial_num]
        t.selection_time = time.time() - t.selection_time
        t.x = x
        t.selection_info = selection_info

    def eval_started(self, trial_num):
        t = self.trials[trial_num]
        t.eval_time = time.time() # use as storage for the start time until evaluation has finished

    def eval_finished(self, trial_num, y, eval_info):
        t = self.trials[trial_num]
        t.y = y
        t.eval_info = eval_info
        t.eval_time = time.time() - t.eval_time
        self.unfinished_trial_nums.remove(trial_num)



    # Utility functions

    def get_sorted_trials(self):
        ''' return a list of (trial_num, Trial) sorted by trial_num (and so sorted by start time) '''
        return sorted(self.trials.items())

    def get_incumbent(self):
        trials = self.get_sorted_trials()
        assert trials, 'no trials'
        costs = [t.y for n, t in trials]
        i = np.argmax(costs) if self.optimiser.is_maximising() else np.argmin(costs)
        return trials[i] # (trial_num, trial)


    def get_data_for_trial(self, trial_num):
        #TODO: for async cannot assume that finished == all trials before trial_num
        finished = [self.trials[n] for n in range(trial_num)]
        trial = self.trials[trial_num]
        return finished, trial

    def get_acquisition_function(self, trial_num):
        ''' see `Optimiser._get_acquisition_function()` '''
        opt = self.optimiser
        acq_type = opt.acq_func_factory.get_type()
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
        acq_fun, acq_info = opt.acq_func_factory(*acq_args)
        return acq_fun

    def remove_unfinished(self):
        ''' remove any unfinished trials. This is useful for still being able to plot after interrupting an Optimiser before it finished '''
        for trial_num in self.unfinished_trial_nums:
            del self.trials[trial_num]
        self.unfinished_trial_nums = set()

