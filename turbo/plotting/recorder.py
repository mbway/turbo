#!/usr/bin/env python3

import turbo.modules as tm

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
        #TODO: re-enable slots (disabled so Jupyter doesn't forget the class between reloads)
        #__slots__ = ('trial_num', 'x', 'y', 'selection_info')
        def __init__(self):
            self.trial_num = None
            self.x = None
            self.y = None
            self.selection_info = None
            self.eval_info = None

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
        self.optimiser = optimiser
        if optimiser is not None:
            optimiser.register_listener(self)

    def registered(self, optimiser):
        assert self.optimiser is None or self.optimiser == optimiser, \
            'cannot use the same PlottingRecorder with multiple optimisers'
        self.optimiser = optimiser

    def selection_started(self, trial_num):
        assert trial_num not in self.trials.keys()
        t = PlottingRecorder.Trial()
        t.trial_num = trial_num
        self.trials[trial_num] = t

    def selection_finished(self, trial_num, x, selection_info):
        t = self.trials[trial_num]
        t.x = x
        t.selection_info = selection_info

    def eval_finished(self, trial_num, y, eval_info):
        t = self.trials[trial_num]
        t.y = y
        t.eval_info = eval_info



    # Utility functions

    def get_sorted_trials(self):
        ''' return a list of (trial_num, Trial) sorted by trial_num (and so sorted by start time) '''
        return sorted(self.trials.items())

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
        return opt.acq_func_factory(*acq_args)

