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
        #__slots__ = ('trial_num', 'x', 'y', 'extra_data')
        def __init__(self):
            self.trial_num = None
            self.x = None
            self.y = None
            self.extra_data = None

    def __init__(self):
        self.trials = {}
        self.optimiser = None

    def registered(self, optimiser):
        assert self.optimiser is None or self.optimiser == optimiser, \
            'cannot use the same PlottingRecorder with multiple optimisers'
        self.optimiser = optimiser

    def selection_started(self, trial_num):
        assert trial_num not in self.trials.keys()
        t = PlottingRecorder.Trial()
        t.trial_num = trial_num
        self.trials[trial_num] = t

    def selection_finished(self, trial_num, x, selection_details):
        t = self.trials[trial_num]
        t.x = x
        t.extra_data = selection_details

    def eval_finished(self, trial_num, y):
        t = self.trials[trial_num]
        t.y = y


    # Utility functions

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
        assert t.extra_data['type'] == 'bayes'
        acq_args = [trial_num, t.extra_data['model'], opt.desired_extremum]
        if acq_type == 'optimism':
            pass # no extra arguments needed
        elif acq_type == 'improvement':
            ys = [f.y for f in finished]
            incumbent_cost = max(ys) if opt.is_maximising() else min(ys)
            acq_args.append(incumbent_cost)
        else:
            raise NotImplementedError('unsupported acquisition function type: {}'.format(acq_type))
        return opt.acq_func_factory(*acq_args)

