#!/usr/bin/env python3

class Plan:
    def __init__(self, pre_phase_trials, checkpoint_interval=1,
                 random_proportion=0.0, re_train_interval=1):
        self.pre_phase_trials = pre_phase_trials
        self.checkpoint_interval = checkpoint_interval
        self.random_proportion = random_proportion
        self.re_train_interval = re_train_interval

    def in_pre_phase(self, iteration):
        return iteration < self.pre_phase_trials

