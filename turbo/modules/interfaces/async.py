
class Async:
    def __init__(self):
        self.trials_started = 0
        self.trials_finished = 0
        self.pending_trial_xs = []

    def get_free_capacity(self):
        pass
    def wait_for_capacity(self):
        '''
        sleep until the Async module is ready to start more trials. This can
        occur due to pending trials finishing, or new resources becoming
        available (e.g. new clients connecting)
        '''
        pass

    def start_trial(self, trial_x):
        pass
    def has_pending_trials(self):
        pass
    def get_finished_trials(self, wait):
        '''

        Args:
            wait: whether to wait for all of the pending trials to finish, or
                return immediately with any finished trials currently available.

        Returns:
            The xs and ys of the trials that were pending
        '''
        pass


