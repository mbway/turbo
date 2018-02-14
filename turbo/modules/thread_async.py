
import interfaces.async

class ThreadAsync(interfaces.async.Async):
    def __init__(self, num_threads):
        self.threads = []
        self.lock

    def reset(self):
        pass

    def get_free_capacity(self):
        pass

    def start_trial(self):
        pass

    def get_finished_trials(self):
        pass


