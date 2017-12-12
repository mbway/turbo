
class UCB_Acquisition_Factory:
    def __init__(self, beta):
        self.beta = beta

    class UCB:
        def __init__(self, beta, model):
            self.beta = beta
            self.model = model

        def __call__(self, p):
            return 0.0

    def new_function(self, iteration, model):
        if callable(self.beta):
            b = self.beta(iteration)
        else:
            b = self.beta # beta is a constant
        return UCB(b, model)

