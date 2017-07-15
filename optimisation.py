import math
import copy
from collections import defaultdict

class dotdict(dict):
    '''
        provide syntactic sugar for accessing dict elements with a dot eg
        mydict = {'somekey': 1}
        d = dotdict(mydict)
        d.somekey   # 1
        d.somekey = 2
        d.somekey   # 2
    '''
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, val):
        self[name] = val



class Range:
    def __init__(self, lower, upper, num=10, scale='linear'):
        '''
        lower: the inclusive lower bound of the range
        upper: the inclusive upper bound of the range
        num: the number of points to consider along the range
        scale: describes how the points are spread out in the range
            - 'linear' => points are spaced evenly (like numpy linspace)
            - 'logarithmic' => points spaced evenly along orders of magnitude (like numpy logspace)
        '''
        self.lower = lower
        self.upper = upper
        assert lower <= upper
        self.range = upper - lower
        self.num = num
        assert type(num) == int and num > 0
        self.scale = scale
        assert scale in ['linear', 'logarithmic']


    def as_list(self):
        return [self.nth(n) for n in range(self.num)]

    def nth(self, n):
        '''
            return the n'th point in the range
                n = 0       => lower
                n = num - 1 => upper
        '''
        if self.num == 1: # avoid divide by 0
            return self.lower
        else:
            if self.scale == 'linear':
                return self.lower + self.range*n/float(self.num-1)
            elif self.scale == 'logarithmic':

                return self.lower + self.range*n/float(self.num-1)


def config_string(config, order=None):
    '''
        a more compact dictionary string
    '''
    string = '{'
    order = sorted(list(config.keys())) if order is None else order
    for p in order:
        string += '{}:{:.2g},'.format(p, config[p])
    string = string[:-1] # remove trailing comma
    string += '}'
    return string


class Optimiser:
    '''
    given a search space and a function to call in order to evaluate the cost at
    a given location, find the minimum of the function in the search space.

    Importantly: an expression for the cost function is not required
    '''
    def __init__(self, ranges, evaluate, logger):
        '''
        ranges: dictionary of parameter names and their ranges
        evaluate: function for evaluating a configuration (returning its cost)
            configuration: dictionary of parameter names to values
        logger: a function which takes a string to be logged and does as it wishes with it
        '''
        self.ranges = dotdict(ranges)
        self.params = ranges.keys() # the hyperparameters to tune
        self.evaluate = evaluate
        # the configurations that have been tested [(configuration, cost)]
        self.samples = []
        self.log = logger

    def best_known_sample(self):
        ''' returns the best known (config, cost) or None if there is none '''
        if len(self.samples) > 0:
            return min(self.samples, key=lambda s: s[1]) # key=cost
        else:
            return None

    def average_for_param(self, param_name):
        ''' marginalise the other parameters and return a list of values for the
            given parameter and the average cost when considered together with
            the other parameters.
            Useful for plotting a 2D graph
        '''
        totals = defaultdict(float)
        counts = defaultdict(int)
        for config,cost in self.samples:
            val = config[param_name]
            totals[val] += cost
            counts[val] += 1

        vals = list(sorted(totals.keys()))
        # average cost for the corresponding value
        averages = [totals[val]/counts[val] for val in vals]
        return vals, averages


    def _next_configuration(self):
        ''' implemented by different optimisation methods
            return the next configuration to try, or None if finished
        '''
        raise NotImplemented

    def run(self):
        self.log('starting optimisation...')
        current_best = math.inf # current best cost

        config = self._next_configuration()
        while config is not None:
            config = dotdict(config)
            cost = self.evaluate(config)

            self.log('configuration {} has cost {:.2g} {}'.format(
                config_string(config), cost,
                ('(current best)' if cost < current_best else '')
            ))

            if cost < current_best:
                current_best = cost
            self.samples.append((config, cost))
            config = self._next_configuration()
        self.log('optimisation finished')

class GridSearchOptimiser(Optimiser):
    '''
        Grid search optimisation strategy: search with some step size over each
        dimension to find the best configuration
    '''
    def __init__(self, ranges, evaluate, log=None, order=None):
        '''
        order: the precedence/importance of the parameters (or None for default)
            appearing earlier => more 'primary' (changes more often)
            default could be any order
        '''
        super(GridSearchOptimiser, self).__init__(ranges, evaluate, log)
        self.order = list(ranges.keys()) if order is None else order
        assert set(self.order) == set(ranges.keys())
        # start at the lower boundary for each parameter
        self.current_config = {param : range.lower for param,range in ranges.items()}

    def _next_configuration(self):
        cur = self.current_config
        next_config = copy.copy(cur)

        # basically an algorithm for adding 1 to a number, but with each 'digit'
        # being of a different base. (note: 'little endian')
        carry = False # carry flag
        for p in self.order:
            r = self.ranges[p]
            if cur[p] + r.step > r.upper: # need to carry
                cur[p] = r.lower # wrap this digit and proceed to the next
                carry = True
            else:
                cur[p] += r.step
                carry = False
                break
        # if the carry flag is true then the whole 'number' has overflowed => finished
        if carry:
            return None
        else:
            return next_config
