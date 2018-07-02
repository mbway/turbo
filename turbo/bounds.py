#!/usr/bin/env python3


class Bounds:
    """ Boundaries of a space """
    def __init__(self, ordered):
        #TODO: do checks like duplicates etc
        #TODO: check that min < max
        self.ordered = ordered
        self.params = set([b[0] for b in ordered])
        self.associative = {b[0]: (b[1], b[2]) for b in ordered}

    def __len__(self):
        return len(self.ordered)

    def get(self, param):
        return self.associative[param]

    def get_param_index(self, param):
        """ Get the index within a point where the value for the given parameter
        can be found.

        For example, given a point `p`:
        `opt.point_to_config(p)[param] == p[opt.bounds.get_param_index(param)]`
        """
        for i, b in enumerate(self.ordered):
            if param == b[0]:
                return i
        raise KeyError()

