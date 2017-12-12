#!/usr/bin/env python3

class Module:
    def __init__(self):
        self.optimiser = None
    def register(self, optimiser):
        self.optimiser = optimiser
    def _is_registered(self):
        return self.optimiser is not None

