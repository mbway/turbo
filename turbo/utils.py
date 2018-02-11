#!/usr/bin/env python3

import sys
import warnings
import numpy as np

def print_err(*args, **kwargs):
    '''print to `stderr` '''
    print(*args, file=sys.stderr, **kwargs)

def row2D(arr):
    ''' convert a numpy array with shape `(l,)` into an array with shape `(1,l)`
        like `np.atleast_2d()`
    '''
    return arr.reshape(1, -1)

def col2D(arr):
    ''' convert a numpy array with shape `(l,)` into an array with shape `(l,1)`
    '''
    return arr.reshape(-1, 1)

def print_warning(warning):
    w = warnings.formatwarning(
        warning.message, warning.category, warning.filename, warning.lineno, warning.line)
    print_err(w)

class IgnoreWarnings(warnings.catch_warnings):
    '''Ignore any warnings that are raised within a block
    '''
    def __init__(self):
        super().__init__(record=True)

class WarningCatcher(warnings.catch_warnings):
    '''
    capture any warnings raised within the with statement and instead of
    printing them, pass them to the given function. Example:
    >>> with WarningCatcher(lambda warn: print(warn)):
    ...     # stuff

    Note: it is possible to nest WarningCatchers, in which case the inner most
        catcher is the only one which receives the warning.

    Note: because of the nature of warnings, `on_warning()` is only called when
        the with statement ends rather than immediately when the warning is
        raised (unlike exceptions).
    '''
    def __init__(self, on_warning):
        '''
        on_warning: a function which takes a warning and does something with it
        '''
        super().__init__(record=True)
        self.on_warning = on_warning
    def __enter__(self):
        self.warning_list = super().__enter__()
    def __exit__(self, *args):
        for warn in self.warning_list:
            self.on_warning(warn)
        super().__exit__(*args)


def close_to_any(x, xs, tol=1e-5):
    '''Whether the point `x` is close to any of the points in `xs`

    Args:
        x: the point to test. `shape=(1, num_attribs)`
        xs: the points to compare with. `shape=(num_points, num_attribs)`
        tol: maximum size of the squared Euclidean distance to be considered 'close'

    Note:
        xs can be empty, in which case return False, however `xs` must still have
        the same number of attributes as `x`
    '''
    assert x.shape[1] == xs.shape[1], 'different number of attributes'
    assert x.shape[0] == 1, 'x must be a single point'
    assert len(x.shape) == len(xs.shape) == 2, 'must be 2D arrays'

    #return np.any(np.linalg.norm(xs - x, axis=1) <= tol)  # l2 norm (Euclidean distance)
    # x is subtracted from each row of xs, each element is squared, each row is
    # summed to leave a 1D array and each sum is checked with the tolerance
    return np.any(np.sum((xs - x)**2, axis=1) <= tol) # squared Euclidean distance

#TODO: test
def unique_rows_close(arr, close_tolerance):
    '''
    Returns:
        a subset of the rows of the given array which are further from each
        other by at least the given closeness tolerance.
    '''
    assert arr.shape[0] > 0
    avoid = np.empty(shape=(0, arr.shape[1]))
    keep_rows = []

    for i, r in enumerate(arr):
        r = row2D(r)
        if not close_to_any(r, avoid, close_tolerance):
            avoid = np.append(avoid, r, axis=0)
            keep_rows.append(i)
    return arr[keep_rows]

def remap(values, range_a, range_b):
    ''' map the values which live in range_a to range_b

    Args:
        values: value or array of values to remap
        range_a: (min, max) of the original domain
        range_b (min, max) of the new domain
    '''
    return np.interp(values, range_a, range_b)
