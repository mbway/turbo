#!/usr/bin/env python3

import sys
import warnings
import numpy as np
import os
import dill as pickle # regular pickle can't pickle lambdas (and has lots of other problems)
import gzip

def detect_pickle_problems(obj):
    ''' use this to descend through a problematic object (manually, one level at
    a time) to find the culprite of a failed pickle or copy
    '''
    for k in obj.__dict__.keys():
        try:
            dill.dumps(getattr(obj, k))
        except Exception:
            print('problematic attribute: {}'.format(k))

def save_compressed(obj, filename, overwrite=False):
    '''save the given object to a compressed file

    the file extension should be '.pkl.gz'

    Note:
        if this fails, dill has some functions to detect problems: https://github.com/uqfoundation/dill/blob/master/dill/detect.py
        such as `dill.detect.badobjects(..., depth=...)`
    '''
    if not overwrite and os.path.isfile(filename):
        raise Exception('file already exists and instructed not to overwrite: {}'.format(filename))
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_compressed(filename):
    ''' load an object from a compressed file created with `save_compressed()` '''
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


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
