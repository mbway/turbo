#!/usr/bin/env python3

import sys
import numpy as np
import os
import dill  # regular pickle can't pickle lambdas (and has lots of other problems)
import gzip


def detect_pickle_problems(obj, quiet=False, always_check_contents=False):
    """ detect problems that may be encountered when trying to either save or
    load the given object using dill (a better pickling library)

    Use this tool to descend through a problematic object (manually, one level
    at a time) to find the culprit of a failed pickle or copy.

    Note: sometimes the overall object will pickle but the contents won't

    also look into things like dill.detect.trace(True); dill.detect.errors(obj)

    Returns:
        a list of [('save|load', attribute_name, exception_caused)]
    """
    problems = []

    saved = None
    try:
        saved = dill.dumps(obj)
    except Exception as e:
        problems.append(('save', '<obj>', e))
        print('problem saving the object')

    if saved is not None:
        try:
            dill.loads(saved)
        except Exception as e:
            print('problem loading the object')
            problems.append(('load', '<obj>', e))

    if not problems and not always_check_contents:
        return None

    if hasattr(obj, '__dict__'):
        items = list(obj.__dict__.items())
    else:
        try:
            keys = list(iter(obj))
            vals = [obj[k] for k in keys]
            items = list(zip(keys, vals))
        except TypeError: # not iterable
            items = []

    for k, v in items:
        try:
            saved = dill.dumps(v)
        except Exception as e:
            problems.append(('save', k, e))
            if not quiet:
                print('problem saving attribute: {}'.format(k))
            continue

        try:
            dill.loads(saved)
        except Exception as e:
            problems.append(('load', k, e))
            if not quiet:
                print('problem loading attribute: {}'.format(k))

    return problems if problems else None


def save_compressed(obj, filename, overwrite=False):
    """ save the given object to a compressed file

    the file extension should be '.pkl.gz'

    Note:
        if this fails, dill has some functions to detect problems: https://github.com/uqfoundation/dill/blob/master/dill/detect.py
        such as `dill.detect.badobjects(..., depth=...)`
    """
    if not overwrite and os.path.isfile(filename):
        raise Exception('file already exists and instructed not to overwrite: {}'.format(filename))
    with gzip.open(filename, 'wb') as f:
        dill.dump(obj, f, protocol=dill.HIGHEST_PROTOCOL)


def load_compressed(filename):
    """ load an object from a compressed file created with `save_compressed()` """
    with gzip.open(filename, 'rb') as f:
        return dill.load(f)


def print_err(*args, **kwargs):
    """print to `stderr` """
    print(*args, file=sys.stderr, **kwargs)


def row_2d(arr):
    """ convert a numpy array with shape `(l,)` into an array with shape `(1,l)`
        like `np.atleast_2d()`
    """
    return arr.reshape(1, -1)


def col_2d(arr):
    """ convert a numpy array with shape `(l,)` into an array with shape `(l,1)`
    """
    return arr.reshape(-1, 1)


def close_to_any(x, xs, tol=1e-5):
    """ Whether the point `x` is close to any of the points in `xs`

    Args:
        x: the point to test. `shape=(1, num_attribs)`
        xs: the points to compare with. `shape=(num_points, num_attribs)`
        tol: maximum size of the squared Euclidean distance to be considered 'close'

    Note:
        xs can be empty, in which case return False, however `xs` must still have
        the same number of attributes as `x`
    """
    assert x.shape[1] == xs.shape[1], 'different number of attributes'
    assert x.shape[0] == 1, 'x must be a single point'
    assert len(x.shape) == len(xs.shape) == 2, 'must be 2D arrays'

    #return np.any(np.linalg.norm(xs - x, axis=1) <= tol)  # l2 norm (Euclidean distance)
    # x is subtracted from each row of xs, each element is squared, each row is
    # summed to leave a 1D array and each sum is checked with the tolerance
    return np.any(np.sum((xs - x)**2, axis=1) <= tol) # squared Euclidean distance


#TODO: test
def unique_rows_close(arr, close_tolerance):
    """
    Returns:
        a subset of the rows of the given array which are further from each
        other by at least the given closeness tolerance.
    """
    assert arr.shape[0] > 0
    avoid = np.empty(shape=(0, arr.shape[1]))
    keep_rows = []

    for i, r in enumerate(arr):
        r = row_2d(r)
        if not close_to_any(r, avoid, close_tolerance):
            avoid = np.append(avoid, r, axis=0)
            keep_rows.append(i)
    return arr[keep_rows]


def remap(values, range_a, range_b):
    """ map the values which live in range_a to range_b

    Args:
        values: value or array of values to remap
        range_a: (min, max) of the original domain
        range_b (min, max) of the new domain
    """
    return np.interp(values, range_a, range_b)


def duration_string(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    h, m = int(h), int(m)
    if h > 0:
        return '{}:{:02d}:{:.1f}'.format(h, m, s)
    else:
        return '{:02d}:{:.1f}'.format(m, s)
