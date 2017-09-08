'''
Compatibility for python 2

Care should be taken with the following differences which cannot be patched:
- must explicitly inherit from object
    - however with the builtins import, super() can be used without any arguments!
- dict.keys() returns a list and so has to be cast to a set
    - with the builtins import, dict behaves as expected, but dictionary _literals_ do not!
- use __future__ to replace some missing features
- comprehensions in python 2 do not create a new scope! Instead they modify the
  local variables around them! (described as a 'brutal source of errors' --
  https://stackoverflow.com/q/4198906) For example:
def my_fun():
    x = 1
    my_list = [(x, y) for x, y in [(4, 5), (6, 7)]]
    print(x) # python3 prints '1', python2 prints '6'
note: this only affects list comprehensions, not the other comprehensions. To
fix, search for the comprehensions and rename the variables to be different to
any variables in the local scope.

Searching in Vim: (yank the line then `/` then C-r-" to paste)
\[\_.\{-}\]
search for strings of the form `[...]` where `...` can be anything
`\_.` is a version of `.` which matches everything including newlines.
`\{-}` is a non-greedy version of `*`. I tried a more elaborate search pattern,
but I got scared of missing some instances that needed to be fixed.
'''
# must be placed at the top of each file
from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys

PY_VERSION = sys.version_info[0] # major python version

if PY_VERSION == 3:
    from math import inf, isclose

elif PY_VERSION == 2:
    # make python 2 behave more like python 3
    from builtins import *

    # move the standard library to match the new locations
    from future import standard_library
    standard_library.install_aliases()

    inf = float('inf')

    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        ''' implementation from the python3 documentation '''
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    print_ = print
    def print(*args, **kwargs):
        ''' based on https://stackoverflow.com/a/35467658 '''
        flush = kwargs.pop('flush', False)
        print_(*args, **kwargs)
        if flush:
            kwargs.get('file', sys.stdout).flush()
else:
    # if a new python version is released, the logic for choosing between python
    # 2 and 3 has to be changed on a case-by-case basis.
    print('unsupported python version: {}'.format(PY_VERSION))


