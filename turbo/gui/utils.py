#!/usr/bin/env python3

def in_jupyter():
    ''' whether the current script is running in IPython/Jupyter '''
    try:
        __IPYTHON__
    except NameError:
        return False
    return True

