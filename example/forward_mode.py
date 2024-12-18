#!/usr/bin/env python3
# File       : forward_mode.py

import numpy as np
from AutoDiff import Forward
"""
PLEASE install the `AutoDiff` package following the instruction on README using the following command before you run the code:
```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple AutoDiff
```
This script demonstrates how to use Forward mode using AutoDiff
"""

x = 5
f = lambda x: np.exp(np.cos(x)+2)
g = Forward(f, x)
print("The value of f =", g.val)
print("The derivative of f =", g.der)

x = [5, 3] # x can be np array or list
def f(x1, x2):
        return [np.exp(np.cos(x1)+2*np.cos(x2)), x1 * x2, 2]
g = Forward(f, *x)
print("The evaluation of f =", g.val)
print("The Jacobian of f =", g.der)