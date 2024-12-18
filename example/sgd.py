#!/usr/bin/env python3
# File       : sgd.py

import numpy as np
from AutoDiff import optim
"""
PLEASE install the `AutoDiff` package following the instruction on README using the following command before you run the code:
```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple AutoDiff
```
This script demonstrates how to use the SGD optimization method using AutoDiff
"""

x0 = [3, 5]
def f(x1, x2):
        return x1**2 + 2*x2**2
eta = 0.15 # default eta=0.1
n_iter = 60000 # default n_iter = 50000
tol = 1e-6 # default tol=1e-5
sol_SGD = optim.SGD(f, *x0, eta=eta, n_iter=n_iter, tol=tol)
assert f(*sol_SGD) - 0 < tol # f(*sol_SGD) should be 0
print(f'The solution is {sol_SGD}.')