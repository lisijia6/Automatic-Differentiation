#!/usr/bin/env python3
# File       : newton.py

from AutoDiff import optim
"""
PLEASE install the `AutoDiff` package following the instruction on README using the following command before you run the code:
```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple AutoDiff
```
This script demonstrates how to use the Newton's method using AutoDiff
"""

x0 = [3, 5]
def f(x1, x2):
        return x1**2 + 2*x2**2
tol = 1e-6 # default tol=1e-5
sol_newton = optim.Newton(f, *x0, tol=tol)
assert f(*sol_newton) - 0 < tol # f(*sol_newton) should be 0
print(f'The solution is {sol_newton}.')