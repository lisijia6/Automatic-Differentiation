# Milestone2b 

## 1.1 What tasks has each group member been assigned to for Milestone 2

During our group meetings, we brainstormed and finalized the overall package framework.
Refined our package user usage and data structure, we split our work into two subgroups and assigned separate implementation tasks.

Catherine and Nancy: 
* `Node` class: functions for overloading elementary operators and reflective elementory operators in node module
* Test for the elementary operators and reflective elementary operators in node module
* Provide interface and other implementation details in documentations.
* Sphinx documentation integration.

Lily and Jiashu: 
* `Forward` class: we aim to complete the forward mode AD for a scalar function of a vector/scalar
* Test class for forward module `TestForward`
* Minimum package requirements: provide `pyproject.toml` and `requirements.txt`. Upload package to PyPI.
* Provide use case (root finding algorithm, e.g. Newton's method).
* CI for pytest, coverage report, inserts a badge reporting pass/fail coverage on `README.md`.

## 1.2 What has each group member done since the submission of Milestone 1

Catherine, Nancy, Lily, and Jiashu: 
* We met frequently to discuss our progress on the implementation.

Catherine:
* Implemented functions for overloading elementary operators and reflective elementory operators in node module (\_\_neg__, \_\_sub__, \_\_truediv__, log, exp, cos, \_\_rsub__, \_\_rtruediv__).
* Wrote test code for the elementary operators and reflective elementary operators in node module (test_sub, test_truediv, test_log, test_exp, test_cos, test_rsub, test_rtruediv).

Nancy:
* Implemented functions for overloading elementary operators and reflective elementory operators in node module (\_\_add__, \_\_mul__, sqrt, \_\_pow__, sin, tan, \_\_radd__, \_\_rmul__).
* Wrote test code for the elementary operators and reflective elementary operators in node module (test_update_node, test_add, test_mul, test_sqrt, test_pow, test_sin, test_tan, test_radd, test_rmul).

Lily:
* Implemented the forward mode AD for a scalar function of a vector, including initialization `__init__()` and the `grad()` method that generates the first derivative of the function and the value of the function given the input value `variables` and function `f`.
* Implemented a portion of the test cases for the `Forward` class, including `f2` in `test_grad_univariate_scalar_func()` and `f2` and `f3` in `test_grad_multivariate_scalar_func()`.

Jiashu:
* Implemented the forward mode AD for a scalar function of a vector, including initialization `__init__()` and the `grad()` method that generates the first derivative of the function and the value of the function given the input value `variables` and function `f`.
* Implemented a portion of the test cases for the `Forward` class, including `f1` in `test_grad_univariate_scalar_func()` and `f1` in `test_grad_multivariate_scalar_func()`.