import numpy as np
from AutoDiff.optim import Newton, SGD
import pytest

class TestOptimization:
    def test_multivariate_newton(self):
        """
        test multivariate Newton
        """
        # taken from https://www.mathworks.com/matlabcentral/fileexchange/104415-multivariate-newton-s-method-newtons_method_n
        x0 = [0, 1]
        def f(x1, x2):
            return [
                2 * x1 + x2 - np.exp(-x1),
                -x1 + 2 * x2 - np.exp(-x2)
            ]

        sol = Newton(f, *x0)
        assert np.allclose(f(*sol), np.zeros(2))

    def test_univariate_newton(self):
        """
        test univariate Newton
        """
        f = lambda x: x**2 - 2
        f_prime = lambda x: 2 * x
        x0 = 1.4

        def newton(f, df, x0, tol=1e-5):
            if abs(f(x0)) < tol:
                return x0
            new_x0 = x0 - f(x0) / df(x0)
            return newton(f, df, new_x0)

        ground_truth = newton(f, f_prime, x0)
        assert np.allclose(ground_truth, np.sqrt(2))
        our_sol = Newton(f, x0)
        assert np.allclose(ground_truth, our_sol)

        x0 = [4, 3]
        def f(x, y):
            return x ** 2 + y ** 2
        sol = Newton(f, *x0)
        assert abs(f(*sol)) < 1e-5
    
    def test_newton_diverge(self):
        """
        test Newton where the function does not converge
        """
        x0 = [4, 3]
        def f(x, y):
            return 5
        with pytest.raises(RuntimeError):
            sol = Newton(f, *x0)

    def test_univariate_sgd(self):
        """
        test univariate SGD
        """
        x0 = 1
        f = lambda x: x**2
        sol = SGD(f, x0)
        assert isinstance(sol, float)
        assert abs(f(sol)) < 1e-5

    def test_multivariate_sgd(self):
        x0 = [4, 3]
        def f(x, y):
            return x ** 2 + y ** 2
        sol = SGD(f, *x0)
        assert abs(f(*sol)) < 1e-5

        x0 = [4, 3]
        def f(x, y):
            return x ** 2 - y ** 2
        with pytest.raises(RuntimeError):
            sol = SGD(f, *x0)