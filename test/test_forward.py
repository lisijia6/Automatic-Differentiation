import numpy as np
from AutoDiff import Forward

class TestForward:
    """This is a class for testing the Forward class
    """
    def test_str(self):
        """Test for str function"""
        x = 50
        def f0(x):
            return [1,2,3]
        g0 = Forward(f0, x)
        assert str(g0) == f'Forward: val={g0.val}, der={g0.der}, and output=({g0.output}).'
    
    def test_repr(self):
        """Test for repr function"""
        x = 50
        def f0(x):
            return [1,2,3]
        g0 = Forward(f0, x)
        assert repr(g0) == f'A Forward object with value of {g0.val}, derivative of {g0.der}, and output of ({g0.output}).'

    def test_grad_univariate_scalar_func(self):
        """
        test gradient of univeriate scalar function ie ℝ1 -> ℝ1
        """
        x = 50
        def f0(x):
            return 0

        g0 = Forward(f0, x)
        assert isinstance(g0, Forward)
        assert g0.val == f0(x)
        assert np.allclose(g0.der, 0)
        assert isinstance(g0.der, (int, float))

        x = 50
        def f1(x):
            return 5 * x

        g1 = Forward(f1, x)
        assert isinstance(g1, Forward)
        assert g1.val == f1(x)
        assert np.allclose(g1.der, 5)
        assert isinstance(g1.der, (int, float))

        def f2(x):
            return np.exp(x)
        
        g2 = Forward(f2, x)
        assert isinstance(g2, Forward)
        assert g2.val == f2(x)
        assert np.allclose(g2.der, f2(x))
        assert isinstance(g2.der, (int, float))

        def f3(x):
            return x
        
        g3 = Forward(f3, x)
        assert isinstance(g3, Forward)
        assert g3.val == f3(x)
        assert np.allclose(g3.der, 1)
        assert isinstance(g3.der, (int, float))

        def f4(x):
            return np.sinh(x) + np.cosh(x)
        
        g4 = Forward(f4, x)
        assert isinstance(g4, Forward)
        assert g4.val == f4(x)
        assert np.allclose(g4.der, np.cosh(x) + np.sinh(x))
        assert isinstance(g4.der, (int, float))

        def f5(x):
            return x + x
        
        g5 = Forward(f5, x)
        assert isinstance(g5, Forward)
        assert g5.val == f5(x)
        assert np.allclose(g5.der, 2)
        assert isinstance(g5.der, (int, float))

    def test_grad_multivariate_scalar_func(self):
        """
        test gradient of multivariate scalar function ie ℝm -> ℝ1
        """
        x = np.array([1, 2])
        def f0(x1, x2):
            return 3

        g0 = Forward(f0, *x)
        assert isinstance(g0, Forward)
        assert g0.val == f0(*x)
        assert np.allclose(g0.der, [0, 0])

        def f1(x1, x2):
            return x1 * x2

        g1 = Forward(f1, *x)
        assert isinstance(g1, Forward)
        assert g1.val == f1(*x)
        assert np.allclose(g1.der, [2, 1])

        def f2(x1, x2):
            return 2 * x1 + x2

        g2 = Forward(f2, *x)
        assert isinstance(g2, Forward)
        assert g2.val == f2(*x)
        assert np.allclose(g2.der, [2, 1])

        def f3(x1, x2):
            return np.cos(x1) + np.sin(x2)

        g3 = Forward(f3, *x)
        assert isinstance(g3, Forward)
        assert g3.val == f3(*x)
        assert np.allclose(g3.der, [-np.sin(1), np.cos(2)])

        def f4(x1, x2):
            x3 = x1 * x2
            return x3 + x2 + x3 * x2
        x = [5, 3]
        g4 = Forward(f4, *x)
        assert g4.val == f4(*x)
        assert np.allclose(g4.der, [12, 36])

    def test_grad_univariate_vector_func(self):
        x = 50
        def f0(x):
            return [1,2,3]
        g0 = Forward(f0, x)
        assert isinstance(g0, Forward)
        assert np.allclose(g0.val, f0(x))
        assert np.allclose(g0.der, [0,0,0])

        def f1(x):
            return [5*x, x + 10]
        g1 = Forward(f1, x)
        assert isinstance(g1, Forward)
        assert np.allclose(g1.val, f1(x))
        assert np.allclose(g1.der, [5, 1])

        def f2(x):
            return [np.sin(x), np.cos(x)]
        g2 = Forward(f2, x)
        assert isinstance(g2, Forward)
        assert np.allclose(g2.val, f2(x))
        assert np.allclose(g2.der, [np.cos(x), -np.sin(x)])

        def f3(x):
            return [np.exp(5*x), x**3, 2*np.sqrt(x)]
        g3 = Forward(f3, x)
        assert isinstance(g3, Forward)
        assert np.allclose(g3.val, f3(x))
        assert np.allclose(g3.der, [np.exp(5*x) * 5, 3 * x**2, 1/np.sqrt(x)])

    def test_grad_multivariate_vector_func(self):
        x = [1, 2]

        def f1(x1, x2):
            return [x1 * x2, 3 * x1 + 9 * x2, 100 * x1]

        g1 = Forward(f1, *x)
        assert isinstance(g1, Forward)
        assert np.allclose(g1.val, f1(*x))
        assert np.allclose(g1.der, [[2, 1], [3, 9], [100, 0]])

        x = [3, 4, 5]

        def f2(x1, x2, x3):
            return [np.sin(x1)-x2-x3, np.cos(x2)*x1/x3]

        g2 = Forward(f2, *x)
        assert isinstance(g2, Forward)
        assert np.allclose(g2.val, f2(*x))
        assert np.allclose(g2.der, [[np.cos(3), -1, -1], 
            [np.cos(4)*1/5, -np.sin(4)*3/5, -3*np.cos(4)/5**2]])
        
        def f3(x1, x2, x3):
            return [np.exp(5*x3), 3, x2**3, 2*np.sqrt(x1)]

        g3 = Forward(f3, *x)
        assert isinstance(g3, Forward)
        assert np.allclose(g3.val, f3(*x))
        assert np.allclose(g3.der, [[0, 0, np.exp(5*5) * 5],[0, 0, 0],
            [0, 3 * 4**2, 0],[1/np.sqrt(3), 0, 0]])

        def f4(x1, x2, x3):
            x4 = x1 + x2
            return [np.exp(5 * x3), x2 ** 3 + x4, 2 * np.sqrt(x1) * x4]

        g4 = Forward(f4, *x)
        assert isinstance(g4, Forward)
        assert np.allclose(g4.val, f4(*x))
        assert np.allclose(g4.der, np.array([
            [0, 0, 3.60024e11],
            [1, 49, 0],
            [7.50555, 3.4641, 0]
        ]))

        x = [1, 2]
        def f5(x1, x2):
            return [x1 * x2 -(x1+x2)**x2, 4]

        g5 = Forward(f5, *x)
        assert isinstance(g5, Forward)
        assert np.allclose(g5.val, f5(*x))
        assert np.allclose(g5.der, [[2-2*3,1-(np.log(3)+2/3)*3**2], [0, 0]])

        x = [2, 3]
        def f6(x1, x2):
            return [(x1+x2)/x2, (x1**x2)/(x1+x2)]

        g6 = Forward(f6, *x)
        assert isinstance(g5, Forward)
        assert np.allclose(g6.val, f6(*x))
        assert np.allclose(g6.der, [[1/3,-2/9], [4*13/25, 8*np.log(2)/5-8/25]])

    def test_newton_method(self):
        """
        test Newton Method
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
        def newton2(f, x0, tol=1e-5):
            if abs(f(x0)) < tol:
                return x0
            g = Forward(f, x0)
            new_x0 = x0 - g.val / g.der
            return newton2(f, new_x0)
        our_sol = newton2(f, x0)
        assert np.allclose(ground_truth, our_sol)
