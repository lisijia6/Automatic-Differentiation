import numpy as np
from AutoDiff import Reverse

class TestReverse:
    """This is a class for testing the Reverse class
    """
    def test_str(self):
        """Test for str function"""
        x = 50
        def f0(x):
            return 0
        g0 = Reverse(f0, x)
        assert str(g0) == 'Reverse: val=0, and der=0.0.'
    
    def test_repr(self):
        """Test for repr function"""
        x = 50
        def f0(x):
            return 0
        g0 = Reverse(f0, x)
        assert repr(g0) == 'A Reverse object with value of 0, and derivative of 0.0.'

    def test_grad_univariate_scalar_func(self):
        """Testing Reverse class with univariate scalar function
        """
        x = 50
        def f0(x):
            return 0

        g0 = Reverse(f0, x)
        assert isinstance(g0, Reverse)
        assert g0.val == f0(x)
        assert np.allclose(g0.der, 0)
        assert isinstance(g0.der, (int, float))

        x = 50
        def f1(x):
            return 5 * x

        g1 = Reverse(f1, x)
        assert isinstance(g1, Reverse)
        assert g1.val == f1(x)
        assert np.allclose(g1.der, 5)
        assert isinstance(g1.der, (int, float))

        def f2(x):
            return np.exp(x)
        
        g2 = Reverse(f2, x)
        assert isinstance(g2, Reverse)
        assert g2.val == f2(x)
        assert np.allclose(g2.der, f2(x))
        assert isinstance(g2.der, (int, float))

        def f3(x):
            return x
        
        g3 = Reverse(f3, x)
        assert isinstance(g3, Reverse)
        assert g3.val == f3(x)
        assert np.allclose(g3.der, 1)
        assert isinstance(g3.der, (int, float))

        def f4(x):
            return np.sinh(x) + 2 * np.cosh(x)
        
        g4 = Reverse(f4, x)
        assert isinstance(g4, Reverse)
        assert g4.val == f4(x)
        print("der", g4.der)
        print("want", np.cosh(x) + 2 * np.sinh(x))
        assert np.allclose(g4.der, np.cosh(x) + 2 * np.sinh(x))
        assert isinstance(g4.der, (int, float))

    def test_grad_multivariate_scalar_func(self):
        """ Testing Reverse class with multivariate scalar function
        """
        x = [1, 2]
        def f0(x1, x2):
            return 3

        g0 = Reverse(f0, *x)
        assert isinstance(g0, Reverse)
        assert g0.val == f0(*x)
        assert np.allclose(g0.der, [0, 0])

        def f1(x1, x2):
            return x1 * x2

        g1 = Reverse(f1, *x)
        assert isinstance(g1, Reverse)
        assert g1.val == f1(*x)
        assert np.allclose(g1.der, [2, 1])

        def f2(x1, x2):
            return 2 * x1 + x2

        g2 = Reverse(f2, *x)
        assert isinstance(g2, Reverse)
        assert g2.val == f2(*x)
        assert np.allclose(g2.der, [2, 1])

        def f3(x1, x2):
            return np.cos(x1) + np.sin(x2) + np.sinh(x1)

        g3 = Reverse(f3, *x)
        assert isinstance(g3, Reverse)
        assert g3.val == f3(*x)
        assert np.allclose(g3.der, [-np.sin(1)+ np.cosh(1), np.cos(2)])

        def f4(x1, x2):
            return x2

        g4 = Reverse(f4, *x)
        assert isinstance(g4, Reverse)
        assert g4.val == f4(*x)
        assert np.allclose(g4.der, [0, 1])


    def test_grad_univariate_vector_func(self):
        """ Testing Reverse class with univariate vector function
        """
        x = 50
        def f0(x):
            return [1,2,3]
        g0 = Reverse(f0, x)
        assert isinstance(g0, Reverse)
        assert np.allclose(g0.val, f0(x))
        assert np.allclose(g0.der, [0,0,0])

        def f1(x):
            return [5*x, x + 10]
        g1 = Reverse(f1, x)
        assert isinstance(g1, Reverse)
        assert np.allclose(g1.val, f1(x))
        assert np.allclose(g1.der, [5, 1])

        def f2(x):
            return [np.sin(x), np.cos(x)]
        g2 = Reverse(f2, x)
        assert isinstance(g2, Reverse)
        assert np.allclose(g2.val, f2(x))
        assert np.allclose(g2.der, [np.cos(x), -np.sin(x)])

        def f3(x):
            return [np.exp(5*x), x**3, 2*np.sqrt(x), x]
        g3 = Reverse(f3, x)
        assert isinstance(g3, Reverse)
        assert np.allclose(g3.val, f3(x))
        assert np.allclose(g3.der, [np.exp(5*x) * 5, 3 * x**2, 1/np.sqrt(x), 1])

    def test_grad_multivariate_vector_func(self):
        """ Testing Reverse class with multivariate vector function
        """
        x = [1, 2]
        def f1(x1, x2):
            return [2 *x2, x1 * x2, 3 * x1 + 9 * x2, 100 * x1]
        g1 = Reverse(f1, *x)
        assert isinstance(g1, Reverse)
        assert np.allclose(g1.val, f1(*x))
        assert np.allclose(g1.der, [[0, 2], [2, 1], [3, 9], [100, 0]])

        x = [3, 4, 5]
        def f2(x1, x2, x3):
            return [np.sin(x1)-x2-x3, x2, np.cos(x2)*x1/x3]
        g2 = Reverse(f2, *x)
        assert isinstance(g2, Reverse)
        assert np.allclose(g2.val, f2(*x))
        assert np.allclose(g2.der, [[np.cos(3), -1, -1], [0, 1, 0],
            [np.cos(4)*1/5, -np.sin(4)*3/5, -3*np.cos(4)/5**2]])
        
        x = [3, 4, 5]
        def f3(x1, x2, x3):
            return [np.exp(5*x3), 3, x2**3, 2*np.sqrt(x1)]
        g3 = Reverse(f3, *x)
        assert isinstance(g3, Reverse)
        assert np.allclose(g3.val, f3(*x))
        print(g3.der)
        assert np.allclose(g3.der, [[0, 0, np.exp(5*5) * 5],[0, 0, 0],
            [0, 3 * 4**2, 0],[1/np.sqrt(3), 0, 0]])
        
        x = [3, 4, 5]
        def f4(x1, x2, x3):
            return [np.exp(5*x3), x2 + 3*x2, x2+2*np.sqrt(x1)-x3+np.exp(x1)]
        g4 = Reverse(f4, *x)
        assert isinstance(g4, Reverse)
        assert np.allclose(g4.val, f4(*x))
        assert np.allclose(g4.der, [[0, 0, np.exp(5*5) * 5],
            [0, 4, 0],[1/np.sqrt(3)+np.exp(3), 1, -1]])
        
        x = [1, 2]
        def f5(x1, x2):
            return [x1 * x2 -(x1+x2)**x2, 4]

        g5 = Reverse(f5, *x)
        assert isinstance(g5, Reverse)
        assert np.allclose(g5.val, f5(*x))
        assert np.allclose(g5.der, [[2-2*3,1-(np.log(3)+2/3)*3**2], [0, 0]])

    def test_newton_method(self):
        """ Testing Reverse class with newton method
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
            g = Reverse(f, x0)
            new_x0 = x0 - g.val / g.der
            return newton2(f, new_x0)
        our_sol = newton2(f, x0)
        assert np.allclose(ground_truth, our_sol)