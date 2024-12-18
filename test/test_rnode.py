import pytest
from AutoDiff import RNode
import numpy as np

class TestRNode:
    """This is a class that tests the Reverse Nodes for reverse mode calculation of
    automatic differentiation and dunparent[0][0] methods to overload built-in operators including 
    negation, addition, subtraction, multiplication, true division, and power. The class 
    also overloads numpy operators including square root, exponential, logarithm, sine, 
    cosine, and tangent. Reflective operations are also tested.
    """

    def test_init(self):
        """Test for constructor method
        """
        x1 = RNode(5)
        assert isinstance(x1, RNode)
        assert x1.val == 5 and x1.der is None and x1.parent == []
        x2 = RNode(3.0)
        assert isinstance(x2, RNode)
        assert x2.val == 3.0 and x2.der is None and x2.parent == []

    def test_clear(self):
        """Test for clear function"""
        x1 = RNode(5)
        x1.der = 7890
        assert isinstance(x1, RNode)
        x1.clear()
        assert x1.der == None and x1.parent == []

    def test_grad_vec(self):
        x1 = RNode(5)
        x2 = RNode(4)
        x3 = RNode(3)
        f1 = np.sin(x1)-x2-x3
        f2 = np.cos(x2)*x1/x3

        assert np.allclose(x1.grad_vec([]), [np.cos(5), np.cos(4)/3])
        assert np.allclose(x2.grad_vec([]), [-1.0, -np.sin(4)*5/3])

    def test_grad(self):
        """Test for grad function"""
        x1 = RNode(5)
        x2 = RNode(3)
        x3 = x1 + x2
        assert x1.grad([]) == 1.0 and x2.grad([]) == 1.0

        x4 = RNode(5)
        x5 = RNode(3)
        x6 = x4 * x5
        assert x4.grad([]) == 3.0 and x5.grad([]) == 5.0
    
    def test_neg(self):
        """Test for overloading negation operator"""
        x1 = RNode(5)
        x2 = -x1
        assert x2.val == -5 
        assert x1.parent[0][0] == -1
        assert x1.parent[0][1] == x2

        x3 = RNode(3.0)
        x4 = -x3
        assert x4.val == -3.0
        assert x3.parent[0][0] == -1
        assert x3.parent[0][1] == x4

    def test_add(self):
        """Test for overloading addition operator"""
        x1 = RNode(5)
        x2 = RNode(3)
        x3 = x1 + x2
        x4 = x1 + 10
        x5 = x1 + 2.5

        assert isinstance(x3, RNode)
        assert isinstance(x4, RNode)
        assert isinstance(x5, RNode)
        
        assert x1.parent[0][0] == 1
        assert x1.parent[0][1] == x3
        assert x2.parent[0][0] == 1
        assert x2.parent[0][1] == x3
        assert x1.parent[1][0] == 1
        assert x1.parent[1][1] == x4
        assert x1.parent[2][0] == 1
        assert x1.parent[2][1] == x5
        
        assert x3.val == 8
        assert x4.val == 15
        assert x5.val == 7.5

        with pytest.raises(TypeError):
            x1 + '1'

        with pytest.raises(TypeError):
            '1' + x1

    def test_sub(self):
        """Test for overloading subtraction operator"""
        x1 = RNode(5)
        x2 = RNode(3.0)
        x3 = x1 - x2
        assert x3.val == 2.0
        assert x1.parent[0][0] == 1
        assert x1.parent[0][1] == x3
        assert x2.parent[0][0] == -1
        assert x2.parent[0][1] == x3

        x4 = RNode(5)
        x5 = 3
        x6 = x4 - x5
        assert x6.val == 2.0
        assert x4.parent[0][0] == 1
        assert x4.parent[0][1] == x6

        with pytest.raises(TypeError):
            x1 - '1'

        with pytest.raises(TypeError):
            '1' - x1

    def test_mul(self):
        """Test for overloading multiplication operator"""
        x1 = RNode(5)
        x2 = RNode(3)
        x3 = x1 * x2
        x4 = x1 * 2
        x5 = x1 * 2.5

        assert isinstance(x3, RNode)
        assert isinstance(x4, RNode)
        assert isinstance(x5, RNode)

        assert x1.parent[0][0] == 3
        assert x1.parent[0][1] == x3
        assert x2.parent[0][0] == 5
        assert x2.parent[0][1] == x3
        assert x1.parent[1][0] == 2
        assert x1.parent[1][1] == x4
        assert x1.parent[2][0] == 2.5
        assert x1.parent[2][1] == x5

        assert x3.val == 15
        assert x4.val == 10
        assert x5.val == 12.5

        with pytest.raises(TypeError):
            x1 * '1'

        with pytest.raises(TypeError):
            '1' * x1
    
    def test_truediv(self):
        """Test for overloading true division operator"""
        x1 = RNode(5)
        x2 = RNode(2.5)
        x3 = x1 / x2
        assert x3.val == 2.0
        assert x1.parent[0][0] == 1/2.5
        assert x1.parent[0][1] == x3
        assert x2.parent[0][0] == -5/2.5**2
        assert x2.parent[0][1] == x3
        
        x4 = RNode(5)
        x5 = 2.5
        x6 = x4 / x5
        assert x6.val == 2.0
        assert x4.parent[0][0] == 1/2.5
        assert x4.parent[0][1] == x6

        with pytest.raises(TypeError):
            x1 / '1'

        with pytest.raises(TypeError):
            '1' / x1
        
        with pytest.raises(TypeError):
            x1.__rtruediv__('1')

        with pytest.raises(ZeroDivisionError):
            x1 / 0
            
        with pytest.raises(ZeroDivisionError):
            x1 / RNode(0)

    def test_sqrt(self):
        """Test for overloading numpy square root operator"""
        x1 = RNode(16)
        x2 = np.sqrt(x1)
        x3 = np.sqrt(x2)
        assert isinstance(x2, RNode)
        assert isinstance(x3, RNode)

        assert x1.parent[0][0] == 1/2*16**(-0.5)
        assert x1.parent[0][1] == x2
        assert x2.parent[0][0] == 1/2*4**(-0.5)
        assert x2.parent[0][1] == x3

        assert x2.val == 4
        assert x3.val == 2

        with pytest.raises(ValueError):
            np.sqrt(RNode(-1))

    def test_logistic(self):
        """Test for standard logistic operator"""
        x1 = RNode(16)
        x2 = x1.logistic()
        assert isinstance(x2, RNode)

        assert np.isclose(x1.parent[0][0], np.e ** (16) / (np.e ** (16) + 1) ** 2)
        assert x1.parent[0][1] == x2

        assert np.isclose(x2.val, 1 / (1 + np.exp(-16)))

    def test_log(self):
        """Test for overloading numpy logarithm operator"""
        x1 = RNode(np.e)
        x2 = np.log(x1)
        x3 = x1.log(2)
        assert np.isclose(x2.val, 1.0)
        assert np.isclose(x1.parent[0][0], 1.0 / np.e)
        assert x1.parent[0][1] == x2
        assert np.isclose(x3.val, 1.0 / np.log(2))
        assert np.isclose(x1.parent[1][0], 1.0 / np.log(2)/ np.e)
        assert x1.parent[1][1] == x3

        with pytest.raises(ValueError):
            np.log(RNode(0))

    def test_pow(self):
        """Test for overloading power operator"""
        x1 = RNode(3)
        x2 = x1 ** 2.0
        x3 = x1 ** 3
        x4 = RNode(2)
        x5 = x1 ** x4

        assert isinstance(x2, RNode)
        assert isinstance(x3, RNode)
        assert isinstance(x4, RNode)
        assert x1.parent[0][0] == 6
        assert x1.parent[0][1] == x2
        assert x1.parent[1][0] == 27
        assert x1.parent[1][1] == x3
        assert x1.parent[2][0] == 6
        assert x1.parent[2][1] == x5
        assert x4.parent[0][0] == np.log(3) * 9
        assert x4.parent[0][1] == x5

        assert x2.val == 9.0
        assert x3.val == 27
        assert x5.val == 9

        with pytest.raises(TypeError):
            x1 ** '1'

        with pytest.raises(TypeError):
            '1' ** x1

    def test_exp(self):
        """Test for overloading numpy exponential operator"""
        x1 = RNode(1)
        x2 = np.exp(x1)
        assert x2.val == np.e
        assert x1.parent[0][0] == np.e
        assert x1.parent[0][1] == x2

    def test_sin(self):
        """Test for overloading numpy sine operator"""
        x1 = RNode(5)
        x2 = np.sin(x1)
        x3 = np.sin(x2)
        assert isinstance(x2, RNode)
        assert isinstance(x3, RNode)
        assert x1.parent[0][0] == np.cos(5)
        assert x1.parent[0][1] == x2
        assert x2.parent[0][0] == np.cos(np.sin(5))
        assert x2.parent[0][1] == x3

        assert x2.val == np.sin(5)
        assert x3.val == np.sin(np.sin(5))

    def test_cos(self):
        """Test for overloading numpy cosine operator"""
        x1 = RNode(np.pi/2)
        x2 = np.cos(x1)
        assert np.isclose(x2.val, 0)
        assert x1.parent[0][0] == -1
        assert x1.parent[0][1] == x2
        
    def test_tan(self):
        """Test for overloading numpy tangent operator"""
        x1 = RNode(5)
        x2 = np.tan(x1)
        x3 = np.tan(x2)
        assert isinstance(x2, RNode)
        assert isinstance(x3, RNode)
        assert x1.parent[0][0] == 1 / np.cos(5) ** 2
        assert x1.parent[0][1] == x2
        assert x2.parent[0][0] == 1 / np.cos(np.tan(5)) ** 2
        assert x2.parent[0][1] == x3

        assert x2.val == np.tan(5)
        assert x3.val == np.tan(np.tan(5))

        with pytest.raises(ValueError):
            np.tan(RNode(np.pi / 2))
            
    def test_arcsin(self):
        """Test for overloading numpy arcsine operator"""
        x1 = RNode(0.5)
        x2 = np.arcsin(x1)
        assert isinstance(x2, RNode)
        assert x1.parent[0][0] == 1 / np.sqrt(1 - 0.5 ** 2)
        assert x1.parent[0][1] == x2

        assert x2.val == np.arcsin(0.5)

        with pytest.raises(ValueError):
            np.arcsin(RNode(-2))

    def test_arccos(self):
        """Test for overloading numpy arccosine operator"""
        x1 = RNode(0.5)
        x2 = np.arccos(x1)
        assert isinstance(x2, RNode)
        assert x1.parent[0][0] == -1 / np.sqrt(1 - 0.5 ** 2)
        assert x1.parent[0][1] == x2

        assert x2.val == np.arccos(0.5)

        with pytest.raises(ValueError):
            np.arccos(RNode(-2))

    def test_arctan(self):
        """Test for overloading numpy arctangent operator"""
        x1 = RNode(0.5)
        x2 = np.arctan(x1)
        assert isinstance(x2, RNode)
        assert x1.parent[0][0] == 1 / (1 + 0.5 ** 2)
        assert x1.parent[0][1] == x2

        assert x2.val == np.arctan(0.5)
        
    def test_sinh(self):
        """Test for overloading numpy hyperbolic sine operator"""
        x1 = RNode(0.5)
        x2 = np.sinh(x1)
        assert isinstance(x2, RNode)
        assert x1.parent[0][0] == np.cosh(0.5)
        assert x1.parent[0][1] == x2

        assert x2.val == np.sinh(0.5)

    def test_cosh(self):
        """Test for overloading numpy hyperbolic cosine operator"""
        x1 = RNode(0.5)
        x2 = np.cosh(x1)
        assert isinstance(x2, RNode)
        assert x1.parent[0][0] == np.sinh(0.5)
        assert x1.parent[0][1] == x2

        assert x2.val == np.cosh(0.5)
        
    def test_tanh(self):
        """Test for overloading numpy hyperbolic tangent operator"""
        x1 = RNode(0.5)
        x2 = np.tanh(x1)
        assert isinstance(x2, RNode)
        assert x1.parent[0][0] == 1 / np.cosh(0.5) ** 2
        assert x1.parent[0][1] == x2

        assert x2.val == np.tanh(0.5)

    def test_radd(self): 
        """Test for overloading reflective addition operator"""
        x1 = RNode(5)
        x2 = 3
        x3 = 2.5
        x21a = x2 + x1
        x31a = x3 + x1

        assert isinstance(x21a, RNode)
        assert isinstance(x31a, RNode)
        assert x1.parent[0][0] == 1
        assert x1.parent[0][1] == x21a
        assert x1.parent[1][0] == 1
        assert x1.parent[1][1] == x31a
        assert x21a.val == 8
        assert x31a.val == 7.5

    def test_rsub(self):
        """Test for overloading reflective subtraction operator"""
        x1 = RNode(5)
        x2 = 3
        x3 = x1 - x2
        x4 = x2 - x1
        assert x1.parent[0][0] == -x1.parent[1][0]
        assert x1.parent[0][1] == x3
        assert x1.parent[1][1] == x4
        assert x3.val == -x4.val

    def test_rmul(self):
        """Test for overloading reflective multiplication operator"""
        x1 = RNode(5)
        x2 = 3
        x3 = 2.5
        x21m = x2 * x1
        x31m = x3 * x1
        
        assert isinstance(x21m, RNode)
        assert isinstance(x31m, RNode)
        assert x1.parent[0][0] == 3
        assert x1.parent[0][1] == x21m
        assert x1.parent[1][0] == 2.5
        assert x1.parent[1][1] == x31m
        assert x21m.val == 15
        assert x31m.val == 12.5

    def test_rtruediv(self):
        """Test for overloading reflective true division operator"""
        x1 = RNode(5)
        x2 = 2.5
        x3 = x2 / x1
        assert x1.parent[0][0] == -2.5 * 5 ** (-2)
        assert x1.parent[0][1] == x3
        assert x3.val == 0.5

        with pytest.raises(ZeroDivisionError):
            1 / RNode(0)

    def test_rpow(self):
        """Test for overloading reflective power operator"""
        x1 = RNode(5)
        x2 = 2.5
        x3 = x2 ** x1
        assert x1.parent[0][0] == np.log(2.5) * 2.5 ** 5
        assert x1.parent[0][1] == x3
        assert x3.val == 2.5 ** 5

    def test_comparison(self):
        """Test for overloading comparison operators"""
        x1 = RNode(3)
        x2 = RNode(5)
        x3 = 4
        x4 = 3
        x5 = RNode(5)
        assert x1 < x2
        assert x1 < x3
        assert x2 > x1
        assert x2 > x3
        assert x1 <= x4
        assert x2 <= x5
        assert x1 >= x4
        assert x2 >= x5
        assert x1 == x4
        assert x2 == x5
        assert x1 != x3
        assert x1 != x5

        with pytest.raises(TypeError):
            x1 < '1'

        with pytest.raises(TypeError):
            '1' < x1

        with pytest.raises(TypeError):
            x1 > '1'

        with pytest.raises(TypeError):
            '1' > x1

        with pytest.raises(TypeError):
            x1 <= '1'

        with pytest.raises(TypeError):
            '1' <= x1

        with pytest.raises(TypeError):
            x1 >= '1'

        with pytest.raises(TypeError):
            '1' >= x1

        with pytest.raises(TypeError):
            x1 == '1'

        with pytest.raises(TypeError):
            '1' == x1

        with pytest.raises(TypeError):
            x1 != '1'

        with pytest.raises(TypeError):
            '1' != x1
    
    def test_str(self):
        x1 = RNode(5)
        assert str(x1) == f'RNode: val={x1.val}, with {len(x1.parent)} parent(s).'

    def test_repr(self):
        x1 = RNode(5)
        assert repr(x1) == f'An RNode object with value of {x1.val}, and {len(x1.parent)} parent(s) with derivatives and locations in {x1.parent}.'