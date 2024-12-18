import pytest
from AutoDiff import Node
import numpy as np

class TestNode:
    """This is a class that tests the Dual numbers and dunder methods to overload 
    build-in operators including negation, addition, subtraction, multiplication, true
    division, and power. The class also tests overloading numpy operators including square root,
    exponential, logarithm, sine, cosine, and tangent. Reflective operations are also 
    tested.
    """

    def test_init(self):
        """Test for constructor method
        """
        x1 = Node(5)
        assert isinstance(x1, Node)
        assert x1.val == 5 and x1.der == 1
        x2 = Node(3.0)
        assert isinstance(x2, Node)
        assert x2.val == 3.0 and x2.der == 1
        x3 = Node(3.0, np.array([0,1]))
        assert isinstance(x3, Node)
        assert x3.val == 3.0 and np.array_equal(x3.der, np.array([0,1]))

    def test_str(self):
        """Test for str function"""
        x1 = Node(5)
        assert str(x1) == f'Node: vindex={x1.v_index}, val={x1.val}, der={x1.der}, parent={x1.parent}, and op={x1.op}.'
    
    def test_repr(self):
        """Test for repr function"""
        x1 = Node(5)
        assert repr(x1) == f'A Node object with index of {x1.v_index}, value of {x1.val}, derivative of {x1.der}, parent of {x1.parent}, and operator of {x1.op}.'
        
    def test_update_node(self):
        """Test for update_node function"""
        x1 = Node(5)
        parent = Node(3)
        x1.update_node([parent], ['+'])
        assert isinstance(x1.parent[0], Node)
        assert x1.parent[0].val == parent.val and x1.parent[0].der == parent.der
        assert x1.op == ['+']
    
    def test_neg(self):
        """Test for overloading negation operator"""
        x1 = Node(5)
        x2 = -x1
        assert x2.val == -5 and x2.der == -1
        x3 = Node(3.0)
        x4 = -x3
        assert x4.val == -3.0 and x4.der == -1
        x5 = Node(3.2,5)
        x6 = -x5
        assert x6.val == -3.2 and x6.der == -5

    def test_add(self):
        """Test for overloading addition operator"""
        x1 = Node(5)
        x2 = Node(3)
        x3 = x1 + x2
        x4 = x1 + 10
        x5 = x1 + 2.5

        assert isinstance(x3, Node)
        assert isinstance(x4, Node)
        assert isinstance(x5, Node)
        
        assert x3.val == 8 and x3.der == 2
        assert x4.val == 15 and x4.der == 1
        assert x5.val == 7.5 and x5.der == 1

        with pytest.raises(TypeError):
            x1 + '1'

        with pytest.raises(TypeError):
            '1' + x1

    def test_sub(self):
        """Test for overloading subtraction operator"""
        x1 = Node(5)
        x2 = Node(3.0)
        x3 = x1 - x2
        assert x3.val == 2.0 and x3.der == 0

        x4 = Node(5, np.array([1,0]))
        x5 = Node(3.0, np.array([0,1]))
        x6 = x4 - x5
        assert x6.val == 2.0 and np.array_equal(x6.der, np.array([1,-1]))

        x7 = Node(5)
        x8 = 3
        x9 = x7 - x8
        assert x9.val == 2.0 and x9.der == 1

        x10 = Node(5, np.array([1,0]))
        x11 = 3
        x12 = x10 - x11
        assert x12.val == 2.0 and np.array_equal(x12.der, np.array([1,0]))

        with pytest.raises(TypeError):
            x1 - '1'

        with pytest.raises(TypeError):
            '1' - x1

    def test_mul(self):
        """Test for overloading multiplication operator"""
        x1 = Node(5)
        x2 = Node(3)
        x3 = x1 * x2
        x4 = x1 * 2
        x5 = x1 * 2.5

        assert isinstance(x3, Node)
        assert isinstance(x4, Node)
        assert isinstance(x5, Node)

        assert x3.val == 15 and x3.der == 8
        assert x4.val == 10 and x4.der == 2
        assert x5.val == 12.5 and x5.der == 2.5

        with pytest.raises(TypeError):
            x1 * '1'

        with pytest.raises(TypeError):
            '1' * x1
    
    def test_truediv(self):
        """Test for overloading true division operator"""
        x1 = Node(5)
        x2 = Node(2.5)
        x3 = x1 / x2
        assert x3.val == 2.0 and x3.der == -0.4

        x4 = Node(5, np.array([1,0]))
        x5 = Node(2.5, np.array([0,1]))
        x6 = x4 / x5
        assert x6.val == 2.0 and np.array_equal(x6.der, np.array([0.4,-0.8]))

        x7 = Node(5)
        x8 = 2.5
        x9 = x7 / x8
        assert x9.val == 2.0 and x9.der == 0.4

        x10 = Node(5, np.array([1,0]))
        x11 = 2.5
        x12 = x10 / x11
        assert x12.val == 2.0 and np.array_equal(x12.der, np.array([0.4,0]))

        with pytest.raises(TypeError):
            x1 / '1'

        with pytest.raises(TypeError):
            '1' / x1
        
        with pytest.raises(TypeError):
            x1.__rtruediv__('1')

        with pytest.raises(ZeroDivisionError):
            x1 / 0
            
        with pytest.raises(ZeroDivisionError):
            x1 / Node(0)

    def test_sqrt(self):
        """Test for overloading numpy square root operator"""
        x1 = Node(16)
        x2 = np.sqrt(x1)
        x3 = np.sqrt(x2)
        assert isinstance(x2, Node)
        assert isinstance(x3, Node)

        assert x2.parent[0].val == 16 and x2.parent[0].der == 1
        assert x3.parent[0].parent[0].val == 16 and x3.parent[0].parent[0].der == 1
        assert x2.val == 4 and x2.der == 0.125
        assert x3.val == 2 and x3.der == 0.03125

        with pytest.raises(ValueError):
            np.sqrt(Node(-1))

    def test_logistic(self):
        """Test for standard logistic operator"""
        x1 = Node(5)
        x2 = x1.logistic()
        assert isinstance(x2, Node)

        assert np.isclose(x2.val, 1 / (1 + np.e ** (-5)))
        assert np.isclose(x2.der, np.e ** (-5) / (np.e ** (-5) + 1) ** 2)

    def test_log(self):
        """Test for overloading numpy logarithm operator"""
        x1 = Node(np.e)
        x2 = np.log(x1)
        x3 = x1.log(2)
        assert np.isclose(x2.val, 1.0) and np.isclose(x2.der, 1.0/np.e)
        assert np.isclose(x3.val, 1.0 / np.log(2)) and np.isclose(x3.der, 1 / (np.e * np.log(2)))

        x3 = Node(np.e, np.array([1,0]))
        x4 = np.log(x3)
        assert np.allclose(x4.val, 1.0) and np.array_equal(x4.der, np.array([1.0/np.e,0]))

        with pytest.raises(ValueError):
            np.log(Node(0))

    def test_pow(self):
        """Test for overloading power operator"""
        x1 = Node(3)
        x2 = x1 ** 2.0
        x3 = x1 ** 3
        x4 = Node(2)
        x5 = x1 ** x4

        assert isinstance(x2, Node)
        assert isinstance(x3, Node)
        assert isinstance(x4, Node)

        assert x2.val == 9.0 and x2.der == 6.0
        assert x3.val == 27 and x3.der == 27
        assert x5.val == 9 and np.isclose(x5.der, 6 + np.log(3)*9) 

        with pytest.raises(TypeError):
            x1 ** '1'

        with pytest.raises(TypeError):
            '1' ** x1

    def test_exp(self):
        """Test for overloading numpy exponential operator"""
        x1 = Node(1)
        x2 = np.exp(x1)
        assert x2.val == np.e and x2.der == np.e

        x3 = Node(1, np.array([1,0]))
        x4 = np.exp(x3)
        assert x4.val == np.e and np.array_equal(x4.der, np.array([np.e,0]))

    def test_sin(self):
        """Test for overloading numpy sine operator"""
        x1 = Node(5)
        x2 = np.sin(x1)
        x3 = np.sin(x2)
        assert isinstance(x2, Node)
        assert isinstance(x3, Node)

        assert x2.parent[0].val == 5 and x2.parent[0].der == 1
        assert x3.parent[0].parent[0].val == 5 and x3.parent[0].parent[0].der == 1
        assert x2.val == np.sin(5) and x2.der == np.cos(5)
        assert x3.val == np.sin(np.sin(5)) and x3.der == np.cos(np.sin(5)) * np.cos(5)

    def test_cos(self):
        """Test for overloading numpy cosine operator"""
        x1 = Node(np.pi/2)
        x2 = np.cos(x1)
        assert np.isclose(x2.val, 0) and x2.der == -1

        x3 = Node(0, np.array([1,0]))
        x4 = np.cos(x3)
        assert x4.val == 1 and np.array_equal(x4.der, np.array([0,0]))
        
    def test_tan(self):
        """Test for overloading numpy tangent operator"""
        x1 = Node(5)
        x2 = np.tan(x1)
        x3 = np.tan(x2)
        assert isinstance(x2, Node)
        assert isinstance(x3, Node)

        assert x2.parent[0].val == 5 and x2.parent[0].der == 1
        assert x3.parent[0].parent[0].val == 5 and x3.parent[0].parent[0].der == 1
        assert x2.val == np.tan(5) and x2.der == 1/(np.cos(5))**2
        assert x3.val == np.tan(np.tan(5))
        assert np.isclose(x3.der, 1/(np.cos(np.tan(5)))**2 * 1/(np.cos(5))**2)

        with pytest.raises(ValueError):
            np.tan(Node(np.pi / 2))

    def test_arcsin(self):
        """Test for overloading numpy arcsine operator"""
        x1 = Node(0.5)
        x2 = np.arcsin(x1)
        assert isinstance(x2, Node)
        assert np.isclose(x2.val, np.arcsin(0.5))
        assert np.isclose(x2.der, 1 / np.sqrt(1 - 0.5 ** 2))

        with pytest.raises(ValueError):
            np.arcsin(Node(-2))

    def test_arccos(self):
        """Test for overloading numpy arccosine operator"""
        x1 = Node(0.5)
        x2 = np.arccos(x1)
        assert isinstance(x2, Node)
        assert np.isclose(x2.val, np.arccos(0.5))
        assert np.isclose(x2.der, -1 / np.sqrt(1 - 0.5 ** 2))

        with pytest.raises(ValueError):
            np.arccos(Node(-2))

    def test_arctan(self):
        """Test for overloading numpy arctangent operator"""
        x1 = Node(0.5)
        x2 = np.arctan(x1)
        assert isinstance(x2, Node)
        assert np.isclose(x2.val, np.arctan(0.5))
        print(x2.val, x2.der)
        assert np.isclose(x2.der, 1 / (1 + 0.5 ** 2))
        
    def test_sinh(self):
        """Test for overloading numpy hyperbolic sine operator"""
        x1 = Node(0.5)
        x2 = np.sinh(x1)
        assert isinstance(x2, Node)
        assert np.isclose(x2.val, np.sinh(0.5))
        assert np.isclose(x2.der, np.cosh(0.5))

    def test_cosh(self):
        """Test for overloading numpy hyperbolic cosine operator"""
        x1 = Node(0.5)
        x2 = np.cosh(x1)
        assert isinstance(x2, Node)
        assert np.isclose(x2.val, np.cosh(0.5))
        assert np.isclose(x2.der, np.sinh(0.5))
        
    def test_tanh(self):
        """Test for overloading numpy hyperbolic tangent operator"""
        x1 = Node(0.5)
        x2 = np.tanh(x1)
        assert isinstance(x2, Node)
        assert np.isclose(x2.val, np.tanh(0.5))
        assert np.isclose(x2.der, 1 / np.cosh(0.5) ** 2)
    
    def test_radd(self): 
        """Test for overloading reflective addition operator"""
        x1 = Node(5)
        x2 = 3
        x3 = 2.5
        x21a = x2 + x1
        x31a = x3 + x1

        assert isinstance(x21a, Node)
        assert isinstance(x31a, Node)
        assert x21a.val == 8 and x21a.der == 1
        assert x31a.val == 7.5 and x31a.der == 1

    def test_rsub(self):
        """Test for overloading reflective subtraction operator"""
        x1 = Node(5)
        x2 = 3
        x3 = x1 - x2
        x4 = x2 - x1
        assert x3.val == -x4.val and x3.der == -x4.der

        x5 = Node(5, np.array([1,0]))
        x6 = 3
        x7 = x5 - x6
        x8 = x6 - x5
        assert x7.val == -x8.val and np.array_equal(x7.der, -x8.der)
        
    def test_rmul(self):
        """Test for overloading reflective multiplication operator"""
        x1 = Node(5)
        x2 = 3
        x3 = 2.5
        x21m = x2 * x1
        x31m = x3 * x1
        
        assert isinstance(x21m, Node)
        assert isinstance(x31m, Node)
        assert x21m.val == 15 and x21m.der == 3
        assert x31m.val == 12.5 and x31m.der == 2.5

    def test_rtruediv(self):
        """Test for overloading reflective true division operator"""
        x1 = Node(5)
        x2 = 2.5
        x3 = x2 / x1
        assert x3.val == 0.5 and x3.der == -0.1

        x4 = Node(5, np.array([1,0]))
        x5 = 2.5
        x6 = x5 / x4
        assert x6.val == 0.5 and np.array_equal(x6.der, np.array([-0.1,0]))

        with pytest.raises(ZeroDivisionError):
            1 / Node(0)
    
    def test_rpow(self):
        """Test for overloading reflective power operator"""
        x1 = Node(5)
        x2 = 2.5
        x3 = x2 ** x1
        assert np.isclose(x3.val, 2.5 ** 5)
        assert np.isclose(x3.der, np.log(2.5) * 2.5 ** 5)

    def test_comparison(self):
        """Test for overloading comparison operators"""
        x1 = Node(3)
        x2 = Node(5)
        x3 = 4
        x4 = 3
        x5 = Node(5)
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