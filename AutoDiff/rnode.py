import numpy as np

class RNode:
    """This is a class that implements the Reverse Nodes for reverse mode calculation of
    automatic differentiation and dunder methods to overload built-in operators including 
    negation, addition, subtraction, multiplication, true division, and power. The class 
    also overloads numpy operators including square root, exponential, logarithm, sine, 
    cosine, and tangent. Reflective operations are also included.

    :param value: The value of the RNode, it can be an interger or float or a 1D numpy array for 1D problem, or a multi-dimensional numpy array for multi-dimensional problem
    :type value: integer or float or numpy array

    :ivar val: The value of the RNode, set using parameter value
    :vartype val: integer or float or numpy array
    
    :ivar der: The parent of the RNode, defaults to None. It can be an interger or float or a 1D numpy array for 1D problem, or a multi-dimensional numpy array for multi-dimensional problem
    :vartype der: integer or float or numpy array
    
    :ivar parent: A list of parent RNodes of the current RNode
    :vartype parent: list

    >>> x1 = RNode(5)
    >>> x1.val
    5
    >>> x1.der
    >>> x1.parent
    []
    """

    _supported_types = (int, float)

    def __init__(self, value):
        self.val = value
        self.der = None
        self.parent = []

    def __str__(self):
        """Print useful information for users.

        :return: A string containing useful information of the RNode object.
        :rtype: string

        >>> x1 = RNode(5)
        >>> print(x1)
        RNode: val=5, with 0 parent(s).
        """
        return f'RNode: val={self.val}, with {len(self.parent)} parent(s).'

    def __repr__(self):
        """Print useful information for developers.

        :return: A string containing useful information of the RNode object.
        :rtype: string

        >>> x1 = RNode(5)
        >>> x1
        An RNode object with value of 5, and 0 parent(s) with derivatives and locations in [].
        """
        return f'An RNode object with value of {self.val}, and {len(self.parent)} parent(s) with derivatives and locations in {self.parent}.'
    
    def clear(self):
        """Clears self's and all self's parent's derivative field.

        >>> x1 = RNode(5)
        >>> x2 = x1 + 1
        >>> x1.der = 3
        >>> x2.der = 5
        >>> x1.der
        3
        >>> x2.der
        5
        >>> x1.clear()
        >>> x1.der
        >>> x2.der
        """
        self.der = None
        for _, p in self.parent:
            p.clear()

    def grad_vec(self, output_depend):
        """Helper function for Reverse AD, produces all gradient for its parents 
        and any parents defined in the intermediate step. See usage in Reverse class.

        :param output_depend: A ordered list of output RNode that each parent of var points to
        :type output_depend: list of RNode objects

        :return: An ordered list of derivatives of each parent of self
        :rtype: list of integers or floats

        >>> x1 = RNode(5)
        >>> x2 = RNode(3)
        >>> f1 = x1 * x2 + x1 ** x2
        >>> f2 = x2 - np.log(x2)
        >>> output_d=[]
        >>> x1.grad_vec(output_d)
        [3.0, 75.0]
        >>> output_d
        [An RNode object with value of 140, and 0 parent(s) with derivatives and locations in []., An RNode object with value of 140, and 0 parent(s) with derivatives and locations in [].]
        >>> output_d2 = []
        >>> x2.grad_vec(output_d2)
        [5.0, 201.17973905426254, -0.3333333333333333, 1.0]
        >>> output_d2 
        [An RNode object with value of 140, and 0 parent(s) with derivatives and locations in []., An RNode object with value of 140, and 0 parent(s) with derivatives and locations in []., An RNode object with value of 1.9013877113318902, and 0 parent(s) with derivatives and locations in []., An RNode object with value of 1.9013877113318902, and 0 parent(s) with derivatives and locations in [].]
        """
        gradient = []
        for i in range(len(self.parent)):
            # for all parents, calculate the gradient
            gradient.append(self.parent[i][0] * self.parent[i][1].grad(output_depend))
        return gradient

    def grad(self, output_depend):
        """Helper function for grad_vec(), see usage in grad().

        :param output_depend: A ordered list of output RNode that each parent of var points to
        :type output_depend: list of RNode objects

        :return: An ordered list of derivatives of each parent of self
        :rtype: list of integers or floats
        """
        if self.parent == []:
            # at the last node, i.e. the node for function, assign derivative to be 1.0
            self.der = 1.0
            output_depend.append(self)
        else:
            self.der = sum(p[0] * p[1].grad(output_depend) for p in self.parent)
        return self.der
        
    def __neg__(self):
        """Overloads the built-in negation operator for handling RNodes in the forward pass 
        and returns a new RNode object with value and parent updated.

        :return: An RNode object after negating of the current RNode, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = -x1
        >>> print(x2)
        RNode: val=-5, with 0 parent(s).
        """
        rnode = RNode(-self.val)
        self.parent.append((-1., rnode))
        return rnode

    def __add__(self, other):
        """Overloads the built-in addition operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated. The addition of an RNode object 
        with integers or floats is also supported.

        :param other: The item to be added to the current RNode object.
        :type other: RNode or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or RNode type).
        
        :return: An RNode object after addition, with value and parent updated.
        :rtype: RNode
        
        >>> x1 = RNode(5)
        >>> x2 = RNode(3)
        >>> x3 = x1 + x2
        >>> print(x3)
        RNode: val=8, with 0 parent(s).
        """
        if isinstance(other, RNode):
            rnode = RNode(self.val + other.val)
            self.parent.append((1., rnode))
            other.parent.append((1., rnode))
            return rnode
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for addition")
        else:
            rnode = RNode(self.val + other)
            self.parent.append((1., rnode))
            return rnode

    def __sub__(self, other):
        """Overloads the built-in subtraction operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated. The subtraction of an RNode object 
        with integers or floats is also supported.

        :param other: The item to be subtracted from the current RNode object.
        :type other: RNode or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or RNode type).
        
        :return: An RNode object after subtraction, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = RNode(3)
        >>> x3 = x1 - x2
        >>> print(x3)
        RNode: val=2, with 0 parent(s).
        2
        """
        if isinstance(other, RNode):
            rnode = RNode(self.val - other.val)
            self.parent.append((1., rnode))
            other.parent.append((-1., rnode))
            return rnode
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for subtraction")
        else:
            rnode = RNode(self.val - other)
            self.parent.append((1., rnode))
            return rnode

    def __mul__(self, other):
        """Overloads the built-in multiplication operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated. The multiplication of an RNode object 
        with integers or floats is also supported.

        :param other: The item to multiply the current RNode object by.
        :type other: RNode or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or RNode type).
        
        :return: An RNode object after multiplication, with value and parent updated.
        :rtype: RNode
        
        >>> x1 = RNode(5)
        >>> x2 = RNode(3)
        >>> x3 = x1 * x2
        >>> print(x3)
        RNode: val=15, with 0 parent(s).  
        """
        if isinstance(other, RNode):
            rnode = RNode(self.val * other.val)
            self.parent.append((other.val, rnode))
            other.parent.append((self.val, rnode))
            return rnode
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for multiplication")
        else:
            rnode = RNode(self.val * other)
            self.parent.append((other, rnode))
            return rnode

    def __truediv__(self, other):
        """Overloads the built-in division operator for handling RNodes in the forward pass 
        and returns a new RNode object with value and parent updated. The division of an RNode object 
        with integers or floats is also supported.

        :param other: The item to divide the current RNode object by.
        :type other: RNode or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or RNode type).
        :raises ZeroDivisionError: Division by zero.
        
        :return: An RNode object after subtraction, with value and parent updated.
        :rtype: RNode
        
        >>> x1 = RNode(5)
        >>> x2 = RNode(3)
        >>> x3 = x1 / x2
        >>> print(x3)
        RNode: val=1.6666666666666667, with 0 parent(s).
        """
        if isinstance(other, RNode):
            if other.val == 0:
                raise ZeroDivisionError('Division by zero')
            rnode = RNode(self.val / other.val)
            self.parent.append((1. / other.val, rnode))
            other.parent.append((-self.val / other.val ** 2, rnode))
            return rnode
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for division")
        else:
            if other == 0:
                raise ZeroDivisionError('Division by zero')
            rnode = RNode(self.val / other)
            self.parent.append((1. / other, rnode))
            return rnode

    def sqrt(self):
        """Overloads the numpy square root operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :raises: ValueError: Cannot take square root of negative number.

        :return: An RNode object after taking the square root of the current RNode, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = np.sqrt(x1)
        >>> print(x2)
        RNode: val=2.23606797749979, with 0 parent(s).
        """
        if self.val < 0:
            raise ValueError('Cannot take square root of negative number.')
        rnode = RNode(np.sqrt(self.val))
        self.parent.append((1/2 * self.val ** (-1/2), rnode))
        return rnode

    def logistic(self):
        """Implements the standard logistic operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :return: An RNode object after taking the exponential of the current RNode, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = x1.logistic()
        >>> print(x2)
        RNode: val=0.9933071490757153, with 0 parent(s).
        """
        rnode = RNode(1 / (1 + np.exp(-self.val)))
        self.parent.append((np.exp(self.val) / ((np.exp(self.val) + 1) ** 2), rnode))
        return rnode

    def log(self, base = np.e):
        """Overloads the numpy logarithm operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :raises: ValueError: Cannot take logarithm of non positive number.

        :return: An RNode object after taking the logarithm of the current RNode, (default is natural logarithm, 
        but user may also pass in a custom base) with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = np.log(x1)
        >>> print(x2)
        RNode: val=1.6094379124341003, with 0 parent(s).
        >>> x3 = x1.log(base=3)
        >>> print(x3)
        RNode: val=1.4649735207179269, with 0 parent(s).
        """
        if self.val <= 0:
            raise ValueError('Cannot take logarithm of negative number.')
        rnode = RNode(np.log(self.val) / np.log(base))
        self.parent.append((1. / (self.val * np.log(base)), rnode))
        return rnode

    def __pow__(self, other):
        """Overloads the built-in power operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated. Raising the RNode object 
        to the power of some integer or float is also supported.

        :param other: The item to raise the power of the current RNode object.
        :type other: RNode or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or RNode type).
        
        :return: An RNode object after take the power, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = RNode(3)
        >>> x3 = x1 ** x2
        >>> print(x3)
        RNode: val=125, with 0 parent(s).
        """
        if isinstance(other, RNode):
            rnode = RNode(self.val ** other.val)
            self.parent.append((other.val * self.val ** (other.val - 1.), rnode))
            other.parent.append((np.log(self.val) * self.val ** other.val, rnode))
            return rnode
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for power")
        else:
            rnode = RNode(self.val ** other)
            self.parent.append((other * self.val ** (other - 1.), rnode))
            return rnode

    def exp(self):
        """Overloads the numpy exponential operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :return: An RNode object after taking the exponential of the current RNode, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = np.exp(x1)
        >>> print(x2)
        RNode: val=148.4131591025766, with 0 parent(s).
        """
        rnode = RNode(np.exp(self.val))
        self.parent.append((np.exp(self.val), rnode))
        return rnode

    def sin(self):
        """Overloads the numpy sine operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :return: An RNode object after taking the sine of the current RNode, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = np.sin(x1)
        >>> print(x2)
        RNode: val=-0.9589242746631385, with 0 parent(s).
        """
        rnode = RNode(np.sin(self.val))
        self.parent.append((np.cos(self.val), rnode))
        return rnode

    def cos(self):
        """Overloads the numpy cosine operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :return: An RNode object after taking the cosine of the current RNode, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = np.cos(x1)
        >>> print(x2)
        RNode: val=0.28366218546322625, with 0 parent(s).
        """
        rnode = RNode(np.cos(self.val))
        self.parent.append((-np.sin(self.val), rnode))
        return rnode

    def tan(self):
        """Overloads the numpy tangent operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :raises ValueError: Cannot take tangent of pi/2 + n * pi, with n being some integer

        :return: An RNode object after taking the tangent of the current RNode, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = np.tan(x1)
        >>> print(x2)
        RNode: val=-3.380515006246585, with 0 parent(s).  
        """
        if np.isclose((self.val - np.pi/2) / np.pi, 0):
            raise ValueError('Cannot take tangent of pi/2 + n * pi, with n being some integer')
        rnode = RNode(np.tan(self.val))
        self.parent.append((1 / np.cos(self.val) ** 2, rnode))
        return rnode

    def arcsin(self):
        """Overloads the numpy arcsine operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :raises ValueError: The value of self is not in the function domain.

        :return: An RNode object after taking the arcsine of the current RNode, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(0.5)
        >>> x2 = np.arcsin(x1)
        >>> print(x2)
        RNode: val=0.5235987755982988, with 0 parent(s).
        """
        if self.val < -1 or self.val > 1:
            raise ValueError(f'The value `{self.val}` is not in the domain')
        rnode = RNode(np.arcsin(self.val))
        self.parent.append((1 / np.sqrt(1 - self.val ** 2), rnode))
        return rnode

    def arccos(self):
        """Overloads the numpy arccosine operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :raises ValueError: The value of self is not in the function domain.

        :return: An RNode object after taking the arccosine of the current RNode, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(0.5)
        >>> x2 = np.arccos(x1)
        >>> print(x2)
        RNode: val=1.0471975511965976, with 0 parent(s).
        """
        if self.val < -1 or self.val > 1:
            raise ValueError(f'The value `{self.val}` is not in the domain')
        rnode = RNode(np.arccos(self.val))
        self.parent.append((-1 / np.sqrt(1 - self.val ** 2), rnode))
        return rnode

    def arctan(self):
        """Overloads the numpy arctangent operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :raises ValueError: The value of self is not in the function domain.

        :return: An RNode object after taking the arctangent of the current RNode, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = np.arctan(x1)
        >>> print(x2)
        RNode: val=1.373400766945016, with 0 parent(s).
        """
        rnode = RNode(np.arctan(self.val))
        self.parent.append((1 / (1 + self.val ** 2), rnode))
        return rnode    

    def sinh(self):
        """Overloads the numpy hyperbolic sine operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :return: An RNode object after taking the hyperbolic sine of the current RNode, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = np.sinh(x1)
        >>> print(x2)
        RNode: val=74.20321057778875, with 0 parent(s).
        """
        rnode = RNode(np.sinh(self.val))
        self.parent.append((np.cosh(self.val), rnode))
        return rnode

    def cosh(self):
        """Overloads the numpy hyperbolic cosine operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :return: An RNode object after taking the hyperbolic cosine of the current RNode, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = np.cosh(x1)
        >>> print(x2)
        RNode: val=74.20994852478785, with 0 parent(s).
        """
        rnode = RNode(np.cosh(self.val))
        self.parent.append((np.sinh(self.val), rnode))
        return rnode

    def tanh(self):
        """Overloads the numpy hyperbolic cosine operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :return: An RNode object after taking the hyperbolic tangent of the current RNode, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = np.tanh(x1)
        >>> print(x2)
        RNode: val=0.9999092042625951, with 0 parent(s).
        """
        rnode = RNode(np.tanh(self.val))
        self.parent.append((1 / np.cosh(self.val) ** 2, rnode))
        return rnode

    def __radd__(self, other):
        """Overloads the built-in reflective addition operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :param other: The integer or float to be added to the current RNode object.
        :type other: integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer or float).
        
        :return: An RNode object after reflective addition, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = 3 + x1
        >>> print(x2)
        RNode: val=8, with 0 parent(s).
        """
        return self.__add__(other)

    def __rsub__(self, other):
        """Overloads the built-in reflective subtraction operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :param other: The integer or float to be subtracted by the current RNode object.
        :type other: integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer or float).
        
        :return: An RNode object after reflective subtraction, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = 3 - x1
        >>> print(x2)
        RNode: val=-2, with 0 parent(s).
        """
        if not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for reflective subtraction")
        else:
            rnode = RNode(other - self.val)
            self.parent.append((-1., rnode))
            return rnode

    def __rmul__(self, other):
        """Overloads the built-in reflective multiplication operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :param other: The integer or float to multiply the current RNode object by.
        :type other: integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer or float).
        
        :return: An RNode object after reflective multiplication, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = 3 * x1
        >>> print(x2)
        RNode: val=15, with 0 parent(s).
        """
        return self.__mul__(other)

    def __rtruediv__(self, other):
        """Overloads the built-in reflective true division operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated.

        :param other: The integer or float to divide the current RNode object by.
        :type other: integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer or float).
        :raises ValueError: Division by zero.

        :return: An RNode object after reflective true division, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = 3 / x1
        >>> print(x2)
        RNode: val=0.6, with 0 parent(s).
        """
        if not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for reflective division")
        else:
            if self.val == 0:
                raise ZeroDivisionError('Division by zero')
            rnode = RNode(other / self.val)
            self.parent.append((-other / self.val ** 2, rnode))
            return rnode

    def __rpow__(self, other):
        """Overloads the built-in reflective power operator for handling RNodes in the forward pass
        and returns a new RNode object with value and parent updated. Raising the RNode object 
        to the power of some integer or float is also supported.

        :param other: The item to raise the reflective power of the current RNode object.
        :type other: RNode or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or RNode type).
        
        :return: An RNode object after take the power, with value and parent updated.
        :rtype: RNode

        >>> x1 = RNode(5)
        >>> x2 = 3 ** x1
        >>> print(x2)
        RNode: val=243, with 0 parent(s).
        """
        if not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for power")
        else:
            rnode = RNode(other ** self.val)
            self.parent.append((np.log(other) * other ** self.val, rnode))
            return rnode

    def __lt__(self, other):
        """Overloads the built-in less than operator for comparisons between RNodes. 
        Comparing the RNode object with some integer or float is also supported.

        :param other: The item to be compared with the current RNode object.
        :type other: RNode or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or RNode type).
        
        :return: A Boolean comparing the current RNode object with other
        :rtype: Boolean

        >>> x1 = RNode(5)
        >>> x2 = RNode(3)
        >>> x1 < x2
        False
        """
        if isinstance(other, RNode):
            return self.val < other.val
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for comparison <")
        else:
            return self.val < other

    def __gt__(self, other):
        """Overloads the built-in greater than operator for comparisons between RNodes. 
        Comparing the RNode object with some integer or float is also supported.

        :param other: The item to be compared with the current RNode object.
        :type other: RNode or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or RNode type).
        
        :return: A Boolean comparing the current RNode object with other
        :rtype: Boolean

        >>> x1 = RNode(5)
        >>> x2 = RNode(3)
        >>> x1 > x2
        True
        """
        if isinstance(other, RNode):
            return self.val > other.val
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for comparison >")
        else:
            return self.val > other

    def __le__(self, other):
        """Overloads the built-in less than or equal to operator for comparisons between RNodes. 
        Comparing the RNode object with some integer or float is also supported.

        :param other: The item to be compared with the current RNode object.
        :type other: RNode or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or RNode type).
        
        :return: A Boolean comparing the current RNode object with other
        :rtype: Boolean

        >>> x1 = RNode(5)
        >>> x2 = RNode(3)
        >>> x1 <= x2
        False
        """
        if isinstance(other, RNode):
            return self.val <= other.val
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for comparison <=")
        else:
            return self.val <= other

    def __ge__(self, other):
        """Overloads the built-in greater than or equal to operator for comparisons between RNodes. 
        Comparing the RNode object with some integer or float is also supported.

        :param other: The item to be compared with the current RNode object.
        :type other: RNode or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or RNode type).
        
        :return: A Boolean comparing the current RNode object with other
        :rtype: Boolean

        >>> x1 = RNode(5)
        >>> x2 = RNode(3)
        >>> x1 >= x2
        True
        """
        if isinstance(other, RNode):
            return self.val >= other.val
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for comparison >=")
        else:
            return self.val >= other

    def __eq__(self, other):
        """Overloads the built-in equal operator for comparisons between RNodes. Comparing the RNode object 
        with some integer or float is also supported.

        :param other: The item to be compared with the current RNode object.
        :type other: RNode or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or RNode type).
        
        :return: A Boolean comparing the current RNode object with other
        :rtype: Boolean

        >>> x1 = RNode(5)
        >>> x2 = RNode(3)
        >>> x1 = x2
        False
        """
        if isinstance(other, RNode):
            return self.val == other.val
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for comparison =")
        else:
            return self.val == other

    def __ne__(self, other):
        """Overloads the built-in not equal operator for comparisons between RNodes. Comparing the RNode object 
        with some integer or float is also supported.

        :param other: The item to be compared with the current RNode object.
        :type other: RNode or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or RNode type).
        
        :return: A Boolean comparing the current RNode object with other
        :rtype: Boolean
        
        >>> x1 = RNode(5)
        >>> x2 = RNode(3)
        >>> x1 != x2
        True
        """
        if isinstance(other, RNode):
            return self.val != other.val
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for comparison =")
        else:
            return self.val != other