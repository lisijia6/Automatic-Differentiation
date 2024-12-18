import numpy as np

class Node:
    """This is a class that implements the Dual numbers and dunder methods to overload 
    built-in operators including negation, addition, subtraction, multiplication, true
    division, and power. The class also overloads numpy operators including square root,
    exponential, logarithm, sine, cosine, and tangent. Reflective operations are also 
    included.

    :param value: The value of the Node, it can be an interger or float or a 1D numpy array for 1D problem, or a multi-dimensional numpy array for multi-dimensional problem
    :type value: integer or float or numpy array

    :param derivative: The derivative of the Node, defaults to 1. It can be an interger or float or a 1D numpy array for 1D problem, or a multi-dimensional numpy array for multi-dimensional problem
    :type derivative: integer or float or numpy array

    :ivar val: The value of the Node, it can be an interger or float or a 1D numpy array for 1D problem, or a multi-dimensional numpy array for multi-dimensional problem
    :vartype val: integer or float or numpy array

    :ivar der: The derivative of the Node, defaults to 1. It can be an interger or float or a 1D numpy array for 1D problem, or a multi-dimensional numpy array for multi-dimensional problem
    :vartype der: integer or float or numpy array

    :ivar parent: A list of parent Nodes of the current Node
    :vartype parent: list

    :ivar op: A list of strings representing operations
    :vartype op: list

    :ivar v_index: An integer used to track variable index in visualization
    :vartype v_index: integer

    >>> x1 = Node(5)
    >>> x1.val
    5
    >>> x1.der
    1
    >>> x1.parent
    []
    >>> x1.op
    []
    >>> x1.v_index
    'v1'
    """
    
    _supported_types = (int, float)
    v_index = 0

    def __init__(self, value, derivative = 1): 
        self.val = value
        self.der = derivative
        self.parent = []
        self.op = []
        type(self).v_index += 1
        self.v_index = f'v{type(self).v_index}'

    def __str__(self):
        """Print useful information for users.

        :return: a string containing useful information of the Node object.
        :rtype: string

        >>> x1 = Node(5)
        >>> print(x1)
        Node: vindex=v1, val=5, der=1, parent=[], and op=[].
        """
        return f'Node: vindex={self.v_index}, val={self.val}, der={self.der}, parent={self.parent}, and op={self.op}.'

    def __repr__(self):
        """Print useful information for developers.

        :return: a string containing useful information of the Node object.
        :rtype: string

        >>> x1 = Node(5)
        >>> x1
        A Node object with index of v1, value of 5, derivative of 1, parent of [], and operator of [].
        """
        return f'A Node object with index of {self.v_index}, value of {self.val}, derivative of {self.der}, parent of {self.parent}, and operator of {self.op}.'

    def update_node(self, parent, op):
        """Update a list of parent Nodes of the current Node with their operations.

        :param parent: A list of parent Nodes of the current Node
        :type parent: list
        :param op: A list of strings representing operations
        :type op: list
        
        :return: the current Node with parents and operations updated
        :rtype: Node

        >>> x1 = Node(5)
        >>> parent = Node(3)
        >>> x1.update_node([parent], ['+'])
        A Node object with index of v2, value of 5, derivative of 1, parent of [A Node object with index of v3, value of 3, derivative of 1, parent of [], and operator of [].], and operator of ['+'].
        >>> x1.parent
        [A Node object with index of v3, value of 3, derivative of 1, parent of [], and operator of [].]
        >>> x1.op
        ['+']
        """
        self.parent = parent
        self.op = op
        return self

    def __neg__(self):
        """Overloads the built-in negation operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated.

        :return: A Node object after negating of the current Node, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(5)
        >>> x2 = -x1
        >>> print(x2)
        Node: vindex=v26, val=-5, der=-1, parent=[A Node object with index of v25, value of 5, derivative of 1, parent of [], and operator of [].], and op=['-1*'].
        """
        value = -self.val
        derivative = -1. * self.der
        return Node(value, derivative).update_node([self], ['-1*'])

    def __add__(self, other):
        """Overloads the built-in addition operator for handling Dual numbers in the Node class
        and returns a new Node object with value and derivative updated. The addition of a Node object 
        with integers or floats is also supported.

        :param other: The item to be added to the current Node object.
        :type other: Node or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or Node type).
        
        :return: A Node object after addition, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(5)
        >>> x2 = Node(3)
        >>> x3 = x1 + x2
        >>> print(x3)
        Node: vindex=v29, val=8, der=2, parent=[A Node object with index of v27, value of 5, derivative of 1, parent of [], and operator of []., A Node object with index of v28, value of 3, derivative of 1, parent of [], and operator of [].], and op=['+'].
        """
        if isinstance(other, Node):
            value = self.val + other.val
            derivative = self.der + other.der
            return Node(value, derivative).update_node([self, other], ['+'])
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for addition")
        else:
            value = self.val + other
            derivative = self.der
            return Node(value, derivative).update_node([self], ['+', other])

    def __sub__(self, other):
        """Overloads the built-in subtraction operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated. The subtraction of a Node object 
        with integers or floats is also supported.

        :param other: The item to be subtracted from the current Node object.
        :type other: Node or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or Node type).
        
        :return: A Node object after subtraction, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(5)
        >>> x2 = Node(3.0)
        >>> x3 = x1 - x2
        >>> print(x3)
        Node: vindex=v32, val=2.0, der=0, parent=[A Node object with index of v30, value of 5, derivative of 1, parent of [], and operator of []., A Node object with index of v31, value of 3.0, derivative of 1, parent of [], and operator of [].], and op=['-'].
        """
        if isinstance(other, Node):
            value = self.val - other.val
            derivative = self.der - other.der
            return Node(value, derivative).update_node([self, other], ['-'])
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for subtraction")
        else:
            value = self.val - other
            derivative = self.der
            return Node(value, derivative).update_node([self], ['-', other])

    def __mul__(self, other):
        """Overloads the built-in multiplication operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated. The multiplication of a Node object 
        with integers or floats is also supported.

        :param other: The item to multiply the current Node object by.
        :type other: Node or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or Node type).
       
        :return: A Node object after multiplication, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(5)
        >>> x2 = Node(3)
        >>> x3 = x1 * x2
        >>> print(x3)
        Node: vindex=v35, val=15, der=8, parent=[A Node object with index of v33, value of 5, derivative of 1, parent of [], and operator of []., A Node object with index of v34, value of 3, derivative of 1, parent of [], and operator of [].], and op=['*'].
        """
        if isinstance(other, Node):
            value = self.val * other.val
            derivative = self.der * other.val + other.der * self.val
            return Node(value, derivative).update_node([self, other], ['*'])
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for multiplication")
        else:
            value = self.val * other
            derivative = self.der * other
            return Node(value, derivative).update_node([self], ['*', other])
    
    def __truediv__(self, other):
        """Overloads the built-in division operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated. The division of a Node object 
        with integers or floats is also supported.

        :param other: The item to divide the current Node object by.
        :type other: Node or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or Node type).
        :raises ZeroDivisionError: Division by zero.
        
        :return: A Node object after subtraction, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(5)
        >>> x2 = Node(2.5)
        >>> x3 = x1 / x2
        >>> print(x3)
        Node: vindex=v38, val=2.0, der=-0.4, parent=[A Node object with index of v36, value of 5, derivative of 1, parent of [], and operator of []., A Node object with index of v37, value of 2.5, derivative of 1, parent of [], and operator of [].], and op=['/'].
        """
        if isinstance(other, Node):
            if other.val == 0:
                raise ZeroDivisionError('Division by zero')
            value = self.val / other.val
            derivative = (self.der * other.val - other.der * self.val) / other.val**2
            return Node(value, derivative).update_node([self, other], ['/'])
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for division")
        else:
            if other == 0:
                raise ZeroDivisionError('Division by zero')
            value = self.val / other
            derivative = self.der / other
            return Node(value, derivative).update_node([self], ['/', other])

    def sqrt(self):
        """Overloads the numpy square root operator for handling Dual numbers in the Node class
        and returns a new Node object with value and derivative updated.

        :return: A Node object after taking the square root of the current Node, with value and derivative updated.
        :raises: ValueError: Cannot take square root of negative number.
        :rtype: Node

        >>> x1 = Node(16)
        >>> x2 = np.sqrt(x1)
        >>> print(x2)
        Node: vindex=v40, val=4.0, der=0.125, parent=[A Node object with index of v39, value of 16, derivative of 1, parent of [], and operator of [].], and op=['sqrt()'].
        """
        if self.val < 0:
            raise ValueError('Cannot take square root of negative number.')
        value = np.sqrt(self.val)
        derivative = 0.5/np.sqrt(self.val) * self.der
        return Node(value, derivative).update_node([self], ['sqrt()'])
    
    def logistic(self):
        """Implements the standard logistic operator for handling Nodes in the forward pass
        and returns a new Node object with value and parent updated.

        :return: An Node object after taking the standard logistic of the current Node, with value and parent updated.
        :rtype: Node

        >>> x1 = Node(5)
        >>> x2 = x1.logistic()
        >>> print(x2)
        Node: vindex=v42, val=0.9933071490757153, der=0.006648056670790156, parent=[A Node object with index of v41, value of 5, derivative of 1, parent of [], and operator of [].], and op=['logistic()'].
        """
        value = 1 / (1 + np.exp(-self.val))
        derivative = np.exp(-self.val) / ((np.exp(-self.val) + 1) ** 2) * self.der
        return Node(value, derivative).update_node([self], ['logistic()'])

    def log(self, base=np.e):
        """Overloads the numpy logarithm operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated.

        :return: A Node object after taking the logarithm of the current Node (default is natural logarithm, but user may also pass in a custom base), with value and derivative updated.
        :raises: ValueError: Cannot take logarithm of non positive number.
        :rtype: Node

        >>> x1 = Node(np.e)
        >>> x2 = np.log(x1)
        >>> print(x2)
        Node: vindex=v44, val=1.0, der=0.36787944117144233, parent=[A Node object with index of v43, value of 2.718281828459045, derivative of 1, parent of [], and operator of [].], and op=['log()'].
        >>> x3 = x1.log(2)
        >>> print(x3)
        Node: vindex=v45, val=1.4426950408889634, der=0.530737845423043, parent=[A Node object with index of v43, value of 2.718281828459045, derivative of 1, parent of [], and operator of [].], and op=['log2()'].
        """
        if self.val <= 0:
            raise ValueError('Cannot take the log of a negative number.')
        if base == np.e:
            value = np.log(self.val)
            derivative = 1 / self.val * self.der
            return Node(value, derivative).update_node([self], ['log()'])
        else:
            value = np.log(self.val) / np.log(base)
            derivative = 1 / (self.val * np.log(base)) * self.der
            return Node(value, derivative).update_node([self], [f'log{base}()'])

    def __pow__(self, other):
        """Overloads the built-in power operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated. Raising the Node object 
        to the power of some integer or float is also supported.

        :param other: The item to raise the power of the current Node object.
        :type other: Node or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or Node type).
        
        :return: A Node object after take the power, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(3)
        >>> x2 = x1 ** 2.0
        >>> print(x2)
        Node: vindex=v47, val=9.0, der=6.0, parent=[A Node object with index of v46, value of 3, derivative of 1, parent of [], and operator of [].], and op=['pow', 2.0].
        """
        if isinstance(other, Node):
            print('here!!')
            value = self.val ** other.val
            derivative = other.val * (self.val ** (other.val - 1)) * self.der + np.log(self.val) * (self.val ** other.val) * other.der
            return Node(value, derivative).update_node([self, other], ['pow'])
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for power")
        else:
            value = self.val ** other
            derivative = other * (self.val ** (other - 1)) * self.der
            return Node(value, derivative).update_node([self], ['pow', other])

    def exp(self):
        """Overloads the numpy exponential operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated.

        :return: A Node object after taking the exponential of the current Node, with value and derivative updated.
        :rtype: Node
        
        >>> x1 = Node(1)
        >>> x2 = np.exp(x1)
        >>> print(x2)
        Node: vindex=v49, val=2.718281828459045, der=2.718281828459045, parent=[A Node object with index of v48, value of 1, derivative of 1, parent of [], and operator of [].], and op=['exp()'].
        """
        value = np.exp(self.val)
        derivative = np.exp(self.val) * self.der
        return Node(value, derivative).update_node([self], ['exp()'])

    def sin(self):
        """Overloads the numpy sine operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated.

        :return: A Node object after taking the sine of the current Node, with value and derivative updated.
        :rtype: Node
        
        >>> x1 = Node(5)
        >>> x2 = np.sin(x1)
        >>> print(x2)
        Node: vindex=v51, val=-0.9589242746631385, der=0.2836621854632263, parent=[A Node object with index of v50, value of 5, derivative of 1, parent of [], and operator of [].], and op=['sin()'].
        """
        value = np.sin(self.val)
        derivative = np.cos(self.val) * self.der
        return Node(value, derivative).update_node([self], ['sin()'])

    def cos(self):
        """Overloads the numpy cosine operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated.

        :return: A Node object after taking the cosine of the current Node, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(np.pi/2)
        >>> x2 = np.cos(x1)
        >>> print(x2)
        Node: vindex=v53, val=6.123233995736766e-17, der=-1.0, parent=[A Node object with index of v52, value of 1.5707963267948966, derivative of 1, parent of [], and operator of [].], and op=['cos()'].
        """
        value = np.cos(self.val)
        derivative = -np.sin(self.val) * self.der
        return Node(value, derivative).update_node([self], ['cos()'])
        
    def tan(self):
        r"""Overloads the numpy tangent operator for handling Dual numbers in the Node class
        and returns a new Node object with value and derivative updated.

        :raises ValueError: Cannot take tangent of :math:`\pi/2 + n \cdot \pi`, with n being some integer

        :return: A Node object after taking the tangent of the current Node, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(5)
        >>> x2 = np.tan(x1)
        >>> print(x2)
        Node: vindex=v55, val=-3.380515006246585, der=12.427881707458349, parent=[A Node object with index of v54, value of 5, derivative of 1, parent of [], and operator of [].], and op=['tan()'].
        """
        if np.isclose((self.val - np.pi/2) / np.pi, 0):
            raise ValueError('Cannot take tangent of pi/2 + n * pi, with n being some integer')
        value = np.tan(self.val)
        derivative = 1 / (np.cos(self.val)) ** 2 * self.der
        return Node(value, derivative).update_node([self], ['tan()'])
    
    def arcsin(self):
        """Overloads the numpy arcsine operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated.

        :raises ValueError: The value of self is not in the function domain.

        :return: A Node object after taking the arcsine of the current Node, with value and derivative updated.
        :rtype: Node
        
        >>> x1 = Node(0.5)
        >>> x2 = np.arcsin(x1)
        >>> print(x2)
        Node: vindex=v57, val=0.5235987755982988, der=1.1547005383792517, parent=[A Node object with index of v56, value of 0.5, derivative of 1, parent of [], and operator of [].], and op=['arcsin()'].
        """
        if self.val < -1 or self.val > 1:
            raise ValueError(f"The value `{self.val}` is not in the domain")
        value = np.arcsin(self.val)
        derivative = 1. / (np.sqrt(1 - self.val ** 2)) * self.der
        return Node(value, derivative).update_node([self], ['arcsin()'])
    
    def arccos(self):
        """Overloads the numpy arccosine operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated.

        :raises ValueError: The value of self is not in the function domain.

        :return: A Node object after taking the arccosine of the current Node, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(0.5)
        >>> x2 = np.arccos(x1)
        >>> print(x2)
        Node: vindex=v59, val=1.0471975511965976, der=-1.1547005383792517, parent=[A Node object with index of v58, value of 0.5, derivative of 1, parent of [], and operator of [].], and op=['arccos()'].
        """
        if self.val < -1 or self.val > 1:
            raise ValueError(f"The value `{self.val}` is not in the domain")
        value = np.arccos(self.val)
        derivative = -1. / (np.sqrt(1 - self.val**2)) * self.der
        return Node(value, derivative).update_node([self], ['arccos()'])
    
    def arctan(self):
        """Overloads the numpy arctangent operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated.

        :return: A Node object after taking the arctangent of the current Node, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(0.5)
        >>> x2 = np.arctan(x1)
        >>> print(x2)
        Node: vindex=v61, val=0.46364760900080615, der=0.8, parent=[A Node object with index of v60, value of 0.5, derivative of 1, parent of [], and operator of [].], and op=['arctan()'].
        """
        value = np.arctan(self.val)
        derivative = 1. / (1. + self.val ** 2) * self.der
        return Node(value, derivative).update_node([self], ['arctan()'])

    def sinh(self):
        """Overloads the numpy sinh operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated.

        :return: A Node object after taking the sinh of the current Node, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(0.5)
        >>> x2 = np.sinh(x1)
        >>> print(x2)
        Node: vindex=v63, val=0.5210953054937474, der=1.1276259652063807, parent=[A Node object with index of v62, value of 0.5, derivative of 1, parent of [], and operator of [].], and op=['sinh()'].
        """
        value = np.sinh(self.val)
        derivative = np.cosh(self.val) * self.der
        return Node(value, derivative).update_node([self], ['sinh()'])
    
    def cosh(self):
        """Overloads the numpy cosh operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated.

        :return: A Node object after taking the cosh of the current Node, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(0.5)
        >>> x2 = np.cosh(x1)
        >>> print(x2)
        Node: vindex=v65, val=1.1276259652063807, der=0.5210953054937474, parent=[A Node object with index of v64, value of 0.5, derivative of 1, parent of [], and operator of [].], and op=['cosh()'].
        """
        value = np.cosh(self.val)
        derivative = np.sinh(self.val) * self.der
        return Node(value, derivative).update_node([self], ['cosh()'])
    
    def tanh(self):
        """Overloads the numpy tanh operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated.

        :return: A Node object after taking the tanh of the current Node, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(0.5)
        >>> x2 = np.tanh(x1)
        >>> print(x2)
        Node: vindex=v67, val=0.46211715726000974, der=0.7864477329659275, parent=[A Node object with index of v66, value of 0.5, derivative of 1, parent of [], and operator of [].], and op=['tanh()'].
        """
        value = np.tanh(self.val)
        derivative = 1. / (np.cosh(self.val) ** 2) * self.der
        return Node(value, derivative).update_node([self], ['tanh()'])

    def __radd__(self, other):
        """Overloads the built-in reflective addition operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated.

        :param other: The integer or float to be added to the current Node object.
        :type other: integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer or float).

        :return: A Node object after reflective addition, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(5)
        >>> x2 = 3
        >>> x21a = x2 + x1
        >>> print(x21a)
        Node: vindex=v69, val=8, der=1, parent=[A Node object with index of v68, value of 5, derivative of 1, parent of [], and operator of [].], and op=['+', 3].
        """
        return self.__add__(other)

    def __rsub__(self, other):
        """Overloads the built-in reflective subtraction operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated.

        :param other: The integer or float to be subtracted by the current Node object.
        :type other: integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer or float).

        :return: A Node object after reflective subtraction, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(5)
        >>> x2 = 3
        >>> x3 = x1 - x2
        >>> print(x3)
        Node: vindex=v71, val=2, der=1, parent=[A Node object with index of v70, value of 5, derivative of 1, parent of [], and operator of [].], and op=['-', 3].
        """
        return self.__sub__(other).__neg__()
        
    def __rmul__(self, other):
        """Overloads the built-in reflective multiplication operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated.

        :param other: The integer or float to multiply the current Node object by.
        :type other: integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer or float).

        :return: A Node object after reflective multiplication, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(5)
        >>> x2 = 3
        >>> x21m = x2 * x1
        >>> print(x21m)
        Node: vindex=v73, val=15, der=3, parent=[A Node object with index of v72, value of 5, derivative of 1, parent of [], and operator of [].], and op=['*', 3].
        """
        return self.__mul__(other)

    def __rtruediv__(self, other):
        """Overloads the built-in reflective true division operator for handling Dual numbers in the Node class 
        and returns a new Node object with value and derivative updated.

        :param other: The integer or float to divide the current Node object by.
        :type other: integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer or float).

        :return: A Node object after reflective true division, with value and derivative updated.
        :rtype: Node

        >>> x1 = Node(5)
        >>> x2 = 2.5
        >>> x3 = x2 / x1
        >>> print(x3)
        Node: vindex=v75, val=0.5, der=-0.1, parent=[A Node object with index of v74, value of 5, derivative of 1, parent of [], and operator of [].], and op=['r/', 2.5].
        """
        if not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for division")
        else:
            value = other / self.val
            derivative =  - self.der * other / self.val ** 2
            return Node(value, derivative).update_node([self], ['r/', other])
    
    def __rpow__(self, other):
        """Overloads the built-in reflective power operator for handling Nodes in the forward pass
        and returns a new Node object with value and parent updated. Raising the Node object 
        to the power of some integer or float is also supported.

        :param other: The item to raise the reflective power of the current Node object.
        :type other: Node or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or Node type).

        :return: An Node object after take the power, with value and parent updated.
        :rtype: Node

        >>> x1 = Node(5)
        >>> x2 = 2.5
        >>> x3 = x2 ** x1
        >>> print(x3)
        Node: vindex=v77, val=97.65625, der=89.48151678458547, parent=[A Node object with index of v76, value of 5, derivative of 1, parent of [], and operator of [].], and op=['rpow', 2.5].
        """
        if not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for power")
        else:
            value = other ** self.val
            derivative = np.log(other) * other ** self.val * self.der
            return Node(value, derivative).update_node([self], ['rpow', other])

    def __lt__(self, other):
        """Overloads the built-in less than operator for comparisons between Nodes. 
        Comparing the Node object with some integer or float is also supported.

        :param other: The item to be compared with the current Node object.
        :type other: Node or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or Node type).

        :return: A Boolean comparing the current Node object with other
        :rtype: Boolean

        >>> x1 = Node(3)
        >>> x2 = Node(5)
        >>> x1 < x2
        True
        """
        if isinstance(other, Node):
            return self.val < other.val
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for comparison <")
        else:
            return self.val < other

    def __gt__(self, other):
        """Overloads the built-in greater than operator for comparisons between Nodes. 
        Comparing the Node object with some integer or float is also supported.

        :param other: The item to be compared with the current Node object.
        :type other: Node or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or Node type).

        :return: A Boolean comparing the current Node object with other
        :rtype: Boolean

        >>> x1 = Node(3)
        >>> x2 = Node(5)
        >>> x2 > x1
        True
        """
        if isinstance(other, Node):
            return self.val > other.val
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for comparison >")
        else:
            return self.val > other

    def __le__(self, other):
        """Overloads the built-in less than or equal to operator for comparisons between Nodes. 
        Comparing the Node object with some integer or float is also supported.

        :param other: The item to be compared with the current Node object.
        :type other: Node or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or Node type).

        :return: A Boolean comparing the current Node object with other
        :rtype: Boolean

        >>> x1 = Node(3)
        >>> x2 = Node(5)
        >>> x1 <= x2
        True
        """
        if isinstance(other, Node):
            return self.val <= other.val
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for comparison <=")
        else:
            return self.val <= other

    def __ge__(self, other):
        """Overloads the built-in greater than or equal to operator for comparisons between Nodes. 
        Comparing the Node object with some integer or float is also supported.

        :param other: The item to be compared with the current Node object.
        :type other: Node or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or Node type).
        
        :return: A Boolean comparing the current Node object with other
        :rtype: Boolean
        
        >>> x1 = Node(3)
        >>> x2 = Node(5)
        >>> x2 >= x1
        True
        """
        if isinstance(other, Node):
            return self.val >= other.val
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for comparison >=")
        else:
            return self.val >= other

    def __eq__(self, other):
        """Overloads the built-in equal operator for comparisons between Nodes. Comparing the Node object 
        with some integer or float is also supported.

        :param other: The item to be compared with the current Node object.
        :type other: Node or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or Node type).
        
        :return: A Boolean comparing the current Node object with other
        :rtype: Boolean

        >>> x1 = Node(3)
        >>> x2 = Node(5)
        >>> x1 == x2
        False
        """
        if isinstance(other, Node):
            return self.val == other.val
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for comparison =")
        else:
            return self.val == other

    def __ne__(self, other):
        """Overloads the built-in not equal operator for comparisons between Nodes. Comparing the Node object 
        with some integer or float is also supported.

        :param other: The item to be compared with the current Node object.
        :type other: Node or integer or float
        :raises TypeError: The type of other is unsupported (i.e., not integer, float, or Node type).
        
        :return: A Boolean comparing the current Node object with other
        :rtype: Boolean

        >>> x1 = Node(3)
        >>> x2 = Node(5)
        >>> x1 != x2
        True
        """
        if isinstance(other, Node):
            return self.val != other.val
        elif not isinstance(other, self._supported_types):
            raise TypeError(f"Type `{type(other)}` is not supported for comparison !=")
        else:
            return self.val != other