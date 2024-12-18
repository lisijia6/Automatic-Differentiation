import numpy as np
from .node import Node

class Forward:
    r"""
    Initialize forward mode AD. For detailed implementation see :py:meth:`AutoDiff.forward.Forward.grad`.

    :param f: A callable function object, the :math:`f: \mathbb{R}^m \mapsto \mathbb{R}^n` function
    :type f: function object

    :param variables: The input for :code:`f`
    :type variables: integer or float or numpy array

    :ivar val: The output of function ``f(*variables)``
    :vartype val: integer or float or numpy array

    :ivar der: The gradient ``f(*variables)`` wrt. ``variables``
            i.e. :math:`\frac{\partial f(\text{variables})}{\partial \text{variables}}`
    :vartype der: integer or float or numpy array

    :ivar output: :py:meth:`AutoDiff.node.Node`; a single node for scalar function or a list of output nodes for vector function.
    :vartype output: Node class object or list of Node class objects

    >>> from AutoDiff import Forward
    >>> import numpy as np
    >>> # Example 1: univariate scalar function
    >>> x0 = 50
    >>> def f0(x):
    >>>     return 0
    >>> g0 = Forward(f0, x0)
    >>> g0.val
    0
    >>> g0.der
    0.0
    >>> ####
    >>> # Example 2: multivariate scalar function
    >>> x1 = [1, 2]
    >>> def f1(x1, x2):
    >>>     return x1 * x2
    >>> g1 = Forward(f1, *x1)
    >>> g1.val
    2
    >>> g1.der
    array([2., 1.])
    >>> ####
    >>> # Example 3: univariate vector function
    >>> x2 = 50
    >>> def f2(x):
    >>>     return [np.sin(x), np.cos(x)]
    >>> g2 = Forward(f2, x2)
    >>> g2.val
    array([-0.26237485,  0.96496603])
    >>> g2.der
    array([0.96496603, 0.26237485])
    >>> ####
    >>> # Example 4: multivariate vector function
    >>> x3 = [3, 4, 5]
    >>> def f3(x1, x2, x3):
    >>>     x4 = x1 + x2
    >>>     return [np.exp(5 * x3), x2 ** 3 + x4, 2 * np.sqrt(x1) * x4]
    >>> g3 = Forward(f3, *x3)
    >>> g3.val
    array([7.20048993e+10, 7.10000000e+01, 2.42487113e+01])
    >>> g3.der
    array([[0.00000000e+00, 0.00000000e+00, 3.60024497e+11],
        [1.00000000e+00, 4.90000000e+01, 0.00000000e+00],
        [7.50555350e+00, 3.46410162e+00, 0.00000000e+00]])
    """
    def __init__(self, f: callable, *variables):
        self.val, self.der, self.output = self.grad(f, *variables)

    def __str__(self):
        """Print useful information for users.

        :return: A string containing useful information of the Forward object.
        :rtype: string

        >>> x = 50
        >>> def f0(x):
        >>>     return 0
        >>> g0 = Forward(f0, x)
        >>> print(g0)
        Forward: val=0, der=0.0, and output=(Node: vindex=v1, val=0, der=[0.], parent=[], and op=[].).
        """
        return f'Forward: val={self.val}, der={self.der}, and output=({self.output}).'

    def __repr__(self):
        """Print useful information for users.

        :return: A string containing useful information of the Forward object.
        :rtype: string

        >>> x = 50
        >>> def f0(x):
        >>>     return 0
        >>> g0 = Forward(f0, x)
        >>> g0
        A Forward object with value of 0, derivative of 0.0, and output of (Node: vindex=v1, val=0, der=[0.], parent=[], and op=[].).
        """
        return f'A Forward object with value of {self.val}, derivative of {self.der}, and output of ({self.output}).'

    @staticmethod
    def grad(f: callable, *variables):
        r"""
        Evaluate the full Jacobian in forward mode. This is the method that is used internally
        by :py:meth:`AutoDiff.forward.Forward.__init__`.
        For each (scalar or multivariate) function ``f``,
        use :math:`m` passes with different seed vector :math:`\mathbf{e}`,
        where each natural basis :math:`\mathbf{e} \in \mathbb{R}^{m}`, and :math:`m` is the number in ``variables``.

        :param f: A callable function object to perform differentiation on
        :type f: function object
        :param variables: The input for variables of function ``f``
        :type variables: integer or float or numpy array or list of intergers or floats

        :return: function evaluation at variable x, Jacobian, a single output node (scalar function) or a list of output nodes (multivariate) function)
            i.e. :math:`\frac{\partial f(\text{variables})}{\partial \text{variables}}`.
            Stack the gradient rows into the full Jacobian.
        :rtype: Node class object or list of Node class objects        
        """
        # Helper function when Forward class is initialized, see usage in init.
        num_variables = len(variables)
        # initialize the intermediate result index
        Node.v_index = -num_variables
        # Convert variables into Nodes and store in a list
        variables = [
            Node(var, derivative = np.eye(num_variables)[i])
            for i, var in enumerate(variables)
        ]
        # Perform the forward mode
        output = f(*variables)
        if isinstance(output, list): # for vector functions (a list of outputs)
            output = [o if isinstance(o, Node) else Node(o, np.zeros(num_variables)) for o in output]
            values = np.array([o.val for o in output])
            ders = np.stack([o.der if len(o.der) > 1 else o.der[0] for o in output])
            return values, ders, output
        else: # for scalar functions (a single output)
            if not isinstance(output, Node):
                output = Node(output, np.zeros(num_variables))
            return output.val, output.der if len(output.der) > 1 else output.der[0], output