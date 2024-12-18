import numpy as np
from .rnode import RNode

class Reverse:
    r"""
    Initialize Reverse. For detailed implementation see :py:meth:`AutoDiff.reverse.Reverse.grad`.
    The structure is similar to :py:class:`AutoDiff.forward.Forward`.
    
    :param f: Function that is callable to perform differentiation on
    :type f: function object

    :param variables: inputs
    :type variables: integer or floats or numpy array or list of integers or floats

    :ivar val: Output of function ``f(*variables)``
    :vartype val: integer or float or numpy array

    :ivar der: gradient ``f(*variables)`` wrt. ``variables``
            i.e. :math:`\frac{\partial f(\text{variables})}{\partial \text{variables}}`
    :vartype der: integer or float or numpy array

    >>> from AutoDiff import Reverse
    >>> import numpy as np
    >>> # Example 1: scalar univariate function
    >>> x1 = 50
    >>> def f1(x):
    >>>    return 5 * x
    >>> g1 = Reverse(f1, x1)
    >>> g1.val
    250
    >>> g1.der
    5.0
    >>> ####
    >>> # Example 2: scalar multivariate function
    >>> x2 = [1, 2]
    >>> def f2(x, y):
    >>>     return x*y
    >>> g2 = Reverse(f2, *x2)
    >>> g2.val
    2
    >>> g2.der
    array([2., 1.])
    >>> ####
    >>> # Example 3: vector univariate function
    >>> x3 = 50
    >>> def f3(x):
    >>>     return [1,2,3]
    >>> g3 = Reverse(f3, x3)
    >>> g3.val
    array([1, 2, 3])
    >>> g3.der
    array([[0., 0., 0.]])
    >>> ####
    >>> # Example 4: vector multivariate function
    >>> x = [1, 2]
    >>> def f1(x1, x2):
    >>>     return [2 *x2, x1 * x2, 3 * x1 + 9 * x2, 100 * x1] 
    >>> g1 = Reverse(f1, *x)
    >>> g1.val
    array([  4,   2,  21, 100])
    >>> g1.der
    array([[  0.,   2.],
            [  2.,   1.],
            [  3.,   9.],
            [100.,   0.]])
    """

    def __init__(self, f: callable, *variables):
        self.val, self.der = self.grad(f, *variables)

    def __str__(self):
        """Print useful information for users.

        :return: A string containing useful information of the Reverse class object.
        :rtype: string

        >>> x = 50
        >>> def f0(x):
        ...     return 0
        ... 
        >>> g0 = Reverse(f0, x)
        >>> print(g0)
        Reverse: val=0, and der=0.0.
        """
        return f'Reverse: val={self.val}, and der={self.der}.'

    def __repr__(self):
        """Print useful information for developers.

        :return: A string containing useful information of the Reverse class object.
        :rtype: string

        >>> x = 50
        >>> def f0(x):
        ...     return 0
        ... 
        >>> g0 = Reverse(f0, x)
        >>> g0
        A Reverse object with value of 0, and derivative of 0.0.
        """
        return f'A Reverse object with value of {self.val}, and derivative of {self.der}.'

    @staticmethod
    def grad(f: callable, *variables):
        r"""
        Evaluate the full Jacobian in reverse mode. This is the method that is used internally
        by :py:meth:`AutoDiff.reverse.Reverse.__init__`.
        For each scalar function, use a forward pass and a reverse pass.
        Stack the partial derivative columns into the full Jacobian.

        :param f: A callable function to perform differentaition on
        :type f: function object

        :param variables: The input for variables of function ``f``
        :type variables: integer or float or numpy array or list of integers or floats

        :return: Jacobian
            i.e. :math:`\frac{\partial f(\text{variables})}{\partial \text{variables}}`.
            Stack the partial derivative columns into the full Jacobian.
        :rtype: integer or float or numpy array
        """
        # Helper function when Reverse class is initialized, see usage in init.
        num_variables = len(variables)
        variables = [RNode(var) for var in variables]
        output = f(*variables)
        if isinstance(output, list): # vector function
            output = [o if isinstance(o, RNode) else RNode(o)for o in output]
        else: # scalar function
            if not isinstance(output, RNode):
                output = RNode(output)
        ders = []

        for var in variables:
            output_depend = []
            var_der = var.grad_vec(output_depend)
            # var_der: ordered list of derivatives of each parent of var
            # output_depend: ordered list of output rnode that each parent of var points to
            var.clear() # clear the paths to prepare for the next iteration of variables

            # sum up the partial derivaties of each scalar function in var_der
            unique_depend = []
            unique_grad = []
            for j, o in enumerate(output_depend):
                if o not in unique_depend:
                    unique_depend.append(o)
                    unique_grad.append(var_der[j])
                else:
                    unique_grad[unique_depend.index(o)] += var_der[j]
            var_der = unique_grad # now var_der is ordered derivatives of each scalar function
            
            if isinstance(output, list): # vector function
                if len(var_der) < len(output):
                    for idx, o in enumerate(output):
                        if o not in output_depend:
                            if id(o) == id(var): # case: f1(x1) = x1
                                var_der.insert(idx, 1.)
                            else: # case: f1(x1) = x2 or f1(x1) = 3
                                var_der.insert(idx, 0.)
            elif output not in output_depend: # scalar function
                if id(output) == id(var): # f(x1) = x1
                    var_der.append(1.)
                else: # case: f(x1) = x2 or f(x1) = 3
                    var_der.append(0.)

            if len(var_der) == 1:
                var_der = var_der[0]
                
            ders.append(var_der)

        ders = np.stack(ders, axis = -1) # stack derivatives of each var
        if isinstance(output, list): # vector function
            values = np.array([o.val for o in output])
            if num_variables == 1: 
                ders = ders.T # reshape column vector to 1d array
        else: # scalar function
            values = output.val
            ders = ders.T
            if len(ders) == 1:
                ders = ders[0]
        return values, ders