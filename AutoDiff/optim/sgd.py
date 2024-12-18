import numpy as np
from .. import Forward

def SGD(f: callable, *x0, eta=1e-1, n_iter=50000, tol=1e-5):
    r"""
    Stochastic gradient descent

    It optimizes the following procedure iteratively

    .. math::
        \mathbf{x} \gets \mathbf{x} - \eta \nabla f(\mathbf{x})

    where :math:`f: \mathbb{R}^n \mapsto \mathbb{R}`

    :param f: A callable function object, the :math:`F: \mathbb{R}^n \mapsto \mathbb{R}` function
    :type f: function object
    :param x0: An initial guess
    :type x0: integer or float or numpy array or list of integers or floats
    :param eta: The learning rate :math:`\eta`, which needed to be picked for each specific optimization task
    :type eta: float
    :param n_iter: After :code:`n_iter` steps the algorithm will terminate
    :type n_iter: integer
    :param tol: The algorithm terminates when it reaches the tolerance, i.e. when :math:`|f(\mathbf{x})| < \text{tol}` is reached
    :type tol: float
    :raises RuntimeError: If the function does not converge in n_iter iterations
    
    :return: The final solution
    :rtype: float or numpy array

    >>> # Example 1: scalar univariate function
    >>> x1 = 1
    >>> f1 = lambda x: x**2
    >>> sol1 = SGD(f1, x1)
    >>> sol1
    0.0030223145490365735
    >>> f(sol1)
    9.134385233318147e-06
    >>> ####
    >>> # Example 2: vector multivariate function
    >>> x2 = [4, 3]
    >>> def f2(x, y):
    ...     return x ** 2 + y ** 2
    ... 
    >>> sol2 = SGD(f2, *x2)
    >>> sol2
    array([0.00202824, 0.00152118])
    >>> f(*sol2)
    6.427752177035966e-06
    """
    i = 0
    while i < n_iter:
        # if the norm of the function is less than tolerance, consider the method converged
        if np.linalg.norm(f(*x0)) < tol:
            # if result list has length 1, return the number without the bracket
            if len(x0) == 1:
                return x0.item()
            return x0
        i += 1
        # use Forward AD for derivative calculation
        g = Forward(f, *x0)
        # apply stochastic gradient descent
        x0 -= eta * g.der
    # if the function does not converge in 50000 iterations, we consider the function does not converge, can raise a Runtime error
    raise RuntimeError(f'The function does not converge in {n_iter} iterations!')
