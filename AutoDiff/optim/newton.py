import numpy as np
from .. import Forward

def Newton(f: callable, *x0, tol=1e-5, max_iter=500, n_iter=1):
    r"""
    Newton's method

    To find :math:`\mathbf{x}` such that :math:`F(\mathbf{x}) = \mathbf{0}`, we need update :math:`\Delta \mathbf{x}_{k}` such that

    .. math::
                  J_F(\mathbf{x}_k)\Delta \mathbf{x}_{k} &= - F(\mathbf{x}_{k}) \\
              \mathbf{x}_{k+1} &\gets \mathbf{x}_{k} + \Delta \mathbf{x}_{k}
    where :math:`J_F(\mathbf{x}_k)` is the Jacobian and :math:`\Delta \mathbf{x}_{k}` is the update.
    
    :param f: A callable function object, the :math:`F: \mathbb{R}^m \mapsto \mathbb{R}^n` function
    :type f: function object
    :param x0: The initial guess, note that Newton is quadratic convergence if initial guess is close to the actual solution
    :type x0: integer or float or numpy array or list of integers or floats
    :param tol: The tolerance, the algorithm terminates when it hits the tolerance i.e. when :math:`\|F(\mathbf{x})\|_F < \text{tol}` is reached, the algorithm terminates
    :type tol: float
    :raises RuntimeError: If the function does not converge in max_iter iterations
    
    :return: The solution :math:`\mathbf{x}`
    :rtype: float or list of floats

    >>> x0 = [0, 1]
    >>> def f(x1, x2):
    >>>     return [
    >>>         2 * x1 + x2 - np.exp(-x1),
    >>>         -x1 + 2 * x2 - np.exp(-x2)
    >>>     ]
    >>> sol = Newton(f, *x0)
    >>> sol
    array([0.19759433, 0.42551406]) 
    >>> f(*sol)
    [-4.3786574366322384e-10, -3.137059279012533e-09]
    """
    n_iter += 1
    if n_iter == max_iter:
        raise RuntimeError(f'The function does not converge in {n_iter} iterations!')
    x0 = np.array(x0)
    # if the norm of the function is less than the tolerance, consider the method converged
    if np.linalg.norm(f(*x0)) < tol:
        # if result list has length 1, return the number without the bracket
        if len(x0) == 1:
            return x0.item()
        return x0
    # use Forward AD for derivative calculation
    g = Forward(f, *x0)
    if isinstance(g.der, (int, float)):
        # if the derivative g is a number, perform Newton's Method in 1D
        new_x = x0 - g.val / g.der
    else:
        # else perform Newton's Method in nD
        if g.der.ndim == 1:
            g.der = g.der.reshape(1, -1)
            g.val = np.array(g.val).reshape(1)
        update, *_ = np.linalg.lstsq(g.der, -g.val, rcond=None)
        new_x = x0 + update
    return Newton(f, *new_x, tol=tol, n_iter=n_iter)