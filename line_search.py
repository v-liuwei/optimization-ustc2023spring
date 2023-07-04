from typing import *

from sympy import Matrix, symbols

from utils import Vec, gradient


def line_search(f: Callable[[Vec[Any]], Any], x: Vec[float], d: Vec[float], rho: float = 0.0001, sigma: float = 0.9, amax: float = None) -> float:
    """Find alpha that satisfies strong Wolfe-Powell conditions.

    Parameters
    ----------
    f: Callable[[Vec[Any]], Any]
        Objective function. It should return a sympy Expr object when receiving a Vec of sympy symbols as argument, where Expr should evaluate to a scalar when all symbols are substituted with numbers.
    x: Vec[float]
        Starting point.
    d: Vec[float]
        Search direction.
    rho: float, optional
        Paramter for the first condition rule, by default 0.0001.
    sigma: float, optional
        Paramter for the second condition rule, by default 0.9.
    amax: float, optional
        Upper bound of alpha, by default None.

    Returns
    -------
    alpha: float or None
        alpha for which ``x_{k+1} = x_{k} + alpha * d_{k}``,
        or None if the line search algorithm did not converge.

    Examples
    --------
    >>> obj_func = lambda x: x[0]**2 + x[1]**2
    >>> x = [1.8, 1.7]
    >>> d = [-1.0, -1.0]
    >>> line_search(obj_func, x, d)
    1.0
    """
    x, d = Matrix(x), Matrix(d)

    phi = lambda alpha: f(x + alpha * d)

    alpha_ = symbols('alpha', real=True)
    derphi = lambda alpha: phi(alpha_).diff(alpha_).subs(alpha_, alpha)

    phi0 = phi(0)
    derphi0 = derphi(0)

    alpha1, alpha2 = 0, amax if amax is not None else float('inf')
    phi1, derphi1 = phi0, derphi0
    alpha = min(1.0, alpha2)

    while True:
        phia = phi(alpha)
        if phia > phi0 + rho * alpha * derphi0:
            alpha_hat = alpha1 + ((alpha1 - alpha) ** 2 * derphi1) / \
                                 (2 * ((phi1 - phia) - (alpha1 - alpha) * derphi1) + 1e-9)
            alpha2 = alpha
            alpha = alpha_hat
        else:
            derphia = derphi(alpha)
            if abs(derphia) > - sigma * derphi0:
                alpha_hat = alpha - (alpha1 - alpha) * derphia / (derphi1 - derphia + 1e-9)
                alpha1, alpha, phi1, derphi1 = alpha, alpha_hat, phia, derphia
            else:
                break

    return alpha


def backtracking_line_search(f: Callable[[Vec[Any]], Any], x: Vec[float], d: Vec[float], dom: Callable[[Vec[float]], bool] = lambda _: True, t0: float = 1.0, alpha: float = 0.3, beta: float = 0.8) -> float:
    """Find step size using backtracking line search.

    Parameters
    ----------
    f: Callable[[Vec[Any]], Any]
        Objective function. It should return a sympy Expr object when receiving a Vec of sympy symbols as argument, where Expr should evaluate to a scalar when all symbols are substituted with numbers.
    x: Vec[float]
        Starting point.
    d: Vec[float]
        Search direction.
    dom: Callable[[Vec[float]], bool], optional
        Domain of the objective function, which is a function that takes a point and returns a boolean value indicating whether the point is in the domain, by default lambda x: True.
    t0: float, optional
        Initial step size, by default 1.0.
    alpha: float, optional
        The fraction of the decrease in f predicted by linear extrapolation that we will accept. Must be in (0, 0.5), by default 0.3.
    beta: float, optional
        The factor by which we decrease step size in each iteration. Must be in (0, 1), by default 0.8.

    Returns
    -------
    t: float
        step size.

    Examples
    --------
    >>> obj_func = lambda x: x[0] ** 2 + x[1] ** 2
    >>> x = [1.8, 1.7]
    >>> d = [-1.0, -1.0]
    >>> backtracking_line_search(obj_func, x, d)
    1.0
    """
    n = len(x)
    x, d = Matrix(x), Matrix(d)

    x_ = symbols(f'x:{n}', real=True)
    g = lambda x: gradient(f(x_), x_).subs(zip(x_, x))

    fx = f(x)
    m = alpha * g(x).dot(d)

    t = t0
    while not dom(x_new := x + t * d) or f(x_new) > fx + t * m:
        t *= beta

    return t


if __name__ == '__main__':
    obj_func = lambda x: x[0] ** 2 + x[1] ** 2
    x = [1.8, 1.7]
    d = [-1.0, -1.0]
    print('line_search:', line_search(obj_func, x, d))
    print('backtracking_line_search:', backtracking_line_search(obj_func, x, d))
