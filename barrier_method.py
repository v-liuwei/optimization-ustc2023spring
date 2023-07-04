"""
Interior point method (barrier method) for convex optimization problem.
"""
from typing import *

from sympy import Matrix, hessian, log, symbols
from sympy.solvers import linsolve

from line_search import backtracking_line_search
from utils import Mat, Vec, gradient


def newton_method_with_equality_constraints(f: Callable[[Vec[Any]], Any], A: Mat[float], b: Vec[float], x0: Vec[float], dom: Callable[[Vec[float]], bool] = lambda _: True, eps: float = 1e-6, max_iter: int = 100) -> Tuple[Vec[float], bool, int]:
    """Solve equality constrained optimization problem using Newton method.

    ```math
      min  f(x)
      s.t. Ax = b
    ```

    Parameters
    ----------
    f : Callable[[Vec[Any]], Any]
        objective function. It should return a sympy Expr object when receiving a Vec of sympy symbols as argument, where Expr should evaluate to a scalar when all symbols are substituted with numbers.
    A : Mat[float]
        constraint matrix
    b : Vec[float]
        constraint vector
    x0 : Vec[float]
        initial point
    eps : float, optional
        tolerance, by default 1e-6
    max_iter : int, optional
        maximum number of iterations, by default 100

    Returns
    -------
    x_k : Vec[float]
        optimal point if converged, otherwise the last point
    success : bool
        whether the algorithm successfully converged
    k : int
        number of iterations

    Examples
    --------
    >>> obj_func = lambda x: x[0]**2 + x[1]**2
    >>> A = [[1, 2]]
    >>> b = [2]
    >>> x0 = [2, 0]
    >>> newton_method_with_equality_constraints(obj_func, A, b, x0)
    (Matrix([
     [2/5],
     [4/5]]),
     True,
     1)
    """
    x0, A, b = Matrix(x0), Matrix(A), Matrix(b)
    p, n = A.shape

    x_ = symbols(f'x:{n}', real=True)
    g = lambda x: gradient(f(x_), x_).subs(zip(x_, x))
    H = lambda x: hessian(f(x_), x_).subs(zip(x_, x))

    success = False
    x_k = x0
    for k in range(max_iter):
        g_k = g(x_k)
        H_k = H(x_k)

        M = Matrix([[H_k, A.T, -g_k], [A, Matrix(p * [(2 * p) * [0]])]])

        d_k = Matrix(linsolve(M).args[0][:n])

        if d_k.T.dot(H_k * d_k) / 2 <= eps:
            success = True
            break

        alpha_k = backtracking_line_search(f, x_k, d_k, dom=dom)

        x_k = x_k + alpha_k * d_k

    return x_k, success, k


def barrier_method(f0: Callable[[Vec[Any]], Any], f_list: List[Callable[[Vec[Any]], Any]], A: Mat[float], b: Vec[float], x0: Vec[float], t0: float = 1, mu: float = 10, eps: float = 1e-6, max_iter: int = 100) -> Tuple[Vec[float], bool, int, int]:
    """Barrier method for inequality constrained optimization problem.

    ```math
      min  f_0(x)
      s.t. f_i(x) <= 0, i = 1, 2, ..., m
           Ax = b
    ```

    Parameters
    ----------
    f0 : Callable[[Vec[Any]], Any]
        objective function. It should return a sympy Expr object when receiving a Vec of sympy symbols as argument, where Expr should evaluate to a scalar when all symbols are substituted with numbers.
    f_list : List[Callable[[Vec[Any]], Any]]
        list of functions for inequality constraints. Each function should return a sympy Expr object when receiving a Vec of sympy symbols as argument, where Expr should evaluate to a scalar when all symbols are substituted with numbers.
    A : Mat[float]
        coefficient matrix of equality constraints
    b : Vec[float]
        constant vector of equality constraints
    x0 : Vec[float]
        initial point
    t0 : float, optional
        initial value of t, by default 1
    mu : float, optional
        factor by which we increase t in each iteration, by default 10
    eps : float, optional
        tolerance, by default 1e-6
    max_iter : int, optional
        maximum number of iterations, by default 100

    Returns
    -------
    x_k : Vec[float]
        Optimal point if converged, otherwise the last point.
    success : bool
        whether the algorithm successfully converged.
    k : int
        number of outer iterations.
    k_nt : int
        total number of inner Newton iterations.

    Examples
    --------
    >>> f0 = lambda x: -x[0] - x[1]
    >>> f_list = [lambda x: x[0] * 2 + x[1] - 2]
    >>> A = [[1, -1]]
    >>> b = [0]
    >>> x0 = [0, 0]
    >>> barrier_method(f0, f_list, A, b, x0)
    (Matrix([
     [0.666666616720613],
     [0.666666616720613]]),
     True,
     8,
     38)
    """
    x0, A, b = Matrix(x0), Matrix(A), Matrix(b)

    phi = lambda x: - sum([log(-f_i(x)) for f_i in f_list])
    dom = lambda x: all([f_i(x) < 0 for f_i in f_list])

    success = False
    k_nt = 0
    x_k, t_k = x0, t0
    for k in range(1, max_iter + 1):
        f = lambda x: t_k * f0(x) + phi(x)
        x_k, success, k_inner = newton_method_with_equality_constraints(f, A, b, x_k, dom=dom, eps=eps)
        k_nt += k_inner

        if not success:
            break

        if len(f_list) / t_k < eps:
            success = True
            break

        t_k *= mu

    return x_k, success, k, k_nt


if __name__ == '__main__':
    f0 = lambda x: - x[0] - x[1]
    f_list = [lambda x: x[0] * 2 + x[1] - 2]
    A = [[1, -1]]
    b = [0]

    x0 = [0, 0]

    print(barrier_method(f0, f_list, A, b, x0))
