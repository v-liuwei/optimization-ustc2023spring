"""
Sequential quadratic programming (SQP) method.

Original proposed by Han (1977).
"""
from typing import *

import cvxopt
import numpy as np
from sympy import Matrix, Max, hessian, symbols

from line_search import backtracking_line_search
from utils import Vec, gradient


def sqp(f: Callable[[Vec[Any]], Any], cons_eq: List[Callable[[Vec[Any]], Any]], cons_ineq: List[Callable[[Vec[Any]], Any]], x0: Vec[float], sigma: float = 0.4, eps: float = 1e-6, max_iter: int = 100) -> Tuple[Vec[float], bool, int]:
    """Solver for constrained optimization problem using Sequential Quadratic Programming (SQP) method.

    ```math
      min  f(x)
      s.t. c_i(x) = 0, i = 1,..., me
           c_i(x) <= 0, i = me+1,..., m
    ```

    Parameters
    ----------
    f : Callable[[Vec[Any]], Any]
        Objective function. It should return a sympy Expr object when receiving a Vec of sympy symbols as argument, where Expr should evaluate to a scalar when all symbols are substituted with numbers.
    cons_eq : List[Callable[[Vec[Any]], Any]]
        List of functions for equality constraints. Each function should return a sympy Expr object when receiving a Vec of sympy symbols as argument, where Expr should evaluate to a scalar when all symbols are substituted with numbers.
    cons_ineq : List[Callable[[Vec[Any]], Any]]
        List of functions for inequality constraints. Each function should return a sympy Expr object when receiving a Vec of sympy symbols as argument, where Expr should evaluate to a scalar when all symbols are substituted with numbers.
    x0 : Vec[float]
        Initial point.
    sigma : float, optional
        Coefficient of penalty function, by default 0.4.
    eps : float, optional
        Tolerance, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations, by default 100.

    Returns
    -------
    x_k : Vec[float]
        Optimal point if converged, otherwise the last point.
    success : bool
        Whether the algorithm successfully converged.
    k : int
        Number of iterations.

    Examples
    --------
    >>> obj_func = lambda x: x[0] ** 2 + x[1] ** 2
    >>> cons_eq = [lambda x: x[0] + x[1] - 1]
    >>> cons_ineq = []
    >>> x0 = [1.8, 1.7]
    >>> sqp(obj_func, cons_eq, cons_ineq, x0)
    (Matrix([
     [0.5],
     [0.5]]),
     True,
     1)
    """
    n, me, mi = len(x0), len(cons_eq), len(cons_ineq)
    m = me + mi

    x0 = Matrix(x0)
    x_ = symbols(f'x:{n}', real=True)

    c = lambda x: Matrix([g(x) for g in cons_eq + cons_ineq])      # constraint function
    A = lambda x: c(x_).jacobian(x_).subs(zip(x_, x))           # Jacobian of constraint function

    l = lambda x, lmbd: f(x) + c(x).dot(Matrix(lmbd))              # Lagrangian function

    g = lambda x: gradient(f(x_), x_).subs(zip(x_, x))
    W = lambda x, lmbd: hessian(l(x_, lmbd), x_).subs(zip(x_, x))

    # L1 Penalty function
    P = lambda x: f(x) + sigma * (sum(abs(c_i(x)) for c_i in cons_eq) + sum(Max(0., c_i(x)) for c_i in cons_ineq))

    success = False
    x_k, lmbd_k = x0, Matrix([0] * m)
    for k in range(max_iter):
        # Solve subproblem
        W_k = W(x_k, lmbd_k)
        g_k = g(x_k)
        A_k = A(x_k)
        c_k = c(x_k)

        # Solve QP
        args: Tuple[Matrix, ...] = W_k, g_k, A_k[me:, :], -c_k[me:, :], A_k[:me, :], -c_k[:me, :]
        res = cvxopt.solvers.qp(*map(lambda x: cvxopt.matrix(np.array(x, dtype=float)), args), options={'show_progress': False})

        # get x and lambda
        d_k = Matrix(np.array(res['x']))
        lmbd_bar_k = Matrix([Matrix(np.array(res['y'])), Matrix(np.array(res['z']))])

        if d_k.norm() <= eps:
            success = True
            break

        # line search
        alpha_k = backtracking_line_search(P, x_k, d_k)

        # update x and lambda
        x_k = x_k + alpha_k * d_k
        lmbd_k = lmbd_bar_k + alpha_k * (lmbd_bar_k - lmbd_k)        

    return x_k, success, k


if __name__ == '__main__':
    rosenbrock = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    cons_eq = [lambda x: 2 * x[0] + x[1] - 1]
    cons_ineq = [lambda x: x[0] + 2 * x[1] - 1, lambda x: x[0] ** 2 + x[1] - 1, lambda x: x[0] ** 2 - x[1] - 1]
    x0 = [0.5, 0]

    print(sqp(rosenbrock, cons_eq, cons_ineq, x0))
