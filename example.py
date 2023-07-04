import argparse
from dataclasses import dataclass
from typing import *

from sympy import Eq, Le, MatMul, Matrix, Transpose, pretty, symbols

from barrier_method import Mat, Vec, barrier_method
from utils import label_str


@dataclass
class Problem:
    f0: Callable[[Vec[Any]], Any]
    f_list: List[Callable[[Vec[Any]], Any]]
    A: Mat[float]
    b: Vec[float]
    x0: Vec[float]

    def __str__(self) -> str:
        x_ = symbols(f'x1:{len(self.x0) + 1}')
        cons = [pretty(Le(f_i(x_), 0)) for f_i in self.f_list] + [pretty(Eq(MatMul(Matrix(self.A), Matrix(x_)), Matrix(self.b)))]
        return label_str(pretty(self.f0(x_)), 'min  ') + \
            '\n' + \
            label_str(cons[0], 's.t. ') + '\n' + label_str('\n'.join(cons[1:]), ' ' * len('s.t. ')) + \
            '\n' + \
            label_str(pretty(Transpose(Matrix(self.x0).T)), '\nstarting point: ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interior Point Method')
    parser.add_argument('--problem', type=int, choices=[1, 2], default=1)
    parser.add_argument('--t0', type=float, default=1)
    parser.add_argument('--mu', type=float, default=10)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--max_iter', type=int, default=100)
    args = parser.parse_args()

    problems = [
        Problem(
            f0=lambda x: - x[0] - x[1],
            f_list=[lambda x: x[0] * 2 + x[1] - 2],
            A=[[1, -1]],
            b=[0],
            x0=[0, 0]
        ),
        Problem(
            f0=lambda x: x[0] ** 2 + x[1] ** 2,
            f_list=[lambda x: x[0] ** 2 * 9 / 16 + (x[1] - 2) ** 2 - 1],
            A=[[1, 1]],
            b=[2],
            x0=[0, 2]
        )
    ]

    p = problems[args.problem - 1]
    print('Problem:')
    print(p)
    print()

    x_k, success, k, k_nt = barrier_method(p.f0, p.f_list, p.A, p.b, p.x0, t0=args.t0, mu=args.mu, eps=args.eps, max_iter=args.max_iter)
    print('Result:')
    print(label_str(pretty(Transpose(x_k.T)), 'x_k = '))
    print(f'{success = }')
    print(f'{k = }')
    print(f'{k_nt = }')
