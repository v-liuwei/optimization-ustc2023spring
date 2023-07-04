from typing import *

from sympy import Expr, Matrix, Symbol

T = TypeVar('T')
Vec = Sequence[T]
Mat = Sequence[Sequence[T]]


def gradient(expr: Expr, x: Sequence[Symbol]) -> Matrix:
    return Matrix([expr]).jacobian(x).T


def label_str(s: str, label: str) -> str:
    """Add label to a string at its center left."""
    from math import ceil
    l_lines = label.split('\n')
    nl = len(l_lines)
    lw = max(len(l) for l in l_lines)
    s_lines = s.split('\n')
    ns = len(s_lines)

    nr = max(nl, ns)
    lines = []
    for i in range(nr):
        line = ''
        if ceil((nr - nl) / 2) <= i < ceil((nr + nl) / 2):
            line += l_lines[i - ceil((nr - nl) / 2)].ljust(lw)
        else:
            line += ' ' * lw
        if ceil((nr - ns) / 2) <= i < ceil((nr + ns) / 2):
            line += s_lines[i - ceil((nr - ns) / 2)]
        lines.append(line)
    return '\n'.join(lines)
