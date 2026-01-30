import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    dot,
    norm,
    vector_add,
    vector_sub,
    scalar_mult,
    mat_vec_mult,
    mat_mult,
    transpose,
)


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    """
    Resolve o sistema Ax = b usando o método dos gradientes conjugados.
    A deve ser simétrica e definida positiva.
    """
    n = len(b)
    if x0 is None:
        x = [0.0] * n
    else:
        x = x0[:]

    if max_iter is None:
        max_iter = n

    r = vector_sub(b, mat_vec_mult(A, x))
    p = r[:]
    rs_old = dot(r, r)

    for k in range(max_iter):
        Ap = mat_vec_mult(A, p)
        alpha = rs_old / dot(p, Ap)

        x = vector_add(x, scalar_mult(alpha, p))
        r = vector_sub(r, scalar_mult(alpha, Ap))

        rs_new = dot(r, r)

        if norm(r) < tol:
            return x, k + 1

        beta = rs_new / rs_old
        p = vector_add(r, scalar_mult(beta, p))
        rs_old = rs_new

    return x, max_iter


def conjugate_gradient_general(A, b, tol=1e-10):
    """
    Resolve Ax = b usando gradientes conjugados,
    transformando o sistema caso A não seja simétrica.
    """
    A_t = transpose(A)
    A_sym = mat_mult(A_t, A)
    b_sym = mat_vec_mult(A_t, b)

    return conjugate_gradient(A_sym, b_sym, tol=tol)
