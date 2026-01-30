import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import transpose, mat_vec_mult, dot
from methods.gram_schmidt import gram_schmidt


def least_squares(A, b):
    """
    Resolve o problema de mínimos quadrados
    min ||Ax - b||^2 usando decomposição QR.
    """
    # Colunas de A como vetores
    A_cols = transpose(A)

    # Ortogonalização de Gram-Schmidt
    Q = gram_schmidt(A_cols)
    Q_t = transpose(Q)

    # Calcula Q^T b
    y = mat_vec_mult(Q_t, b)

    # Matriz R
    R = [[dot(Q[i], A_cols[j]) for j in range(len(A_cols))] for i in range(len(Q))]

    # Resolução de Rx = y (substituição retroativa)
    n = len(R)
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        s = sum(R[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / R[i][i]

    return x
