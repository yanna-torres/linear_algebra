import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import mat_mult, identity
from methods.qr_decomposition import qr_decomposition
from methods.householder import householder_similarity


def qr_eigenvalues_symmetric(A, tol=1e-8, max_iter=1000):
    """
    Método QR para autovalores e autovetores
    de matrizes simétricas.
    """
    n = len(A)
    A_k = [row[:] for row in A]
    Q_acc = identity(n)

    for _ in range(max_iter):
        Q, R = qr_decomposition(A_k)
        A_k = mat_mult(R, Q)
        Q_acc = mat_mult(Q_acc, Q)

        off_diag = sum(abs(A_k[i][j]) for i in range(n) for j in range(n) if i != j)
        if off_diag < tol:
            break

    eigenvalues = [A_k[i][i] for i in range(n)]
    eigenvectors = Q_acc

    return eigenvalues, eigenvectors


def qr_with_householder(A, tol=1e-8):
    """
    Aplica Householder seguido do método QR
    para matrizes simétricas.
    """
    A_h, H = householder_similarity(A)
    eigenvalues, Q = qr_eigenvalues_symmetric(A_h, tol)

    eigenvectors = mat_mult(H, Q)
    return eigenvalues, eigenvectors


def qr_eigenvalues_general(A, tol=1e-8, max_iter=1000):
    """
    Método QR para matrizes não simétricas.
    Retorna a matriz BUT aproximada.
    """
    A_k = [row[:] for row in A]
    n = len(A)

    for _ in range(max_iter):
        Q, R = qr_decomposition(A_k)
        A_k = mat_mult(R, Q)

        off_subdiag = sum(abs(A_k[i][i - 1]) for i in range(1, n))
        if off_subdiag < tol:
            break

    return A_k


def eigenvalues_from_but(B, tol=1e-8):
    """
    Extrai autovalores a partir de uma matriz
    triangular superior em blocos (BUT).
    """
    n = len(B)
    eigenvalues = []
    i = 0

    while i < n:
        if i < n - 1 and abs(B[i + 1][i]) > tol:
            a, b = B[i][i], B[i][i + 1]
            c, d = B[i + 1][i], B[i + 1][i + 1]

            trace = a + d
            det = a * d - b * c
            delta = trace**2 - 4 * det

            eigenvalues.append((trace + delta**0.5) / 2)
            eigenvalues.append((trace - delta**0.5) / 2)
            i += 2
        else:
            eigenvalues.append(B[i][i])
            i += 1

    return eigenvalues
