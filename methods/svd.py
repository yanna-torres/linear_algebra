from ..utils import mat_mult, transpose, print_matrix
from .qr_eigen import qr_eigenvalues_symmetric as qr_eigen
from .gram_schmidt import gram_schmidt
import math


def svd(A, tol=1e-8):
    """
    Decomposição SVD de uma matriz A (m x n)
    Retorna U, Sigma, V
    """
    m = len(A)
    n = len(A[0])

    print("Matriz original A:")
    print_matrix(A)

    # 1. Calcula A^T A
    At = transpose(A)
    AtA = mat_mult(At, A)

    # 2. Autovalores e autovetores de A^T A
    eigenvalues, V = qr_eigen(AtA, tol)

    # 3. Valores singulares
    singular_values = [math.sqrt(ev) if ev > 0 else 0.0 for ev in eigenvalues]

    # 4. Ordena em ordem decrescente
    idx = sorted(
        range(len(singular_values)), key=lambda i: singular_values[i], reverse=True
    )

    singular_values = [singular_values[i] for i in idx]
    V = [[V[row][i] for i in idx] for row in range(n)]

    # 5. Monta Sigma (m x n)
    Sigma = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(min(m, n)):
        Sigma[i][i] = singular_values[i]

    # 6. Calcula U = A V Sigma^{-1}
    U = []
    for i, sigma in enumerate(singular_values):
        if sigma < tol:
            continue

        v_i = [V[row][i] for row in range(n)]
        Av = mat_mult(A, [[x] for x in v_i])
        u_i = [Av[row][0] / sigma for row in range(m)]
        U.append(u_i)

    # 7. Completa base de U se necessário
    if len(U) < m:
        U = gram_schmidt(U, m)

    # Transforma U em matriz m x m
    U = transpose(U)

    print("\nMatriz U:")
    print_matrix(U)

    print("\nMatriz Sigma:")
    print_matrix(Sigma)

    print("\nMatriz V:")
    print_matrix(V)

    # 8. Verificação
    print("\nVerificação U . Sigma . V^T:")
    US = mat_mult(U, Sigma)
    A_rec = mat_mult(US, transpose(V))
    print_matrix(A_rec)

    return U, Sigma, V
