import math

def cholesky_decomposition(A):
    """
    Realiza a decomposição de Cholesky de uma matriz A,
    assumindo que A é simétrica e positiva definida.
    Retorna a matriz L tal que A = L L^T.
    """
    n = len(A)

    # Inicialização da matriz L
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                value = A[i][i] - sum(L[i][k] ** 2 for k in range(j))
                if value <= 0:
                    raise ValueError("A matriz não é positiva definida.")
                L[i][j] = math.sqrt(value)
            else:
                L[i][j] = (
                    A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
                ) / L[j][j]

    return L
