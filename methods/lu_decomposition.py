def lu_decomposition(A):
    """
    Realiza a decomposição LU de uma matriz A,
    assumindo que A admite decomposição sem pivotação.
    """
    n = len(A)

    # Inicialização das matrizes L e U
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0  # diagonal de L igual a 1

    for k in range(n):
        # Cálculo da linha k de U
        for j in range(k, n):
            U[k][j] = A[k][j] - sum(L[k][m] * U[m][j] for m in range(k))

        if U[k][k] == 0:
            raise ValueError("A matriz não admite decomposição LU sem pivotação")

        # Cálculo da coluna k de L
        for i in range(k + 1, n):
            L[i][k] = (A[i][k] - sum(L[i][m] * U[m][k] for m in range(k))) / U[k][k]

    return L, U


def forward_substitution(L, b):
    """
    Resolve Ly = b por substituição direta.
    """
    n = len(b)
    y = [0.0] * n

    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

    return y


def backward_substitution(U, y):
    """
    Resolve Ux = y por substituição regressiva.
    """
    n = len(y)
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

    return x


def solve_lu(A, b):
    """
    Resolve Ax = b utilizando decomposição LU.
    """
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

if __name__ == "__main__":
    A = [
        [2, 3, 1],
        [4, 7, 7],
        [-2, 4, 5]
    ]

    b = [1, 3, 2]

    solution = solve_lu(A, b)
    print("Solution:", solution)
