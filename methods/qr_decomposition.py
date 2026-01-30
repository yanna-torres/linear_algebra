from ..utils import dot, norm, scalar_mult, vector_sub


def qr_decomposition(A):
    """
    Realiza a decomposição QR de uma matriz quadrada A
    utilizando o processo de Gram-Schmidt.
    """
    n = len(A)
    m = len(A[0])

    # Colunas de A
    columns = [[A[i][j] for i in range(n)] for j in range(m)]

    Q_columns = []
    R = [[0.0] * m for _ in range(m)]

    for j in range(m):
        v = columns[j][:]

        for i in range(j):
            R[i][j] = dot(Q_columns[i], columns[j])
            v = vector_sub(v, scalar_mult(R[i][j], Q_columns[i]))

        R[j][j] = norm(v)
        if R[j][j] == 0:
            raise ValueError("As colunas da matriz são linearmente dependentes")

        q = scalar_mult(1 / R[j][j], v)
        Q_columns.append(q)

    # Monta Q a partir das colunas
    Q = [[Q_columns[j][i] for j in range(m)] for i in range(n)]

    return Q, R
