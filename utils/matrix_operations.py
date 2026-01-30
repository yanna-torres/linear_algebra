from .vector_operations import dot


def transpose(matrix):
    """
    Retorna a transposta de uma matriz.
    """
    return list(map(list, zip(*matrix)))


def mat_vec_mult(matrix, vector):
    """
    Multiplica uma matriz por um vetor.
    """
    return [dot(row, vector) for row in matrix]


def mat_mult(A, B):
    """
    Multiplica duas matrizes A e B.
    """
    return [
        [sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
        for i in range(len(A))
    ]


def identity(n):
    """
    Retorna a matriz identidade n x n.
    """
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def print_matrix(M):
    """
    Imprime a matriz M de forma formatada.
    """
    for row in M:
        print(["{:.4f}".format(v) for v in row])
