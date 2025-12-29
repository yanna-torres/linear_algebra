def rref(matrix, tol=1e-10):
    """
    Calcula a forma escalonada reduzida por linhas (RREF)
    de uma matriz m x n.
    """
    A = [row[:] for row in matrix]
    m = len(A)
    n = len(A[0])

    row = 0  # linha do pivô
    pivot_columns = []

    for col in range(n):
        # Encontra o maior pivô (em módulo) na coluna atual
        pivot_row = None
        max_value = tol

        for i in range(row, m):
            if abs(A[i][col]) > max_value:
                max_value = abs(A[i][col])
                pivot_row = i

        if pivot_row is None:
            continue

        # Troca de linhas
        A[row], A[pivot_row] = A[pivot_row], A[row]

        # Normaliza a linha do pivô
        pivot = A[row][col]
        for j in range(n):
            A[row][j] /= pivot

        # Eliminação acima e abaixo do pivô
        for i in range(m):
            if i != row:
                factor = A[i][col]
                for j in range(n):
                    A[i][j] -= factor * A[row][j]

        pivot_columns.append(col)
        row += 1

        if row == m:
            break

    return A, pivot_columns


def matrix_rank(rref_matrix, tol=1e-10):
    """
    Calcula o posto da matriz a partir do RREF.
    """
    rank = 0
    for row in rref_matrix:
        if any(abs(value) > tol for value in row):
            rank += 1
    return rank


def null_space_dimension(num_columns, rank):
    """
    Calcula a dimensão do espaço nulo.
    """
    return num_columns - rank


def analyze_matrix(matrix):
    """
    Retorna:
    - RREF
    - posto (rank)
    - dimensão do espaço nulo
    """
    rref_matrix, pivot_columns = rref(matrix)
    rank = matrix_rank(rref_matrix)
    nullity = null_space_dimension(len(matrix[0]), rank)

    return rref_matrix, rank, nullity


if __name__ == "__main__":
    # Exemplo de uso
    A = [
        [1, 2, 3, 4],
        [2, 4, 6, 8],
        [1, 1, 1, 1],
    ]

    rref_matrix, rank, nullity = analyze_matrix(A)

    print("Matriz RREF:")
    for row in rref_matrix:
        print(row)
    print(f"Posto (rank): {rank}")
    print(f"Dimensão do espaço nulo: {nullity}")