def gauss_jordan_partial(A, b):
    """
    Resolve o sistema Ax = b usando o método de Gauss-Jordan
    com pivotação parcial.
    """
    n = len(b)

    # Criação da matriz aumentada
    augmented_matrix = [A[i][:] + [b[i]] for i in range(n)]

    for k in range(n):
        # Pivotação parcial: maior elemento da coluna k
        max_row = max(range(k, n), key=lambda i: abs(augmented_matrix[i][k]))

        if augmented_matrix[max_row][k] == 0:
            raise ValueError("Sistema sem solução única")

        # Troca de linhas
        augmented_matrix[k], augmented_matrix[max_row] = (
            augmented_matrix[max_row],
            augmented_matrix[k],
        )

        # Normalização do pivô
        pivot = augmented_matrix[k][k]
        for j in range(k, n + 1):
            augmented_matrix[k][j] /= pivot

        # Eliminação acima e abaixo do pivô
        for i in range(n):
            if i != k:
                factor = augmented_matrix[i][k]
                for j in range(k, n + 1):
                    augmented_matrix[i][j] -= factor * augmented_matrix[k][j]

    # A solução está na última coluna
    solution = [augmented_matrix[i][n] for i in range(n)]
    return solution


def gauss_jordan_total(A, b):
    """
    Resolve o sistema Ax = b usando o método de Gauss-Jordan
    com pivotação total.
    """
    n = len(b)

    # Criação da matriz aumentada
    augmented_matrix = [A[i][:] + [b[i]] for i in range(n)]

    # Vetor de permutação das variáveis
    permutation = list(range(n))

    for k in range(n):
        # Busca do maior elemento na submatriz
        max_value = 0
        max_row, max_col = k, k

        for i in range(k, n):
            for j in range(k, n):
                if abs(augmented_matrix[i][j]) > max_value:
                    max_value = abs(augmented_matrix[i][j])
                    max_row, max_col = i, j

        if max_value == 0:
            raise ValueError("Sistema sem solução única")

        # Troca de linhas
        augmented_matrix[k], augmented_matrix[max_row] = (
            augmented_matrix[max_row],
            augmented_matrix[k],
        )

        # Troca de colunas
        for i in range(n):
            augmented_matrix[i][k], augmented_matrix[i][max_col] = (
                augmented_matrix[i][max_col],
                augmented_matrix[i][k],
            )

        permutation[k], permutation[max_col] = (
            permutation[max_col],
            permutation[k],
        )

        # Normalização do pivô
        pivot = augmented_matrix[k][k]
        for j in range(k, n + 1):
            augmented_matrix[k][j] /= pivot

        # Eliminação acima e abaixo do pivô
        for i in range(n):
            if i != k:
                factor = augmented_matrix[i][k]
                for j in range(k, n + 1):
                    augmented_matrix[i][j] -= factor * augmented_matrix[k][j]

    # Reorganização da solução devido à troca de colunas
    solution = [0] * n
    for i in range(n):
        solution[permutation[i]] = augmented_matrix[i][n]

    return solution
