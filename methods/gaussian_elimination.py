def gaussian_elimination_partial(A, b):
    """
    Resolve o sistema Ax = b usando eliminação de Gauss
    com pivotação parcial.
    """
    n = len(b)
    # Cria uma matriz aumentada
    Ab = [A[i][:] + [b[i]] for i in range(n)]

    # Eliminação para frente com pivotação parcial
    for i in range(n - 1):
        # Encontrar o pivô máximo na coluna i
        max_row = max(range(i, n), key=lambda r: abs(Ab[r][i]))

        if Ab[max_row][i] == 0:
            raise ValueError("Sistema sem solução única")

        # Trocar a linha atual com a linha do pivô máximo
        Ab[i], Ab[max_row] = Ab[max_row], Ab[i]

        # Eliminação dos elementos abaixo do pivô
        for j in range(i + 1, n):
            factor = Ab[j][i] / Ab[i][i]
            for k in range(i, n + 1):
                Ab[j][k] -= factor * Ab[i][k]

    # Substituição regressiva
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i][n] / Ab[i][i]
        for j in range(i + 1, n):
            x[i] -= (Ab[i][j] / Ab[i][i]) * x[j]

    return x


def gaussian_elimination_total(A, b):
    """
    Resolve o sistema Ax = b usando eliminação de Gauss
    com pivotação total.
    """
    n = len(b)

    # Criação da matriz aumentada
    augmented_matrix = [A[i][:] + [b[i]] for i in range(n)]

    # Vetor de permutação das variáveis
    permutation = list(range(n))

    # Fase de eliminação
    for k in range(n - 1):
        # Busca do maior elemento (em módulo) na submatriz
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

        # Eliminação dos elementos abaixo do pivô
        for i in range(k + 1, n):
            factor = augmented_matrix[i][k] / augmented_matrix[k][k]
            for j in range(k, n + 1):
                augmented_matrix[i][j] -= factor * augmented_matrix[k][j]

    # Substituição regressiva
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (
            augmented_matrix[i][n]
            - sum(augmented_matrix[i][j] * x[j] for j in range(i + 1, n))
        ) / augmented_matrix[i][i]

    # Reorganização da solução por causa da troca de colunas
    final_solution = [0] * n
    for i in range(n):
        final_solution[permutation[i]] = x[i]

    return final_solution