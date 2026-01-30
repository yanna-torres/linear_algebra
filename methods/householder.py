from ..utils import norm, mat_mult, identity, scalar_mult, vector_sub, print_matrix


def householder_matrix(x):
    """
    Constrói a matriz de Householder associada ao vetor x.
    """
    n = len(x)
    e1 = [1.0] + [0.0] * (n - 1)

    alpha = norm(x)
    if alpha == 0:
        return identity(n)

    v = vector_sub(x, scalar_mult(alpha, e1))
    v_norm = norm(v)

    if v_norm == 0:
        return identity(n)

    v = scalar_mult(1 / v_norm, v)

    H = identity(n)
    for i in range(n):
        for j in range(n):
            H[i][j] -= 2 * v[i] * v[j]

    return H


def householder_similarity(A):
    """
    Aplica o método de Householder por transformações de similaridade.
    """
    n = len(A)
    A_k = [row[:] for row in A]
    Q = identity(n)

    print("Matriz original:")
    print_matrix(A_k)

    for k in range(n - 2):
        # Extrai o vetor abaixo da diagonal
        x = [A_k[i][k] for i in range(k + 1, n)]

        H_k_small = householder_matrix(x)

        # Expande H_k para dimensão n
        H_k = identity(n)
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                H_k[i][j] = H_k_small[i - k - 1][j - k - 1]

        # Atualiza A_k = H_k A_k H_k
        A_k = mat_mult(mat_mult(H_k, A_k), H_k)

        # Acumula Q
        Q = mat_mult(Q, H_k)

        print(f"\nPasso {k + 1}")
        print("Matriz de Householder:")
        print_matrix(H_k)

        print("Matriz transformada:")
        print_matrix(A_k)

        print("Matriz de Householder acumulada:")
        print_matrix(Q)

    return A_k, Q
