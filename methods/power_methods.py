from ..utils import mat_vec_mult, norm, scalar_mult, vector_sub
from .lu_decomposition import solve_lu


def power_method(A, x0, eps=1e-8, max_iter=1000):
    """
    Método da Potência Regular.
    Aproxima o autovalor dominante e o autovetor associado.
    """
    x = scalar_mult(1 / norm(x0), x0)

    for _ in range(max_iter):
        y = mat_vec_mult(A, x)
        y_norm = norm(y)

        if y_norm == 0:
            raise ValueError("Vetor nulo encontrado durante iteração")

        x_new = scalar_mult(1 / y_norm, y)

        if norm(vector_sub(x_new, x)) < eps:
            eigenvalue = y_norm
            return eigenvalue, x_new

        x = x_new

    raise RuntimeError("Método da potência não convergiu")


def inverse_power_method(A, x0, eps=1e-8, max_iter=1000):
    """
    Método da Potência Inversa.
    Aproxima o autovalor de menor módulo.
    """
    x = scalar_mult(1 / norm(x0), x0)

    for _ in range(max_iter):
        # Resolve A y = x
        y = solve_lu(A, x)
        y_norm = norm(y)

        if y_norm == 0:
            raise ValueError("Vetor nulo encontrado durante iteração")

        x_new = scalar_mult(1 / y_norm, y)

        if norm(vector_sub(x_new, x)) < eps:
            eigenvalue = 1 / y_norm
            return eigenvalue, x_new

        x = x_new

    raise RuntimeError("Método da potência inversa não convergiu")


def shifted_power_method(A, x0, mu, eps=1e-8, max_iter=1000):
    """
    Método da Potência com Deslocamento.
    Aproxima o autovalor mais próximo de mu.
    """
    n = len(A)

    # Construção de A - mu I
    A_shifted = [
        [A[i][j] - (mu if i == j else 0) for j in range(n)]
        for i in range(n)
    ]

    eigenvalue_shifted, eigenvector = inverse_power_method(
        A_shifted, x0, eps, max_iter
    )

    eigenvalue = eigenvalue_shifted + mu
    return eigenvalue, eigenvector
