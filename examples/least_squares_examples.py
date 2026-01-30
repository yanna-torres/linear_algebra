import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.least_squares import least_squares
from utils import mat_vec_mult, norm, vector_sub, print_matrix


def test_least_squares():
    """
    Testa a solução de mínimos quadrados.
    """
    A = [[1, 1], [1, 2], [1, 3]]
    b = [1, 2, 2]

    print("Matriz A:")
    print_matrix(A)
    print("Vetor b:", b)

    x_star = least_squares(A, b)
    print("\nSolução de mínimos quadrados x*:")
    print([round(v, 4) for v in x_star])

    # Verifica erro
    Ax = mat_vec_mult(A, x_star)
    error_norm = norm(vector_sub(Ax, b))
    print("Norma do erro ||Ax* - b||:", round(error_norm, 4))


if __name__ == "__main__":
    test_least_squares()
