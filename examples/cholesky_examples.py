import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.cholesky_decomposition import cholesky_decomposition
from utils import print_matrix, transpose, mat_mult


def test_positive_definite():
    """
    Teste com matriz simétrica e positiva definida.
    """
    A = [
        [4, 12, -16],
        [12, 37, -43],
        [-16, -43, 98]
    ]
    print_matrix(A)
    print("\n=== Matriz simétrica e positiva definida ===")
    try:
        L = cholesky_decomposition(A)
        print("Matriz L (Cholesky):")
        print_matrix(L)
        print("\nVerificação (L * L^T):")
        LLT = mat_mult(L, transpose(L))
        print_matrix(LLT)
    except ValueError as e:
        print(e)
    print()


def test_not_positive_definite():
    """
    Teste com matriz simétrica, mas não positiva definida.
    """
    A = [
        [1, 2],
        [2, 1]
    ]
    print("=== Matriz simétrica, mas não positiva definida ===")
    try:
        L = cholesky_decomposition(A)
        print("Matriz L (Cholesky):")
        print_matrix(L)
    except ValueError as e:
        print(e)
    print()


if __name__ == "__main__":
    test_positive_definite()
    test_not_positive_definite()
