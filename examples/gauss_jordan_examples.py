import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.matrix_operations import print_matrix
from methods.gauss_jordan import (
    gauss_jordan_partial,
    gauss_jordan_total,
)


def test_small_system():
    """
    Teste com um sistema pequeno 3x3.
    """
    A = [
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0],
    ]
    b = [8.0, -11.0, -3.0]

    print("=== Sistema pequeno 3x3 ===")
    print_matrix(A)
    print("Vetor b:")
    print(b)

    x_partial = gauss_jordan_partial(A, b)
    print("\nSolução (Gauss-Jordan + Pivotação Parcial):", x_partial)

    x_total = gauss_jordan_total(A, b)
    print("Solução (Gauss-Jordan + Pivotação Total):  ", x_total)


def test_10x10_system():
    """
    Teste com um sistema 10x10.
    """
    A = [
        [10.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 11.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [3.0, 4.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [4.0, 5.0, 1.0, 13.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [5.0, 1.0, 2.0, 3.0, 14.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 15.0, 7.0, 8.0, 9.0, 10.0],
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 16.0, 9.0, 10.0, 11.0],
        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 17.0, 11.0, 12.0],
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 18.0, 13.0],
        [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 19.0],
    ]

    b = [1.0] * 10

    print("\n=== Sistema 10x10 ===")
    print_matrix(A)
    print("Vetor b:")
    print(b)

    x_partial = gauss_jordan_partial(A, b)
    print("\nSolução (Gauss-Jordan + Pivotação Parcial):")
    print(x_partial)

    x_total = gauss_jordan_total(A, b)
    print("\nSolução (Gauss-Jordan + Pivotação Total):")
    print(x_total)


if __name__ == "__main__":
    test_small_system()
    test_10x10_system()
