import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.lu_decomposition import lu_decomposition, solve_lu
from utils import print_matrix


def test_small_system():
    """
    Teste com um sistema pequeno 3x3.
    """
    A = [[2.0, 3.0, 1.0], [4.0, 7.0, 7.0], [-2.0, 4.0, 5.0]]
    b = [1.0, 3.0, 2.0]

    print("=== Sistema pequeno 3x3 ===")

    # Solução usando a função solve_lu
    x = solve_lu(A, b)
    print("Solução:", x)

    # Mostrar as matrizes L e U
    L, U = lu_decomposition(A)
    print("\nMatriz L:")
    print_matrix(L)

    print("\nMatriz U:")
    print_matrix(U)


def test_10x10_system():
    """
    Teste com um sistema 10x10.
    """
    # Exemplo simples, diagonais dominantes para garantir LU sem pivotação
    A = [[10 + i if i == j else 1 for j in range(10)] for i in range(10)]
    b = [i + 1 for i in range(10)]

    print("\n=== Sistema 10x10 ===")

    x = solve_lu(A, b)
    print("Solução:", x)

    L, U = lu_decomposition(A)
    print("\nMatriz L:")
    print_matrix(L)

    print("\nMatriz U:")
    print_matrix(U)


if __name__ == "__main__":
    test_small_system()
    test_10x10_system()
