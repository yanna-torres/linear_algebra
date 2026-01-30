import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.rref import analyze_matrix
from utils import print_matrix


def test_m_greater_n():
    """
    Caso m > n
    """
    A = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [1, 0, 1]
    ]
    print("=== Matriz m > n ===")
    rref_matrix, rank, nullity = analyze_matrix(A)
    print("RREF:")
    print_matrix(rref_matrix)
    print(f"Rank: {rank}, Dimensão do espaço nulo: {nullity}\n")


def test_m_less_n():
    """
    Caso m < n
    """
    A = [
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10]
    ]
    print("=== Matriz m < n ===")
    rref_matrix, rank, nullity = analyze_matrix(A)
    print("RREF:")
    print_matrix(rref_matrix)
    print(f"Rank: {rank}, Dimensão do espaço nulo: {nullity}\n")


def test_m_equals_n():
    """
    Caso m = n
    """
    A = [
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ]
    print("=== Matriz m = n ===")
    rref_matrix, rank, nullity = analyze_matrix(A)
    print("RREF:")
    print_matrix(rref_matrix)
    print(f"Rank: {rank}, Dimensão do espaço nulo: {nullity}\n")


if __name__ == "__main__":
    test_m_greater_n()
    test_m_less_n()
    test_m_equals_n()
