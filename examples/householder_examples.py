import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.householder import householder_similarity


def test_householder():
    """
    Testa o método de Householder para matrizes simétricas e não simétricas.
    """
    matrices = {
        "Simétrica 3x3": [[4, 1, 2], [1, 3, 0], [2, 0, 1]],
        "Não Simétrica 3x3": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "Simétrica 4x4": [[6, 2, 1, 0], [2, 3, 1, 0], [1, 1, 1, 0], [0, 0, 0, 2]],
        "Não Simétrica 4x4": [[1, 2, 0, 1], [0, 1, 2, 3], [3, 0, 1, 4], [2, 1, 0, 1]],
    }

    for name, A in matrices.items():
        print(f"\n{'=' * 40}\nTeste: {name}\n{'=' * 40}")
        H_A, Q = householder_similarity(A)


if __name__ == "__main__":
    test_householder()
