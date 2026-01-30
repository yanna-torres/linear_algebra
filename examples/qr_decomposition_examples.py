import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.qr_decomposition import qr_decomposition
from utils import mat_mult, print_matrix


def test_qr():
    """
    Testa a decomposição QR em várias matrizes quadradas.
    """
    matrices = {
        "Matriz 3x3": [[12, -51, 4], [6, 167, -68], [-4, 24, -41]],
        "Matriz 4x4": [[1, 1, 1, 1], [1, 2, 3, 4], [1, 3, 6, 10], [1, 4, 10, 20]],
    }

    for name, A in matrices.items():
        print(f"\n{'=' * 40}\nTeste: {name}\n{'=' * 40}")
        Q, R = qr_decomposition(A)

        print("\nMatriz Q (ortogonal):")
        print_matrix(Q)

        print("\nMatriz R (triangular superior):")
        print_matrix(R)

        print("\nProduto QR (deve ser igual à matriz original):")
        QR = mat_mult(Q, R)
        print_matrix(QR)

        print("Matriz original A:")
        print_matrix(A)


if __name__ == "__main__":
    test_qr()
