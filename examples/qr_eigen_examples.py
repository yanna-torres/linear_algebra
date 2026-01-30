import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.qr_eigen import (
    qr_eigenvalues_symmetric,
    qr_with_householder,
    qr_eigenvalues_general,
    eigenvalues_from_but,
)
from utils import print_matrix


def test_qr_symmetric():
    """
    Testa o método QR para autovalores em matriz simétrica.
    """
    A = [[4, 1, -2, 2], [1, 2, 0, 1], [-2, 0, 3, -2], [2, 1, -2, -1]]

    print("\n=== Matriz Simétrica ===")
    print_matrix(A)

    # QR direto
    eigenvalues, eigenvectors = qr_eigenvalues_symmetric(A)
    print("\nAutovalores (QR direto):", eigenvalues)
    print("\nAutovetores (colunas da matriz):")
    print_matrix(eigenvectors)

    # QR após Householder
    eigenvalues_h, eigenvectors_h = qr_with_householder(A)
    print("\nAutovalores (Householder + QR):", eigenvalues_h)
    print("\nAutovetores (Householder + QR):")
    print_matrix(eigenvectors_h)


def test_qr_general():
    """
    Testa o método QR para matrizes não simétricas.
    """
    A = [[2, 1, 0], [1, 2, 1], [0, 1, 2]]

    print("\n=== Matriz Geral (não simétrica) ===")
    print_matrix(A)

    B = qr_eigenvalues_general(A)
    print("\nMatriz BUT aproximada:")
    print_matrix(B)

    eigenvalues = eigenvalues_from_but(B)
    print("\nAutovalores extraídos da matriz BUT:", eigenvalues)


if __name__ == "__main__":
    test_qr_symmetric()
    test_qr_general()
