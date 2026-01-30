import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.svd import svd

def test_svd_square():
    """
    Testa a decomposição SVD em uma matriz quadrada.
    """
    A = [
        [1, 2, 0],
        [2, 0, 2],
        [0, 2, 1]
    ]

    print("\n=== Matriz Quadrada ===")
    U, Sigma, V = svd(A)


def test_svd_rectangular_m_gt_n():
    """
    Testa SVD em matriz retangular com m > n.
    """
    A = [
        [1, 0],
        [0, 1],
        [1, 1]
    ]

    print("\n=== Matriz Retangular m>n ===")
    U, Sigma, V = svd(A)


def test_svd_rectangular_m_lt_n():
    """
    Testa SVD em matriz retangular com m < n.
    """
    A = [
        [1, 0, 2],
        [0, 1, 1]
    ]

    print("\n=== Matriz Retangular m<n ===")
    U, Sigma, V = svd(A)


if __name__ == "__main__":
    test_svd_square()
    test_svd_rectangular_m_gt_n()
    test_svd_rectangular_m_lt_n()
