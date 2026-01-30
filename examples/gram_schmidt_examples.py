import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.gram_schmidt import extend_to_basis, gram_schmidt


def test_extend_and_orthonormalize():
    """
    Testa a extensão de vetores e a ortonormalização.
    """
    # Conjunto inicial de vetores (2 vetores em R^3)
    vectors = [[1, 1, 0], [0, 1, 1]]
    m = 3

    print("=== Conjunto inicial de vetores ===")
    for v in vectors:
        print(v)
    print()

    # Extende para formar uma base completa
    basis = extend_to_basis(vectors, m)
    print("=== Base completa ===")
    for v in basis:
        print(v)
    print()

    # Aplica Gram-Schmidt para obter base ortonormal
    ortho_basis = gram_schmidt(basis)
    print("=== Base ortonormal ===")
    for v in ortho_basis:
        print([round(x, 4) for x in v])
    print()


if __name__ == "__main__":
    test_extend_and_orthonormalize()
