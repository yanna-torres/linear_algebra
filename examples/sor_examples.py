import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.sor import sor


def test_sor():
    """
    Teste do método SOR em um sistema linear.
    """
    # Sistema de teste
    A = [[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]]
    b = [15, 10, 10, 10]
    x0 = [0, 0, 0, 0]

    omegas = [0.5, 1.0, 1.25, 1.5, 1.75, 1.9]

    for omega in omegas:
        print(f"\n--- Teste com ω = {omega} ---")
        x = sor(A, b, x0, omega, tol=1e-8, max_iter=1000)
        print("Resultado final:", x)


if __name__ == "__main__":
    test_sor()
