import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.conjugate_gradient import conjugate_gradient, conjugate_gradient_general
from methods.sor import sor
from utils import norm, vector_sub


def test_conjugate_gradient():
    """
    Teste do método de Gradientes Conjugados em um sistema linear.
    """
    # Sistema de teste
    A = [[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]]
    b = [15, 10, 10, 10]
    x0 = [0, 0, 0, 0]

    print("\n--- Gradientes Conjugados (sistema simétrico) ---")
    x_cg, steps = conjugate_gradient(A, b, x0)
    print(f"Solução: {x_cg}")
    print(f"Passos realizados: {steps}")

    # Comparação com SOR
    omegas = [1.0, 1.25, 1.5, 1.75, 1.9]
    for omega in omegas:
        print(f"\n--- SOR com ω = {omega} ---")
        x_sor = sor(A, b, x0, omega, tol=1e-10, max_iter=1000)
        error = norm(vector_sub(x_cg, x_sor))
        print(f"Erro em relação ao gradiente conjugado: {error}")


def test_conjugate_gradient_general():
    """
    Teste do método de Gradientes Conjugados para matriz não simétrica.
    """
    A = [[2, 1, 1], [0, 1, -1], [1, 2, 3]]
    b = [4, 1, 7]

    print("\n--- Gradientes Conjugados (sistema não simétrico) ---")
    x_cg_gen, steps = conjugate_gradient_general(A, b)
    print(f"Solução: {x_cg_gen}")
    print(f"Passos realizados: {steps}")


if __name__ == "__main__":
    test_conjugate_gradient()
    test_conjugate_gradient_general()
