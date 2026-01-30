import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.power_methods import power_method, inverse_power_method, shifted_power_method
from utils import norm, print_matrix

def test_power_methods():
    """
    Testa os métodos de potência: regular, inverso e com deslocamento.
    """
    A = [
        [4, 1, 0],
        [1, 3, 1],
        [0, 1, 2]
    ]
    x0 = [1, 1, 1]
    mu = 2.5
    eps = 1e-8

    print("Matriz A:")
    print_matrix(A)
    print("\nVetor inicial x0:", x0)

    # Potência regular
    eigenvalue, eigenvector = power_method(A, x0, eps)
    print("\n=== Potência Regular ===")
    print("Autovalor dominante:", round(eigenvalue, 6))
    print("Autovetor associado:", [round(v, 6) for v in eigenvector])

    # Potência inversa
    eigenvalue_inv, eigenvector_inv = inverse_power_method(A, x0, eps)
    print("\n=== Potência Inversa ===")
    print("Autovalor de menor módulo:", round(eigenvalue_inv, 6))
    print("Autovetor associado:", [round(v, 6) for v in eigenvector_inv])

    # Potência com deslocamento
    eigenvalue_shift, eigenvector_shift = shifted_power_method(A, x0, mu, eps)
    print("\n=== Potência com Deslocamento ===")
    print(f"Autovalor mais próximo de {mu}:", round(eigenvalue_shift, 6))
    print("Autovetor associado:", [round(v, 6) for v in eigenvector_shift])

    # Verificação da norma
    for name, val, vec in [
        ("Regular", eigenvalue, eigenvector),
        ("Inversa", eigenvalue_inv, eigenvector_inv),
        ("Deslocamento", eigenvalue_shift, eigenvector_shift)
    ]:
        residual = [sum(A[i][j] * vec[j] for j in range(len(A))) - val * vec[i] for i in range(len(A))]
        print(f"\nNorma do resíduo ({name}): {round(norm(residual), 8)}")

if __name__ == "__main__":
    test_power_methods()
