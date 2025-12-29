from utils import generate_system
from methods import (
    gaussian_elimination_partial,
    gaussian_elimination_total,
    gauss_jordan_partial,
    gauss_jordan_total,
)

if __name__ == "__main__":
    A10, b10 = generate_system(4)

    print("Sistema gerado:")
    print("Matriz A:")
    for row in A10:
        print(row)
    print("Vetor b:")
    print(b10)

    print("Solução com pivotação parcial:")
    solution_partial = gaussian_elimination_partial(A10, b10)
    print(solution_partial)

    print("Solução com pivotação total:")
    solution_total = gaussian_elimination_total(A10, b10)
    print(solution_total)
