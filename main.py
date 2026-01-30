from utils import generate_system
from methods import *

if __name__ == "__main__":
    A10, b10 = generate_system(4)

    print("Sistema gerado:")
    print("Matriz A:")
    for row in A10:
        print(row)
    print("Vetor b:")
    print(b10)

    print("")
    print("Solução com pivotação parcial:")
    solution_partial = gaussian_elimination_partial(A10, b10)
    print(solution_partial)

    print("")
    print("Solução com pivotação total:")
    solution_total = gaussian_elimination_total(A10, b10)
    print(solution_total)

    print("")
    print("Decomposição de Cholesky:")
    A = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
    L = cholesky_decomposition(A)
    print("Matriz L:", L)
