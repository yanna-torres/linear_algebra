from ..utils import norm, vector_sub, print_vector


def sor(A, b, x0, omega, tol=1e-8, max_iter=1000):
    """
    Método SOR para resolver Ax = b
    """
    n = len(A)
    x = x0[:]

    for k in range(max_iter):
        x_old = x[:]

        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j < i:
                    sigma += A[i][j] * x[j]
                elif j > i:
                    sigma += A[i][j] * x_old[j]

            x[i] = (1 - omega) * x_old[i] + (omega / A[i][i]) * (b[i] - sigma)

        # critério de parada
        error = norm(vector_sub(x, x_old))
        if error < tol:
            print(f"Convergiu em {k + 1} iterações.")
            print("Solução aproximada:")
            print_vector(x)
            return x

    print("Número máximo de iterações atingido.")
    return x
