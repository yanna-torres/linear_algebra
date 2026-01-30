import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import dot, norm, scalar_mult, vector_sub


def extend_to_basis(vectors, m):
    """
    Estende um conjunto de vetores linearmente independentes
    para formar uma base de R^m.
    """
    basis = [v[:] for v in vectors]

    # Vetores da base canônica
    for i in range(m):
        e = [0.0] * m
        e[i] = 1.0

        # Verifica independência linear de forma simples
        try:
            gram_schmidt(basis + [e])
            basis.append(e)
        except ZeroDivisionError:
            continue

        if len(basis) == m:
            break

    return basis


def gram_schmidt(vectors):
    """
    Aplica o processo de Gram-Schmidt e retorna
    uma base ortonormal.
    """
    ortho = []

    for v in vectors:
        w = v[:]
        for u in ortho:
            proj = scalar_mult(dot(w, u), u)
            w = vector_sub(w, proj)

        w_norm = norm(w)
        if w_norm == 0:
            raise ZeroDivisionError("Vetores linearmente dependentes")

        ortho.append(scalar_mult(1 / w_norm, w))

    return ortho
