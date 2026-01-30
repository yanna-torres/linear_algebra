import math


def dot(u, v):
    """Produto interno entre dois vetores."""
    return sum(ui * vi for ui, vi in zip(u, v))


def norm(v):
    """Norma euclidiana de um vetor."""
    return math.sqrt(dot(v, v))


def scalar_mult(c, v):
    """Multiplicação de um vetor por um escalar."""
    return [c * vi for vi in v]


def vector_sub(u, v):
    """Subtração de vetores."""
    return [ui - vi for ui, vi in zip(u, v)]


def print_vector(v, precision=4):
    """
    Imprime um vetor com formatação.
    """
    formatted = [f"{x:.{precision}f}" for x in v]
    print("[", "  ".join(formatted), "]")


def vector_add(u, v):
    """Adição de vetores."""
    if len(u) != len(v):
        raise ValueError("Vetores devem ter o mesmo tamanho")
    return [ui + vi for ui, vi in zip(u, v)]
