import random

def generate_system(size):
    """Generates a random system of linear equations Ax = b."""
    A = [[random.uniform(1, 10) for _ in range(size)] for _ in range(size)]
    b = [random.uniform(1, 10) for _ in range(size)]
    return A, b