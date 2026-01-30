
# Linear Algebra

This repository contains Python implementations of several methods developed as part of a practical study on Computational Linear Algebra.

## Project Structure

```bash
linear_algebra/
│
├── main.py                # Main script for running tests
├── methods/               # Implementations of the algorithms
├── utils/                 # Helper functions
└── examples/              # Test cases and example matrices for each method
```

## Methods

### Linear Equations

- [Gaussian Elimination](methods/gaussian_elimination.py)
- [Gauss-Jordan](methods/gauss_jordan.py)
- [LU Decomposition](methods/lu_decomposition.py)
- [RREF](methods/rref.py)
- [Cholesky Decomposition](methods/cholesky_decomposition.py)
- [Gram-Schmidt](methods/gram_schmidt.py)
- [Least Squares](methods/least_squares.py)
- [Power Methods](methods/power_methods.py)
- [Householder](methods/householder.py)
- [QR Decomposition](methods/qr_decomposition.py)
- [QR Method](methods/qr_eigen.py)
- [SVD](methods/svd.py)
- [SOR](methods/sor.py)
- [Conjugate Gradient](methods/conjugate_gradient.py)

## How to Run

You can import any method in the file `main.py` and run following the correspondent instruction.

```bash
python main.py
```

For specific study cases, you can check the folder [`examples`](examples) and run test cases for each method.

The folder follows the same structure of the methods folder, where each file is related to a method.