import numpy as np
from typing import Union

Array = Union[np.ndarray, list]


def check_var(a: float) -> bool:
    """
    Checks if the variable a is within the range [0, 1].
    """
    return 0 <= a <= 1


def impl(a: Array, b: Array) -> np.ndarray:
    """
    Gödel implication: 1 if a <= b else b.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(a <= b, 1.0, b)


def equ(u: Array, v: Array) -> np.ndarray:
    """
    Biequivalence: max(min(u, v), min(1 - u, 1 - v)).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return np.maximum(np.minimum(u, v), np.minimum(1 - u, 1 - v))


def mmp(A: Array, x: Array) -> np.ndarray:
    """
    Max-min product: mmp(A, x)_i = max_j min(A_ij, x_j).
    """
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)

    if A.ndim == 1:
        if A.shape[0] != x.shape[0]:
            raise ValueError("The size of A must match the size of x.")
        return np.max(np.minimum(A, x))

    elif A.ndim == 2:
        if A.shape[1] != x.shape[0]:
            raise ValueError("The number of columns in A must match the size of x.")
        return np.max(np.minimum(A, x), axis=1)

    else:
        raise ValueError("A must be either a 1D or 2D array.")


def mip(A: Array, x: Array) -> np.ndarray:
    """
    Min-implication product: mip(A, x)_i = min_j impl(A_ij, x_j).
    """
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)

    if A.ndim == 1:
        if A.shape[0] != x.shape[0]:
            raise ValueError("The size of A must match the size of x.")
        return np.min(impl(A, x))

    elif A.ndim == 2:
        if A.shape[1] != x.shape[0]:
            raise ValueError("The number of columns in A must match the size of x.")
        return np.min(impl(A, x), axis=1)

    else:
        raise ValueError("A must be either a 1D or 2D array.")


if __name__ == "__main__":

    # Just to check that the equ function behaves as showed in the article
    u = np.linspace(0, 1, 11)
    assert np.allclose(equ(u, 1), u), "equ(u, 1) should equal u"
    assert np.allclose(equ(u, 0), 1 - u), "equ(u, 0) should equal 1 - u"
    print("equ(u, 1) = u and equ(u, 0) = 1 - u verified.")
