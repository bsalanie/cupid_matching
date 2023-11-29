"""This module contains some utility programs used by the package."""

from typing import Final, TypeAlias

import numpy as np
from bs_python_utils.bsutils import bs_error_abort

Nest: TypeAlias = list[int]
NestsList: TypeAlias = list[Nest]

# for numerical derivatives
_EPS: Final = 1e-6
_TWO_EPS: Final = 2.0 * _EPS


def make_XY_K_mat(xyk_array: np.ndarray) -> np.ndarray:
    """Reshapes an (X,Y,K) array to an (XY,K) matrix.

    Args:
        xyk_array: an (X, Y, K) array of bases

    Returns:
        the same,  (XY, K)-reshaped
    """
    X, Y, K = xyk_array.shape
    XY = X * Y
    xy_k_mat = np.zeros((XY, K))
    for k in range(K):
        xy_k_mat[:, k] = xyk_array[:, :, k].ravel()
    return xy_k_mat


def reshape4_to2(array4: np.ndarray) -> np.ndarray:
    """Reshapes an array (X,Y,Z,T) to a matrix (XY,ZT).

    Args:
        array4: an (X, Y, Z, T) array

    Returns:
        the same,  (XY, ZT)-reshaped
    """
    if array4.ndim != 4:
        bs_error_abort(f"array4 should have 4 dimensions not {array4.ndim}")
    X, Y, Z, T = array4.shape
    XY, ZT = X * Y, Z * T
    array2 = np.zeros((XY, ZT))
    xy = 0
    for x in range(X):
        for y in range(Y):
            array2[xy, :] = array4[x, y, :, :].ravel()
            xy += 1
    return array2


def change_indices(nests: NestsList):
    """subtracts 1 from the indices within the nest structure

    Args:
        nests: the nest structure

    Returns:
        a similar list
    """
    return [[nest_i - 1 for nest_i in nest] for nest in nests]


def find_nest_of(nests: NestsList, y: int) -> int:
    """find the index of the nest that contains y, or return -1

    Args:
        nests: a nest structure
        y: the type we are looking for

    Returns:
        the nest of y, or -1 if not found
    """
    for i_n, nest in enumerate(nests):
        if y in nest:
            return i_n
    return -1  # if not found
