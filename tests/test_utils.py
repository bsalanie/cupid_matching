import numpy as np

from cupid_matching.utils import find_nest_of, make_XY_K_mat, reshape4_to2


def test_make_XY_K_mat():
    array3 = np.arange(12).reshape((2, 3, 2))
    array2_th = np.zeros((6, 2))
    array2_th[0, :] = array3[0, 0, :]
    array2_th[1, :] = array3[0, 1, :]
    array2_th[2, :] = array3[0, 2, :]
    array2_th[3, :] = array3[1, 0, :]
    array2_th[4, :] = array3[1, 1, :]
    array2_th[5, :] = array3[1, 2, :]
    array2 = make_XY_K_mat(array3)
    assert np.allclose(array2, array2_th)


def test_reshape4_to2():
    array4 = np.arange(48).reshape((3, 2, 2, 4))
    array2_th = np.zeros((6, 8))
    array2_th[0, :] = array4[0, 0, :, :].reshape(8)
    array2_th[1, :] = array4[0, 1, :, :].reshape(8)
    array2_th[2, :] = array4[1, 0, :, :].reshape(8)
    array2_th[3, :] = array4[1, 1, :, :].reshape(8)
    array2_th[4, :] = array4[2, 0, :, :].reshape(8)
    array2_th[5, :] = array4[2, 1, :, :].reshape(8)
    array2 = reshape4_to2(array4)
    assert np.allclose(array2, array2_th)


def test_find_nest_of():
    nest_list = [[1, 3, 5], [6, 2, 4], [7, 9]]
    assert find_nest_of(nest_list, 4) == 1
    assert find_nest_of(nest_list, 8) == -1
