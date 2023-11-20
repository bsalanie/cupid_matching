from math import isclose

import numpy as np
from pytest import fixture

from cupid_matching.matching_utils import (
    Matching,
    compute_margins,
    get_singles,
    var_divide,
    variance_muhat,
)


@fixture(scope="module")
def _matching_example():
    muxy = np.array([[1.0, 2.0, 3.0], [6.0, 1.0, 9.0]])
    n = np.array([13.0, 37.0])
    m = np.array([12.0, 6.0, 23.5])
    mux0 = np.array([7.0, 21.0])
    mu0y = np.array([5.0, 3.0, 11.5])
    return muxy, mux0, mu0y, n, m


def test_get_singles(_matching_example):
    muxy, mux0, mu0y, n, m = _matching_example
    mux0_th = mux0
    mu0y_th = mu0y
    mux0, mu0y = get_singles(muxy, n, m)
    assert np.allclose(mux0, mux0_th)
    assert np.allclose(mu0y, mu0y_th)


def test_compute_margins(_matching_example):
    muxy, mux0, mu0y, n, m = _matching_example
    n_th = n
    m_th = m
    n, m = compute_margins(muxy, mux0, mu0y)
    assert np.allclose(n, n_th)
    assert np.allclose(m, m_th)


def test_init_matching(_matching_example):
    muxy_th, mux0_th, mu0y_th, n_th, m_th = _matching_example
    mus = Matching(muxy_th, n_th, m_th)
    _, mux0, mu0y, *_ = mus.unpack()
    assert np.allclose(mux0, mux0_th)
    assert np.allclose(mu0y, mu0y_th)


def test_variancematching(_matching_example):
    muxy, mux0_0, mu0y_0, n, m = _matching_example
    mus = Matching(muxy, n, m)
    muxy, mux0, mu0y, n, m = mus.unpack()
    assert np.allclose(mux0, mux0_0)
    assert np.allclose(mu0y, mu0y_0)
    n_h = mus.n_households
    assert isclose(n_h, 69.5)
    varmus = variance_muhat(mus)
    (
        v_xyzt,
        v_xyz0,
        v_xy0t,
        v_x0z0,
        v_x00t,
        v_0y0t,
        v_xyn,
        v_xym,
        v_nn,
        v_nm,
        v_mm,
    ) = varmus.unpack()
    v_xyz0 = varmus.var_xyz0
    muxyr = muxy.ravel()
    v_xyzt_th = np.diag(muxyr) - np.outer(muxyr, muxyr) / n_h
    assert np.allclose(v_xyzt, v_xyzt_th)
    v_xyz0_th = -np.outer(muxyr, mux0) / n_h
    assert np.allclose(v_xyz0, v_xyz0_th)
    v_xy0t_th = -np.outer(muxyr, mu0y) / n_h
    assert np.allclose(v_xy0t, v_xy0t_th)
    v_x0z0_th = np.diag(mux0) - np.outer(mux0, mux0) / n_h
    assert np.allclose(v_x0z0, v_x0z0_th)
    v_x00t_th = -np.outer(mux0, mu0y) / n_h
    assert np.allclose(v_x00t, v_x00t_th)
    v_0y0t_th = np.diag(mu0y) - np.outer(mu0y, mu0y) / n_h
    assert np.allclose(v_0y0t, v_0y0t_th)

    # v3n0_th = -muxy[1, 0] * n[0] / n_h
    # assert isclose(v_xyn[3, 0], v3n0_th)
    v4m1_th = muxy[1, 1] * (1.0 - m[1] / n_h)
    assert isclose(v_xym[4, 1], v4m1_th)
    # vn0m1_th = muxy[0, 1] - n[0] * m[1] / n_h
    # assert isclose(v_nm[0, 1], vn0m1_th)
    # vn0n1_th = -n[0] * n[1] / n_h
    # assert isclose(v_nn[0, 1], vn0n1_th)
    vn1n1_th = n[1] * (1.0 - n[1] / n_h)
    assert isclose(v_nn[1, 1], vn1n1_th)
    vm2m2_th = m[2] * (1.0 - m[2] / n_h)
    assert isclose(v_mm[2, 2], vm2m2_th)


def test_var_allmus(_matching_example):
    muxy, *_, n, m = _matching_example
    mus = Matching(muxy, n, m)
    varmus = variance_muhat(mus)
    v_allmus = varmus.var_allmus
    v_xyz0, v_xy0t = varmus.var_xyz0, varmus.var_xy0t
    assert isclose(v_allmus[5, 7], v_xyz0[5, 1])
    assert isclose(v_allmus[9, 3], v_xy0t[3, 1])


def test_var_munm(_matching_example):
    muxy, *_, n, m = _matching_example
    mus = Matching(muxy, n, m)
    varmus = variance_muhat(mus)
    v_munm = varmus.var_munm
    v_xyn, v_xym = varmus.var_xyn, varmus.var_xym
    assert isclose(v_munm[5, 7], v_xyn[5, 1])
    assert isclose(v_munm[9, 3], v_xym[3, 1])


def test_div_variance(_matching_example):
    muxy, *_, n, m = _matching_example
    mus = Matching(muxy, n, m)
    varmus = variance_muhat(mus)
    vardiv = var_divide(varmus, 10.0)
    assert np.allclose(vardiv.var_xyzt[5, 3], varmus.var_xyzt[5, 3] / 10.0)
