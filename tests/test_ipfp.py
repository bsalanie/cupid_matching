import numpy as np
from pytest import fixture

from cupid_matching.ipfp_solvers import (
    ipfp_gender_heteroskedastic_solver,
    ipfp_heteroskedastic_solver,
    ipfp_homoskedastic_no_singles_solver,
    ipfp_homoskedastic_solver,
)
from cupid_matching.matching_utils import Matching


@fixture
def _matching_phi():
    X, Y = 4, 3
    muxy = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1], [1, 1, 3]])
    n = np.arange(7, 7 + X)
    m = np.arange(6, 6 + Y)
    mus = Matching(muxy, n, m)
    _, mux0, mu0y, *_ = mus.unpack()
    phi = 2.0 * np.log(muxy) - np.log(mu0y)
    phi -= np.log(mux0).reshape((-1, 1))
    return mus, phi


@fixture
def _matching_phi_no_singles():
    muxy = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1], [1, 1, 3]])
    n = np.sum(muxy, 1)
    m = np.sum(muxy, 0)
    mus = Matching(muxy, n, m, no_singles=True)
    phi = 2.0 * np.log(muxy)
    return mus, phi


@fixture
def _matching_phi_gender_hetero():
    tau = 0.7
    X, Y = 4, 3
    muxy = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1], [1, 1, 3]])
    n = np.arange(7, 7 + X)
    m = np.arange(6, 6 + Y)
    mus = Matching(muxy, n, m)
    _, mux0, mu0y, *_ = mus.unpack()
    phi = (1.0 + tau) * np.log(muxy) - tau * np.log(mu0y)
    phi -= np.log(mux0).reshape((-1, 1))
    return mus, phi, tau


@fixture
def _matching_phi_hetero():
    X, Y = 4, 3
    sigx = np.array([1.2, 0.9, 1.4])
    tauy = np.array([0.8, 0.5, 1.3])
    muxy = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1], [1, 1, 3]])
    n = np.arange(7, 7 + X)
    m = np.arange(6, 6 + Y)
    mus = Matching(muxy, n, m)
    _, mux0, mu0y, *_ = mus.unpack()
    sigx1 = np.concatenate((np.ones(1), sigx))
    sigtau = np.add.outer(sigx1, tauy)
    phi = sigtau * np.log(muxy) - tauy * np.log(mu0y)
    phi -= (sigx1 * np.log(mux0)).reshape((-1, 1))
    return mus, phi, sigx1, tauy


def test_ipfp_homo(_matching_phi):
    mus_th, phi = _matching_phi
    muxy_th, mux0_th, mu0y_th, n_th, m_th = mus_th.unpack()
    mus, *_ = ipfp_homoskedastic_solver(phi, n_th, m_th)
    assert np.allclose(mus.muxy, muxy_th)
    assert np.allclose(mus.mux0, mux0_th)
    assert np.allclose(mus.mu0y, mu0y_th)


def test_ipfp_homo_no_singles(_matching_phi_no_singles):
    mus_th, phi = _matching_phi_no_singles
    muxy_th, *_, n_th, m_th = mus_th.unpack()
    muxy, *_ = ipfp_homoskedastic_no_singles_solver(phi, n_th, m_th)
    assert np.allclose(muxy, muxy_th)


def test_ipfp_gender_hetero(_matching_phi_gender_hetero):
    mus_th, phi, tau = _matching_phi_gender_hetero
    muxy_th, mux0_th, mu0y_th, n_th, m_th = mus_th.unpack()
    mus, *_ = ipfp_gender_heteroskedastic_solver(phi, n_th, m_th, tau=tau)
    assert np.allclose(mus.muxy, muxy_th)
    assert np.allclose(mus.mux0, mux0_th)
    assert np.allclose(mus.mu0y, mu0y_th)


def test_ipfp_hetero(_matching_phi_hetero):
    mus_th, phi, sigx1, tauy = _matching_phi_hetero
    muxy_th, mux0_th, mu0y_th, n_th, m_th = mus_th.unpack()
    mus, *_ = ipfp_heteroskedastic_solver(phi, n_th, m_th, sigma_x=sigx1, tau_y=tauy)
    assert np.allclose(mus.muxy, muxy_th)
    assert np.allclose(mus.mux0, mux0_th)
    assert np.allclose(mus.mu0y, mu0y_th)
