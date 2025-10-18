import numpy as np

from cupid_matching.model_classes import NestedLogitPrimitives
from cupid_matching.matching_utils import compute_margins


def test_nested_logit_solver_respects_margins():
    Phi = np.array([[0.5, -0.2], [0.7, 0.1]])
    n = np.array([40.0, 60.0])
    m = np.array([55.0, 45.0])
    nests_for_each_x = [[1], [2]]
    nests_for_each_y = [[1], [2]]
    true_alphas = np.array([0.9, 1.1, 0.85, 1.2])

    nested_logit = NestedLogitPrimitives(
        Phi,
        n,
        m,
        nests_for_each_x=nests_for_each_x,
        nests_for_each_y=nests_for_each_y,
        true_alphas=true_alphas,
    )
    mus = nested_logit.ipfp_solve()
    muxy, mux0, mu0y, _, _ = mus.unpack()
    n_sim, m_sim = compute_margins(muxy, mux0, mu0y)

    assert np.allclose(n_sim, n)
    assert np.allclose(m_sim, m)
    assert np.all(muxy > 0.0)
    assert np.all(mux0 > 0.0)
    assert np.all(mu0y > 0.0)
