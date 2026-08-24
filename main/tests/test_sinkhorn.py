"""Sinkhorn correctness: marginals, the entropic limit, and sparse == dense."""

import numpy as np
import pytest

from cellstateadj.cost import build_support
from cellstateadj.sinkhorn import sinkhorn_dense, sinkhorn_sparse
from cellstateadj.utils import uniform_weights


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    Za = rng.standard_normal((40, 3))
    Zb = rng.standard_normal((35, 3)) + 0.3
    return Za, Zb


def test_dense_marginals(toy):
    Za, Zb = toy
    sup, _ = build_support(Za, Zb, dtau=1.0, dense=True)
    C = np.zeros(sup.shape)
    C[sup.rows, sup.cols] = sup.cost
    a, b = uniform_weights(40), uniform_weights(35)
    res = sinkhorn_dense(C, a, b, epsilon=0.1)
    P = res.to_dense()
    assert np.allclose(P.sum(1), a, atol=1e-9)
    assert np.allclose(P.sum(0), b, atol=1e-9)
    assert abs(P.sum() - 1.0) < 1e-10


def test_large_epsilon_gives_independent_coupling(toy):
    """Degeneracy 2: as eps -> inf, P -> a b^T and fingerprints all coincide."""
    Za, Zb = toy
    sup, _ = build_support(Za, Zb, dtau=1.0, dense=True)
    C = np.zeros(sup.shape)
    C[sup.rows, sup.cols] = sup.cost
    a, b = uniform_weights(40), uniform_weights(35)
    P = sinkhorn_dense(C, a, b, epsilon=5e3).to_dense()
    assert np.abs(P - np.outer(a, b)).max() < 1e-6


def test_small_epsilon_is_sharp(toy):
    Za, Zb = toy
    sup, _ = build_support(Za, Zb, dtau=1.0, dense=True)
    C = np.zeros(sup.shape)
    C[sup.rows, sup.cols] = sup.cost
    a, b = uniform_weights(40), uniform_weights(35)
    P_sharp = sinkhorn_dense(C, a, b, epsilon=0.005).to_dense()
    P_soft = sinkhorn_dense(C, a, b, epsilon=1.0).to_dense()
    ent = lambda P: -(P[P > 0] * np.log(P[P > 0])).sum()
    assert ent(P_sharp) < ent(P_soft)


def test_sparse_matches_dense_on_full_support(toy):
    Za, Zb = toy
    sup_d, scale = build_support(Za, Zb, dtau=1.0, dense=True)
    sup_s, _ = build_support(Za, Zb, dtau=1.0, kappa=35, cost_scale=scale)
    a, b = uniform_weights(40), uniform_weights(35)
    C = np.zeros(sup_d.shape)
    C[sup_d.rows, sup_d.cols] = sup_d.cost
    Pd = sinkhorn_dense(C, a, b, 0.1).to_dense()
    Ps = sinkhorn_sparse(sup_s, a, b, 0.1).to_dense()
    assert np.abs(Pd - Ps).max() < 1e-8


def test_symmetrised_support_covers_every_row_and_column(toy):
    Za, Zb = toy
    sup, _ = build_support(Za, Zb, dtau=1.0, kappa=5)
    assert sup.min_row_degree() >= 5
    assert sup.min_col_degree() >= 5


def test_sparse_support_is_feasible_at_moderate_kappa(toy):
    Za, Zb = toy
    sup, _ = build_support(Za, Zb, dtau=1.0, kappa=10)
    a, b = uniform_weights(40), uniform_weights(35)
    res = sinkhorn_sparse(sup, a, b, 0.1, max_iter=5000, tol=1e-12)
    assert res.marginal_error < 1e-8
