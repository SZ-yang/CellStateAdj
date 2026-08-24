"""Stability machinery: state alignment and edge support."""

import numpy as np

from cellstateadj.config import CouplingConfig, ModelConfig, OptimConfig
from cellstateadj.optimize import fit
from cellstateadj.reference import build_reference_chain
from cellstateadj.stability import align_states, edge_support


def test_align_states_recovers_a_permutation():
    mu_ref = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    perm_true = np.array([2, 0, 1])
    mu_other = mu_ref[perm_true] + 0.01
    perm = align_states(mu_ref, mu_other)
    assert list(perm) == list(perm_true)


def _problem(seed=0, T=3, n=40, d=3):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((3, d)) * 3
    Z = [c[rng.integers(0, 3, n)] + 0.3 * rng.standard_normal((n, d)) for _ in range(T)]
    chain = build_reference_chain(Z, np.arange(T, dtype=float),
                                  CouplingConfig(epsilon=0.1, support="dense",
                                                 tol=1e-12), verbose=0)
    return chain, Z


def test_edge_support_is_a_probability_and_is_one_for_identical_fits():
    import torch
    chain, Z = _problem()
    fits, mus = [], []
    for _ in range(3):
        res = fit(chain, Z, ModelConfig(K=3, dtype="float64"),
                  OptimConfig(max_iter=50, verbose=0, seed=0))
        with torch.no_grad():
            M = res.model.memberships()
            g = res.model.state_masses(M)
            mus.append([m.detach().numpy()
                        for m in res.model.expression_prototypes(M, g)])
        fits.append(res)
    rep = edge_support(fits, mus, delta_A=0.05)
    assert rep.n_fits == 3
    for S in rep.edge_support:
        assert np.all(S >= 0) and np.all(S <= 1.0 + 1e-12)
    # identical fits: every edge that exists is supported by all of them
    assert max(S.max() for S in rep.edge_support) == 1.0
    assert "not biological uncertainty" in rep.caveat().lower() or \
           "NOT" in rep.caveat()
