"""Baselines must run through the same machinery as the model."""

import numpy as np

from cellstateadj.baselines import (expression_clustering, induced_from_labels,
                                    membership_from_labels, spectral_coarse_grain)
from cellstateadj.config import CouplingConfig, ModelConfig
from cellstateadj.reference import build_reference_chain


def problem(T=3, n=40, d=3, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((3, d)) * 3
    Z = [c[rng.integers(0, 3, n)] + 0.3 * rng.standard_normal((n, d)) for _ in range(T)]
    chain = build_reference_chain(Z, np.arange(T, dtype=float),
                                  CouplingConfig(epsilon=0.1, support="dense",
                                                 tol=1e-12), verbose=0)
    return chain, Z


def test_membership_from_labels_is_strictly_positive_and_normalised():
    M = membership_from_labels(np.array([0, 1, 2, 0]), K=4)
    assert np.all(M > 0)
    assert np.allclose(M.sum(1), 1.0)


def test_expression_clustering_baseline_gives_a_valid_objective():
    chain, Z = problem()
    labels = expression_clustering(Z, K=3, method="kmeans")
    model = induced_from_labels(chain, Z, labels, K=3,
                                cfg=ModelConfig(K=3, dtype="float64"))
    loss, terms = model.objective()
    assert np.isfinite(float(loss.detach()))
    assert terms.compress >= -1e-9
    Ts, A, B, g = model.induced_transitions()
    assert np.allclose(A[0].sum(1).detach().numpy(), 1.0, atol=1e-8)


def test_spectral_coarse_grain_returns_one_label_array_per_timepoint():
    chain, Z = problem()
    labels = spectral_coarse_grain(chain, K=3)
    assert len(labels) == chain.T
    assert all(len(l) == n for l, n in zip(labels, chain.n_cells))
