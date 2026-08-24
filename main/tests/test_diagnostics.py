"""V/G diagnostics, branching counts, and the DAG export."""

import numpy as np

from cellstateadj.config import (CouplingConfig, DiagnosticsConfig, ModelConfig,
                                 OptimConfig)
from cellstateadj.diagnostics import (compute_diagnostics, geometric_null,
                                      nadaraya_watson_crossfit)
from cellstateadj.optimize import fit
from cellstateadj.reference import build_reference_chain


def problem(T=4, n=50, d=3, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((3, d)) * 3
    Z = [c[rng.integers(0, 3, n)] + 0.4 * rng.standard_normal((n, d)) for _ in range(T)]
    chain = build_reference_chain(Z, np.arange(T, dtype=float),
                                  CouplingConfig(epsilon=0.1, support="dense",
                                                 tol=1e-12), verbose=0)
    return chain, Z


def test_diagnostics_shapes_and_ranges():
    chain, Z = problem()
    res = fit(chain, Z, ModelConfig(K=3, dtype="float64"),
              OptimConfig(max_iter=60, verbose=0))
    d = compute_diagnostics(res.model, DiagnosticsConfig(null_n_folds=3, seed=0))
    assert len(d.g) == chain.T
    assert len(d.T_mat) == chain.T - 1
    assert all(np.all(v >= -1e-12) for v in d.V_plus if v is not None)
    assert all(np.all(v >= -1e-12) for v in d.G_plus if v is not None)
    for t, A in enumerate(d.A):
        assert np.allclose(A.sum(1), 1.0, atol=1e-8)
        nch = d.n_child[t]
        assert np.all(nch >= 1.0 - 1e-8)
        assert np.all(nch <= A.shape[1] + 1e-8)


def test_geometric_null_is_smaller_when_fingerprints_are_smooth_in_z():
    """G is low when the fingerprint is a smooth function of expression."""
    rng = np.random.default_rng(0)
    n, K = 300, 3
    Z = rng.standard_normal((n, 2))
    logits = np.stack([Z[:, 0], Z[:, 1], np.zeros(n)], 1) * 2.0
    F = np.exp(logits)
    F /= F.sum(1, keepdims=True)
    a = np.full(n, 1.0 / n)
    M = np.full((n, 1), 1.0)
    g = np.array([1.0])
    cfg = DiagnosticsConfig(null_n_folds=4, seed=0)
    G_smooth = geometric_null(Z, F, a, M, g, cfg)[0]

    perm = rng.permutation(n)          # destroy the relation to z
    G_rough = geometric_null(Z, F[perm], a, M, g, cfg)[0]
    assert G_smooth < G_rough


def test_crossfit_predictor_stays_in_the_simplex():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((80, 2))
    F = rng.dirichlet(np.ones(4), size=80)
    folds = rng.integers(0, 4, 80)
    Fh = nadaraya_watson_crossfit(Z, F, folds, bandwidth=0.7)
    assert np.all(Fh >= 0)
    assert np.allclose(Fh.sum(1), 1.0, atol=1e-10)


def test_event_table_and_dag_edges():
    chain, Z = problem()
    res = fit(chain, Z, ModelConfig(K=3, dtype="float64"),
              OptimConfig(max_iter=60, verbose=0))
    d = compute_diagnostics(res.model, DiagnosticsConfig(geometric_null=False))
    tbl = d.event_table()
    edges = d.dag_edges(min_mass=1e-6)
    assert len(edges) > 0
    assert all(0 <= e["forward"] <= 1 + 1e-9 for e in edges)
    assert len(tbl) > 0


def test_normalized_transition_ratios_are_in_range():
    from cellstateadj.diagnostics import normalized_transition_ratios
    chain, Z = problem()
    res = fit(chain, Z, ModelConfig(K=3, dtype="float64"),
              OptimConfig(max_iter=60, verbose=0))
    r = normalized_transition_ratios(res.model)
    for key in ("R_plus", "R_minus"):
        vals = [v for v in r[key] if np.isfinite(v)]
        assert len(vals) == chain.T - 1
        # conditional information cannot exceed the total it is a share of
        assert all(-1e-9 <= v <= 1.0 + 1e-6 for v in vals), (key, vals)
