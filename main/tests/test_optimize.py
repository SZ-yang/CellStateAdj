"""Optimiser behaviour: line search really is monotone, and it converges."""

import numpy as np
import pytest
import torch

from cellstateadj.config import CouplingConfig, ModelConfig, OptimConfig
from cellstateadj.optimize import fit, initialize_logits
from cellstateadj.reference import build_reference_chain


def small_problem(T=4, n=40, d=3, seed=0, eps=0.1):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((3, d)) * 3
    Z, labels = [], []
    for t in range(T):
        lab = rng.integers(0, 3, size=n)
        Z.append(centers[lab] + 0.4 * rng.standard_normal((n, d)) + 0.2 * t)
        labels.append(lab)
    tau = np.arange(T, dtype=float)
    chain = build_reference_chain(Z, tau, CouplingConfig(epsilon=eps, support="dense",
                                                         tol=1e-12), verbose=0)
    return chain, Z, labels


def test_line_search_is_monotone_with_fingerprint_terms():
    chain, Z, _ = small_problem()
    cfg = ModelConfig(K=4, dtype="float64", lambda_plus=1.0, lambda_minus=1.0)
    res = fit(chain, Z, cfg, OptimConfig(max_iter=25, verbose=0, seed=0))
    obj = res.history_array("total")
    assert np.all(np.diff(obj) <= 1e-10), obj
    assert res.monotone


def test_converges_and_reports_instrumentation():
    chain, Z, _ = small_problem()
    cfg = ModelConfig(K=4, dtype="float64")
    res = fit(chain, Z, cfg, OptimConfig(max_iter=200, verbose=0))
    assert res.converged
    for key in ("k_eff_mean", "g_min", "floor_fraction", "mean_V_plus"):
        assert key in res.history[-1]
    assert 1.0 <= res.terms.k_eff[0] <= cfg.K + 1e-9


def test_block_coordinate_also_decreases():
    chain, Z, _ = small_problem(T=3, n=30)
    cfg = ModelConfig(K=3, dtype="float64", lambda_plus=0.5, lambda_minus=0.5)
    res = fit(chain, Z, cfg,
              OptimConfig(method="block_coordinate", max_iter=8, damping=0.7,
                          verbose=0))
    obj = res.history_array("total")
    assert obj[-1] <= obj[0] + 1e-10


def test_recovers_well_separated_clusters():
    chain, Z, labels = small_problem(T=4, n=60, seed=2)
    cfg = ModelConfig(K=3, dtype="float64", lambda_compress=1.0, lambda_x=1.0)
    res = fit(chain, Z, cfg, OptimConfig(max_iter=200, verbose=0))
    from sklearn.metrics import adjusted_rand_score
    aris = [adjusted_rand_score(l, m.argmax(1)) for l, m in zip(labels, res.M)]
    assert np.mean(aris) > 0.9, aris


def test_turning_on_fingerprint_terms_changes_the_solution():
    """Handoff step 4, check (ii)."""
    chain, Z, _ = small_problem(T=4, n=50, seed=7)
    base = fit(chain, Z, ModelConfig(K=4, dtype="float64"),
               OptimConfig(max_iter=150, verbose=0))
    withfp = fit(chain, Z,
                 ModelConfig(K=4, dtype="float64", lambda_plus=5.0, lambda_minus=5.0),
                 OptimConfig(max_iter=150, verbose=0))
    from cellstateadj.diagnostics import membership_sensitivity
    s = membership_sensitivity(base.M, withfp.M)
    assert s["mean_l1_membership_change"] > 1e-3, s


def test_multiple_initialisations_report_agreement():
    chain, Z, _ = small_problem(T=3, n=40)
    res = fit(chain, Z, ModelConfig(K=3, dtype="float64"),
              OptimConfig(max_iter=60, n_init=3, verbose=0))
    assert len(res.restarts) == 3
    assert all("mean_ari_to_others" in r for r in res.restarts)
