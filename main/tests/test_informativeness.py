"""Step 1 diagnostics: the epsilon curve must show both degeneracies."""

import numpy as np

from cellstateadj.config import CouplingConfig
from cellstateadj.informativeness import (
    coupling_information, epsilon_scan, fingerprint_information,
    outgoing_fingerprints,
)
from cellstateadj.cost import build_support
from cellstateadj.sinkhorn import sinkhorn_dense
from cellstateadj.utils import uniform_weights


def _pair(seed=0, n=60, d=3):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((3, d)) * 3
    Za = c[rng.integers(0, 3, n)] + 0.3 * rng.standard_normal((n, d))
    Zb = c[rng.integers(0, 3, n)] + 0.3 * rng.standard_normal((n, d))
    return Za, Zb


def test_information_decreases_with_epsilon():
    Za, Zb = _pair()
    sup, _ = build_support(Za, Zb, 1.0, dense=True)
    C = np.zeros(sup.shape)
    C[sup.rows, sup.cols] = sup.cost
    a, b = uniform_weights(len(Za)), uniform_weights(len(Zb))
    vals = [coupling_information(sinkhorn_dense(C, a, b, e), a, b)
            for e in (0.01, 0.1, 1.0, 10.0, 1000.0)]
    assert all(x >= y - 1e-9 for x, y in zip(vals, vals[1:])), vals
    assert vals[-1] < 1e-4


def test_fingerprint_information_vanishes_at_large_epsilon():
    """Degeneracy 2 measured the way the method actually feels it."""
    Za, Zb = _pair()
    sup, _ = build_support(Za, Zb, 1.0, dense=True)
    C = np.zeros(sup.shape)
    C[sup.rows, sup.cols] = sup.cost
    a, b = uniform_weights(len(Za)), uniform_weights(len(Zb))
    labels = np.random.default_rng(0).integers(0, 5, len(Zb))
    lo = fingerprint_information(outgoing_fingerprints(
        sinkhorn_dense(C, a, b, 0.05), labels, 5), a)
    hi = fingerprint_information(outgoing_fingerprints(
        sinkhorn_dense(C, a, b, 1000.0), labels, 5), a)
    assert lo > 1e-3
    # collapse by orders of magnitude is the claim; the residual is roundoff
    assert hi < 1e-5
    assert hi < lo / 1e4


def test_epsilon_scan_runs_and_recommends_a_window():
    rng = np.random.default_rng(1)
    Z = [rng.standard_normal((50, 3)) + 0.5 * t for t in range(3)]
    scan = epsilon_scan(Z, np.arange(3.0), epsilons=(0.02, 0.1, 0.5, 5.0),
                        cfg=CouplingConfig(support="dense", tol=1e-11),
                        provisional_K=5, n_pairs=200, n_resample=2,
                        verbose=0, seed=0)
    assert scan.metrics["I_cell"].shape == (4, 2)
    assert np.all(np.isfinite(scan.metrics["I_cell"]))
    rec = scan.recommend()
    assert "window" in rec
    # informativeness must be monotone decreasing in epsilon
    curve = scan.mean_curve("I_cell")
    assert all(x >= y - 1e-8 for x, y in zip(curve, curve[1:])), curve
