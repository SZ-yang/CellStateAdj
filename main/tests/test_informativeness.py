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


# ---------------------------------------------------------------------------
# regression: interval-level NaN must not be averaged away (issue 1)
# ---------------------------------------------------------------------------

def _scan_with(stability_resample, n_intervals=2, eps=(0.1,)):
    """Minimal EpsilonScanResult carrying only what recommend() reads."""
    import numpy as np
    from cellstateadj.informativeness import EpsilonScanResult

    eps = np.asarray(eps, dtype=float)
    shape = (len(eps), n_intervals)
    ones = np.ones(shape)
    metrics = {
        "I_cell_normalized": ones * 0.5,      # comfortably over min_norm_info
        "I_fingerprint_plus": ones * 0.5,     # comfortably over min_fingerprint_info
        "stability_cost": ones,               # perfect
        "feasible": ones,                     # every plan feasible
        "stability_resample": np.asarray(stability_resample, dtype=float).reshape(shape),
    }
    return EpsilonScanResult(epsilons=eps, intervals=list(range(n_intervals)),
                             metrics=metrics)


def test_recommend_rejects_epsilon_with_an_unevaluated_interval():
    """One interval measured, one NaN => NOT admissible.

    ``mean_curve`` is a nanmean, so [0.95, nan] averages to 0.95 and clears the
    0.8 stability threshold on the strength of half the evidence.  The stated
    policy is that an unevaluated required criterion counts as a failure, so
    this epsilon must be rejected and the gap reported.
    """
    import numpy as np

    scan = _scan_with([[0.95, np.nan]])
    assert np.isclose(scan.mean_curve("stability_resample")[0], 0.95)   # the trap

    rec = scan.recommend()
    assert rec["epsilon_star"] is None, rec
    assert "stability_resample" in rec["unevaluated"]
    # the report names the epsilon and the specific interval that was missing
    assert rec["unevaluated"]["stability_resample"]["per_epsilon"]["0.1"] == [1]


def test_recommend_accepts_when_every_interval_is_evaluated():
    """Same numbers, nothing missing => admissible.  Guards over-rejection."""
    scan = _scan_with([[0.95, 0.9]])
    rec = scan.recommend()
    assert rec["epsilon_star"] == 0.1, rec
    assert rec["unevaluated"] == {}


def test_recommend_rejects_only_the_epsilon_with_the_gap():
    """A NaN at one epsilon must not disqualify a fully-evaluated one."""
    import numpy as np

    scan = _scan_with([[0.95, np.nan], [0.95, 0.9]], eps=(0.1, 0.2))
    rec = scan.recommend()
    assert rec["epsilon_star"] == 0.2, rec
    assert rec["admissible"] == [0.2]
    assert rec["unevaluated"]["stability_resample"]["per_epsilon"] == {"0.1": [1]}


def test_unevaluated_intervals_reports_interval_ids_not_columns():
    """Interval ids come from self.intervals, so a subset scan reports real ids."""
    import numpy as np
    from cellstateadj.informativeness import EpsilonScanResult

    metrics = {"stability_resample": np.array([[0.9, np.nan, 0.9]])}
    scan = EpsilonScanResult(epsilons=np.array([0.1]), intervals=[5, 11, 17],
                             metrics=metrics)
    assert scan.unevaluated_intervals("stability_resample") == {0: [11]}
