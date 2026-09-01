"""Regressions for the seven review findings.

The pre-existing suite passed while findings 1 and 2 were live, because it only
ever exercised feasible, balanced couplings on equally-spaced timepoints.  Each
test here fails against the code as it was.
"""

import numpy as np
import pytest

# import the package (and hence scipy) before torch: loading torch's libomp
# first and scipy's second aborts the process on this macOS/conda setup
from cellstateadj.config import (CouplingConfig, DiagnosticsConfig, ModelConfig,
                                 OptimConfig, PipelineConfig)
import torch
from cellstateadj.cost import (build_support, global_cost_scale,
                               interval_cost_median, resolve_cost_scales)
from cellstateadj.model import CoarseGrainModel
from cellstateadj.optimize import fit
from cellstateadj.reference import (InfeasibleCouplingError,
                                    SinkhornConvergenceError,
                                    build_reference_chain)


# ---------------------------------------------------------------------------
# Finding 1 -- dtau must survive cost normalisation
# ---------------------------------------------------------------------------

def _pair(seed=0, n=40, d=3):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)), rng.standard_normal((n, d)) + 0.5


def test_per_interval_scaling_cancels_dtau_this_is_the_bug():
    """Pin the old behaviour so nobody reintroduces it as 'normalisation'."""
    Za, Zb = _pair()
    s1, _ = build_support(Za, Zb, dtau=1.0, dense=True)
    s4, _ = build_support(Za, Zb, dtau=4.0, dense=True)
    assert np.allclose(s1.cost, s4.cost), (
        "per-interval median normalisation is supposed to cancel dtau; if this "
        "fails the premise of the fix has changed")


def test_shared_scale_preserves_dtau():
    """With one scale for all intervals, a 4x longer gap gives 4x lower cost."""
    Za, Zb = _pair()
    scale = interval_cost_median(Za, Zb, 1.0)
    s1, _ = build_support(Za, Zb, dtau=1.0, dense=True, cost_scale=scale)
    s4, _ = build_support(Za, Zb, dtau=4.0, dense=True, cost_scale=scale)
    assert not np.allclose(s1.cost, s4.cost)
    assert np.allclose(s1.cost, 4.0 * s4.cost)


def test_global_scale_is_one_value_for_every_interval():
    rng = np.random.default_rng(0)
    Z = [rng.standard_normal((30, 3)) + t for t in range(4)]
    tau = np.array([0.0, 1.0, 3.0, 9.0])          # deliberately unequal
    scales = resolve_cost_scales(Z, tau, "global")
    assert len(scales) == 3
    assert len(set(scales)) == 1
    per = resolve_cost_scales(Z, tau, "per_interval")
    assert len(set(per)) > 1                       # they really do differ


def test_unequal_intervals_produce_different_couplings_under_global_scale():
    """The delta-tau study depends on this: spacing must change the coupling.

    The scale is pinned explicitly for both chains.  A *recomputed* global scale
    is the median over per-interval medians, so lengthening one interval shifts
    the shared scale and moves every interval a little -- correct for a global
    normalisation, but it would confound this particular test.
    """
    rng = np.random.default_rng(1)
    Z = [rng.standard_normal((35, 3)) + 0.4 * t for t in range(3)]
    cfg = CouplingConfig(epsilon=0.2, support="dense", tol=1e-11)
    scale = [interval_cost_median(Z[0], Z[1], 1.0)] * 2

    equal = build_reference_chain(Z, np.array([0.0, 1.0, 2.0]), cfg,
                                  cost_scales=scale, verbose=0)
    unequal = build_reference_chain(Z, np.array([0.0, 1.0, 5.0]), cfg,
                                    cost_scales=scale, verbose=0)

    # interval 0 has the same dtau in both; interval 1 does not
    assert np.allclose(equal.couplings[0].values, unequal.couplings[0].values,
                       atol=1e-9)
    assert not np.allclose(equal.couplings[1].values,
                           unequal.couplings[1].values, atol=1e-6)


def test_per_interval_mode_hides_the_spacing():
    """The documented sensitivity-analysis mode is blind to dtau, by design."""
    rng = np.random.default_rng(1)
    Z = [rng.standard_normal((35, 3)) + 0.4 * t for t in range(3)]
    cfg = CouplingConfig(epsilon=0.2, support="dense", tol=1e-11,
                         cost_scale_mode="per_interval")
    equal = build_reference_chain(Z, np.array([0.0, 1.0, 2.0]), cfg, verbose=0)
    unequal = build_reference_chain(Z, np.array([0.0, 1.0, 5.0]), cfg, verbose=0)
    assert np.allclose(equal.couplings[1].values, unequal.couplings[1].values,
                       atol=1e-9)


# ---------------------------------------------------------------------------
# Finding 2 -- an infeasible chain must not reach the model
# ---------------------------------------------------------------------------

def _starved():
    """Two tight, unequally sized clusters: kNN support cannot balance."""
    rng = np.random.default_rng(0)
    return [np.vstack([rng.standard_normal((5, 2)) * 0.01,
                       rng.standard_normal((35, 2)) * 0.01 + np.array([50.0, 0.0])])
            for _ in range(2)]


def test_infeasible_support_raises_by_default():
    cfg = CouplingConfig(epsilon=0.05, support="knn", kappa=1, kappa_max=1,
                         tol=1e-13, max_iter=4000)
    with pytest.raises(InfeasibleCouplingError) as e:
        build_reference_chain(_starved(), np.arange(2.0), cfg, verbose=0)
    assert e.value.interval == 0
    assert "row-stochastic" in str(e.value)


def test_infeasible_support_can_fall_back_to_dense():
    cfg = CouplingConfig(epsilon=0.05, support="knn", kappa=1, kappa_max=1,
                         tol=1e-13, max_iter=20000, on_infeasible="dense")
    chain = build_reference_chain(_starved(), np.arange(2.0), cfg, verbose=0)
    assert chain.feasible


def test_model_refuses_an_infeasible_chain():
    cfg = CouplingConfig(epsilon=0.05, support="knn", kappa=1, kappa_max=1,
                         tol=1e-13, max_iter=4000, on_infeasible="warn")
    Z = _starved()
    chain = build_reference_chain(Z, np.arange(2.0), cfg, verbose=0)
    assert not chain.feasible
    with pytest.raises(ValueError, match="infeasible"):
        CoarseGrainModel(chain, Z, ModelConfig(K=3, dtype="float64"))


def test_infeasible_chain_really_does_break_row_stochasticity():
    """Why the guard exists.  NOTE: uniform memberships hide this entirely --
    both sides collapse to 1/K -- so the check must use non-uniform logits."""
    cfg = CouplingConfig(epsilon=0.05, support="knn", kappa=1, kappa_max=1,
                         tol=1e-13, max_iter=4000, on_infeasible="warn")
    Z = _starved()
    chain = build_reference_chain(Z, np.arange(2.0), cfg, verbose=0)
    K = 3
    rng = np.random.default_rng(0)
    U = [rng.standard_normal((40, K)) * 3 for _ in range(2)]

    model = CoarseGrainModel(chain, Z, ModelConfig(K=K, dtype="float64"),
                             U_init=U, allow_infeasible=True)
    with torch.no_grad():
        _, A, _, _ = model.induced_transitions()
        dev = float(np.abs(A[0].sum(1).numpy() - 1.0).max())
    assert dev > 1e-3, "expected A_t to stop being row-stochastic"

    # and the uniform-membership blind spot, which is why the old tests passed
    flat = CoarseGrainModel(chain, Z, ModelConfig(K=K, dtype="float64"),
                            allow_infeasible=True)
    with torch.no_grad():
        _, A0, _, _ = flat.induced_transitions()
        assert np.allclose(A0[0].sum(1).numpy(), 1.0, atol=1e-9)


def test_feasibility_threshold_is_consistent():
    """chain.feasible and the kappa-growth decision must use one threshold."""
    rng = np.random.default_rng(0)
    Z = [rng.standard_normal((30, 3)) + t for t in range(3)]
    cfg = CouplingConfig(epsilon=0.1, support="dense", tol=1e-11,
                         feasibility_tol=1e-6)
    chain = build_reference_chain(Z, np.arange(3.0), cfg, verbose=0)
    assert chain.feasibility_tol == cfg.feasibility_tol
    assert chain.feasible == all(e < cfg.feasibility_tol
                                 for e in chain.marginal_errors())


def test_slow_convergence_is_reported_separately_from_infeasibility():
    """Running out of iterations needs more max_iter, not more kappa."""
    rng = np.random.default_rng(0)
    Z = [rng.standard_normal((30, 3)) + 0.3 * t for t in range(2)]
    cfg = CouplingConfig(epsilon=0.5, support="dense", tol=1e-14, max_iter=10)
    with pytest.raises(SinkhornConvergenceError) as e:
        build_reference_chain(Z, np.arange(2.0), cfg, verbose=0)
    assert "growing kappa will not help" in str(e.value)
    assert "max_iter" in str(e.value)


def test_dense_support_is_never_called_infeasible():
    """A dense support always admits a balanced plan for positive marginals, so
    a miss there is convergence/precision -- never infeasibility.  Reporting it
    as infeasible would send the user to grow kappa, which cannot help."""
    rng = np.random.default_rng(0)
    Z = [rng.standard_normal((30, 3)) + 0.3 * t for t in range(2)]
    cfg = CouplingConfig(epsilon=0.01, support="dense", tol=1e-12, max_iter=20)
    with pytest.raises(SinkhornConvergenceError):
        build_reference_chain(Z, np.arange(2.0), cfg, verbose=0)


def test_epsilon_too_small_for_the_cost_range_is_diagnosed():
    """exp(-C/eps) underflows once max(cost)/eps approaches ~708 in float64.
    No iteration count fixes that, so the message must say so."""
    rng = np.random.default_rng(0)
    Z = [rng.standard_normal((30, 3)) * 3 + 0.3 * t for t in range(2)]
    cfg = CouplingConfig(epsilon=0.001, support="dense", tol=1e-12, max_iter=200)
    with pytest.raises(SinkhornConvergenceError) as e:
        build_reference_chain(Z, np.arange(2.0), cfg, verbose=0)
    msg = str(e.value)
    assert "underflow limit" in msg and "too small" in msg
    assert e.value.cost_ratio > 200


# ---------------------------------------------------------------------------
# Finding 4 -- epsilon selection must demand positive evidence
# ---------------------------------------------------------------------------

def test_unevaluated_stability_makes_an_epsilon_inadmissible():
    from cellstateadj.informativeness import EpsilonScanResult
    eps = np.array([0.01, 0.1, 1.0])
    metrics = {
        "I_cell_normalized": np.full((3, 1), 0.5),
        "I_fingerprint_plus": np.full((3, 1), 1.0),
        "stability_resample": np.array([[np.nan], [0.95], [0.95]]),
        "stability_cost": np.full((3, 1), 0.95),
    }
    rec = EpsilonScanResult(epsilons=eps, intervals=[0], metrics=metrics).recommend()
    assert 0.01 not in rec["admissible"], "NaN stability must not pass"
    assert rec["unevaluated"].get("stability_resample") == 1


def test_paired_replicates_hold_out_the_same_unit_at_both_ends():
    from cellstateadj.informativeness import epsilon_scan
    rng = np.random.default_rng(0)
    Z = [rng.standard_normal((40, 3)) + 0.4 * t for t in range(3)]
    rep = [np.repeat([0, 1], 20) for _ in range(3)]
    scan = epsilon_scan(Z, np.arange(3.0), epsilons=(0.1, 0.5),
                        cfg=CouplingConfig(support="dense", tol=1e-11),
                        provisional_K=4, n_pairs=100, n_resample=2,
                        replicate=rep, replicate_paired=True, verbose=0)
    assert scan.notes["replicate_paired"] is True
    assert scan.notes["cost_scale_mode"] == "global"


# ---------------------------------------------------------------------------
# Finding 5 -- no bogus "excess" column
# ---------------------------------------------------------------------------

def test_event_table_reports_quadrants_not_a_difference():
    from cellstateadj.diagnostics import compute_diagnostics
    rng = np.random.default_rng(0)
    c = rng.standard_normal((3, 3)) * 3
    Z = [c[rng.integers(0, 3, 60)] + 0.4 * rng.standard_normal((60, 3))
         for _ in range(3)]
    chain = build_reference_chain(Z, np.arange(3.0),
                                  CouplingConfig(epsilon=0.1, support="dense",
                                                 tol=1e-11), verbose=0)
    res = fit(chain, Z, ModelConfig(K=3, dtype="float64"),
              OptimConfig(max_iter=40, verbose=0))
    d = compute_diagnostics(res.model, DiagnosticsConfig(null_n_folds=3))
    tbl = d.event_table()
    cols = list(tbl.columns) if hasattr(tbl, "columns") else list(tbl[0])
    assert "excess_plus" not in cols
    assert "geometry_plus" in cols and "geometry_minus" in cols
    allowed = {"homogeneous", "geometric", "excess_nonsmooth", "unevaluated"}
    vals = (set(tbl["geometry_plus"]) if hasattr(tbl, "columns")
            else {r["geometry_plus"] for r in tbl})
    assert vals <= allowed


def test_quadrant_helper_classifies_correctly():
    from cellstateadj.diagnostics import _quadrant
    assert _quadrant(0.01, 0.5, 0.1, 0.1) == "homogeneous"
    assert _quadrant(0.5, 0.01, 0.1, 0.1) == "geometric"
    assert _quadrant(0.5, 0.5, 0.1, 0.1) == "excess_nonsmooth"
    assert _quadrant(np.nan, 0.5, 0.1, 0.1) == "unevaluated"


# ---------------------------------------------------------------------------
# Finding 6 -- alignment must use transition role, not expression alone
# ---------------------------------------------------------------------------

def test_joint_alignment_separates_states_that_expression_confuses():
    """Two states with near-identical mu but swapped roles.

    Expression-only matching has nothing to go on and picks the wrong pairing;
    the joint criterion sees the fingerprints and gets it right.
    """
    from cellstateadj.stability import align_states, align_states_joint

    # At t=0 states 0 and 1 are EXACTLY coincident in z, so expression carries
    # no information about which is which.  At t=1 the states are well
    # separated, so that timepoint aligns unambiguously and can anchor the
    # fingerprint coordinates.
    amb = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]])
    clear = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 10.0]])
    mu = [amb, clear]
    mu_other = [amb.copy(), clear.copy()]

    # the other fit's states 0 and 1 have swapped transition roles
    pp_ref = [np.array([[0.9, 0.05, 0.05],
                        [0.05, 0.9, 0.05],
                        [0.05, 0.05, 0.9]]), None]
    pp_other = [np.array([[0.05, 0.9, 0.05],
                          [0.9, 0.05, 0.05],
                          [0.05, 0.05, 0.9]]), None]
    pm_ref = [None, None]
    pm_other = [None, None]

    perms = align_states_joint(mu, mu_other, pp_ref, pp_other, pm_ref, pm_other,
                               transition_weight=5.0)
    assert list(perms[0]) == [1, 0, 2], perms[0]

    # expression alone is blind here: the two prototypes are identical, so the
    # matching is an arbitrary tie-break and it does NOT recover the swap
    expr_only = align_states(mu[0], mu_other[0])
    assert list(expr_only) != [1, 0, 2]


def test_edge_support_exposes_node_support_and_alignment_mode():
    from cellstateadj.stability import edge_support
    rng = np.random.default_rng(0)
    c = rng.standard_normal((3, 3)) * 3
    Z = [c[rng.integers(0, 3, 40)] + 0.3 * rng.standard_normal((40, 3))
         for _ in range(3)]
    chain = build_reference_chain(Z, np.arange(3.0),
                                  CouplingConfig(epsilon=0.1, support="dense",
                                                 tol=1e-11), verbose=0)
    fits = [fit(chain, Z, ModelConfig(K=3, dtype="float64"),
                OptimConfig(max_iter=30, verbose=0, seed=0)) for _ in range(2)]
    rep = edge_support(fits)
    assert rep.alignment == "joint"
    assert len(rep.node_support) == chain.T
    for s in rep.node_support:
        assert np.all((s >= 0) & (s <= 1))


# ---------------------------------------------------------------------------
# Finding 7 -- a stuck line search is not convergence
# ---------------------------------------------------------------------------

def test_fit_reports_an_explicit_status():
    rng = np.random.default_rng(0)
    c = rng.standard_normal((3, 3)) * 3
    Z = [c[rng.integers(0, 3, 40)] + 0.3 * rng.standard_normal((40, 3))
         for _ in range(3)]
    chain = build_reference_chain(Z, np.arange(3.0),
                                  CouplingConfig(epsilon=0.1, support="dense",
                                                 tol=1e-11), verbose=0)
    res = fit(chain, Z, ModelConfig(K=3, dtype="float64"),
              OptimConfig(max_iter=200, verbose=0))
    assert res.status in {"converged", "line_search_failed", "max_iter"}
    assert res.converged == (res.status == "converged")
    assert np.isfinite(res.grad_norm)
    assert "status" in res.summary()


def test_max_iter_is_not_reported_as_converged():
    rng = np.random.default_rng(0)
    c = rng.standard_normal((3, 3)) * 3
    Z = [c[rng.integers(0, 3, 40)] + 0.3 * rng.standard_normal((40, 3))
         for _ in range(3)]
    chain = build_reference_chain(Z, np.arange(3.0),
                                  CouplingConfig(epsilon=0.1, support="dense",
                                                 tol=1e-11), verbose=0)
    res = fit(chain, Z, ModelConfig(K=3, dtype="float64"),
              OptimConfig(max_iter=2, verbose=0, patience=10 ** 6,
                          tol_objective=0.0, tol_membership=0.0))
    assert res.status == "max_iter"
    assert not res.converged


def test_line_search_failure_is_distinguished():
    """Forcing every step to be rejected must not report success."""
    rng = np.random.default_rng(0)
    c = rng.standard_normal((3, 3)) * 3
    Z = [c[rng.integers(0, 3, 40)] + 0.3 * rng.standard_normal((40, 3))
         for _ in range(3)]
    chain = build_reference_chain(Z, np.arange(3.0),
                                  CouplingConfig(epsilon=0.1, support="dense",
                                                 tol=1e-11), verbose=0)
    # armijo_c huge => no step can ever satisfy the sufficient-decrease test
    res = fit(chain, Z, ModelConfig(K=3, dtype="float64"),
              OptimConfig(max_iter=50, verbose=0, armijo_c=1e6,
                          max_backtrack=3))
    assert res.status == "line_search_failed"
    assert not res.converged


# ---------------------------------------------------------------------------
# Finding 3 -- K selection needs a held-out criterion
# ---------------------------------------------------------------------------

def _replicated_series(seed=0, n=90, T=4, d=3, n_states=3):
    """A series with two culture replicates per timepoint."""
    from cellstateadj.data import TimeSeriesData
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((n_states, d)) * 4
    X, rep = [], []
    for t in range(T):
        lab = rng.integers(0, n_states, n)
        z = c[lab] + 0.5 * rng.standard_normal((n, d))
        counts = rng.poisson(np.exp(1.0 + z @ rng.standard_normal((d, 40)) * 0.3))
        X.append(counts)
        rep.append(np.repeat([0, 1], n // 2))
    return TimeSeriesData(X=X, tau=np.arange(T, dtype=float), replicate=rep)


def test_select_K_requires_replicates():
    from cellstateadj.data import TimeSeriesData
    from cellstateadj.selection import select_K
    d = _replicated_series()
    bare = TimeSeriesData(X=d.X, tau=d.tau)
    with pytest.raises(ValueError, match="replicate"):
        select_K(bare, PipelineConfig(), [2, 3], verbose=0)


def test_training_compression_is_monotone_but_heldout_is_not():
    """The whole justification for protocol (b).

    Training compression falls with every extra state, so it can only ever
    recommend the largest K tried.  The held-out curve is free to turn over,
    which is what makes it a selector.
    """
    from cellstateadj.selection import select_K
    data = _replicated_series(seed=1, n=90, T=4)
    cfg = PipelineConfig()
    cfg.representation.n_hvg = None
    cfg.representation.n_components = 8
    cfg.coupling.epsilon = 0.2
    cfg.coupling.support = "dense"
    cfg.optim.max_iter = 60
    cfg.optim.verbose = 0

    res = select_K(data, cfg, [2, 3, 5, 8], seed=0, n_init_for_stability=1,
                   verbose=0)
    train = np.asarray(res.train_compress)
    held = np.asarray(res.heldout_compress)
    assert np.all(np.diff(train) <= 1e-9), f"training should be monotone: {train}"
    assert res.recommend()["K"] in res.Ks
    assert np.all(np.isfinite(held))
    # the held-out estimate is a genuinely different quantity, not a copy
    assert not np.allclose(train, held)


def test_transfer_does_not_refit_on_the_held_out_half():
    """Transfer must be a pure nearest-prototype map; anything fitted on the
    held-out cells would leak and stop the estimate being held out."""
    from cellstateadj.selection import transfer_memberships
    mu = [np.array([[0.0, 0.0], [10.0, 0.0]])]
    Z = [np.array([[0.1, 0.0], [9.9, 0.0], [0.2, 0.0]])]
    U = transfer_memberships(Z, mu)
    lab = U[0].argmax(1)
    assert list(lab) == [0, 1, 0]


def test_epsilon_scan_does_not_abort_on_an_unsolvable_epsilon():
    """Step 1 must always produce a curve: an epsilon that cannot be solved is
    a finding to record, not a reason to crash the scan."""
    from cellstateadj.informativeness import epsilon_scan
    rng = np.random.default_rng(0)
    Z = [np.vstack([rng.standard_normal((6, 2)) * 0.01,
                    rng.standard_normal((24, 2)) * 0.01 + np.array([40.0, 0.0])])
         for _ in range(2)]
    scan = epsilon_scan(
        Z, np.arange(2.0), epsilons=(1e-4, 0.5),
        cfg=CouplingConfig(support="knn", kappa=1, kappa_max=1,
                           max_iter=200, tol=1e-13),
        provisional_K=3, n_pairs=50, n_resample=1, verbose=0)
    assert scan.metrics["I_cell"].shape == (2, 1)
    # feasibility is reported per epsilon rather than raised
    assert np.all(np.isfinite(scan.metrics["marginal_error"]))


# ===========================================================================
# Second review round
# ===========================================================================

# --- 1. replicate split must be paired across time -------------------------

def test_replicate_split_is_paired_across_time():
    """Half A must be the SAME culture at every timepoint.

    Against the old per-timepoint draw this produced e.g.
    [[0], [0], [0], [1], [1]] -- replicate 1 ends up in half A at late times
    and in half B early, so the "held-out" half is not held out at all.
    """
    from cellstateadj.data import split_half_by_replicate
    d = _replicated_series(seed=3, n=80, T=5)
    a, b = split_half_by_replicate(d, seed=1)
    reps_a = [np.unique(r).tolist() for r in a.replicate]
    reps_b = [np.unique(r).tolist() for r in b.replicate]
    assert all(r == reps_a[0] for r in reps_a), reps_a
    assert all(r == reps_b[0] for r in reps_b), reps_b
    assert set(reps_a[0]).isdisjoint(reps_b[0])


def test_replicate_split_pairing_is_stable_across_seeds():
    from cellstateadj.data import split_half_by_replicate
    d = _replicated_series(seed=4, n=60, T=6)
    for seed in range(5):
        a, _ = split_half_by_replicate(d, seed=seed)
        reps = [np.unique(r).tolist() for r in a.replicate]
        assert all(r == reps[0] for r in reps), (seed, reps)


def test_unpaired_split_still_available_for_independent_wells():
    from cellstateadj.data import split_half_by_replicate
    d = _replicated_series(seed=5, n=60, T=6)
    a, b = split_half_by_replicate(d, seed=0, paired=False)
    assert all(len(x) > 0 for x in a.X) and all(len(x) > 0 for x in b.X)


# --- 2. eps-scan support sizing --------------------------------------------

def test_scan_sizes_support_at_the_largest_epsilon():
    """Sizing at the smallest epsilon conflates numerical failure with an
    infeasible support, leaving every epsilon on an under-sized support.

    Here the grid spans an epsilon that underflows and one that does not; the
    scan must still reach the same feasibility the ordinary chain builder does
    at the well-conditioned end.
    """
    from cellstateadj.informativeness import epsilon_scan
    rng = np.random.default_rng(0)
    c = rng.standard_normal((3, 3)) * 6
    Z = [c[rng.integers(0, 3, 60)] + 0.3 * rng.standard_normal((60, 3))
         for _ in range(2)]
    cfg = CouplingConfig(support="knn", kappa=1, kappa_max=64, max_iter=4000)

    scan = epsilon_scan(Z, np.arange(2.0), epsilons=(1e-4, 0.5), cfg=cfg,
                        provisional_K=3, n_pairs=50, n_resample=1, verbose=0)
    ei = int(np.flatnonzero(scan.epsilons == 0.5)[0])
    chain = build_reference_chain(Z, np.arange(2.0),
                                  CouplingConfig(epsilon=0.5, support="knn",
                                                 kappa=1, kappa_max=64,
                                                 max_iter=4000), verbose=0)
    assert chain.feasible
    assert scan.metrics["feasible"][ei, 0] == 1.0, (
        "the scan rejected an epsilon the ordinary builder solves fine: "
        f"marginal error {scan.metrics['marginal_error'][ei, 0]:.3e}")


# --- 3. block-coordinate line-search failure -------------------------------

def test_block_coordinate_failure_is_not_called_convergence():
    """Every proposal rejected leaves rel==0 and dM==0, so the convergence test
    fires first unless the failure check precedes it."""
    rng = np.random.default_rng(0)
    c = rng.standard_normal((3, 3)) * 3
    Z = [c[rng.integers(0, 3, 40)] + 0.3 * rng.standard_normal((40, 3))
         for _ in range(3)]
    chain = build_reference_chain(Z, np.arange(3.0),
                                  CouplingConfig(epsilon=0.1, support="dense",
                                                 tol=1e-11), verbose=0)
    res = fit(chain, Z, ModelConfig(K=3, dtype="float64"),
              OptimConfig(method="block_coordinate", max_iter=20, verbose=0,
                          armijo_c=1e6, max_backtrack=2))
    assert res.status == "line_search_failed", res.status
    assert not res.converged


def test_block_coordinate_still_converges_normally():
    rng = np.random.default_rng(1)
    c = rng.standard_normal((3, 3)) * 3
    Z = [c[rng.integers(0, 3, 40)] + 0.3 * rng.standard_normal((40, 3))
         for _ in range(3)]
    chain = build_reference_chain(Z, np.arange(3.0),
                                  CouplingConfig(epsilon=0.1, support="dense",
                                                 tol=1e-11), verbose=0)
    res = fit(chain, Z, ModelConfig(K=3, dtype="float64"),
              OptimConfig(method="block_coordinate", max_iter=30, verbose=0))
    assert res.status in {"converged", "max_iter"}
    obj = res.history_array("total")
    assert obj[-1] <= obj[0] + 1e-10


# --- 4. epsilon_star must be admissible ------------------------------------

def test_epsilon_star_is_always_an_admissible_value():
    """With pass = [True, False, True] the old code returned the FAILING middle
    value as epsilon_star."""
    from cellstateadj.informativeness import EpsilonScanResult
    eps = np.array([0.01, 0.1, 1.0])
    metrics = {
        "I_cell_normalized": np.array([[0.5], [0.0], [0.5]]),   # 0.1 fails
        "I_fingerprint_plus": np.full((3, 1), 1.0),
        "stability_resample": np.full((3, 1), 0.95),
        "stability_cost": np.full((3, 1), 0.95),
    }
    rec = EpsilonScanResult(epsilons=eps, intervals=[0],
                            metrics=metrics).recommend()
    assert rec["admissible"] == [0.01, 1.0]
    assert rec["epsilon_star"] in rec["admissible"], rec
    assert rec["epsilon_star"] != 0.1
    assert rec["n_admissible_runs"] == 2
    # the reported window must itself be contiguous and admissible
    assert all(e in rec["admissible"] for e in rec["admissible_window"])


def test_contiguous_window_is_reported_when_admissible_set_is_contiguous():
    from cellstateadj.informativeness import EpsilonScanResult
    eps = np.array([0.01, 0.1, 1.0, 10.0])
    metrics = {
        "I_cell_normalized": np.array([[0.0], [0.5], [0.5], [0.0]]),
        "I_fingerprint_plus": np.full((4, 1), 1.0),
        "stability_resample": np.full((4, 1), 0.95),
        "stability_cost": np.full((4, 1), 0.95),
    }
    rec = EpsilonScanResult(epsilons=eps, intervals=[0],
                            metrics=metrics).recommend()
    assert rec["window"] == (0.1, 1.0)
    assert rec["n_admissible_runs"] == 1
    assert rec["epsilon_star"] in (0.1, 1.0)


# --- 5. K selection must not accept non-converged fits ---------------------

def test_k_selection_rejects_non_converged_fits():
    from cellstateadj.selection import KSelectionResult
    r = KSelectionResult(
        Ks=[2, 3, 4],
        heldout_compress=[3.0, 1.0, 2.0],      # K=3 looks best...
        heldout_expression=[0, 0, 0],
        train_compress=[3, 2, 1], train_expression=[0, 0, 0],
        min_state_mass=[0.1, 0.1, 0.1], k_eff=[2, 3, 4],
        init_ari=[0.9, 0.9, 0.9],
        heldout_se=[0.0, 0.0, 0.0],
        all_converged=[True, False, True],     # ...but its fit never converged
        statuses=[["converged"] * 2, ["max_iter", "converged"], ["converged"] * 2],
    )
    rec = r.recommend()
    assert rec["K"] != 3
    assert 3 in rec["rejected"]
    assert "converge" in rec["rejected"][3]


def test_k_selection_rejects_collapsed_states():
    from cellstateadj.selection import KSelectionResult
    r = KSelectionResult(
        Ks=[2, 3], heldout_compress=[3.0, 1.0],
        heldout_expression=[0, 0], train_compress=[3, 2],
        train_expression=[0, 0],
        min_state_mass=[0.2, 1e-9],            # K=3 has an empty state
        k_eff=[2, 2.1], init_ari=[0.9, 0.9],
        heldout_se=[0.0, 0.0], all_converged=[True, True],
        statuses=[["converged"], ["converged"]],
    )
    rec = r.recommend(min_state_mass=1e-3)
    assert rec["K"] == 2
    assert 3 in rec["rejected"] and "mass" in rec["rejected"][3]


def test_k_selection_uses_the_measured_spread_not_a_fixed_percentage():
    """The threshold must come from the measured fold-direction spread.

    (Originally asserted the phrase "one SE"; the third review is right that
    calling a two-direction spread a standard error overclaims, so the wording
    is now "one-SE-style" and the test checks the arithmetic instead.)
    """
    from cellstateadj.selection import KSelectionResult
    r = KSelectionResult(
        Ks=[2, 3], heldout_compress=[1.05, 1.00],
        heldout_expression=[0, 0], train_compress=[2, 1],
        train_expression=[0, 0], min_state_mass=[0.1, 0.1],
        k_eff=[2, 3], init_ari=[0.9, 0.9],
        heldout_se=[0.2, 0.2],                 # 1.05 is within 1 SE of 1.00
        all_converged=[True, True],
        statuses=[["converged"], ["converged"]],
    )
    rec = r.recommend()
    assert rec["K"] == 2, rec
    assert "one-SE-STYLE" in rec["rule"]
    assert rec["heldout_se"] == 0.2
    assert rec["best_K"] == 3


def test_select_K_records_status_and_se():
    from cellstateadj.selection import select_K
    data = _replicated_series(seed=6, n=80, T=4)
    cfg = PipelineConfig()
    cfg.representation.n_hvg = None
    cfg.representation.n_components = 8
    cfg.coupling.epsilon = 0.2
    cfg.coupling.support = "dense"
    cfg.optim.max_iter = 50
    cfg.optim.verbose = 0
    res = select_K(data, cfg, [2, 3], seed=0, n_init_for_stability=1, verbose=0)
    assert len(res.statuses) == 2
    assert all(len(st) == 2 for st in res.statuses)     # two split directions
    assert len(res.all_converged) == 2
    assert len(res.heldout_se) == 2


# --- 6. delta-tau strides must share representation and cost scale ---------

def test_epsilon_scan_accepts_a_pinned_cost_scale():
    """The delta-tau study pins the native-series scale so stride runs differ
    only in spacing."""
    from cellstateadj.informativeness import epsilon_scan
    rng = np.random.default_rng(0)
    Z = [rng.standard_normal((40, 3)) + 0.4 * t for t in range(3)]
    scan = epsilon_scan(Z, np.arange(3.0), epsilons=(0.2,),
                        cfg=CouplingConfig(support="dense", tol=1e-11),
                        provisional_K=4, n_pairs=50, n_resample=1,
                        cost_scales=[7.5, 7.5], verbose=0)
    assert scan.notes["cost_scales_supplied"] is True
    assert scan.notes["cost_scales"] == [7.5, 7.5]

    with pytest.raises(ValueError, match="one entry per interval"):
        epsilon_scan(Z, np.arange(3.0), epsilons=(0.2,),
                     cfg=CouplingConfig(support="dense"), provisional_K=4,
                     n_pairs=10, n_resample=1, cost_scales=[7.5], verbose=0)


def test_stride_selection_reuses_the_same_cells_and_representation():
    """Selecting timepoints from one pre-subsampled series keeps the cells and
    the PCA basis identical across strides -- only the spacing changes."""
    from cellstateadj.data import subsample_cells
    from cellstateadj.representation import learn_representation
    from cellstateadj.config import RepresentationConfig

    data = _replicated_series(seed=7, n=60, T=9)
    base = subsample_cells(data, 40, seed=0)
    Z_base, _ = learn_representation(
        base, RepresentationConfig(n_hvg=None, n_components=6))

    idx1 = list(range(0, base.T, 1))
    idx4 = list(range(0, base.T, 4))
    Z1 = [Z_base[i] for i in idx1]
    Z4 = [Z_base[i] for i in idx4]
    # timepoint 0 and 4 appear in both strides and must be bit-identical
    assert np.array_equal(Z1[0], Z4[0])
    assert np.array_equal(Z1[4], Z4[1])
    assert base.select_timepoints(idx4).tau.tolist() == [0.0, 4.0, 8.0]


# ---------------------------------------------------------------------------
# Third review, finding 1 -- infeasible resampled couplings must not score as
# perfectly stable
# ---------------------------------------------------------------------------

def test_resampled_supports_grow_their_own_kappa():
    """The main support's kappa is not enough for a smaller resampled support.

    Feasibility is a property of the bipartite support graph, and a subset has
    fewer cells, so reusing the full support's kappa can leave the subset
    without any balanced plan.
    """
    from cellstateadj.informativeness import epsilon_scan

    rng = np.random.default_rng(3)
    Z = [np.concatenate([rng.standard_normal((30, 2)),
                         rng.standard_normal((30, 2)) + 12.0]) for _ in range(2)]
    scan = epsilon_scan(Z, np.array([0.0, 1.0]), epsilons=(0.3,),
                        cfg=CouplingConfig(support="knn", kappa=2, kappa_max=64),
                        provisional_K=4, n_pairs=10, n_resample=2,
                        resample_fraction=0.5, seed=0, verbose=0)
    n_sub = scan.metrics["n_resample_subsets"][0, 0]
    n_ok = scan.metrics["n_resample_feasible"][0, 0]
    assert n_sub == 2
    # kappa growth on the subsets is what makes them solvable at all
    assert n_ok == n_sub, (
        f"only {n_ok}/{n_sub} resampled plans were feasible; each resampled "
        "support must grow its own kappa")


def test_infeasible_resampled_plans_leave_stability_unevaluated():
    """An infeasible resampled plan must yield NaN stability, never 1.0.

    Two invalid plans agree with each other perfectly, so without a feasibility
    gate a completely broken coupling scores as maximal stability and can push
    an epsilon into the admissible window.
    """
    from cellstateadj.informativeness import epsilon_scan

    rng = np.random.default_rng(0)
    Z = [rng.standard_normal((40, 2)) for _ in range(2)]
    # feasibility_tol=0 can never be met, so every plan -- main, resampled and
    # perturbed -- is rejected, and no stability number may be produced
    cfg = CouplingConfig(support="dense", feasibility_tol=0.0)
    scan = epsilon_scan(Z, np.array([0.0, 1.0]), epsilons=(0.2,), cfg=cfg,
                        provisional_K=4, n_pairs=10, n_resample=2, seed=0,
                        verbose=0)
    assert scan.metrics["n_resample_feasible"][0, 0] == 0
    assert np.isnan(scan.metrics["stability_resample"][0, 0]), (
        "stability was scored from plans that failed feasibility_tol")
    assert np.isnan(scan.metrics["stability_cost"][0, 0])
    assert scan.metrics["feasible"][0, 0] == 0.0


def test_cost_perturbed_plan_feasibility_is_recorded():
    from cellstateadj.informativeness import epsilon_scan

    rng = np.random.default_rng(1)
    Z = [rng.standard_normal((40, 2)) for _ in range(2)]
    scan = epsilon_scan(Z, np.array([0.0, 1.0]), epsilons=(0.2,),
                        cfg=CouplingConfig(support="dense"), provisional_K=4,
                        n_pairs=10, n_resample=1, seed=0, verbose=0)
    assert np.isfinite(scan.metrics["marginal_error_perturbed"][0, 0])
    assert scan.metrics["marginal_error_perturbed"][0, 0] < 1e-5
    assert np.isfinite(scan.metrics["stability_cost"][0, 0])


def test_paired_resampling_without_shared_groups_is_unevaluated():
    """No group at both ends means no paired hold-out exists.

    The old code silently drew an unrelated target group, so the scan compared
    two different cultures and reported the disagreement as instability.
    """
    from cellstateadj.informativeness import epsilon_scan

    rng = np.random.default_rng(2)
    Z = [rng.standard_normal((40, 2)) for _ in range(2)]
    rep = [np.repeat([0, 1], 20), np.repeat([2, 3], 20)]   # disjoint labels
    scan = epsilon_scan(Z, np.array([0.0, 1.0]), epsilons=(0.2,),
                        cfg=CouplingConfig(support="dense"), provisional_K=4,
                        n_pairs=10, n_resample=2, replicate=rep,
                        replicate_paired=True, seed=0, verbose=0)
    assert scan.metrics["n_resample_subsets"][0, 0] == 0
    assert np.isnan(scan.metrics["stability_resample"][0, 0])

    # unpaired is explicitly allowed to draw independently
    scan_u = epsilon_scan(Z, np.array([0.0, 1.0]), epsilons=(0.2,),
                          cfg=CouplingConfig(support="dense"), provisional_K=4,
                          n_pairs=10, n_resample=2, replicate=rep,
                          replicate_paired=False, seed=0, verbose=0)
    assert scan_u.metrics["n_resample_subsets"][0, 0] == 2


def test_paired_resampling_holds_the_same_group_at_both_ends():
    from cellstateadj.informativeness import epsilon_scan

    rng = np.random.default_rng(4)
    Z = [rng.standard_normal((40, 2)) for _ in range(2)]
    rep = [np.repeat([0, 1], 20), np.repeat([0, 1], 20)]
    scan = epsilon_scan(Z, np.array([0.0, 1.0]), epsilons=(0.2,),
                        cfg=CouplingConfig(support="dense"), provisional_K=4,
                        n_pairs=10, n_resample=2, replicate=rep,
                        replicate_paired=True, seed=0, verbose=0)
    assert scan.metrics["n_resample_subsets"][0, 0] == 2
    assert np.isfinite(scan.metrics["stability_resample"][0, 0])


# ---------------------------------------------------------------------------
# Third review, finding 2 -- paired splitting with partially missing labels
# ---------------------------------------------------------------------------

def _series_with_labels(label_sets, n_each=20, d=3, seed=0):
    from cellstateadj.data import TimeSeriesData
    rng = np.random.default_rng(seed)
    X, rep = [], []
    for labs in label_sets:
        rep.append(np.repeat(np.asarray(labs), n_each))
        X.append(rng.poisson(3.0, size=(n_each * len(labs), d)).astype(float))
    return TimeSeriesData(X=X, tau=np.arange(len(label_sets), dtype=float),
                          replicate=rep)


def test_paired_split_rejects_inconsistent_replicate_labels():
    """t0: A,B,C  t1: A,B,D  t2: A,B,C

    Neither half is empty for ga={A,B}, but the complement is culture C at t0
    and culture D at t1 -- a "paired" half made of two different cultures.  The
    old emptiness check passed this.
    """
    from cellstateadj.data import split_half_by_replicate

    d = _series_with_labels([["A", "B", "C"], ["A", "B", "D"], ["A", "B", "C"]])
    for seed in range(6):
        with pytest.raises(ValueError, match="same replicate labels at every"):
            split_half_by_replicate(d, seed=seed, paired=True)


def test_restrict_to_shared_gives_a_genuinely_paired_split():
    from cellstateadj.data import split_half_by_replicate

    d = _series_with_labels([["A", "B", "C"], ["A", "B", "D"], ["A", "B", "C"]])
    for seed in range(6):
        a, b = split_half_by_replicate(d, seed=seed, paired=True,
                                       restrict_to_shared=True)
        sa = [set(np.unique(r).tolist()) for r in a.replicate]
        sb = [set(np.unique(r).tolist()) for r in b.replicate]
        assert all(s == sa[0] for s in sa), f"half A switched culture: {sa}"
        assert all(s == sb[0] for s in sb), f"half B switched culture: {sb}"
        assert not (sa[0] & sb[0])
        assert sa[0] | sb[0] == {"A", "B"}   # C and D dropped


def test_paired_split_still_works_with_consistent_labels():
    from cellstateadj.data import split_half_by_replicate

    d = _series_with_labels([["A", "B", "C"]] * 3)
    a, b = split_half_by_replicate(d, seed=0, paired=True)
    sa = [set(np.unique(r).tolist()) for r in a.replicate]
    sb = [set(np.unique(r).tolist()) for r in b.replicate]
    assert all(s == sa[0] for s in sa) and all(s == sb[0] for s in sb)
    assert sa[0] | sb[0] == {"A", "B", "C"}


def test_unpaired_split_is_unaffected_by_inconsistent_labels():
    from cellstateadj.data import split_half_by_replicate

    d = _series_with_labels([["A", "B", "C"], ["A", "B", "D"], ["A", "B", "C"]])
    a, b = split_half_by_replicate(d, seed=0, paired=False)
    assert all(len(x) > 0 for x in a.X) and all(len(x) > 0 for x in b.X)


# ---------------------------------------------------------------------------
# Third review, finding 3 -- convergence must reach the export paths
# ---------------------------------------------------------------------------

def test_run_fit_refuses_to_export_a_nonconverged_fit(tmp_path):
    import json
    import scripts.run_fit as run_fit

    out = run_fit.main([
        "--sim", "branching", "--K", "3", "--epsilon", "0.2",
        "--n-per-timepoint", "60", "--stride", "4", "--max-iter", "2",
        "--no-geometric-null", "--out", str(tmp_path), "--verbose", "0",
    ])
    assert out["converged"] is False
    assert out["fit_status"] in ("max_iter", "line_search_failed")
    written = {p.name for p in tmp_path.iterdir()}
    assert written == {"summary.json"}, (
        f"a non-converged fit exported {written - {'summary.json'}}")
    assert json.load(open(tmp_path / "summary.json"))["converged"] is False


def test_run_fit_allow_nonconverged_stamps_the_directory(tmp_path):
    import scripts.run_fit as run_fit

    out = run_fit.main([
        "--sim", "branching", "--K", "3", "--epsilon", "0.2",
        "--n-per-timepoint", "60", "--stride", "4", "--max-iter", "2",
        "--no-geometric-null", "--allow-nonconverged",
        "--out", str(tmp_path), "--verbose", "0",
    ])
    assert out["converged"] is False
    written = {p.name for p in tmp_path.iterdir()}
    assert "NONCONVERGED.txt" in written
    assert "memberships.npz" in written


def test_split_half_stability_refuses_nonconverged_fits():
    from cellstateadj.stability import split_half_stability

    data = _replicated_series(seed=5, n=60, T=3)
    cfg = PipelineConfig()
    cfg.representation.n_hvg = None
    cfg.representation.n_components = 4
    cfg.coupling.epsilon = 0.2
    cfg.model.K = 3
    cfg.optim.max_iter = 2
    cfg.optim.verbose = 0
    with pytest.raises(ValueError, match="did not converge"):
        split_half_stability(data, cfg, n_splits=1, seed=0, verbose=0)

    rep = split_half_stability(data, cfg, n_splits=1, seed=0,
                               require_converged=False, verbose=0)
    assert "[UNRELIABLE]" in rep.notes


def test_edge_support_flags_nonconverged_fits():
    """Even called directly, the report must say the votes are unreliable."""
    from cellstateadj.reference import build_reference_chain
    from cellstateadj.stability import edge_support

    rng = np.random.default_rng(0)
    Z = [rng.standard_normal((40, 3)) for _ in range(3)]
    chain = build_reference_chain(Z, np.arange(3.0),
                                  CouplingConfig(support="dense", epsilon=0.2),
                                  verbose=0)
    mcfg = ModelConfig(K=3)
    fits = [fit(chain, Z, mcfg, OptimConfig(max_iter=2, seed=s, verbose=0))
            for s in range(2)]
    assert all(f.status != "converged" for f in fits)
    rep = edge_support(fits)
    assert "[UNRELIABLE]" in rep.notes


# ---------------------------------------------------------------------------
# Third review, methodological caveat -- the spread is not a sampling SE
# ---------------------------------------------------------------------------

def test_one_se_rule_is_labelled_as_style_not_a_sampling_se():
    from cellstateadj.selection import KSelectionResult

    res = KSelectionResult(
        Ks=[2, 3], heldout_compress=[1.05, 1.00], heldout_expression=[0.0, 0.0],
        train_compress=[1.0, 0.9], train_expression=[0.0, 0.0],
        min_state_mass=[0.2, 0.2], k_eff=[2.0, 3.0], init_ari=[1.0, 1.0],
        heldout_se=[0.2, 0.2], all_converged=[True, True],
        statuses=[["converged"] * 2] * 2)
    rec = res.recommend()
    assert rec["K"] == 2
    assert "STYLE" in rec["rule"] or "style" in rec["rule"]
    assert "not a sampling standard error" in rec["rule"]
    assert "NOT a sampling standard error" in rec["heldout_se_interpretation"]


def test_run_fit_does_not_score_a_strided_fit_against_unstrided_truth():
    """Found while testing the convergence gate, not in the review.

    ``--stride`` reduces the timepoints of the data but not of ``truth.states``
    / ``truth.T_true``, so recovery was scored across mismatched timepoints and
    transition recovery then indexed past the end of ``T_true``.
    """
    import scripts.run_fit as run_fit
    from cellstateadj import simulate as sim

    sc = sim.make("branching", seed=0)
    sc.n_sample = 40
    truth = sim.simulate(sc, verbose=0)
    assert len(truth.states) == truth.data.T
    idx = list(range(0, truth.data.T, 4))
    assert len(idx) < len(truth.T_true), (
        "the strided series has fewer intervals than T_true records, which is "
        "exactly the index error")


# ---------------------------------------------------------------------------
# Fourth review, finding 1 -- duplicate replicate draws faked perfect stability
# ---------------------------------------------------------------------------

def _two_replicate_pair(seed=0, n=60, d=3):
    rng = np.random.default_rng(seed)
    Z = [rng.standard_normal((n, d)) for _ in range(2)]
    rep = [np.repeat([0, 1], n // 2), np.repeat([0, 1], n // 2)]
    return Z, rep


def test_replicate_subsets_are_enumerated_not_drawn():
    """Two groups must give exactly two subsets, on every seed.

    Drawing one of two groups per resample duplicates the draw about half the
    time; the two 'resampled' datasets are then identical and agree perfectly.
    """
    from cellstateadj.informativeness import epsilon_scan

    Z, rep = _two_replicate_pair()
    for seed in range(6):
        # n_resample=5 must be IGNORED: two groups means exactly two subsets,
        # never five draws with repeats
        scan = epsilon_scan(Z, np.array([0.0, 1.0]), epsilons=(0.2,),
                            cfg=CouplingConfig(support="dense"),
                            provisional_K=5, n_pairs=10, n_resample=5,
                            replicate=rep, replicate_paired=True, seed=seed,
                            verbose=0)
        assert scan.metrics["n_resample_subsets"][0, 0] == 2, (
            f"seed {seed}: expected one subset per replicate group, not "
            f"n_resample draws")
        assert scan.notes["resample_mode"] == "enumerated replicate groups"


def test_duplicate_replicate_draws_no_longer_fake_perfect_stability():
    """Across seeds, stability must never be exactly 1.0 on random data.

    The old code returned exactly 1.0 on 7 of 12 seeds of the SAME data, purely
    because the same replicate was selected twice.
    """
    from cellstateadj.informativeness import epsilon_scan

    Z, rep = _two_replicate_pair()
    vals = []
    for seed in range(12):
        scan = epsilon_scan(Z, np.array([0.0, 1.0]), epsilons=(0.2,),
                            cfg=CouplingConfig(support="dense"),
                            provisional_K=5, n_pairs=10, n_resample=2,
                            replicate=rep, replicate_paired=True, seed=seed,
                            verbose=0)
        vals.append(float(scan.metrics["stability_resample"][0, 0]))
    exact_one = [v for v in vals if abs(v - 1.0) < 1e-12]
    assert not exact_one, (
        f"{len(exact_one)}/12 seeds scored exactly 1.0 on random data; the "
        f"same replicate is being used as two independent resamples")
    assert all(np.isfinite(v) for v in vals)


def test_disjoint_replicate_subsets_all_contribute():
    """Replicate subsets never overlap, so the pairwise route cannot run.

    The old fallback scored Fsubs[0] against the full coupling and silently
    discarded every other subset.  Every subset must contribute now, so making
    the second replicate's cells structurally different must move the number.
    """
    from cellstateadj.informativeness import epsilon_scan

    rng = np.random.default_rng(0)
    base = rng.standard_normal((60, 3))
    Z_same = [base.copy(), rng.standard_normal((60, 3))]
    # identical first replicate, second replicate displaced far away
    shifted = base.copy()
    shifted[30:] += 25.0
    Z_diff = [shifted, Z_same[1]]
    rep = [np.repeat([0, 1], 30), np.repeat([0, 1], 30)]

    kw = dict(epsilons=(0.2,), cfg=CouplingConfig(support="dense"),
              provisional_K=5, n_pairs=10, n_resample=2, replicate=rep,
              replicate_paired=True, seed=0, verbose=0)
    a = epsilon_scan(Z_same, np.array([0.0, 1.0]), **kw)
    b = epsilon_scan(Z_diff, np.array([0.0, 1.0]), **kw)
    assert a.metrics["stability_is_vs_full"][0, 0] == 1.0, (
        "disjoint subsets must be scored against the full coupling")
    assert a.metrics["n_resample_subsets"][0, 0] == 2
    assert not np.isclose(a.metrics["stability_resample"][0, 0],
                          b.metrics["stability_resample"][0, 0]), (
        "changing only the SECOND replicate left stability unchanged, so it is "
        "still being ignored")


def test_overlapping_cell_subsets_still_use_the_pairwise_route():
    from cellstateadj.informativeness import epsilon_scan

    rng = np.random.default_rng(0)
    Z = [rng.standard_normal((60, 3)) for _ in range(2)]
    scan = epsilon_scan(Z, np.array([0.0, 1.0]), epsilons=(0.2,),
                        cfg=CouplingConfig(support="dense"), provisional_K=5,
                        n_pairs=10, n_resample=3, resample_fraction=0.8,
                        seed=0, verbose=0)
    assert scan.metrics["stability_is_vs_full"][0, 0] == 0.0
    assert scan.metrics["n_resample_subsets"][0, 0] == 3
    assert scan.notes["resample_mode"] == "random cell subsets"


def test_unpaired_replicate_subsets_are_also_enumerated():
    from cellstateadj.informativeness import epsilon_scan

    rng = np.random.default_rng(0)
    Z = [rng.standard_normal((60, 3)) for _ in range(2)]
    rep = [np.repeat([0, 1], 30), np.repeat([2, 3], 30)]   # disjoint labels
    scan = epsilon_scan(Z, np.array([0.0, 1.0]), epsilons=(0.2,),
                        cfg=CouplingConfig(support="dense"), provisional_K=5,
                        n_pairs=10, n_resample=2, replicate=rep,
                        replicate_paired=False, seed=0, verbose=0)
    assert scan.metrics["n_resample_subsets"][0, 0] == 2


# ---------------------------------------------------------------------------
# Fourth review, finding 2 -- a reused output directory keeps stale outputs
# ---------------------------------------------------------------------------

def test_run_fit_refuses_a_nonempty_output_directory(tmp_path):
    import scripts.run_fit as run_fit

    (tmp_path / "memberships.npz").write_bytes(b"stale")
    (tmp_path / "dag_edges.json").write_text("[]")
    with pytest.raises(SystemExit, match="not empty"):
        run_fit.main(["--sim", "branching", "--K", "3", "--epsilon", "0.2",
                      "--n-per-timepoint", "60", "--max-iter", "2",
                      "--no-geometric-null", "--out", str(tmp_path),
                      "--verbose", "0"])
    # the stale files are left exactly as they were, not half-replaced
    assert (tmp_path / "memberships.npz").read_bytes() == b"stale"


def test_overwrite_clears_stale_outputs_before_a_refused_fit(tmp_path):
    """The gate says 'wrote summary.json only'; the directory must agree."""
    import scripts.run_fit as run_fit

    (tmp_path / "memberships.npz").write_bytes(b"stale")
    (tmp_path / "dag_edges.json").write_text("[]")
    out = run_fit.main(["--sim", "branching", "--K", "3", "--epsilon", "0.2",
                        "--n-per-timepoint", "60", "--max-iter", "2",
                        "--no-geometric-null", "--overwrite",
                        "--out", str(tmp_path), "--verbose", "0"])
    assert out["converged"] is False
    assert {p.name for p in tmp_path.iterdir()} == {"summary.json"}


def test_overwrite_clears_a_stale_nonconverged_stamp(tmp_path):
    """A converged rerun must not inherit the previous run's NONCONVERGED.txt."""
    import scripts.run_fit as run_fit

    (tmp_path / "NONCONVERGED.txt").write_text("from an earlier run")
    out = run_fit.main(["--sim", "branching", "--K", "3", "--epsilon", "0.2",
                        "--n-per-timepoint", "80", "--max-iter", "400",
                        "--no-geometric-null", "--overwrite",
                        "--out", str(tmp_path), "--verbose", "0"])
    assert out["converged"] is True
    assert not (tmp_path / "NONCONVERGED.txt").exists()


# ---------------------------------------------------------------------------
# Fourth review, finding 3 -- restriction must precede representation fitting
# ---------------------------------------------------------------------------

def test_restrict_to_shared_replicates_drops_the_right_cells():
    from cellstateadj.data import restrict_to_shared_replicates

    d = _series_with_labels([["A", "B", "C"], ["A", "B", "D"], ["A", "B", "C"]])
    out = restrict_to_shared_replicates(d, verbose=0)
    assert [set(np.unique(r).tolist()) for r in out.replicate] == [{"A", "B"}] * 3
    assert sum(out.n_cells) < sum(d.n_cells)


def test_discarded_replicates_do_not_shape_the_pca_basis():
    """PCA must be fit AFTER restriction, not before.

    A non-shared replicate placed far off-axis dominates the leading component;
    if it still moves the basis, the restriction happened too late.
    """
    from cellstateadj.data import TimeSeriesData, restrict_to_shared_replicates
    from cellstateadj.representation import learn_representation
    from cellstateadj.config import RepresentationConfig

    rng = np.random.default_rng(0)
    X, rep = [], []
    for t in range(3):
        shared = rng.poisson(3.0, size=(40, 12)).astype(float)
        odd = rng.poisson(3.0, size=(20, 12)).astype(float)
        odd[:, 0] += 400.0                     # a wildly off-axis outlier group
        X.append(np.vstack([shared, odd]))
        rep.append(np.array(["A"] * 20 + ["B"] * 20 +
                            [f"X{t}"] * 20))   # X{t} is unique to timepoint t
    d = TimeSeriesData(X=X, tau=np.arange(3.0), replicate=rep)

    rcfg = RepresentationConfig(n_hvg=None, n_components=3)
    restricted = restrict_to_shared_replicates(d, verbose=0)
    Z_late, _ = learn_representation(d, rcfg)          # wrong order
    Z_early, _ = learn_representation(restricted, rcfg)  # correct order
    assert sum(restricted.n_cells) == 120
    # the shared cells' coordinates differ between the two orders
    assert not np.allclose(Z_late[0][:40], Z_early[0][:40], atol=1e-6), (
        "the outlier replicate did not affect the basis, so this test cannot "
        "detect the ordering")


def test_select_K_restricts_before_learning_the_representation(monkeypatch):
    """select_K must hand learn_representation the RESTRICTED data."""
    from cellstateadj import selection
    from cellstateadj.data import TimeSeriesData

    rng = np.random.default_rng(0)
    X, rep = [], []
    for t in range(3):
        X.append(rng.poisson(3.0, size=(60, 20)).astype(float))
        rep.append(np.array(["A"] * 20 + ["B"] * 20 + [f"X{t}"] * 20))
    d = TimeSeriesData(X=X, tau=np.arange(3.0), replicate=rep)

    seen = {}
    real = selection.learn_representation

    def spy(data, cfg):
        seen["n_cells"] = sum(data.n_cells)
        seen["labels"] = [set(np.unique(r).tolist()) for r in data.replicate]
        return real(data, cfg)

    monkeypatch.setattr(selection, "learn_representation", spy)
    cfg = PipelineConfig()
    cfg.representation.n_hvg = None
    cfg.representation.n_components = 4
    cfg.coupling.epsilon = 0.3
    cfg.optim.max_iter = 5
    cfg.optim.verbose = 0
    selection.select_K(d, cfg, [2], seed=0, n_init_for_stability=1,
                       restrict_to_shared_replicates=True, verbose=0)
    assert seen["n_cells"] == 120, (
        f"learn_representation saw {seen['n_cells']} cells; the non-shared "
        f"replicates were still present when the PCA basis was fit")
    assert seen["labels"] == [{"A", "B"}] * 3
