"""Regression tests for the analysis/wot_serum runner layer.

These cover the four deterministic fixes that guard scientific correctness:
serum-arm enforcement, destination reservation, K-sweep compatibility, and the
provenance fingerprint.  Run with the repo's pytest:

    python -m pytest analysis/wot_serum/test_wot_serum.py -q
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csa_wot
from csa_wot import (ARM_SPLIT_DAY, compare_fingerprints, fingerprint_hash,
                     reserve_destinations, serum_arm_mask)


# ---------------------------------------------------------------------------
# issue 4 -- the parent AnnData must not be able to smuggle 2i cells in
# ---------------------------------------------------------------------------

def _toy_adata(include_2i: bool, with_arm: bool = True, n_per: int = 6, seed: int = 0):
    """Miniature stand-in for the WOT file: days either side of the split."""
    ad = pytest.importorskip("anndata")
    import pandas as pd

    rng = np.random.default_rng(seed)
    days, arms = [], []
    for d in (7.5, 8.0, 8.5, 9.0):
        if d <= ARM_SPLIT_DAY:
            days += [d] * n_per
            arms += ["shared"] * n_per
        else:
            days += [d] * n_per
            arms += ["serum"] * n_per
            if include_2i:
                days += [d] * n_per
                arms += ["2i"] * n_per
    n = len(days)
    obs = pd.DataFrame({
        "day": np.asarray(days, dtype=float),
        "batch": pd.Categorical(np.tile(["1", "2"], n // 2 + 1)[:n]),
        "cell_sets": ["x"] * n,
    }, index=[f"c{i}" for i in range(n)])
    if with_arm:
        obs["arm"] = pd.Categorical(arms)
    a = ad.AnnData(X=rng.standard_normal((n, 3)).astype("float32"), obs=obs)
    a.obsm["X_model"] = rng.standard_normal((n, 4)).astype("float32")
    return a


def test_serum_arm_mask_drops_2i_and_keeps_the_conditional_trajectory():
    days = np.array([7.0, 8.0, 8.5, 8.5, 9.0])
    arm = np.array(["shared", "shared", "serum", "2i", "serum"])
    assert serum_arm_mask(days, arm).tolist() == [True, True, True, False, True]


def test_serum_arm_mask_drops_post_split_shared_and_pre_split_serum():
    """Only the two intended (day, arm) combinations survive."""
    days = np.array([7.0, 9.0, 7.0])
    arm = np.array(["serum", "shared", "shared"])
    assert serum_arm_mask(days, arm).tolist() == [False, False, True]


def test_load_serum_excludes_2i_from_a_parent_file(tmp_path):
    """A day-only filter would let post-day-8 2i cells through; this must not."""
    a = _toy_adata(include_2i=True)
    p = tmp_path / "parent.h5ad"
    a.write_h5ad(p)

    data, Z, obs = csa_wot.load_serum(str(p), day_min=0.0, day_max=18.0, verbose=0)

    arms = np.concatenate([np.asarray(obs["arm"].reindex(data.obs[t]["index"]).astype(str))
                           for t in range(data.T)])
    assert "2i" not in set(arms), f"2i cells survived: {sorted(set(arms))}"
    # and the serum cells that should be there still are
    assert set(arms) == {"shared", "serum"}
    assert sum(data.n_cells) < a.n_obs      # something really was dropped


def test_load_serum_refuses_a_file_without_an_arm_column(tmp_path):
    """No arm column => refuse, rather than silently falling back to days."""
    a = _toy_adata(include_2i=True, with_arm=False)
    p = tmp_path / "no_arm.h5ad"
    a.write_h5ad(p)
    with pytest.raises(ValueError, match="no obs\\['arm'\\] column"):
        csa_wot.load_serum(str(p), verbose=0)


# ---------------------------------------------------------------------------
# issue 2 -- nothing may be written before every destination is validated
# ---------------------------------------------------------------------------

def test_reserve_destinations_rejects_an_occupied_directory(tmp_path):
    occupied = tmp_path / "fit_main"
    occupied.mkdir()
    (occupied / "summary.json").write_text("{}")
    fresh = tmp_path / "chain.npz"
    with pytest.raises(SystemExit) as e:
        reserve_destinations([str(fresh), str(occupied)])
    assert "fit_main" in str(e.value)


def test_reserve_destinations_does_not_create_or_modify_anything(tmp_path):
    """A failed collision check must leave the previous run untouched."""
    occupied = tmp_path / "fit_main"
    occupied.mkdir()
    (occupied / "summary.json").write_text('{"keep": 1}')
    chain = tmp_path / "reference_chain.npz"

    with pytest.raises(SystemExit):
        reserve_destinations([str(chain), str(occupied)])

    assert not chain.exists(), "reserve must not create the chain it was checking"
    assert json.loads((occupied / "summary.json").read_text()) == {"keep": 1}


def test_reserve_destinations_allows_empty_dirs_and_new_paths(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    reserve_destinations([str(empty), str(tmp_path / "new.npz")])   # must not raise


def test_reserve_destinations_overwrite_permits_occupied(tmp_path):
    occupied = tmp_path / "d"
    occupied.mkdir()
    (occupied / "f").write_text("x")
    reserve_destinations([str(occupied)], overwrite=True)           # must not raise


# ---------------------------------------------------------------------------
# issue 3 -- incompatible K-sweep files must be rejected, not averaged
# ---------------------------------------------------------------------------

def _fp(**over):
    base = {
        "h5ad": "/data/serum.h5ad",
        "representation": {"obsm_key": "X_model", "sha1": "abc123",
                           "n_cells": [10, 10], "n_dims": 4},
        "arm_policy": "shared<=8, serum>8",
        "day_min": 0.0, "day_max": 18.0, "stride": 1,
        "n_per_timepoint": None, "sampling_seed": 0,
        "epsilon": 0.05, "support": "knn", "kappa": 400, "kappa_max": 1000,
        "cost_scale_mode": "global", "coupling_dtype": "float64",
        "lambda_compress": 1.0, "lambda_x": 1.0,
        "optim": {"method": "full_gradient", "direction": "lbfgs", "max_iter": 800,
                  "n_init": 2, "tol_objective": 1e-7, "tol_membership": 1e-5},
        "seed_policy": "seed=0",
    }
    base.update(over)
    return base


def _write_k(directory, K, fp):
    from csa_wot import provenance_block
    rec = {"K": K, "heldout_compress": 1.0 / K, "heldout_expression": 1.0,
           "train_compress": 0.5 / K, "train_expression": 1.0,
           "heldout_compress_sd": 0.01, "heldout_se": 0.01,
           "min_state_mass": 0.05, "k_eff": K * 0.9, "init_ari": 0.9,
           "statuses": ["converged", "converged"], "all_converged": True,
           "Ks_requested": [4, 8], "provenance": provenance_block(fp)}
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"K_{K:03d}.json"), "w") as fh:
        json.dump(rec, fh)


def _reducer():
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "k_reduce", os.path.join(here, "02_k_reduce.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_reducer_rejects_k_files_from_different_epsilons(tmp_path):
    d = tmp_path / "k_sweep"
    _write_k(str(d), 4, _fp())
    _write_k(str(d), 8, _fp(epsilon=0.5))          # stale file from another run
    with pytest.raises(SystemExit) as e:
        _reducer().load_and_validate(str(d))
    msg = str(e.value)
    assert "incompatible configurations" in msg
    assert "epsilon" in msg


def test_reducer_rejects_k_files_from_different_day_ranges(tmp_path):
    d = tmp_path / "k_sweep"
    _write_k(str(d), 4, _fp())
    _write_k(str(d), 8, _fp(day_max=8.0))
    with pytest.raises(SystemExit) as e:
        _reducer().load_and_validate(str(d))
    assert "day_max" in str(e.value)


def test_reducer_rejects_a_different_representation(tmp_path):
    """Same knobs, different underlying X_model content => not comparable."""
    d = tmp_path / "k_sweep"
    _write_k(str(d), 4, _fp())
    _write_k(str(d), 8, _fp(representation={"obsm_key": "X_model", "sha1": "deadbe",
                                            "n_cells": [10, 10], "n_dims": 4}))
    with pytest.raises(SystemExit) as e:
        _reducer().load_and_validate(str(d))
    assert "representation" in str(e.value)


def test_reducer_rejects_files_without_a_fingerprint(tmp_path):
    d = tmp_path / "k_sweep"
    os.makedirs(d)
    with open(d / "K_004.json", "w") as fh:
        json.dump({"K": 4, "heldout_compress": 0.2}, fh)     # pre-fix format
    with pytest.raises(SystemExit, match="no configuration fingerprint"):
        _reducer().load_and_validate(str(d))


def test_reducer_accepts_a_consistent_sweep(tmp_path):
    d = tmp_path / "k_sweep"
    _write_k(str(d), 4, _fp())
    _write_k(str(d), 8, _fp())
    rows, h = _reducer().load_and_validate(str(d))
    assert sorted(r["K"] for r in rows) == [4, 8]
    assert h == fingerprint_hash(_fp())


def test_reducer_rejects_duplicate_K(tmp_path):
    d = tmp_path / "k_sweep"
    _write_k(str(d), 4, _fp())
    os.replace(os.path.join(str(d), "K_004.json"), os.path.join(str(d), "K_04.json"))
    _write_k(str(d), 4, _fp())
    with pytest.raises(SystemExit, match="duplicate K"):
        _reducer().load_and_validate(str(d))


# ---------------------------------------------------------------------------
# the fingerprint itself
# ---------------------------------------------------------------------------

def test_fingerprint_hash_is_stable_and_config_sensitive():
    assert fingerprint_hash(_fp()) == fingerprint_hash(_fp())
    for change in ({"epsilon": 0.1}, {"stride": 2}, {"support": "dense"},
                   {"n_per_timepoint": 500}, {"kappa": 200}, {"day_min": 1.0},
                   {"sampling_seed": 1}, {"lambda_x": 2.0}):
        assert fingerprint_hash(_fp(**change)) != fingerprint_hash(_fp()), change


def test_compare_fingerprints_names_the_offending_field():
    diffs = compare_fingerprints(_fp(), _fp(stride=4))
    assert len(diffs) == 1 and diffs[0].startswith("stride:")


def test_representation_fingerprint_tracks_content_not_just_name():
    rng = np.random.default_rng(0)
    Z1 = [rng.standard_normal((5, 3)), rng.standard_normal((5, 3))]
    Z2 = [z.copy() for z in Z1]
    same = csa_wot.representation_fingerprint(Z1)["sha1"]
    assert csa_wot.representation_fingerprint(Z2)["sha1"] == same
    Z2[0][0, 0] += 1.0
    assert csa_wot.representation_fingerprint(Z2)["sha1"] != same


# ---------------------------------------------------------------------------
# fixed-anchor CMI machinery (ADDITIONAL_EXPERIMENTS Stage 2)
# ---------------------------------------------------------------------------

import csa_anchors as ca


def _toy_fingerprints(n=300, n_anchor=20, K=6, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.dirichlet(np.ones(n_anchor) * 0.4, size=n)
    a = np.full(n, 1.0 / n)
    U = rng.standard_normal((n, K)) * 2.0
    M = np.exp(U); M /= M.sum(1, keepdims=True)
    return F, a, M


def test_cmi_chain_rule_is_exact():
    """I(I;A) = I(Z;A) + I(I;A|Z) must hold to floating point.

    The whole fixed-anchor evaluation rests on this decomposition, because
    ``retained = I(Z;A)/I(I;A)`` is only a fraction if the parts sum to the whole.
    """
    F, a, M = _toy_fingerprints()
    r = ca.state_cmi(M, F, a)
    assert abs(r["decomposition_error"]) < 1e-9, r
    assert abs(r["i_cell_anchor"] - (r["i_state_anchor"] + r["cmi"])) < 1e-9


def test_cmi_fast_form_equals_the_direct_sum():
    """The entropy form must equal the literal sum_ik a_i M_ik KL(f_i || phi_k)."""
    F, a, M = _toy_fingerprints()
    g = M.T @ a
    phi = (M.T @ (a[:, None] * F)) / g[:, None]
    KL = ((F * np.log(np.maximum(F, 1e-300))).sum(1)[:, None]
          - F @ np.log(np.maximum(phi, 1e-300)).T)
    direct = float((a[:, None] * M * KL).sum())
    assert abs(direct - ca.state_cmi(M, F, a)["cmi"]) < 1e-9


def test_cmi_at_K1_equals_total_information_and_retains_nothing():
    """One state can explain nothing, so CMI == I(cell;A) and retained == 0.

    This is the fixed-anchor analogue of Degeneracy 3's K=1 limit -- but note the
    contrast: with a LEARNED neighbour space L_pm goes to 0 at K=1, whereas here
    CMI goes to its MAXIMUM. That inversion is exactly why fixed anchors cannot be
    gamed by collapse.
    """
    F, a, _ = _toy_fingerprints()
    r = ca.state_cmi(np.ones((len(F), 1)), F, a)
    assert abs(r["cmi"] - r["i_cell_anchor"]) < 1e-9
    ret, why = ca.retained_information(r["cmi"], r["i_cell_anchor"])
    assert abs(ret) < 1e-9, (ret, why)


def test_retained_information_refuses_an_uninformative_denominator():
    """A low CMI on an uninformative coupling is not a success -- must be NaN."""
    F, a, M = _toy_fingerprints()
    flat = ca.independence_null(F, a)
    r = ca.state_cmi(M, flat, a)
    assert r["i_cell_anchor"] < 1e-9
    val, why = ca.retained_information(r["cmi"], r["i_cell_anchor"])
    assert np.isnan(val)
    assert "uninformative" in why or "no cell-level" in why


def test_independence_null_has_zero_cell_information():
    F, a, _ = _toy_fingerprints()
    assert ca.cell_information(ca.independence_null(F, a), a) < 1e-9


def test_fixed_anchors_are_frozen_across_assignments():
    """Anchors learned on a train split must reproduce exactly when re-applied."""
    rng = np.random.default_rng(0)
    Z = [rng.standard_normal((120, 5)) for _ in range(3)]
    cen = ca.make_anchors(Z, 8, seed=0)
    A1 = ca.assign_anchors(Z, cen)
    A2 = ca.assign_anchors(Z, cen)
    for x, y in zip(A1, A2):
        assert np.array_equal(x, y)
    # and each row is a distribution
    for x in A1:
        assert np.allclose(x.sum(1), 1.0)


def test_logits_from_memberships_round_trips_through_softmax():
    """Warm starts must reproduce the membership they came from."""
    _, _, M = _toy_fingerprints()
    U = ca.logits_from_memberships([M])[0]
    back = np.exp(U); back /= back.sum(1, keepdims=True)
    assert np.allclose(back, M, atol=1e-7)


def test_logits_from_memberships_survives_saturation():
    """Memberships hit exactly 1.000 in the real fits; log(0) must not appear."""
    M = np.zeros((10, 4)); M[:, 0] = 1.0
    U = ca.logits_from_memberships([M])[0]
    assert np.isfinite(U).all()


def test_gaussian_variant_is_the_sse_objective_at_a_specific_lambda_x():
    """B1 with FROZEN sigma^2 is a reparameterisation, not a new objective.

    L_gauss = SSE/(2 s2) + T*(d/2)*log(2 pi s2) = (d/(2 s2)) * (SSE/d) + const,
    and the constant does not depend on M -- so the argmin is that of the SSE
    objective at lambda_x = d/(2 s2). Guards the claim the ablation rests on.
    """
    import importlib.util as u
    spec = u.spec_from_file_location(
        "abl", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "06_expression_ablation.py"))
    m = u.module_from_spec(spec); spec.loader.exec_module(m)

    d, T, s2 = 30, 22, 4.7333
    assert m.lambda_x_for("none", s2, d, 1.0) == 0.0
    assert m.lambda_x_for("sse", s2, d, 1.0) == 1.0
    assert abs(m.lambda_x_for("gaussian", s2, d, 1.0) - d / (2 * s2)) < 1e-12

    # the two objectives must agree up to the M-independent constant
    L_expr_sse = 3.5                               # = SSE/d for some assignment
    SSE = L_expr_sse * d
    lam = m.lambda_x_for("gaussian", s2, d, 1.0)
    direct = SSE / (2 * s2) + m.gaussian_constant(s2, d, T)
    viaLam = lam * L_expr_sse + m.gaussian_constant(s2, d, T)
    assert abs(direct - viaLam) < 1e-9


# ===========================================================================
# scientific-readiness revision (2026-09-08 review)
# ===========================================================================

# --- issue 1: exact chain/data identity ------------------------------------

class _FakeChain:
    """Minimal stand-in with the attributes validate_chain_identity reads."""
    def __init__(self, Z, tau, feasible=True):
        self.Z = [np.asarray(z, float) for z in Z]
        self.tau = np.asarray(tau, float)
        self.T = len(Z)
        self.n_cells = [len(z) for z in Z]
        self.feasibility_tol = 1e-5
        self._feasible = feasible

    @property
    def feasible(self):
        return self._feasible

    def infeasible_intervals(self):
        return [] if self._feasible else [0]


class _FakeData:
    def __init__(self, Z, tau, ids):
        self.T = len(Z)
        self.n_cells = [len(z) for z in Z]
        self.tau = np.asarray(tau, float)
        self.obs = [{"index": np.asarray(i)} for i in ids]
        self.replicate = [np.array(["1"] * len(z)) for z in Z]


def _identity_fixture(seed=0, n=30, T=3, d=4):
    rng = np.random.default_rng(seed)
    Z = [rng.standard_normal((n, d)) for _ in range(T)]
    tau = np.arange(float(T))
    ids = [[f"t{t}_c{i}" for i in range(n)] for t in range(T)]
    return Z, tau, ids


def test_identity_accepts_the_matching_chain():
    Z, tau, ids = _identity_fixture()
    data = _FakeData(Z, tau, ids)
    ident = ca.identity_payload(data, Z, h5ad="/x.h5ad", day_min=0, day_max=2,
                                stride=1, n_per_timepoint=None, seed=0,
                                arm_policy="serum")
    ca.validate_chain_identity(_FakeChain(Z, tau), data, Z, identity=ident,
                               require_identity_file=False)


def test_identity_rejects_same_counts_but_different_cell_ids():
    """The exact failure the n_cells-only check missed."""
    Z, tau, ids = _identity_fixture()
    data = _FakeData(Z, tau, ids)
    ident = ca.identity_payload(data, Z, h5ad="/x.h5ad", day_min=0, day_max=2,
                               stride=1, n_per_timepoint=None, seed=0,
                               arm_policy="serum")
    other_ids = [[f"OTHER_{i}" for i in range(len(Z[0]))] for _ in range(len(Z))]
    data2 = _FakeData(Z, tau, other_ids)          # SAME counts, SAME Z
    with pytest.raises(SystemExit) as e:
        ca.validate_chain_identity(_FakeChain(Z, tau), data2, Z, identity=ident,
                                   require_identity_file=False)
    assert "cell identities" in str(e.value)


def test_identity_rejects_same_cells_in_a_different_order():
    Z, tau, ids = _identity_fixture()
    data = _FakeData(Z, tau, ids)
    ident = ca.identity_payload(data, Z, h5ad="/x.h5ad", day_min=0, day_max=2,
                               stride=1, n_per_timepoint=None, seed=0,
                               arm_policy="serum")
    perm = np.random.default_rng(1).permutation(len(Z[0]))
    shuffled_ids = [list(np.asarray(i)[perm]) for i in ids]
    data2 = _FakeData(Z, tau, shuffled_ids)
    with pytest.raises(SystemExit) as e:
        ca.validate_chain_identity(_FakeChain(Z, tau), data2, Z, identity=ident,
                                   require_identity_file=False)
    assert "DIFFERENT ORDER" in str(e.value)


def test_identity_rejects_a_different_representation_at_equal_shapes():
    Z, tau, ids = _identity_fixture()
    data = _FakeData(Z, tau, ids)
    Z2 = [z + 0.01 for z in Z]                    # same shapes, different values
    with pytest.raises(SystemExit) as e:
        ca.validate_chain_identity(_FakeChain(Z, tau), data, Z2,
                                   require_identity_file=False)
    assert "representation content hash" in str(e.value)


def test_identity_rejects_an_infeasible_chain():
    Z, tau, ids = _identity_fixture()
    data = _FakeData(Z, tau, ids)
    with pytest.raises(SystemExit, match="INFEASIBLE"):
        ca.validate_chain_identity(_FakeChain(Z, tau, feasible=False), data, Z,
                                   require_identity_file=False)


def test_identity_requires_the_sidecar_by_default(tmp_path):
    Z, tau, ids = _identity_fixture()
    data = _FakeData(Z, tau, ids)
    with pytest.raises(SystemExit, match="identity sidecar"):
        ca.validate_chain_identity(_FakeChain(Z, tau), data, Z,
                                   chain_path=str(tmp_path / "chain.npz"))


def test_check_identity_args_rejects_a_different_day_range():
    Z, tau, ids = _identity_fixture()
    data = _FakeData(Z, tau, ids)
    ident = ca.identity_payload(data, Z, h5ad="/x.h5ad", day_min=8.25, day_max=18,
                               stride=1, n_per_timepoint=None, seed=0,
                               arm_policy="serum")
    with pytest.raises(SystemExit, match="day_max"):
        ca.check_identity_args(ident, h5ad="/x.h5ad", day_min=8.25, day_max=11.0,
                               stride=1, n_per_timepoint=None, seed=0)


# --- issue 3: simplex-aware cluster tendency ------------------------------

def test_hopkins_style_tendency_is_retired():
    with pytest.raises(NotImplementedError):
        ca.cluster_tendency([np.eye(3)])


def test_cluster_stability_negative_control_on_unclustered_simplex():
    """NEGATIVE control: unclustered Dirichlet must NOT look reproducibly clustered."""
    rng = np.random.default_rng(0)
    X = rng.dirichlet(np.ones(20), size=600)
    r = ca.cluster_stability_resampled([X], 2, np.random.default_rng(1), n_rep=6)
    assert r["mean_ari"] < 0.5, r


def test_cluster_stability_positive_control_on_two_clusters():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.dirichlet(np.r_[np.ones(3) * 8, np.ones(17) * 0.1], 300),
                   rng.dirichlet(np.r_[np.ones(17) * 0.1, np.ones(3) * 8], 300)])
    r = ca.cluster_stability_resampled([X], 2, np.random.default_rng(1), n_rep=6)
    assert r["mean_ari"] > 0.9, r


def test_bimodality_separates_the_simplex_controls():
    rng = np.random.default_rng(0)
    Xu = rng.dirichlet(np.ones(20), size=600)
    Xg = np.vstack([rng.dirichlet(np.r_[np.ones(3) * 8, np.ones(17) * 0.1], 300),
                    rng.dirichlet(np.r_[np.ones(17) * 0.1, np.ones(3) * 8], 300)])
    bu = ca.bimodality_clr([Xu])["bimodality_pc1"]
    bg = ca.bimodality_clr([Xg])["bimodality_pc1"]
    assert bu < 0.555 < bg, (bu, bg)


def test_clr_maps_to_the_sum_zero_hyperplane():
    rng = np.random.default_rng(0)
    X = rng.dirichlet(np.ones(6), size=50)
    assert np.allclose(ca.clr(X).sum(1), 0.0, atol=1e-9)


# --- issue 4: expression-correspondence statistics ------------------------

def _three_regimes(seed=0, n=500, kA=20, d=6):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, d))
    F_ind = rng.dirichlet(np.ones(kA) * 0.4, size=n)
    lg = Z @ rng.standard_normal((d, kA)) * 2.0
    F_sm = np.exp(lg); F_sm /= F_sm.sum(1, keepdims=True)
    F_div = F_sm.copy()
    loc = np.argsort(((Z - Z[0]) ** 2).sum(1))[:100]
    F_div[loc] = rng.dirichlet(np.ones(kA) * 0.2, size=len(loc))
    return Z, F_ind, F_sm, F_div


def test_old_permutation_statistic_is_retired():
    with pytest.raises(NotImplementedError):
        ca.local_neighbourhood_permutation(np.eye(3), np.eye(3), np.ones(3) / 3)


def test_concordance_is_flat_when_fingerprints_are_independent_of_expression():
    Z, F_ind, _, _ = _three_regimes()
    r = ca.expression_neighbour_concordance(Z, F_ind, seed=0)
    assert 0.9 < r["ratio"] < 1.1, r


def test_concordance_drops_when_fingerprints_are_smooth_in_expression():
    Z, _, F_sm, _ = _three_regimes()
    r = ca.expression_neighbour_concordance(Z, F_sm, seed=0)
    assert r["ratio"] < 0.7, r


def test_residual_fraction_separates_all_three_regimes():
    """independent ~1, smooth << 1, and local divergence in between."""
    Z, F_ind, F_sm, F_div = _three_regimes()
    ri = ca.expression_predicts_fingerprint(Z, F_ind, seed=0)["residual_fraction"]
    rs = ca.expression_predicts_fingerprint(Z, F_sm, seed=0)["residual_fraction"]
    rd = ca.expression_predicts_fingerprint(Z, F_div, seed=0)["residual_fraction"]
    assert ri > 0.8, ri
    assert rs < 0.5, rs
    assert rs < rd < ri, (rs, rd, ri)


def test_expression_predictor_can_be_grouped_to_block_batch_leakage():
    Z, _, F_sm, _ = _three_regimes()
    g = np.tile(["a", "b"], len(Z) // 2)
    r = ca.expression_predicts_fingerprint(Z, F_sm, seed=0, groups=g)
    assert r["grouped"] is True and r["n_folds"] == 2


# --- issue 6: strict convergence ------------------------------------------

def _ablation_module():
    import importlib.util as u
    spec = u.spec_from_file_location(
        "abl2", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "06_expression_ablation.py"))
    m = u.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class _FakeRes:
    def __init__(self, status="converged", objective=1.0):
        self.status = status
        self.objective = objective


def test_strict_converged_rejects_a_plateau_that_is_not_stationary():
    """The package can report 'converged' on objective plateau alone."""
    m = _ablation_module()
    ok, why = m.strict_converged(_FakeRes(), grad_norm=1.0, kkt=1.0, dM=1.0,
                                 tol_grad=1e-3, tol_kkt=1e-4, tol_dM=1e-4)
    assert not ok
    assert any("grad norm" in w for w in why)
    assert any("KKT" in w for w in why)
    assert any("membership change" in w for w in why)


def test_strict_converged_rejects_a_nonconverged_status():
    m = _ablation_module()
    ok, why = m.strict_converged(_FakeRes(status="max_iter"), 1e-9, 1e-9, 1e-9,
                                 1e-3, 1e-4, 1e-4)
    assert not ok and any("max_iter" in w for w in why)


def test_strict_converged_accepts_a_genuinely_stationary_point():
    m = _ablation_module()
    ok, why = m.strict_converged(_FakeRes(), 1e-9, 1e-11, 1e-9, 1e-3, 1e-4, 1e-4)
    assert ok and why == []


def test_strict_converged_rejects_non_finite_quantities():
    m = _ablation_module()
    ok, _ = m.strict_converged(_FakeRes(objective=float("nan")), 1e-9, 1e-11,
                               1e-9, 1e-3, 1e-4, 1e-4)
    assert not ok
    ok2, _ = m.strict_converged(_FakeRes(), float("inf"), 1e-11, 1e-9,
                                1e-3, 1e-4, 1e-4)
    assert not ok2


def test_heldout_expression_nll_is_finite_and_scales_with_sigma():
    m = _ablation_module()
    rng = np.random.default_rng(0)
    Z = [rng.standard_normal((40, 5)) for _ in range(3)]
    M = [np.eye(4)[rng.integers(0, 4, 40)] for _ in range(3)]
    idx = [np.arange(40) for _ in range(3)]
    a = m._heldout_expression_nll(Z, M, idx, 1.0)
    b = m._heldout_expression_nll(Z, M, idx, 2.0)
    assert np.isfinite(a) and a > 0
    assert abs(b - a / 2) < 1e-9      # NLL scales as 1/(2 sigma^2)
