"""Fixed-anchor fingerprint geometry and CMI evaluation.

Why this module exists
----------------------
In the joint objective, the future variable at time ``t`` is the LEARNED state
``Z[t+1]``, so the optimiser can lower ``L_plus`` by collapsing or redefining
``Z[t+1]`` rather than by grouping cells that genuinely share a future.  That is
Degeneracy 3, and the 2026-09-07 grids showed the optimiser finds it: ``L_pm``
fell from ~3.0 to ~0.0001 while compression degraded 3x.

Everything here replaces the learned neighbour memberships with a **fixed,
frozen anchor system** ``A[t]`` -- a fine expression microclustering computed once
on a designated training split and never re-fitted.  Because the target alphabet
cannot be changed by whatever is being evaluated, a low CMI against fixed anchors
requires real predictive grouping and cannot be bought by collapse.

The exact decomposition this rests on (unit-tested in ``test_wot_serum.py``):

    I(I ; A) = I(Z ; A) + I(I ; A | Z)

with, for a candidate soft assignment ``M`` and fixed fingerprints ``f``,

    phi_k        = sum_i a_i M_ik f_i / g_k          (KL barycentre)
    I(I ; A|Z)   = sum_ik a_i M_ik KL(f_i || phi_k)
                 = sum_k g_k H(phi_k) - sum_i a_i H(f_i)
    I(I ; A)     = H(fbar) - sum_i a_i H(f_i),   fbar = sum_i a_i f_i
    I(Z ; A)     = H(fbar) - sum_k g_k H(phi_k)

so ``retained = I(Z;A) / I(I;A) = 1 - CMI / I(I;A)`` exactly.  The ratio is only
meaningful when the denominator is demonstrably nonzero; ``retained_information``
returns NaN and a reason rather than a flattering number when it is not.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from csa_wot import MAIN  # noqa: F401  (puts main/ on sys.path)

_EPS = 1e-30


# --------------------------------------------------------------------------
# anchors: a frozen coordinate system for describing origins and destinations
# --------------------------------------------------------------------------

def make_anchors(Z_train: Sequence[np.ndarray], n_anchors: int,
                 seed: int = 0, n_init: int = 4) -> List[np.ndarray]:
    """Per-timepoint expression microcluster centroids, learned ONCE.

    Anchors are bins, not proposed biological states.  They are learned on a
    designated training split and then frozen: every later comparison (bootstrap,
    batch hold-out, neighbouring epsilon, different support) must describe its
    fingerprints in the SAME coordinate system, or the distributions are not
    comparable and any "instability" is just relabelling.
    """
    from sklearn.cluster import KMeans

    out = []
    for t, z in enumerate(Z_train):
        z = np.asarray(z, dtype=np.float64)
        k = int(min(n_anchors, len(z)))
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed + 1000 * t)
        km.fit(z)
        out.append(km.cluster_centers_.astype(np.float64))
    return out


def assign_anchors(Z: Sequence[np.ndarray], centroids: Sequence[np.ndarray],
                   soft: bool = False, temperature: float = 1.0) -> List[np.ndarray]:
    """Assign cells to FROZEN anchor centroids.  Returns ``(n_t, n_anchor)`` rows.

    Hard (one-hot) by default, matching the plan's first pass.  ``soft=True`` gives
    a softmax over negative scaled squared distance, for a resolution-robustness
    check -- hard assignment can make a fingerprint jumpy when a cell sits on an
    anchor boundary, which would show up as false instability.
    """
    out = []
    for z, C in zip(Z, centroids):
        z = np.asarray(z, dtype=np.float64)
        d2 = ((z[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        if soft:
            scale = max(float(np.median(d2)), _EPS) * float(temperature)
            W = np.exp(-(d2 - d2.min(1, keepdims=True)) / scale)
            A = W / np.maximum(W.sum(1, keepdims=True), _EPS)
        else:
            A = np.zeros_like(d2)
            A[np.arange(len(z)), d2.argmin(1)] = 1.0
        out.append(A)
    return out


def anchors_from_labels(labels: Sequence[np.ndarray],
                        categories: Optional[Sequence] = None) -> List[np.ndarray]:
    """One-hot anchors from a categorical annotation (e.g. ``obs['cell_sets']``).

    Evaluated as an ADDITIONAL anchor system only: published annotations may be
    far too coarse to expose transition heterogeneity, so a null result against
    them is not evidence that no structure exists.
    """
    cats = (list(categories) if categories is not None
            else sorted({str(v) for lab in labels for v in np.asarray(lab)}))
    idx = {c: i for i, c in enumerate(cats)}
    out = []
    for lab in labels:
        lab = np.asarray(lab).astype(str)
        A = np.zeros((len(lab), len(cats)))
        for i, v in enumerate(lab):
            if v in idx:
                A[i, idx[v]] = 1.0
        out.append(A)
    return out, cats


# --------------------------------------------------------------------------
# chain <-> data identity (issue 1)
# --------------------------------------------------------------------------

IDENTITY_SUFFIX = ".identity.json"


def representation_hash(Z: Sequence[np.ndarray]) -> str:
    """Content hash of a representation, elementwise and order-sensitive."""
    import hashlib
    h = hashlib.sha1()
    for z in Z:
        h.update(np.ascontiguousarray(np.asarray(z, dtype=np.float64)).tobytes())
    return h.hexdigest()


def cell_id_hash(obs_index_per_t: Sequence[Sequence[str]]) -> str:
    """Order-sensitive hash of the cell identifiers, per timepoint."""
    import hashlib
    h = hashlib.sha1()
    for ids in obs_index_per_t:
        h.update("\x00".join(str(x) for x in ids).encode())
        h.update(b"\x01")
    return h.hexdigest()


def identity_payload(data, Z, *, h5ad, day_min, day_max, stride,
                     n_per_timepoint, seed, arm_policy) -> dict:
    """What a chain must be able to prove about the data it was built from.

    ``ReferenceChain.save`` cannot carry cell identifiers, so this travels beside
    the chain as ``chain_eps<e>.identity.json`` and is also copied into
    ``chain_manifest.json``.
    """
    ids = [[str(x) for x in np.asarray(data.obs[t]["index"])] for t in range(data.T)]
    return {
        "h5ad": os.path.abspath(h5ad),
        "day_min": float(day_min), "day_max": float(day_max),
        "stride": int(stride),
        "n_per_timepoint": (None if n_per_timepoint is None else int(n_per_timepoint)),
        "sampling_seed": int(seed),
        "arm_policy": arm_policy,
        "T": int(data.T),
        "n_cells": list(data.n_cells),
        "tau": np.asarray(data.tau, dtype=float).tolist(),
        "representation_sha1": representation_hash(Z),
        "cell_id_sha1": cell_id_hash(ids),
        "cell_ids": ids,
    }


def validate_chain_identity(chain, data, Z, identity: Optional[dict] = None,
                            chain_path: Optional[str] = None,
                            require_feasible: bool = True,
                            require_identity_file: bool = True) -> dict:
    """Abort unless the chain was built from EXACTLY this data, in this order.

    [CRITICAL] ``chain.n_cells == data.n_cells`` is not sufficient.  The serum-only
    and cross-arm H5ADs can hold the same number of cells per timepoint while
    differing in identity, order, or representation, and every fingerprint,
    membership and CMI number downstream is then silently attached to the wrong
    cells.  Checked here, in increasing order of strength:

      1. T, per-timepoint counts, tau
      2. ``chain.Z`` against the loaded representation, elementwise
      3. the sidecar identity file: cell ids AND their order, representation hash,
         h5ad path, day range, stride, subsample, seed, arm policy
      4. feasibility -- an infeasible chain makes A_t non-row-stochastic, so every
         transition quantity built on it is wrong

    Returns the identity dict that was verified, for provenance.
    """
    problems: List[str] = []

    if chain.T != data.T:
        problems.append(f"T: chain {chain.T} != data {data.T}")
    elif list(chain.n_cells) != list(data.n_cells):
        bad = [(t, c, d) for t, (c, d) in
               enumerate(zip(chain.n_cells, data.n_cells)) if c != d]
        problems.append(f"per-timepoint cell counts differ at {bad[:5]}")
    else:
        ct = np.asarray(chain.tau, dtype=float)
        dt = np.asarray(data.tau, dtype=float)
        if ct.shape != dt.shape or not np.allclose(ct, dt, rtol=0, atol=1e-9):
            problems.append(f"tau differs: chain {ct[:4]}... vs data {dt[:4]}...")
        if chain.Z is None:
            problems.append("chain carries no Z, so the representation cannot be "
                            "verified; rebuild the chain with Stage 0")
        else:
            hz_chain = representation_hash(chain.Z)
            hz_data = representation_hash(Z)
            if hz_chain != hz_data:
                problems.append(
                    f"representation content hash differs: chain {hz_chain[:12]} vs "
                    f"data {hz_data[:12]} -- same shapes can still be different "
                    f"cells, a different order, or a different PCA basis")

    if identity is None and chain_path:
        cand = str(chain_path) + IDENTITY_SUFFIX
        if os.path.exists(cand):
            with open(cand) as fh:
                identity = json.load(fh)
        elif require_identity_file:
            problems.append(
                f"no identity sidecar at {cand}. Stage-0 chains are written with one; "
                f"a chain without it cannot be shown to match this data. Rebuild with "
                f"00_stage0_chain.py, or pass require_identity_file=False knowingly.")

    if identity:
        ids = [[str(x) for x in np.asarray(data.obs[t]["index"])]
               for t in range(data.T)]
        h_now = cell_id_hash(ids)
        if identity.get("cell_id_sha1") != h_now:
            # locate the first difference so the message is actionable
            where = "unknown"
            rec = identity.get("cell_ids")
            if rec and len(rec) == len(ids):
                for t, (x, y) in enumerate(zip(rec, ids)):
                    if list(x) != list(y):
                        if sorted(x) == sorted(y):
                            where = f"timepoint {t}: same cells, DIFFERENT ORDER"
                        else:
                            miss = set(x) ^ set(y)
                            where = (f"timepoint {t}: {len(miss)} cell ids differ "
                                     f"(e.g. {sorted(miss)[:3]})")
                        break
            else:
                where = (f"recorded {len(rec) if rec else 0} timepoints vs "
                         f"{len(ids)} now")
            problems.append(f"cell identities/order differ -- {where}")
        if identity.get("representation_sha1") != representation_hash(Z):
            problems.append("representation hash differs from the identity record")
        for key, now in (("h5ad", None), ("day_min", None), ("day_max", None),
                         ("stride", None), ("n_per_timepoint", None),
                         ("sampling_seed", None), ("arm_policy", None)):
            pass  # compared by the caller, which knows its own arguments

    if require_feasible and not chain.feasible:
        problems.append(
            f"chain is INFEASIBLE at intervals {chain.infeasible_intervals()} "
            f"(marginal errors exceed {chain.feasibility_tol:.1e}); A_t is not "
            f"row-stochastic so every transition quantity would be wrong")

    if problems:
        raise SystemExit(
            "chain/data identity check FAILED -- refusing to run:\n  "
            + "\n  ".join(problems)
            + "\n\nThe chain must be built from exactly this h5ad, day range, "
              "stride, subsample and seed. Rebuild Stage 0 with matching arguments, "
              "or point --chain at the chain that matches.")
    return identity or {}


def check_identity_args(identity: dict, *, h5ad, day_min, day_max, stride,
                        n_per_timepoint, seed) -> None:
    """Abort if the caller's own arguments disagree with the chain's record."""
    if not identity:
        return
    want = {"h5ad": os.path.abspath(h5ad), "day_min": float(day_min),
            "day_max": float(day_max), "stride": int(stride),
            "n_per_timepoint": (None if n_per_timepoint is None else int(n_per_timepoint)),
            "sampling_seed": int(seed)}
    bad = {k: (identity.get(k), v) for k, v in want.items()
           if k in identity and identity.get(k) != v}
    if bad:
        raise SystemExit(
            "the chain was built with different arguments than this job is using:\n  "
            + "\n  ".join(f"{k}: chain={a!r} but this job has {b!r}"
                           for k, (a, b) in bad.items())
            + "\n\nRe-run with the chain's arguments, or rebuild Stage 0.")


# --------------------------------------------------------------------------
# fixed-anchor fingerprints
# --------------------------------------------------------------------------

def _coupling_csr(res, shape):
    import scipy.sparse as sp
    return sp.coo_matrix((res.values, (res.rows, res.cols)), shape=shape).tocsr()


def fixed_fingerprints(chain, A: Sequence[np.ndarray]) -> Tuple[List, List]:
    """``(f_plus, f_minus)`` against FROZEN anchors, from ``P^ref`` only.

    Mirrors ``model.fingerprints()`` but substitutes the fixed ``A`` for the
    learned ``M``, which is the whole point: these do not move when a candidate
    state assignment changes.

    ``f_plus[t]`` is ``(n_t, n_anchor_{t+1})`` for ``t < T-1`` and None at the last
    timepoint; ``f_minus[t]`` is ``(n_t, n_anchor_{t-1})`` for ``t > 0`` and None at
    the first.
    """
    T = chain.T
    Fp: List[Optional[np.ndarray]] = [None] * T
    Fm: List[Optional[np.ndarray]] = [None] * T
    for t in range(T - 1):
        P = _coupling_csr(chain.couplings[t], chain.couplings[t].shape)
        rs = np.asarray(P.sum(axis=1)).ravel()
        cs = np.asarray(P.sum(axis=0)).ravel()
        num = P @ A[t + 1]
        Fp[t] = num / np.maximum(rs, _EPS)[:, None]
        numT = P.T @ A[t]
        Fm[t + 1] = numT / np.maximum(cs, _EPS)[:, None]
    return Fp, Fm


# --------------------------------------------------------------------------
# information quantities
# --------------------------------------------------------------------------

def _H(p, axis=-1):
    p = np.asarray(p, dtype=np.float64)
    return -(p * np.log(np.maximum(p, _EPS))).sum(axis=axis)


def cell_information(F: np.ndarray, a: np.ndarray) -> float:
    """``I(cell_identity ; anchor) = H(fbar) - sum_i a_i H(f_i)``, in nats.

    The denominator of every "retained information" statement.  If this is near
    zero the coupling carries no cell-level transition signal, and a low CMI then
    means nothing -- there was nothing to lose.
    """
    F = np.asarray(F, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    fbar = (a[:, None] * F).sum(0)
    fbar = fbar / max(fbar.sum(), _EPS)
    return float(_H(fbar) - (a * _H(F, axis=1)).sum())


def state_cmi(M: np.ndarray, F: np.ndarray, a: np.ndarray) -> Dict[str, float]:
    """Fixed-anchor CMI for a candidate assignment ``M``, plus its decomposition.

    Returns ``cmi`` = ``I(I;A|Z)``, ``i_state_anchor`` = ``I(Z;A)``,
    ``i_cell_anchor`` = ``I(I;A)``, and the exact residual of the chain rule as
    ``decomposition_error`` -- if that is not ~0 something upstream is wrong.
    """
    M = np.asarray(M, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)

    g = M.T @ a                                     # state masses
    phi = (M.T @ (a[:, None] * F)) / np.maximum(g, _EPS)[:, None]
    phi = phi / np.maximum(phi.sum(1, keepdims=True), _EPS)

    H_f = (a * _H(F, axis=1)).sum()
    H_phi = (g * _H(phi, axis=1)).sum()
    fbar = (a[:, None] * F).sum(0)
    fbar = fbar / max(fbar.sum(), _EPS)

    cmi = float(H_phi - H_f)
    i_cell = float(_H(fbar) - H_f)
    i_state = float(_H(fbar) - H_phi)
    return {"cmi": cmi, "i_state_anchor": i_state, "i_cell_anchor": i_cell,
            "decomposition_error": float(i_cell - (i_state + cmi)),
            "k_eff": float(np.exp(_H(g, axis=0))),
            "min_state_mass": float(g.min()),
            "rate_upper_bound_log_K": float(np.log(M.shape[1]))}


def retained_information(cmi: float, i_cell: float,
                         min_information: float = 1e-3) -> Tuple[float, str]:
    """``1 - CMI / I(cell;anchor)``, or NaN with a reason.

    [CRITICAL] The ratio is unstable when the denominator is near zero, and in
    that regime a small CMI is not a success -- it means the coupling had no
    cell-level information to preserve.  Returning NaN with a reason keeps that
    case from being reported as a good result.
    """
    if not np.isfinite(i_cell) or i_cell < min_information:
        return (float("nan"),
                f"I(cell;anchor)={i_cell:.3g} < {min_information:g}: no cell-level "
                f"transition information to retain, so the ratio is uninformative")
    return (1.0 - cmi / i_cell, "ok")


# --------------------------------------------------------------------------
# fingerprint geometry
# --------------------------------------------------------------------------

def js_divergence(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Row-wise Jensen-Shannon divergence (nats). Bounded, symmetric, finite."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    Mm = 0.5 * (P + Q)
    return 0.5 * ((P * (np.log(np.maximum(P, _EPS)) - np.log(np.maximum(Mm, _EPS)))).sum(1)
                  + (Q * (np.log(np.maximum(Q, _EPS)) - np.log(np.maximum(Mm, _EPS)))).sum(1))


def hellinger(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Row-wise Hellinger distance in [0, 1]."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    return np.sqrt(np.maximum(0.5 * ((np.sqrt(P) - np.sqrt(Q)) ** 2).sum(1), 0.0))


def pairwise_sample(F: np.ndarray, n_pairs: int, rng) -> np.ndarray:
    """JS divergence over a random sample of cell pairs."""
    n = len(F)
    if n < 2:
        return np.zeros(0)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    m = i != j
    if not m.any():
        return np.zeros(0)
    return js_divergence(F[i[m]], F[j[m]])


def independence_null(F: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Every cell gets the marginal fingerprint -- the eps -> infinity limit.

    The reference for "is this fingerprint variation real": under an independent
    coupling all cells share ``fbar``, so ``I(cell;anchor) = 0`` and pairwise
    divergence is 0.
    """
    fbar = (np.asarray(a)[:, None] * np.asarray(F)).sum(0)
    fbar = fbar / max(fbar.sum(), _EPS)
    return np.tile(fbar, (len(F), 1))


def local_neighbourhood_permutation(*args, **kwargs):
    """REMOVED -- it did not test what A3 claims.  Use the functions below.

    The old statistic recomputed the GLOBAL ``I(cell;anchor)`` after replacing each
    fingerprint with a random expression-neighbour's.  ``I(cell;anchor)`` is a
    function of the multiset of fingerprint rows and of ``fbar``, so it responds
    mainly to fingerprint DIVERSITY rather than to whether fingerprints line up
    with expression coordinates -- and its sign does not map onto "structure beyond
    expression", which is the A3 question.  Measured on synthetic data it gave
    z=+0.19 for fingerprints independent of expression and z=+15.7 for fingerprints
    that are a smooth function of expression: it separates those cases, but by
    responding to concentration, not correspondence.

    Replaced by :func:`expression_neighbour_concordance` (does expression locality
    predict fingerprint similarity?) and :func:`expression_predicts_fingerprint`
    (cross-fitted: how much fingerprint variation survives an expression-based
    predictor?).
    """
    raise NotImplementedError(local_neighbourhood_permutation.__doc__)


def expression_neighbour_concordance(Z: np.ndarray, F: np.ndarray,
                                     n_neighbors: int = 30, n_pairs: int = 20000,
                                     seed: int = 0) -> Dict[str, float]:
    """Are expression neighbours more similar in fingerprint than random pairs?

    Directly tests the expression-fingerprint CORRESPONDENCE that A3 is about:
    mean JS between each cell and its expression neighbours, against mean JS
    between matched random pairs.  ``ratio`` near 1 means expression locality says
    nothing about transition role; ratio << 1 means fingerprints are largely a
    readout of expression position (the handoff's density concern); an intermediate
    value with a large residual is the interesting case.
    """
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z, float); F = np.asarray(F, float)
    n = len(Z)
    k = int(min(n_neighbors, n - 1))
    if n < 3 or k < 1:
        return {"js_neighbour": float("nan"), "js_random": float("nan"),
                "ratio": float("nan"), "n_cells": int(n)}
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nbr = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]

    m = int(min(n_pairs, n * k))
    rows = rng.integers(0, n, size=m)
    cols = nbr[rows, rng.integers(0, k, size=m)]
    js_nb = js_divergence(F[rows], F[cols])

    r2 = rng.integers(0, n, size=m)
    c2 = rng.integers(0, n, size=m)
    ok = r2 != c2
    js_rnd = js_divergence(F[r2[ok]], F[c2[ok]])

    mn, mr = float(js_nb.mean()), float(js_rnd.mean())
    return {"js_neighbour": mn, "js_random": mr,
            "ratio": float(mn / mr) if mr > 0 else float("nan"),
            "n_neighbors": k, "n_pairs": int(m), "n_cells": int(n)}


def expression_predicts_fingerprint(Z: np.ndarray, F: np.ndarray,
                                    n_folds: int = 5, n_neighbors: int = 30,
                                    seed: int = 0,
                                    groups: Optional[np.ndarray] = None
                                    ) -> Dict[str, float]:
    """Cross-fitted: how much fingerprint variation survives expression prediction?

    Fits a kNN (Nadaraya-Watson style, uniform weights over the k nearest training
    cells) predictor of the fingerprint from expression on out-of-fold data, then
    reports the held-out KL(f_i || fhat_i).  Baselines:

      * ``kl_to_marginal``  -- predicting the global mean fingerprint (no model)
      * ``kl_to_expression`` -- predicting from expression

    ``residual_fraction = kl_to_expression / kl_to_marginal`` is the share of
    cell-level fingerprint variation that expression geometry does NOT explain.
    Near 0 means fingerprints are an expression readout; near 1 means expression
    carries almost none of the transition signal.

    ``groups`` (e.g. technical batch) keeps a cell and its batch out of its own
    training fold, so a batch-specific quirk cannot leak into the "smooth
    geometry" prediction.
    """
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z, float); F = np.asarray(F, float)
    n = len(Z)
    if n < n_folds + 2:
        return {"kl_to_expression": float("nan"), "kl_to_marginal": float("nan"),
                "residual_fraction": float("nan"), "n_cells": int(n)}

    if groups is not None:
        g = np.asarray(groups)
        uniq = np.unique(g)
        assign = {v: i % min(n_folds, len(uniq)) for i, v in enumerate(rng.permutation(uniq))}
        folds = np.array([assign[v] for v in g])
    else:
        folds = rng.integers(0, n_folds, size=n)

    fhat = np.zeros_like(F)
    for f in np.unique(folds):
        te = np.flatnonzero(folds == f)
        tr = np.flatnonzero(folds != f)
        if tr.size == 0:
            fhat[te] = F.mean(0)
            continue
        k = int(min(n_neighbors, tr.size))
        d2 = ((Z[te][:, None, :] - Z[tr][None, :, :]) ** 2).sum(-1)
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        pred = F[tr][idx].mean(axis=1)
        fhat[te] = pred / np.maximum(pred.sum(1, keepdims=True), _EPS)

    fbar = F.mean(0)
    fbar = fbar / max(fbar.sum(), _EPS)

    def _kl(P, Q):
        return float((P * (np.log(np.maximum(P, _EPS))
                           - np.log(np.maximum(Q, _EPS)))).sum(1).mean())

    kl_expr = _kl(F, fhat)
    kl_marg = _kl(F, np.tile(fbar, (n, 1)))
    return {"kl_to_expression": kl_expr, "kl_to_marginal": kl_marg,
            "residual_fraction": float(kl_expr / kl_marg) if kl_marg > 0 else float("nan"),
            "n_folds": int(len(np.unique(folds))), "n_neighbors": int(n_neighbors),
            "grouped": groups is not None, "n_cells": int(n)}


# --------------------------------------------------------------------------
# cluster tendency, respecting the simplex (issue 3)
# --------------------------------------------------------------------------

def clr(F: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Centred log-ratio transform: takes the simplex to a hyperplane in R^k.

    Compositional data live on the simplex, so Euclidean tools applied to raw
    proportions are measuring the wrong geometry.  CLR is the standard remedy; the
    image is the sum-zero hyperplane, which is why the null below is built inside
    the CLR span rather than a bounding box.
    """
    F = np.maximum(np.asarray(F, float), floor)
    F = F / F.sum(1, keepdims=True)
    L = np.log(F)
    return L - L.mean(1, keepdims=True)


def cluster_tendency(*args, **kwargs):
    """REMOVED. Hopkins-style tendency statistics did not work here; see below.

    Two implementations were tried and both failed a control:

    1. **Hopkins with a bounding-box null** (the original).  Fingerprints lie on
       one or two probability simplices, so uniform box samples land OFF the
       simplex and are unnaturally far from the data.  On UNCLUSTERED Dirichlet
       samples it returned 0.77-0.84, and on genuinely 2-clustered simplex data
       only 0.91 -- so a reported 0.945 was barely above its own null and was NOT
       evidence of clustering.
    2. **CLR + covariance-matched Gaussian null**.  This fixed the negative
       control (0.508-0.523 on unclustered Dirichlet, correctly ~0.5) but failed
       the POSITIVE control: genuinely 2-clustered data also gave 0.517, because
       two separated clusters are largely described by their second moments, which
       the matched null therefore reproduces.

    Use :func:`cluster_stability_resampled` (does a K-way partition reproduce?) and
    :func:`bimodality_clr` (is the leading CLR direction multimodal?).  On the same
    controls those give 0.15-0.21 vs 1.00 and 0.32 vs 0.76 respectively.
    """
    raise NotImplementedError(cluster_tendency.__doc__)


def bimodality_clr(F_blocks: Sequence[np.ndarray], max_pc: int = 3,
                   max_cells: int = 3000, rng=None) -> Dict[str, float]:
    """Multimodality of the leading CLR directions -- descriptive, simplex-aware.

    Sarle's bimodality coefficient ``(skew^2 + 1) / kurtosis`` on each of the top
    CLR principal components; above ~0.555 suggests bimodality, below suggests a
    unimodal spread.  Controls: 0.32 on unclustered Dirichlet, 0.76 on two clear
    simplex clusters.

    Descriptive only.  A single projection being bimodal does not establish
    discrete biological states, and a unimodal leading direction does not rule out
    structure in a direction that is not leading.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    X = np.hstack([clr(np.asarray(B, float)) for B in F_blocks])
    if len(X) > max_cells:
        X = X[rng.choice(len(X), max_cells, replace=False)]
    Xc = X - X.mean(0)
    if len(Xc) < 20:
        return {"bimodality_pc1": float("nan"), "n_cells": int(len(Xc))}
    try:
        V = np.linalg.svd(Xc, full_matrices=False)[2]
    except np.linalg.LinAlgError:
        return {"bimodality_pc1": float("nan"), "n_cells": int(len(Xc))}
    out = {"n_cells": int(len(Xc)), "clr_dim": int(X.shape[1]),
           "threshold": 0.555,
           "interpretation": ("Sarle's bimodality coefficient; >0.555 suggests "
                              "bimodal. Descriptive only -- does not establish "
                              "discrete biological states.")}
    for i in range(min(max_pc, V.shape[0])):
        pc = Xc @ V[i]
        sd = pc.std()
        if sd <= 0:
            out[f"bimodality_pc{i+1}"] = float("nan")
            continue
        sk = float(((pc - pc.mean()) ** 3).mean() / sd ** 3)
        ku = float(((pc - pc.mean()) ** 4).mean() / sd ** 4)
        out[f"bimodality_pc{i+1}"] = float((sk ** 2 + 1) / ku) if ku > 0 else float("nan")
    return out


def cluster_stability_resampled(F_blocks: Sequence[np.ndarray], K: int, rng,
                                n_rep: int = 10, frac: float = 0.8,
                                max_cells: int = 1500) -> Dict[str, float]:
    """Do k-means partitions of fingerprint space reproduce under resampling?

    The alternative the plan asks for: rather than asking whether the cloud "looks
    clustered", ask whether a K-way partition of it is REPRODUCIBLE.  Pairs of
    subsamples are compared by ARI on their overlap, in CLR coordinates.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    X = np.hstack([clr(np.asarray(B, float)) for B in F_blocks])
    if len(X) > max_cells:
        X = X[rng.choice(len(X), max_cells, replace=False)]
    n = len(X)
    if n < max(4 * K, 40):
        return {"mean_ari": float("nan"), "K": int(K), "n_cells": int(n)}
    labs = []
    for r in range(n_rep):
        idx = rng.choice(n, int(frac * n), replace=False)
        km = KMeans(n_clusters=K, n_init=4, random_state=int(rng.integers(1 << 30)))
        km.fit(X[idx])
        full = km.predict(X)
        labs.append(full)
    aris = [adjusted_rand_score(labs[i], labs[j])
            for i in range(len(labs)) for j in range(i + 1, len(labs))]
    return {"mean_ari": float(np.mean(aris)), "sd_ari": float(np.std(aris)),
            "K": int(K), "n_rep": int(n_rep), "frac": float(frac),
            "n_cells": int(n)}


# --------------------------------------------------------------------------
# splits and warm starts
# --------------------------------------------------------------------------

def batch_split(data, batch: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Row indices for ``replicate == batch`` and its complement, per timepoint."""
    inside, outside = [], []
    for t in range(data.T):
        rep = np.asarray(data.replicate[t]).astype(str)
        inside.append(np.flatnonzero(rep == str(batch)))
        outside.append(np.flatnonzero(rep != str(batch)))
    return inside, outside


def bootstrap_indices(data, frac: float, seed: int) -> List[np.ndarray]:
    """Per-timepoint subsample WITHOUT replacement, stratified by replicate.

    Without replacement because the couplings are balanced over distinct cells:
    duplicated cells would give a coupling with a different support structure, not
    a resampled estimate of the same one.
    """
    rng = np.random.default_rng(seed)
    out = []
    for t in range(data.T):
        rep = np.asarray(data.replicate[t]).astype(str)
        keep = []
        for g in np.unique(rep):
            pool = np.flatnonzero(rep == g)
            k = max(1, int(round(frac * len(pool))))
            keep.append(rng.choice(pool, size=k, replace=False))
        out.append(np.sort(np.concatenate(keep)))
    return out


def logits_from_memberships(M: Sequence[np.ndarray], floor: float = 1e-8
                            ) -> List[np.ndarray]:
    """``U = log M`` so that ``softmax(U) = M`` -- for lambda continuation warm starts.

    Softmax is shift-invariant per row, so the additive constant is irrelevant.  The
    floor matters: the 2026-09-07 grids had memberships saturating at exactly 1.000,
    and ``log(0)`` would hand the optimiser a -inf start.
    """
    return [np.log(np.maximum(np.asarray(m, dtype=np.float64), floor))
            for m in M]
