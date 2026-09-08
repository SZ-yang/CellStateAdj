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


def local_neighbourhood_permutation(Z: np.ndarray, F: np.ndarray, a: np.ndarray,
                                    n_neighbors: int = 30, n_perm: int = 20,
                                    seed: int = 0) -> Dict[str, float]:
    """Is fingerprint divergence larger than smooth expression geometry predicts?

    Permutes each cell's fingerprint with that of a random expression neighbour and
    recomputes the cell-level information.  If the observed value sits inside this
    null, the fingerprint variation is compatible with smooth geometry plus noise
    and is NOT evidence of transition-specific structure.
    """
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z, float)
    n = len(Z)
    k = int(min(n_neighbors, n - 1))
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    nbr = np.argsort(d2, axis=1)[:, 1:k + 1]

    obs = cell_information(F, a)
    null = np.empty(n_perm)
    for r in range(n_perm):
        pick = nbr[np.arange(n), rng.integers(0, k, size=n)]
        null[r] = cell_information(F[pick], a)
    return {"observed": obs, "null_mean": float(null.mean()),
            "null_sd": float(null.std()),
            "p_value": float((null >= obs).mean()),
            "z": float((obs - null.mean()) / max(null.std(), _EPS))}


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
