"""Evaluation against simulator ground truth.

State labels are time-local and arbitrary, so every comparison against a
ground-truth state set goes through an explicit matching step (Hungarian on the
joint mass).  Transition-matrix error is only meaningful after that matching.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np


def _ari(a, b) -> float:
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(a, b))
    except Exception:  # pragma: no cover
        return float("nan")


def _nmi(a, b) -> float:
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(a, b))
    except Exception:  # pragma: no cover
        return float("nan")


def state_recovery(M: Sequence[np.ndarray], true_states: Sequence[np.ndarray]) -> dict:
    """ARI / NMI of hard assignments against the ground-truth state per timepoint."""
    aris = [_ari(np.asarray(m).argmax(1), np.asarray(s)) for m, s in zip(M, true_states)]
    nmis = [_nmi(np.asarray(m).argmax(1), np.asarray(s)) for m, s in zip(M, true_states)]
    return {"ari_per_t": aris, "nmi_per_t": nmis,
            "mean_ari": float(np.mean(aris)), "mean_nmi": float(np.mean(nmis))}


def match_states(M: np.ndarray, true_states: np.ndarray, n_true: int,
                 a: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Match learned states to ground-truth states by maximum joint mass.

    Returns ``(assignment, joint)`` where ``assignment[k]`` is the true state
    matched to learned state k (-1 when unmatched) and ``joint`` is the
    (K, n_true) mass table.
    """
    M = np.asarray(M)
    n, K = M.shape
    a = np.full(n, 1.0 / n) if a is None else np.asarray(a)
    joint = np.zeros((K, n_true))
    np.add.at(joint.T, np.asarray(true_states).astype(int), (a[:, None] * M))
    assign = np.full(K, -1, dtype=int)
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-joint)
        assign[r] = c
    except Exception:  # pragma: no cover
        assign = joint.argmax(1)
    return assign, joint


def transition_recovery(
    A_pred: Sequence[np.ndarray],
    g_pred: Sequence[np.ndarray],
    M: Sequence[np.ndarray],
    true_states: Sequence[np.ndarray],
    T_true: Sequence[np.ndarray],
    n_true: int,
    a: Optional[Sequence[np.ndarray]] = None,
) -> dict:
    """Compare the induced state-level flow with the ground-truth lineage flow.

    The learned states are first collapsed onto ground-truth states via the
    joint mass table, so the predicted flow is projected into the true state
    space rather than the true flow being permuted into an arbitrary one.
    """
    errs, corrs = [], []
    for t, Tt_true in enumerate(T_true):
        Mt, Mt1 = np.asarray(M[t]), np.asarray(M[t + 1])
        at = (np.full(Mt.shape[0], 1.0 / Mt.shape[0]) if a is None
              else np.asarray(a[t]))
        at1 = (np.full(Mt1.shape[0], 1.0 / Mt1.shape[0]) if a is None
               else np.asarray(a[t + 1]))
        # R_t[k, s] = P(true state s | learned state k)
        _, J0 = match_states(Mt, true_states[t], n_true, at)
        _, J1 = match_states(Mt1, true_states[t + 1], n_true, at1)
        R0 = J0 / np.maximum(J0.sum(1, keepdims=True), 1e-30)
        R1 = J1 / np.maximum(J1.sum(1, keepdims=True), 1e-30)
        Tt_pred = np.asarray(A_pred[t]) * np.asarray(g_pred[t])[:, None]
        proj = R0.T @ Tt_pred @ R1                     # (n_true, n_true)
        proj = proj / max(proj.sum(), 1e-30)
        truth = Tt_true / max(Tt_true.sum(), 1e-30)
        errs.append(float(np.abs(proj - truth).sum()))
        corrs.append(float(np.corrcoef(proj.ravel(), truth.ravel())[0, 1])
                     if proj.std() > 0 and truth.std() > 0 else float("nan"))
    return {"l1_error_per_t": errs, "mean_l1_error": float(np.mean(errs)),
            "corr_per_t": corrs, "mean_corr": float(np.nanmean(corrs))}


def coupling_recovery(P_ref, P_true: np.ndarray) -> dict:
    """How well P^ref reproduces the ground-truth (partial) cell coupling.

    Reported as the mass of P^ref that lands on true ancestor-descendant pairs,
    against the mass an independent coupling would put there.
    """
    if P_true is None:
        return {}
    n, m = P_true.shape
    hit = np.zeros(P_true.shape, dtype=bool)
    hit[P_true > 0] = True
    dense = np.zeros((n, m))
    dense[P_ref.rows, P_ref.cols] = P_ref.values
    captured = float(dense[hit].sum())
    a = dense.sum(1, keepdims=True)
    b = dense.sum(0, keepdims=True)
    baseline = float((a @ b)[hit].sum())
    return {"captured_mass": captured, "independent_baseline": baseline,
            "enrichment": captured / max(baseline, 1e-30)}


def fingerprint_recovery(F_pred: np.ndarray, ancestor_state: np.ndarray,
                         assign: np.ndarray, n_true: int) -> dict:
    """Do the incoming fingerprints point at the true ancestor state?

    ``assign`` maps learned state index -> true state index.  Accuracy is the
    fraction of cells whose fingerprint argmax matches the true ancestor.
    """
    if ancestor_state is None:
        return {}
    pick = np.asarray(F_pred).argmax(1)
    mapped = np.array([assign[p] if 0 <= p < len(assign) else -1 for p in pick])
    ok = mapped == np.asarray(ancestor_state)
    return {"accuracy": float(ok.mean()), "n": int(len(ok))}
