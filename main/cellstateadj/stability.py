"""Stability reporting (spec 1.21).

[CRITICAL, handoff s5] For the WOT dataset the duplicate samples are
culture/well-level replicates from ONE embryo.  The effective biological n for
the whole experiment is one animal.  What is available:

* duplicate-sample split-half stability (n = 2 per timepoint);
* cross-experiment reproducibility (fit on one experiment, check the structure
  is recovered in the other);
* initialisation stability (handled in ``optimize.fit`` via ``n_init``);
* epsilon / K / cost sensitivity.

What is NOT available: a proper R~100 biological-replicate bootstrap.  Do not
build that machinery for this dataset, and label everything honestly --
"culture-replicate stability", never "biological uncertainty".  WOT itself used
these duplicate batches to estimate interpolation variability, so there is
precedent for the weaker claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import numpy as np

from .config import PipelineConfig
from .data import (TimeSeriesData, split_half_by_replicate,
                   restrict_to_shared_replicates as restrict_shared)
from .optimize import FitResult

_EPS = 1e-30


def _assign(D: np.ndarray) -> np.ndarray:
    """Hungarian matching on a cost matrix; ``out[k]`` indexes the reference."""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(D)
        perm = np.full(D.shape[0], -1, dtype=int)
        perm[r] = c
        return perm
    except Exception:  # pragma: no cover
        return D.argmin(1)


def align_states(mu_ref: np.ndarray, mu_other: np.ndarray,
                 g_ref: Optional[np.ndarray] = None,
                 g_other: Optional[np.ndarray] = None) -> np.ndarray:
    """Match states by expression prototype ALONE.

    Kept for the K-selection transfer step and as the initialiser for
    :func:`align_states_joint`.  Do NOT use it on its own for stability
    reporting: two states can share an expression prototype and differ entirely
    in temporal role, which is exactly the distinction this method exists to
    make, so an expression-only matching can swap them and report the swap as
    instability (or, worse, hide real instability).
    """
    D = ((mu_other[:, None, :] - mu_ref[None, :, :]) ** 2).sum(-1)
    return _assign(D)


def _kl_matrix(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """KL(P_k || Q_l) for every pair of rows."""
    P = np.maximum(P, _EPS)
    Q = np.maximum(Q, _EPS)
    return (P * np.log(P)).sum(1)[:, None] - P @ np.log(Q).T


def align_states_joint(
    mu_ref: Sequence[np.ndarray],
    mu_other: Sequence[np.ndarray],
    phi_plus_ref: Sequence[Optional[np.ndarray]],
    phi_plus_other: Sequence[Optional[np.ndarray]],
    phi_minus_ref: Sequence[Optional[np.ndarray]],
    phi_minus_other: Sequence[Optional[np.ndarray]],
    transition_weight: float = 1.0,
    n_iter: int = 3,
) -> List[np.ndarray]:
    """Align two fits on expression AND transition role, jointly across time.

    The difficulty is circular: phi+_tk is a distribution over the states at
    t+1, so comparing two fits' fingerprints requires already knowing how their
    t+1 states correspond.  We break the circle by iterating --

    1. align every timepoint on expression prototypes alone;
    2. use the current alignment to permute the other fit's phi+/phi- into the
       reference fit's state coordinates;
    3. re-align on expression distance + KL between the mapped fingerprints;
    4. repeat until the permutations stop changing.

    Expression distances are normalised by their mean so ``transition_weight``
    is a like-for-like trade-off rather than a units-dependent one.
    """
    T = len(mu_ref)
    perms = [align_states(mu_ref[t], mu_other[t]) for t in range(T)]

    def _permute_cols(phi: np.ndarray, perm: np.ndarray, K: int) -> np.ndarray:
        """Re-index phi's COLUMNS (neighbour states) into reference coordinates."""
        out = np.zeros((phi.shape[0], K))
        for k in range(phi.shape[1]):
            if 0 <= perm[k] < K:
                out[:, perm[k]] += phi[:, k]
        return out

    for _ in range(max(0, n_iter)):
        new_perms = []
        for t in range(T):
            K = mu_ref[t].shape[0]
            D = ((mu_other[t][:, None, :] - mu_ref[t][None, :, :]) ** 2).sum(-1)
            D = D / max(float(D.mean()), _EPS)

            if transition_weight > 0:
                if phi_plus_ref[t] is not None and phi_plus_other[t] is not None:
                    mapped = _permute_cols(phi_plus_other[t], perms[t + 1],
                                           phi_plus_ref[t].shape[1])
                    mapped = mapped / np.maximum(mapped.sum(1, keepdims=True), _EPS)
                    Kp = _kl_matrix(mapped, phi_plus_ref[t])
                    D = D + transition_weight * Kp / max(float(Kp.mean()), _EPS)
                if phi_minus_ref[t] is not None and phi_minus_other[t] is not None:
                    mapped = _permute_cols(phi_minus_other[t], perms[t - 1],
                                           phi_minus_ref[t].shape[1])
                    mapped = mapped / np.maximum(mapped.sum(1, keepdims=True), _EPS)
                    Km = _kl_matrix(mapped, phi_minus_ref[t])
                    D = D + transition_weight * Km / max(float(Km.mean()), _EPS)
            new_perms.append(_assign(D))
        if all(np.array_equal(a, b) for a, b in zip(perms, new_perms)):
            perms = new_perms
            break
        perms = new_perms
    return perms


@dataclass
class StabilityReport:
    kind: str                                   # what was resampled
    n_fits: int
    edge_support: List[np.ndarray] = field(default_factory=list)   # (T-1,) K x K
    mean_edge_support: float = float("nan")
    node_support: List[np.ndarray] = field(default_factory=list)   # (T,) K
    state_mass_sd: List[np.ndarray] = field(default_factory=list)
    per_fit: List[dict] = field(default_factory=list)
    alignment: str = "joint"
    notes: str = ""

    def caveat(self) -> str:
        return (f"{self.kind} stability over {self.n_fits} fits. "
                "This is algorithmic / culture-replicate stability, NOT "
                "biological uncertainty, unless the resampled units are "
                "independent biological replicates.")


def _fit_prototypes(fit: FitResult):
    """``(mu, phi_plus, phi_minus)`` for one fit, as numpy."""
    import torch
    with torch.no_grad():
        M = fit.model.memberships()
        g = fit.model.state_masses(M)
        mu = [x.cpu().numpy() for x in fit.model.expression_prototypes(M, g)]
        Fp, Fm = fit.model.fingerprints(M)
        pp, pm = [], []
        for t in range(fit.model.T):
            pp.append(None if Fp[t] is None
                      else fit.model.prototypes(M, Fp[t], g, t).cpu().numpy())
            pm.append(None if Fm[t] is None
                      else fit.model.prototypes(M, Fm[t], g, t).cpu().numpy())
    return mu, pp, pm


def edge_support(
    fits: Sequence[FitResult],
    mus: Optional[Sequence[Sequence[np.ndarray]]] = None,
    delta_A: float = 0.05,
    reference: int = 0,
    delta_g: float = 1e-3,
    transition_weight: float = 1.0,
    alignment: str = "joint",
) -> StabilityReport:
    """Bootstrap support of Eq. 36 for aligned state-level edges.

        S_tkl = (1/R) sum_r 1[ A^(r)_tkl > delta_A ]

    States are aligned with :func:`align_states_joint`, i.e. on expression AND
    transition role.  Expression-only alignment (``alignment='expression'``,
    kept for comparison) can swap two states that look alike but behave
    differently, which is precisely the confusion this method is built to
    avoid.  Unmatched states contribute no support.

    Report edge width as transition mass and edge opacity as this support.
    ``node_support`` is the analogous per-state quantity: the fraction of fits
    in which the aligned state carries mass above ``delta_g``.
    """
    import torch

    R = len(fits)
    ref = fits[reference]
    K = ref.model.K
    T = ref.model.T
    n_bad = sum(1 for f in fits if getattr(f, "status", "converged") != "converged")

    protos = [_fit_prototypes(f) for f in fits]
    if mus is not None:
        for r in range(R):
            protos[r] = ([np.asarray(x) for x in mus[r]], protos[r][1], protos[r][2])
    mu_ref, pp_ref, pm_ref = protos[reference]

    counts = [np.zeros((K, K)) for _ in range(T - 1)]
    node_counts = [np.zeros(K) for _ in range(T)]
    masses = [np.zeros((R, K)) for _ in range(T)]

    for r, f in enumerate(fits):
        with torch.no_grad():
            _, A_r, _, g_r = f.model.induced_transitions()
        mu_r, pp_r, pm_r = protos[r]
        if alignment == "joint":
            perms = align_states_joint(mu_ref, mu_r, pp_ref, pp_r, pm_ref, pm_r,
                                       transition_weight=transition_weight)
        elif alignment == "expression":
            perms = [align_states(mu_ref[t], mu_r[t]) for t in range(T)]
        else:
            raise ValueError(f"unknown alignment {alignment!r}")

        for t in range(T):
            gr = g_r[t].cpu().numpy()
            for k in range(K):
                if 0 <= perms[t][k] < K:
                    masses[t][r, perms[t][k]] += gr[k]
                    if gr[k] > delta_g:
                        node_counts[t][perms[t][k]] += 1.0
        for t in range(T - 1):
            Ar = A_r[t].cpu().numpy()
            for k in range(K):
                kk = perms[t][k]
                if not (0 <= kk < K):
                    continue
                for l in range(K):
                    ll = perms[t + 1][l]
                    if 0 <= ll < K and Ar[k, l] > delta_A:
                        counts[t][kk, ll] += 1.0

    support = [c / max(R, 1) for c in counts]
    return StabilityReport(
        kind="resampled-fit",
        n_fits=R,
        notes=("" if n_bad == 0 else
               f"[UNRELIABLE] {n_bad} of {R} fits did not converge; each still "
               f"contributes one vote to node and edge support."),
        alignment=alignment,
        node_support=[c / max(R, 1) for c in node_counts],
        edge_support=support,
        mean_edge_support=float(np.mean([s.max(initial=0.0) for s in support])),
        state_mass_sd=[m.std(0) for m in masses],
        per_fit=[f.summary() for f in fits],
    )


def split_half_stability(
    data: TimeSeriesData,
    cfg: PipelineConfig,
    n_splits: int = 1,
    seed: int = 0,
    require_converged: bool = True,
    restrict_to_shared_replicates: bool = False,
    verbose: int = 1,
) -> StabilityReport:
    """Refit on each culture-replicate half and compare the recovered structure.

    Each half gets its OWN reference chain -- the couplings are defined on the
    cell set being used, so reusing the full-data chain would leak information
    across the split.  The frozen representation is shared, because it is fit
    once by construction and refitting it per half would confound
    representation variability with state variability.

    ``require_converged`` (default True): every half must produce a converged
    fit.  Node and edge support count how often a state or edge reappears across
    fits, so a fit that stopped at ``max_iter`` contributes a vote from wherever
    the optimiser happened to be -- it lowers support for real structure and can
    manufacture support for structure that is only a partial-optimisation
    artefact.  Set it False only to inspect the failure, and read the resulting
    ``notes``.
    """
    from .pipeline import run_pipeline
    from .representation import learn_representation

    if data.replicate is None:
        raise ValueError("split_half_stability requires replicate labels")

    # [CRITICAL] Restrict before the representation is fit, not after: the PCA
    # basis is built from the cells it is handed, so restricting afterwards
    # would let the discarded replicates shape the space the halves are
    # compared in.
    if restrict_to_shared_replicates:
        data = restrict_shared(data, verbose=verbose)

    Z_full, rep_info = learn_representation(data, cfg.representation)
    fits, mus, per_fit = [], [], []
    bad: List[tuple] = []

    for s in range(n_splits):
        halves = split_half_by_replicate(data, seed=seed + s)  # already restricted
        for h, half in enumerate(halves):
            if verbose:
                print(f"[split-half] split {s} half {h}: n={half.n_cells}")
            # push the held-out cells through the SAME frozen representation
            from .representation import apply_representation
            if rep_info.get("method") == "pca":
                Zh = [apply_representation(x, rep_info) for x in half.X]
            else:
                Zh = None
            res = run_pipeline(half, cfg, Z=Zh, with_diagnostics=True,
                               check_degeneracy=False, verbose=max(0, verbose - 1))
            st = getattr(res.fit, "status", "converged")
            if st != "converged":
                bad.append((s, h, st, res.fit.n_iter))
                if verbose:
                    print(f"[split-half]   NOT CONVERGED: status={st} "
                          f"after {res.fit.n_iter} iterations")
            fits.append(res.fit)
            mus.append([m for m in res.diagnostics.mu])
            per_fit.append(res.fit.summary())

    if bad and require_converged:
        detail = ", ".join(f"split {s} half {h}: {st} at {n} iters"
                           for s, h, st, n in bad)
        raise ValueError(
            f"{len(bad)} of {len(fits)} split-half fits did not converge "
            f"({detail}). Support counts votes across fits, so a fit that never "
            f"met its tolerances votes from wherever the optimiser stopped. "
            f"Raise cfg.optim.max_iter, or pass require_converged=False to "
            f"inspect the failure -- the report is then marked unreliable."
        )

    rep = edge_support(fits, mus=mus)
    rep.kind = "culture-replicate split-half"
    rep.notes = ("Duplicate samples are culture/well-level replicates. "
                 "Report as culture-replicate stability, not biological "
                 "uncertainty.")
    if bad:
        rep.notes += (f" [UNRELIABLE] {len(bad)} of {len(fits)} fits did not "
                      f"converge and were counted anyway "
                      f"(require_converged=False).")
    return rep


def epsilon_sensitivity(
    Z: Sequence[np.ndarray],
    tau: np.ndarray,
    cfg: PipelineConfig,
    epsilons: Sequence[float],
    verbose: int = 1,
) -> List[dict]:
    """Refit states across an interval of epsilon and report the spread.

    No single epsilon is treated as uniquely correct (spec 1.4), so results are
    reported across the informative interval rather than at one value.
    """
    from dataclasses import replace as dc_replace
    from .optimize import fit as fit_states
    from .reference import build_reference_chain

    out = []
    ref_labels = None
    for eps in epsilons:
        ccfg = dc_replace(cfg.coupling, epsilon=float(eps))
        chain = build_reference_chain(Z, tau, ccfg, verbose=0)
        res = fit_states(chain, Z, cfg.model, cfg.optim)
        rec = res.summary()
        rec["epsilon"] = float(eps)
        rec["feasible"] = chain.feasible
        if ref_labels is not None:
            try:
                from sklearn.metrics import adjusted_rand_score as ari
                rec["ari_to_first"] = float(np.mean(
                    [ari(a, b) for a, b in zip(ref_labels, res.labels)]))
            except Exception:
                rec["ari_to_first"] = float("nan")
        else:
            ref_labels = res.labels
            rec["ari_to_first"] = 1.0
        if verbose:
            print(f"[eps sensitivity] eps={eps:<8g} L={rec['total']:.6e} "
                  f"ARI_to_first={rec['ari_to_first']:.3f}")
        out.append(rec)
    return out
