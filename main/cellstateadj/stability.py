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
from .data import TimeSeriesData, split_half_by_replicate
from .optimize import FitResult


def align_states(mu_ref: np.ndarray, mu_other: np.ndarray,
                 g_ref: Optional[np.ndarray] = None,
                 g_other: Optional[np.ndarray] = None) -> np.ndarray:
    """Match one fit's states to a reference fit's by expression prototype.

    Both fits live in the SAME frozen z space, so the prototypes mu_tk are
    directly comparable even when the two fits saw different cells.  Returns
    ``perm`` with ``perm[k]`` = the reference state matched to state k.
    """
    D = ((mu_other[:, None, :] - mu_ref[None, :, :]) ** 2).sum(-1)
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(D)
        perm = np.full(mu_other.shape[0], -1, dtype=int)
        perm[r] = c
        return perm
    except Exception:  # pragma: no cover
        return D.argmin(1)


@dataclass
class StabilityReport:
    kind: str                                   # what was resampled
    n_fits: int
    edge_support: List[np.ndarray] = field(default_factory=list)   # (T-1,) K x K
    mean_edge_support: float = float("nan")
    state_mass_sd: List[np.ndarray] = field(default_factory=list)
    per_fit: List[dict] = field(default_factory=list)
    notes: str = ""

    def caveat(self) -> str:
        return (f"{self.kind} stability over {self.n_fits} fits. "
                "This is algorithmic / culture-replicate stability, NOT "
                "biological uncertainty, unless the resampled units are "
                "independent biological replicates.")


def edge_support(
    fits: Sequence[FitResult],
    mus: Sequence[Sequence[np.ndarray]],
    delta_A: float = 0.05,
    reference: int = 0,
) -> StabilityReport:
    """Bootstrap support of Eq. 36 for aligned state-level edges.

        S_tkl = (1/R) sum_r 1[ A^(r)_tkl > delta_A ]

    Every fit's states are first aligned to the reference fit's by expression
    prototype; unmatched states contribute no support.  Report edge width as
    transition mass and edge opacity as this support.
    """
    import torch

    R = len(fits)
    ref = fits[reference]
    with torch.no_grad():
        _, A_ref, _, g_ref = ref.model.induced_transitions()
        K = ref.model.K
        T = ref.model.T

    counts = [np.zeros((K, K)) for _ in range(T - 1)]
    masses = [np.zeros((R, K)) for _ in range(T)]

    for r, f in enumerate(fits):
        import torch
        with torch.no_grad():
            _, A_r, _, g_r = f.model.induced_transitions()
        perms = [align_states(np.asarray(mus[reference][t]), np.asarray(mus[r][t]))
                 for t in range(T)]
        for t in range(T):
            gr = g_r[t].cpu().numpy()
            for k in range(K):
                if 0 <= perms[t][k] < K:
                    masses[t][r, perms[t][k]] += gr[k]
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
    verbose: int = 1,
) -> StabilityReport:
    """Refit on each culture-replicate half and compare the recovered structure.

    Each half gets its OWN reference chain -- the couplings are defined on the
    cell set being used, so reusing the full-data chain would leak information
    across the split.  The frozen representation is shared, because it is fit
    once by construction and refitting it per half would confound
    representation variability with state variability.
    """
    from .pipeline import run_pipeline
    from .representation import learn_representation

    if data.replicate is None:
        raise ValueError("split_half_stability requires replicate labels")

    Z_full, rep_info = learn_representation(data, cfg.representation)
    fits, mus, per_fit = [], [], []

    for s in range(n_splits):
        halves = split_half_by_replicate(data, seed=seed + s)
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
            fits.append(res.fit)
            mus.append([m for m in res.diagnostics.mu])
            per_fit.append(res.fit.summary())

    rep = edge_support(fits, mus)
    rep.kind = "culture-replicate split-half"
    rep.notes = ("Duplicate samples are culture/well-level replicates. "
                 "Report as culture-replicate stability, not biological "
                 "uncertainty.")
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
