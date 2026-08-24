"""STEP 1 -- the epsilon-informativeness curve (spec 1.4, handoff s4, s8).

This runs before any state learning and answers the question that can kill the
project cheaply: *is there an epsilon at which the couplings are both
informative and stable?*

Two degeneracies bracket the usable window.

* eps -> infinity: P^eps -> a a^T, every cell gets the same fingerprint,
  I_cell -> 0 and L_pm is vacuous.
* eps -> 0: the plan is near-deterministic and dominated by sampling noise.

Quantities computed per interval and per epsilon:

* ``I_cell = KL(P^eps || a_t a_{t+1}^T)`` (Eq. 3) and its normalised version
  (Eq. 4) -- the mutual information between source and target cell identity;
* ``I_fingerprint`` -- ``sum_i a_i KL(f+_i || phibar+)`` at a provisional fine
  clustering of the target timepoint.  This is I(I_t; Z_{t+1}) and is the
  direct measure of whether fingerprints carry *cell-level* variation, which is
  what L_pm actually needs;
* the distribution of pairwise KL between subsampled cell fingerprints;
* the effective number of targets per source, exp(H(P(i,:)/a_i));
* stability under cell / culture-replicate resampling and under multiplicative
  cost perturbation, both measured on cells common to the two fits so the
  fingerprints are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import numpy as np

from .config import CouplingConfig, DEFAULT_EPSILON_GRID
from .cost import Support, build_support, perturb_cost
from .sinkhorn import SinkhornResult, sinkhorn_dense, sinkhorn_sparse
from .utils import entropy, uniform_weights


# ---------------------------------------------------------------------------
# scalar diagnostics on a solved coupling
# ---------------------------------------------------------------------------

def coupling_information(res: SinkhornResult, a: np.ndarray, b: np.ndarray) -> float:
    """I_cell = sum_{ij: P>0} P_ij log( P_ij / (a_i b_j) )   (Eq. 3)."""
    v = res.values
    m = v > 0
    v = v[m]
    return float((v * (np.log(v) - np.log(a[res.rows[m]]) - np.log(b[res.cols[m]]))).sum())


def normalized_coupling_information(res, a, b) -> float:
    """Eq. 4: I_cell / min(H(a), H(b))."""
    denom = min(float(entropy(a, axis=0)), float(entropy(b, axis=0)))
    return coupling_information(res, a, b) / max(denom, 1e-30)


def outgoing_fingerprints(res: SinkhornResult, labels_target: np.ndarray,
                          K: int) -> np.ndarray:
    """f+_i over provisional target classes, from P^ref only (Eq. 15)."""
    n = res.shape[0]
    F = np.zeros((n, K))
    np.add.at(F, (res.rows, labels_target[res.cols]), res.values)
    rs = F.sum(1, keepdims=True)
    rs[rs <= 0] = 1.0
    return F / rs


def incoming_fingerprints(res: SinkhornResult, labels_source: np.ndarray,
                          K: int) -> np.ndarray:
    """f-_j over provisional source classes (Eq. 16)."""
    m = res.shape[1]
    F = np.zeros((m, K))
    np.add.at(F, (res.cols, labels_source[res.rows]), res.values)
    cs = F.sum(1, keepdims=True)
    cs[cs <= 0] = 1.0
    return F / cs


def fingerprint_information(F: np.ndarray, a: np.ndarray) -> float:
    """sum_i a_i KL(f_i || phibar) with phibar = sum_i a_i f_i.

    Equals I(I_t; Z_{t+1}) at the provisional resolution: the total transition
    information available to be explained by states.  If this is ~0, L_pm has
    nothing to work with no matter how M is chosen.
    """
    phibar = (a[:, None] * F).sum(0)
    phibar = phibar / max(phibar.sum(), 1e-30)
    return float((a * (F * (np.log(np.maximum(F, 1e-30))
                            - np.log(np.maximum(phibar, 1e-30)))).sum(1)).sum())


def pairwise_kl_sample(F: np.ndarray, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """KL(f_i || f_j) for random cell pairs -- the spread of fingerprints."""
    n = F.shape[0]
    if n < 2:
        return np.array([])
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    P, Q = np.maximum(F[i], 1e-30), np.maximum(F[j], 1e-30)
    return (P * (np.log(P) - np.log(Q))).sum(1)


def effective_targets(res: SinkhornResult, a: np.ndarray) -> float:
    """Mean over source cells of exp(H(P(i,:)/a_i)) -- coupling sharpness."""
    n = res.shape[0]
    v = res.values
    rs = np.bincount(res.rows, weights=v, minlength=n)
    rs[rs <= 0] = 1.0
    p = v / rs[res.rows]
    contrib = np.zeros(n)
    m = p > 0
    np.add.at(contrib, res.rows[m], -p[m] * np.log(p[m]))
    return float((a * np.exp(contrib)).sum() / max(a.sum(), 1e-30))


# ---------------------------------------------------------------------------
# the scan
# ---------------------------------------------------------------------------

@dataclass
class EpsilonScanResult:
    epsilons: np.ndarray
    intervals: List[int]
    metrics: Dict[str, np.ndarray]        # name -> (n_eps, n_interval)
    pairwise_kl: Dict[int, Dict[float, np.ndarray]] = field(default_factory=dict)
    provisional_K: int = 0
    notes: Dict[str, object] = field(default_factory=dict)

    def mean_curve(self, name: str) -> np.ndarray:
        return np.nanmean(self.metrics[name], axis=1)

    def to_frame(self):
        """Long-format pandas frame (pandas optional)."""
        import pandas as pd
        rows = []
        for ei, e in enumerate(self.epsilons):
            for ii, t in enumerate(self.intervals):
                rec = {"epsilon": e, "interval": t}
                for k, v in self.metrics.items():
                    rec[k] = v[ei, ii]
                rows.append(rec)
        return pd.DataFrame(rows)

    def recommend(
        self,
        min_norm_info: float = 0.02,
        min_fingerprint_info: float = 1e-3,
        min_stability: float = 0.8,
    ) -> dict:
        """Pick the informative interval of epsilon (spec 1.4).

        Criteria, applied to the across-interval mean curve: nonzero coupling
        informativeness, fingerprints that actually vary across cells, and
        stability under both resampling and cost perturbation.  No single value
        is treated as uniquely correct -- an interval is returned and results
        should be reported across it.
        """
        ok = np.ones(len(self.epsilons), dtype=bool)
        ok &= self.mean_curve("I_cell_normalized") >= min_norm_info
        if "I_fingerprint_plus" in self.metrics:
            ok &= self.mean_curve("I_fingerprint_plus") >= min_fingerprint_info
        for key in ("stability_resample", "stability_cost"):
            if key in self.metrics:
                c = self.mean_curve(key)
                ok &= np.where(np.isnan(c), True, c >= min_stability)
        if "feasible" in self.metrics:
            ok &= self.mean_curve("feasible") > 0.999
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            return {"window": None, "epsilon_star": None,
                    "reason": "no epsilon satisfies all criteria -- "
                              "the informative window may not exist at this spacing"}
        lo, hi = float(self.epsilons[idx[0]]), float(self.epsilons[idx[-1]])
        # geometric centre of the admissible window
        star = float(np.exp(0.5 * (np.log(lo) + np.log(hi))))
        star = float(self.epsilons[np.argmin(np.abs(np.log(self.epsilons) - np.log(star)))])
        return {"window": (lo, hi), "epsilon_star": star,
                "admissible": self.epsilons[idx].tolist()}


def _provisional_labels(Z: np.ndarray, K: int, seed: int) -> np.ndarray:
    from .optimize import _kmeans
    return _kmeans(Z, int(min(K, Z.shape[0])), seed=seed)


def _solve(sup: Support, a, b, eps, cfg: CouplingConfig, f0=None, g0=None):
    if sup.dense:
        C = np.zeros(sup.shape)
        C[sup.rows, sup.cols] = sup.cost
        return sinkhorn_dense(C, a, b, eps, max_iter=cfg.max_iter, tol=cfg.tol,
                              f_init=f0, g_init=g0, device=cfg.device, dtype=cfg.dtype)
    return sinkhorn_sparse(sup, a, b, eps, max_iter=cfg.max_iter, tol=cfg.tol,
                           f_init=f0, g_init=g0, device=cfg.device, dtype=cfg.dtype)


def _fingerprint_agreement(F1: np.ndarray, F2: np.ndarray, a: np.ndarray) -> float:
    """Agreement in [0, 1]: 1 - normalised mean symmetric KL between two fits.

    Normalised by the information content of the fingerprints themselves, so a
    coupling that carries no information cannot score as "perfectly stable".
    """
    P, Q = np.maximum(F1, 1e-30), np.maximum(F2, 1e-30)
    sym = 0.5 * ((P * (np.log(P) - np.log(Q))).sum(1)
                 + (Q * (np.log(Q) - np.log(P))).sum(1))
    disagreement = float((a * sym).sum() / max(a.sum(), 1e-30))
    scale = 0.5 * (fingerprint_information(F1, a) + fingerprint_information(F2, a))
    if scale <= 1e-12:
        return float("nan")
    return float(max(0.0, 1.0 - disagreement / (2.0 * scale)))


def epsilon_scan(
    Z: Sequence[np.ndarray],
    tau: np.ndarray,
    epsilons: Sequence[float] = DEFAULT_EPSILON_GRID,
    cfg: CouplingConfig = CouplingConfig(),
    intervals: Optional[Sequence[int]] = None,
    provisional_K: int = 30,
    n_pairs: int = 20000,
    n_resample: int = 2,
    resample_fraction: float = 0.7,
    cost_perturbation: float = 0.05,
    replicate: Optional[Sequence[np.ndarray]] = None,
    seed: int = 0,
    verbose: int = 1,
) -> EpsilonScanResult:
    """Run the epsilon scan.  No memberships, no objective, no state learning.

    ``resample_fraction`` subsets *cells* (or whole culture replicates when
    ``replicate`` is given); fingerprints are then compared on the cells present
    in both fits, against a provisional target clustering that is fixed once on
    the full data so the two fits share a label space.
    """
    rng = np.random.default_rng(seed)
    Z = [np.asarray(z, dtype=np.float64) for z in Z]
    tau = np.asarray(tau, dtype=float)
    T = len(Z)
    intervals = list(range(T - 1)) if intervals is None else list(intervals)
    epsilons = np.asarray(list(epsilons), dtype=float)

    labels = [_provisional_labels(Z[t], provisional_K, seed=seed + t) for t in range(T)]
    Kprov = provisional_K

    names = ["I_cell", "I_cell_normalized", "I_fingerprint_plus",
             "I_fingerprint_minus", "eff_targets", "pairwise_kl_median",
             "pairwise_kl_p90", "marginal_error", "feasible", "n_iter",
             "stability_resample", "stability_cost"]
    metrics = {k: np.full((len(epsilons), len(intervals)), np.nan) for k in names}
    pairwise: Dict[int, Dict[float, np.ndarray]] = {t: {} for t in intervals}

    for ii, t in enumerate(intervals):
        dtau = float(tau[t + 1] - tau[t])
        a = uniform_weights(Z[t].shape[0])
        b = uniform_weights(Z[t + 1].shape[0])
        sup, scale = build_support(Z[t], Z[t + 1], dtau,
                                   kappa=None if cfg.support == "dense" else cfg.kappa,
                                   dense=(cfg.support == "dense"),
                                   normalize=cfg.normalize_cost)
        sup_pert = perturb_cost(sup, cost_perturbation, rng)

        if verbose:
            print(f"[eps-scan] interval {t} (tau {tau[t]:g}->{tau[t+1]:g}, "
                  f"dtau={dtau:g}) n={Z[t].shape[0]}x{Z[t+1].shape[0]} "
                  f"nnz={sup.nnz} cost_scale={scale:.3g}")

        # resampled subsets, reused across epsilon so the comparison is clean
        subsets = []
        for r in range(n_resample):
            if replicate is not None:
                groups_a = np.unique(replicate[t])
                groups_b = np.unique(replicate[t + 1])
                ga = rng.choice(groups_a, size=max(1, len(groups_a) // 2), replace=False)
                gb = rng.choice(groups_b, size=max(1, len(groups_b) // 2), replace=False)
                ia = np.flatnonzero(np.isin(replicate[t], ga))
                ib = np.flatnonzero(np.isin(replicate[t + 1], gb))
            else:
                ia = np.sort(rng.choice(Z[t].shape[0],
                                        size=max(2, int(resample_fraction * Z[t].shape[0])),
                                        replace=False))
                ib = np.sort(rng.choice(Z[t + 1].shape[0],
                                        size=max(2, int(resample_fraction * Z[t + 1].shape[0])),
                                        replace=False))
            sub_sup, _ = build_support(Z[t][ia], Z[t + 1][ib], dtau,
                                       kappa=None if cfg.support == "dense" else cfg.kappa,
                                       dense=(cfg.support == "dense"),
                                       normalize=cfg.normalize_cost, cost_scale=scale)
            subsets.append((ia, ib, sub_sup))

        f0 = g0 = None
        for ei, eps in enumerate(sorted(epsilons, reverse=True)):
            e_idx = int(np.flatnonzero(epsilons == eps)[0])
            res = _solve(sup, a, b, eps, cfg, f0, g0)
            f0, g0 = res.f, res.g   # warm start the next (smaller) epsilon

            metrics["I_cell"][e_idx, ii] = coupling_information(res, a, b)
            metrics["I_cell_normalized"][e_idx, ii] = normalized_coupling_information(res, a, b)
            metrics["eff_targets"][e_idx, ii] = effective_targets(res, a)
            metrics["marginal_error"][e_idx, ii] = res.marginal_error
            metrics["feasible"][e_idx, ii] = float(res.marginal_error < cfg.feasibility_tol)
            metrics["n_iter"][e_idx, ii] = res.n_iter

            Fp = outgoing_fingerprints(res, labels[t + 1], Kprov)
            Fm = incoming_fingerprints(res, labels[t], Kprov)
            metrics["I_fingerprint_plus"][e_idx, ii] = fingerprint_information(Fp, a)
            metrics["I_fingerprint_minus"][e_idx, ii] = fingerprint_information(Fm, b)

            kl = pairwise_kl_sample(Fp, n_pairs, rng)
            if kl.size:
                pairwise[t][float(eps)] = kl
                metrics["pairwise_kl_median"][e_idx, ii] = float(np.median(kl))
                metrics["pairwise_kl_p90"][e_idx, ii] = float(np.percentile(kl, 90))

            # -- stability under resampling -------------------------------
            agrees = []
            Fsubs = []
            for (ia, ib, sub_sup) in subsets:
                sub_res = _solve(sub_sup, uniform_weights(len(ia)),
                                 uniform_weights(len(ib)), eps, cfg)
                Fsub = outgoing_fingerprints(sub_res, labels[t + 1][ib], Kprov)
                Fsubs.append((ia, Fsub))
            for x in range(len(Fsubs)):
                for y in range(x + 1, len(Fsubs)):
                    ia_x, Fx = Fsubs[x]
                    ia_y, Fy = Fsubs[y]
                    common, px, py = np.intersect1d(ia_x, ia_y, return_indices=True)
                    if len(common) >= 10:
                        w = uniform_weights(len(common))
                        agrees.append(_fingerprint_agreement(Fx[px], Fy[py], w))
            if not agrees and len(Fsubs) >= 1:
                ia_x, Fx = Fsubs[0]
                agrees.append(_fingerprint_agreement(Fx, Fp[ia_x], uniform_weights(len(ia_x))))
            if agrees:
                metrics["stability_resample"][e_idx, ii] = float(np.nanmean(agrees))

            # -- stability under cost perturbation ------------------------
            res_p = _solve(sup_pert, a, b, eps, cfg)
            Fp_p = outgoing_fingerprints(res_p, labels[t + 1], Kprov)
            metrics["stability_cost"][e_idx, ii] = _fingerprint_agreement(Fp, Fp_p, a)

            if verbose > 1:
                print(f"    eps={eps:<8g} Ibar={metrics['I_cell_normalized'][e_idx, ii]:.4f} "
                      f"Ifp+={metrics['I_fingerprint_plus'][e_idx, ii]:.4f} "
                      f"eff_tgt={metrics['eff_targets'][e_idx, ii]:.1f} "
                      f"stab_rs={metrics['stability_resample'][e_idx, ii]:.3f} "
                      f"stab_C={metrics['stability_cost'][e_idx, ii]:.3f}")

    return EpsilonScanResult(epsilons=epsilons, intervals=intervals, metrics=metrics,
                             pairwise_kl=pairwise, provisional_K=Kprov,
                             notes={"cfg": cfg.__dict__.copy(),
                                    "resample_fraction": resample_fraction,
                                    "cost_perturbation": cost_perturbation,
                                    "replicate_based": replicate is not None})
