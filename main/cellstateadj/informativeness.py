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

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence
import numpy as np

from .config import CouplingConfig, DEFAULT_EPSILON_GRID
from .cost import Support, build_support, perturb_cost, resolve_cost_scales
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

        A criterion that could not be EVALUATED (NaN) counts as FAILED, not as
        passed.  The spec requires positive evidence of stability; treating a
        missing measurement as satisfying it is how an epsilon gets declared
        admissible with no stability evidence behind it at all.  Which criteria
        went unevaluated is reported in ``unevaluated``.
        """
        ok = np.ones(len(self.epsilons), dtype=bool)
        unevaluated: Dict[str, int] = {}

        def _apply(curve: np.ndarray, threshold: float, name: str) -> None:
            nonlocal ok
            nan = np.isnan(curve)
            if nan.any():
                unevaluated[name] = int(nan.sum())
            ok = ok & np.where(nan, False, curve >= threshold)

        _apply(self.mean_curve("I_cell_normalized"), min_norm_info, "I_cell_normalized")
        if "I_fingerprint_plus" in self.metrics:
            _apply(self.mean_curve("I_fingerprint_plus"), min_fingerprint_info,
                   "I_fingerprint_plus")
        for key in ("stability_resample", "stability_cost"):
            if key in self.metrics:
                _apply(self.mean_curve(key), min_stability, key)
        if "feasible" in self.metrics:
            _apply(self.mean_curve("feasible"), 0.999, "feasible")

        idx = np.flatnonzero(ok)
        if idx.size == 0:
            return {"window": None, "epsilon_star": None,
                    "unevaluated": unevaluated,
                    "reason": "no epsilon satisfies all criteria -- "
                              "the informative window may not exist at this spacing"
                              + (f" (note: {unevaluated} could not be evaluated "
                                 f"and were counted as failures)" if unevaluated else "")}

        # The admissible set need not be contiguous.  Taking (first, last) as a
        # window and picking its geometric centre can land on an epsilon that
        # FAILED -- e.g. pass = [True, False, True] would recommend the middle
        # value.  So: split into contiguous runs, keep the longest (ties -> the
        # one with the best I_cell_normalized), and choose epsilon_star from
        # inside that run, always a value that actually passed.
        order = np.argsort(self.epsilons)          # runs are in epsilon order
        ok_sorted = ok[order]
        runs, start = [], None
        for pos in range(len(order)):
            if ok_sorted[pos] and start is None:
                start = pos
            if start is not None and (pos == len(order) - 1 or not ok_sorted[pos + 1]):
                runs.append((start, pos))
                start = None

        info = self.mean_curve("I_cell_normalized")[order]
        def _run_key(r):
            lo_i, hi_i = r
            seg = info[lo_i:hi_i + 1]
            return (hi_i - lo_i + 1, float(np.nanmax(seg)) if seg.size else -np.inf)
        best_run = max(runs, key=_run_key)
        lo_i, hi_i = best_run
        run_idx = order[lo_i:hi_i + 1]
        run_eps = self.epsilons[run_idx]

        # geometric centre of the run, snapped to the nearest ADMISSIBLE value
        target = float(np.exp(0.5 * (np.log(run_eps[0]) + np.log(run_eps[-1]))))
        star = float(run_eps[np.argmin(np.abs(np.log(run_eps) - np.log(target)))])

        return {"window": (float(run_eps[0]), float(run_eps[-1])),
                "epsilon_star": star,
                "admissible": sorted(float(e) for e in self.epsilons[idx]),
                "admissible_window": [float(e) for e in run_eps],
                "n_admissible_runs": len(runs),
                "unevaluated": unevaluated}


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
    replicate_paired: bool = True,
    cost_scales: Optional[Sequence[float]] = None,
    seed: int = 0,
    verbose: int = 1,
) -> EpsilonScanResult:
    """Run the epsilon scan.  No memberships, no objective, no state learning.

    ``resample_fraction`` subsets *cells* (or whole culture replicates when
    ``replicate`` is given); fingerprints are then compared on the cells present
    in both fits, against a provisional target clustering that is fixed once on
    the full data so the two fits share a label space.

    ``replicate_paired`` (default True) holds the SAME replicate group out at
    both ends of an interval.  WOT ran parallel time courses, so a replicate
    label plausibly tracks one culture lineage across time; drawing the source
    and target groups independently would then leave the same experimental unit
    on both sides of the split and overstate stability.  Set it False only if
    the labels really are independent wells harvested per timepoint.  Groups
    come from the labels shared by both ends of the interval; if fewer than two
    are shared, no paired hold-out exists and ``stability_resample`` is left
    unevaluated (NaN, which fails admissibility) rather than falling back to an
    unrelated target group.

    [CRITICAL] With replicate labels the subsets are ENUMERATED -- one per
    replicate group, deterministically -- and ``n_resample`` does not apply.
    Drawing groups at random duplicates them: with two groups, two independent
    draws of one group coincide about half the time, the two "resampled"
    datasets are bit-identical, and the agreement is exactly 1.0 for no reason
    at all.  Replicate subsets are also disjoint, so there is no shared cell for
    a pairwise comparison; each subset is instead scored against the full
    coupling on its own cells and the scores averaged over ALL subsets.
    ``stability_is_vs_full`` records which of the two comparisons was used
    (1.0 = against the full coupling, 0.0 = pairwise between overlapping
    subsets), and ``n_resample_subsets`` the number actually evaluated.

    Every resampled and cost-perturbed plan is held to the same
    ``cfg.feasibility_tol`` as the main one, and each resampled support grows its
    own kappa: an infeasible plan is not a noisy coupling, it is not a coupling,
    and two of them agreeing is not stability.  ``n_resample_feasible`` records
    how many survived at each epsilon.

    The support is grown until a balanced plan exists, using the same helper and
    the same ``cfg.feasibility_tol`` as the final chain builder, so an epsilon
    cannot pass here under a weaker standard than it will later face.
    """
    from .reference import (InfeasibleCouplingError,
                            SinkhornConvergenceError, solve_interval)

    rng = np.random.default_rng(seed)
    Z = [np.asarray(z, dtype=np.float64) for z in Z]
    tau = np.asarray(tau, dtype=float)
    T = len(Z)
    intervals = list(range(T - 1)) if intervals is None else list(intervals)
    epsilons = np.asarray(list(epsilons), dtype=float)

    # One scale for every interval, matching build_reference_chain: a
    # per-interval median would cancel dtau and make the scan blind to spacing.
    #
    # ``cost_scales`` lets a caller pin the scale computed from a DIFFERENT
    # series -- the delta-tau study needs this, because recomputing the scale
    # per stride would make stride-1 and stride-4 differ by cost normalisation
    # as well as by spacing.
    if cost_scales is None:
        scales = resolve_cost_scales(Z, tau, cfg.cost_scale_mode)
    else:
        scales = [float(x) for x in cost_scales]
        if len(scales) != len(Z) - 1:
            raise ValueError(
                f"cost_scales must have one entry per interval "
                f"({len(Z) - 1}), got {len(scales)}")

    labels = [_provisional_labels(Z[t], provisional_K, seed=seed + t) for t in range(T)]
    Kprov = provisional_K

    names = ["I_cell", "I_cell_normalized", "I_fingerprint_plus",
             "I_fingerprint_minus", "eff_targets", "pairwise_kl_median",
             "pairwise_kl_p90", "marginal_error", "feasible", "n_iter",
             "stability_resample", "stability_cost",
             "n_resample_subsets", "n_resample_feasible",
             "marginal_error_perturbed", "stability_is_vs_full"]
    metrics = {k: np.full((len(epsilons), len(intervals)), np.nan) for k in names}
    pairwise: Dict[int, Dict[float, np.ndarray]] = {t: {} for t in intervals}

    for ii, t in enumerate(intervals):
        dtau = float(tau[t + 1] - tau[t])
        a = uniform_weights(Z[t].shape[0])
        b = uniform_weights(Z[t + 1].shape[0])
        # Size the support at the LARGEST epsilon on the grid.
        #
        # Whether a support admits a balanced plan at all is a combinatorial
        # property of the bipartite graph (a Hall-type condition on the
        # marginals) and does not depend on epsilon.  What DOES depend on
        # epsilon is conditioning: at small epsilon exp(-C/eps) underflows and
        # Sinkhorn cannot hit the tolerance no matter how good the support is.
        #
        # Sizing at the smallest epsilon therefore conflates the two: the solve
        # fails for numerical reasons, kappa growth is abandoned, and every
        # epsilon is then scored on an under-sized support -- so the scan can
        # report "no informative window" when a perfectly good one exists.
        # Sizing at the largest epsilon isolates the combinatorial question,
        # which is the one that determines kappa.
        try:
            _, sup, scale = solve_interval(
                Z[t], Z[t + 1], dtau, a, b,
                replace(cfg, epsilon=float(np.max(epsilons)),
                        on_infeasible="warn"),
                cost_scale=scales[t], interval=t, verbose=max(0, verbose - 1),
            )
        except (InfeasibleCouplingError, SinkhornConvergenceError) as exc:
            # The scan must never abort: an epsilon that cannot be solved is a
            # RESULT to record (feasible=0 for that row), not a crash.
            if verbose:
                print(f"  [interval {t}] support sizing at eps="
                      f"{np.max(epsilons):g} did not settle ({type(exc).__name__}); "
                      f"falling back to kappa={cfg.kappa} and reporting "
                      f"feasibility per epsilon")
            sup, scale = build_support(
                Z[t], Z[t + 1], dtau,
                kappa=None if cfg.support == "dense" else cfg.kappa,
                dense=(cfg.support == "dense"), cost_scale=scales[t])
        sup_pert = perturb_cost(sup, cost_perturbation, rng)

        if verbose:
            print(f"[eps-scan] interval {t} (tau {tau[t]:g}->{tau[t+1]:g}, "
                  f"dtau={dtau:g}) n={Z[t].shape[0]}x{Z[t+1].shape[0]} "
                  f"nnz={sup.nnz} kappa={sup.kappa} cost_scale={scale:.3g}")

        # resampled subsets, reused across epsilon so the comparison is clean.
        #
        # [CRITICAL] Each subset grows its OWN kappa.  A subset has fewer cells,
        # so the kappa that made the FULL support feasible can leave the subset
        # without a balanced plan -- and an infeasible plan still produces
        # fingerprints, which still agree with each other, which still scores as
        # perfect stability.  A completely invalid pair of plans could therefore
        # push an epsilon over the stability threshold and into the admissible
        # window.  Feasibility is a combinatorial property of the support, so
        # kappa is grown once at the largest epsilon, exactly as for the main
        # support above.
        # [CRITICAL] With replicate labels the subsets are ENUMERATED, not
        # drawn.  Randomly drawing one of two groups per resample draws the SAME
        # group twice about half the time; the two "resampled" datasets are then
        # bit-identical and agree perfectly, so stability_resample came out at
        # exactly 1.0 on 7 of 12 seeds of the same data -- and when the draws
        # differed, the two subsets were disjoint, the pairwise comparison had
        # no shared cells to run on, and the fallback scored only the FIRST
        # subset against the full coupling and ignored the second.  Neither
        # number measured stability, and both could admit an epsilon.
        #
        # One subset per replicate group, deterministically, is the honest
        # version of "resample the replicates" when the replicates are the
        # sampling unit: it uses every group exactly once and cannot duplicate.
        # ``n_resample`` therefore does not apply here -- the number of subsets
        # is the number of replicate groups.
        #
        # Each subset also grows its OWN kappa.  A subset has fewer cells, so
        # the kappa that made the FULL support feasible can leave it without a
        # balanced plan -- and an infeasible plan still produces fingerprints,
        # which still agree with each other, which still scores as stable.
        # Feasibility is combinatorial, so kappa is grown once at the largest
        # epsilon, exactly as for the main support above.
        subsets = []
        group_ids: List[object] = []
        draws = []
        if replicate is not None:
            groups_a = np.unique(replicate[t])
            groups_b = np.unique(replicate[t + 1])
            if replicate_paired:
                shared = np.intersect1d(groups_a, groups_b)
                if shared.size < 2:
                    # No group is present at BOTH ends, so no paired hold-out
                    # exists.  Silently drawing an unrelated target group would
                    # compare two different cultures and call the disagreement
                    # "instability"; leaving stability unevaluated (NaN) is the
                    # honest result, and NaN already fails admissibility.
                    if verbose:
                        print(f"  [interval {t}] paired resampling impossible: "
                              f"{shared.size} replicate group(s) shared between "
                              f"tau={tau[t]:g} and tau={tau[t+1]:g}; "
                              f"stability_resample left unevaluated")
                else:
                    for gid in shared:
                        draws.append((gid, np.asarray([gid]), np.asarray([gid])))
            else:
                # independent wells per timepoint: pair them up by position,
                # still one subset per group and still without replacement
                n_pair = min(len(groups_a), len(groups_b))
                for j in range(n_pair):
                    draws.append((f"{groups_a[j]}|{groups_b[j]}",
                                  np.asarray([groups_a[j]]),
                                  np.asarray([groups_b[j]])))
        else:
            for r in range(n_resample):
                draws.append((r, None, None))

        for (gid, ga, gb) in draws:
            if replicate is not None:
                ia = np.flatnonzero(np.isin(replicate[t], ga))
                ib = np.flatnonzero(np.isin(replicate[t + 1], gb))
            else:
                ia = np.sort(rng.choice(Z[t].shape[0],
                                        size=max(2, int(resample_fraction * Z[t].shape[0])),
                                        replace=False))
                ib = np.sort(rng.choice(Z[t + 1].shape[0],
                                        size=max(2, int(resample_fraction * Z[t + 1].shape[0])),
                                        replace=False))
            if len(ia) < 2 or len(ib) < 2:
                continue
            try:
                _, sub_sup, _ = solve_interval(
                    Z[t][ia], Z[t + 1][ib], dtau,
                    uniform_weights(len(ia)), uniform_weights(len(ib)),
                    replace(cfg, epsilon=float(np.max(epsilons)),
                            on_infeasible="warn"),
                    cost_scale=scale, interval=t, verbose=0,
                )
            except (InfeasibleCouplingError, SinkhornConvergenceError):
                sub_sup, _ = build_support(
                    Z[t][ia], Z[t + 1][ib], dtau,
                    kappa=None if cfg.support == "dense" else sup.kappa,
                    dense=(cfg.support == "dense"), cost_scale=scale)
            subsets.append((ia, ib, sub_sup))
            group_ids.append(gid)
        if len(set(map(str, group_ids))) != len(group_ids):   # cannot happen
            raise AssertionError(f"duplicate resample subsets: {group_ids}")
        metrics["n_resample_subsets"][:, ii] = float(len(subsets))

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
            #
            # Every resampled plan must clear the SAME feasibility_tol as the
            # main one.  A plan whose marginals are wrong is not a worse
            # estimate of the coupling, it is not the coupling at all, and two
            # such plans agreeing with each other is not evidence of stability.
            main_feasible = res.marginal_error < cfg.feasibility_tol
            agrees = []
            Fsubs = []
            n_sub_feasible = 0
            for (ia, ib, sub_sup) in subsets:
                sub_res = _solve(sub_sup, uniform_weights(len(ia)),
                                 uniform_weights(len(ib)), eps, cfg)
                if sub_res.marginal_error >= cfg.feasibility_tol:
                    continue
                n_sub_feasible += 1
                Fsub = outgoing_fingerprints(sub_res, labels[t + 1][ib], Kprov)
                Fsubs.append((ia, Fsub))
            metrics["n_resample_feasible"][e_idx, ii] = float(n_sub_feasible)
            if not main_feasible:
                Fsubs = []
            for x in range(len(Fsubs)):
                for y in range(x + 1, len(Fsubs)):
                    ia_x, Fx = Fsubs[x]
                    ia_y, Fy = Fsubs[y]
                    common, px, py = np.intersect1d(ia_x, ia_y, return_indices=True)
                    if len(common) >= 10:
                        w = uniform_weights(len(common))
                        agrees.append(_fingerprint_agreement(Fx[px], Fy[py], w))
            if agrees:
                metrics["stability_is_vs_full"][e_idx, ii] = 0.0
            if not agrees:
                # Replicate subsets are DISJOINT by construction, so there is no
                # shared cell to compare them on.  Score each subset against the
                # full coupling restricted to that subset's own cells and
                # average -- every subset contributes, which the old code did
                # not do: it scored Fsubs[0] alone and discarded the rest.
                for ia_x, Fx in Fsubs:
                    agrees.append(_fingerprint_agreement(
                        Fx, Fp[ia_x], uniform_weights(len(ia_x))))
                metrics["stability_is_vs_full"][e_idx, ii] = 1.0
            if agrees:
                metrics["stability_resample"][e_idx, ii] = float(np.nanmean(agrees))

            # -- stability under cost perturbation ------------------------
            # Same rule: an infeasible perturbed plan leaves this unevaluated
            # rather than scoring it.
            res_p = _solve(sup_pert, a, b, eps, cfg)
            metrics["marginal_error_perturbed"][e_idx, ii] = res_p.marginal_error
            if main_feasible and res_p.marginal_error < cfg.feasibility_tol:
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
                                    "replicate_based": replicate is not None,
                                    "resample_mode": ("enumerated replicate "
                                                      "groups"
                                                      if replicate is not None
                                                      else "random cell subsets"),
                                    "replicate_paired": replicate_paired,
                                    "cost_scale_mode": cfg.cost_scale_mode,
                                    "cost_scales_supplied": cost_scales is not None,
                                    "cost_scales": scales})
