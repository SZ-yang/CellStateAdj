#!/usr/bin/env python
"""Workstream A -- fixed-anchor cell-level fingerprint geometry (A1-A5).

Asks whether reproducible transition-role information EXISTS in the frozen chain,
before asking any joint objective to discover it.  The learned neighbour
memberships are replaced by a frozen fine anchor system, so nothing measured here
can be gamed by collapsing the target alphabet (Degeneracy 3).

Analyses, per the plan:

  A1 informativeness  -- I(cell;anchor), fingerprint entropy, effective
                         destination/origin counts, pairwise JS, independence null
  A2 reproducibility  -- batch hold-out, bootstrap, neighbouring epsilon, support
                         perturbation, anchor resolution; always compared in the
                         SAME frozen anchor coordinates
  A3 vs expression    -- expression distance vs fingerprint distance, a
                         neighbourhood-preserving permutation null, and how well
                         expression alone predicts the fingerprint held out
  A4 geometry         -- embeddings, local intrinsic dimension, cluster tendency
                         and multiresolution stability
  A5 held-out states  -- fingerprint clustering vs expression clustering at matched
                         K, scored by held-out fixed-anchor CMI

Interval-level results are always kept alongside aggregates: the 2026-09-07 memo's
central error was averaging over intervals that behaved differently.

    python 05_fixed_anchor_fingerprints.py --chain <stage0>/chain_eps0.2.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import csa_anchors as ca
import csa_wot
from csa_wot import (DEFAULT_H5AD, RESULTS_ROOT, jdump, load_serum, make_cfg,
                     provenance_block, run_fingerprint)

from cellstateadj.reference import ReferenceChain, build_reference_chain
from cellstateadj.utils import uniform_weights


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", default=DEFAULT_H5AD)
    p.add_argument("--chain", default=None,
                   help="Stage-0 chain .npz; if omitted it is BUILT here and a "
                        "warning is printed, because the plan requires one shared chain")
    p.add_argument("--day-min", type=float, default=8.25)
    p.add_argument("--day-max", type=float, default=18.0)
    p.add_argument("--epsilon", type=float, default=0.2,
                   help="used to build a chain if --chain is absent, and as the "
                        "centre of the epsilon sensitivity in A2")
    p.add_argument("--epsilon-neighbours", type=float, nargs="*", default=[0.1, 0.5],
                   help="A2: neighbouring epsilons for the sensitivity check")
    p.add_argument("--anchors", type=int, nargs="+", default=[20, 40, 80],
                   help="A2/A4: anchor resolutions")
    p.add_argument("--support", default="knn", choices=["knn", "dense"])
    p.add_argument("--kappa", type=int, default=400)
    p.add_argument("--kappa-perturb", type=int, nargs="*", default=[200, 800],
                   help="A2: support-size perturbations")
    p.add_argument("--cost-scale-mode", default="global",
                   choices=["global", "per_interval", "none"])
    p.add_argument("--train-batch", default="1",
                   help="anchors are learned on this batch and FROZEN")
    p.add_argument("--n-bootstrap", type=int, default=3)
    p.add_argument("--bootstrap-frac", type=float, default=0.7)
    p.add_argument("--n-pairs", type=int, default=20000)
    p.add_argument("--perm-neighbors", type=int, default=30)
    p.add_argument("--perm-reps", type=int, default=20)
    p.add_argument("--stability-K", type=int, nargs="+", default=[4, 8],
                   help="A4: K values for resampled partition reproducibility")
    p.add_argument("--stability-reps", type=int, default=8)
    p.add_argument("--match-K", type=int, nargs="+", default=[4, 8, 12, 20],
                   help="A5: matched complexity for fingerprint vs expression states")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["A1", "A2", "A3", "A4", "A5"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(RESULTS_ROOT, "fixed_anchor"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


# --------------------------------------------------------------------------

def _flush(state, out):
    """Write after every analysis: these are long jobs and partials are useful."""
    jdump(state, os.path.join(out, "summary.json"))
    try:
        import pandas as pd
        if state.get("interval_rows"):
            pd.DataFrame(state["interval_rows"]).to_csv(
                os.path.join(out, "interval_metrics.csv"), index=False)
        if state.get("summary_rows"):
            pd.DataFrame(state["summary_rows"]).to_csv(
                os.path.join(out, "summary.csv"), index=False)
    except Exception:
        pass


def _chain_for(Z, tau, eps, support, kappa, cost_scale_mode, scales, verbose):
    cfg = make_cfg(epsilon=eps, support=support, kappa=kappa,
                   cost_scale_mode=cost_scale_mode, verbose=0)
    return build_reference_chain(Z, tau, cfg.coupling,
                                 cost_scales=scales, verbose=max(0, verbose - 1))


def _fp_stats(F, a, n_pairs, rng, label):
    """A1 for one (interval, direction): information, entropy, spread, null."""
    if F is None:
        return None
    ic = ca.cell_information(F, a)
    ent = -(F * np.log(np.maximum(F, 1e-30))).sum(1)
    eff = np.exp(ent)
    kl = ca.pairwise_sample(F, n_pairs, rng)
    null = ca.independence_null(F, a)
    return {
        "direction": label,
        "n_anchors": int(F.shape[1]),
        "i_cell_anchor": ic,
        "i_cell_anchor_null": ca.cell_information(null, a),
        "mean_entropy": float((a * ent).sum()),
        "eff_targets_mean": float((a * eff).sum()),
        "eff_targets_median": float(np.median(eff)),
        "js_median": float(np.median(kl)) if kl.size else float("nan"),
        "js_p90": float(np.percentile(kl, 90)) if kl.size else float("nan"),
    }



def _resolve_outdir(base, cfg_hash, args):
    """Config-hashed output directory that refuses to mix runs (issue 8).

    ``--overwrite`` on a shared directory is not enough: an earlier run at a
    different K, epsilon or lambda grid leaves membership NPZ and CSV files behind
    that a later glob or reducer will happily pick up.  Keying the directory on the
    configuration hash means a collision can only mean "this exact configuration
    already ran", and even then the directory must be empty or explicitly
    overwritten.
    """
    out = os.path.join(base, f"cfg_{cfg_hash}")
    os.makedirs(out, exist_ok=True)
    existing = os.listdir(out)
    if existing and not getattr(args, "overwrite", False):
        raise SystemExit(
            f"{out} already contains {len(existing)} file(s). Runs are keyed by "
            f"configuration hash, so this means the same configuration already ran. "
            f"Use a fresh --out, or --overwrite to replace it (which may leave "
            f"stale artifacts from a different lambda/K grid beside the new ones).")
    print(f"[out] config hash {cfg_hash} -> {out}")
    return out

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    data, Z, obs = load_serum(args.h5ad, day_min=args.day_min, day_max=args.day_max,
                              seed=args.seed, verbose=args.verbose)
    T = data.T
    a_full = [uniform_weights(n) for n in data.n_cells]

    from cellstateadj.cost import resolve_cost_scales
    scales = resolve_cost_scales(Z, data.tau,
                                 make_cfg(cost_scale_mode=args.cost_scale_mode,
                                          verbose=0).coupling.cost_scale_mode)

    # ---- the chain -------------------------------------------------------
    if not args.chain:
        raise SystemExit(
            "--chain is required. The plan mandates ONE shared Stage-0 chain "
            "(requirement 7: 'do not silently rebuild it inside individual jobs'); "
            "building one here would make these results incomparable with "
            "Workstream B. Run 00_stage0_chain.py first.")
    chain = ReferenceChain.load(args.chain)
    identity = ca.validate_chain_identity(chain, data, Z, chain_path=args.chain)
    ca.check_identity_args(identity, h5ad=args.h5ad, day_min=args.day_min,
                           day_max=args.day_max, stride=1, n_per_timepoint=None,
                           seed=args.seed)
    chain_src = os.path.abspath(args.chain)
    print(f"[A] Stage-0 chain {os.path.basename(args.chain)} (eps={chain.epsilon:g}) "
          f"-- identity verified: {chain.T} timepoints, cell ids and order, tau, "
          f"representation hash, feasibility")

    cfg_fp = make_cfg(epsilon=chain.epsilon, support=args.support, kappa=args.kappa,
                      cost_scale_mode=args.cost_scale_mode, seed=args.seed, verbose=0)
    prov = provenance_block(run_fingerprint(
        Z, cfg_fp, h5ad=args.h5ad, day_min=args.day_min, day_max=args.day_max,
        stride=1, n_per_timepoint=None, sampling_seed=args.seed))
    args.out = _resolve_outdir(args.out, prov["fingerprint_hash"], args)

    state = {"provenance": prov, "chain_source": chain_src,
             "epsilon": float(chain.epsilon), "support": args.support,
             "kappa": args.kappa, "cost_scale_mode": args.cost_scale_mode,
             "anchor_resolutions": args.anchors, "train_batch": args.train_batch,
             "tau": np.asarray(data.tau).tolist(),
             "interval_rows": [], "summary_rows": [], "analyses": {}}

    # ---- frozen anchors, learned on the training batch only --------------
    tr_idx, te_idx = ca.batch_split(data, args.train_batch)
    print(f"[A] anchors learned on batch {args.train_batch} "
          f"({sum(len(i) for i in tr_idx)} cells), frozen, then applied to all")
    centroids = {}
    for nA in args.anchors:
        Z_tr = [Z[t][tr_idx[t]] for t in range(T)]
        centroids[nA] = ca.make_anchors(Z_tr, nA, seed=args.seed)
    np.savez_compressed(
        os.path.join(args.out, "anchor_centroids.npz"),
        **{f"nA{nA}_t{t}": centroids[nA][t] for nA in args.anchors for t in range(T)})
    jdump({"train_batch": args.train_batch,
           "train_idx": {str(t): tr_idx[t].tolist() for t in range(T)},
           "test_idx": {str(t): te_idx[t].tolist() for t in range(T)},
           "cell_ids": {str(t): np.asarray(data.obs[t]["index"]).tolist()
                        for t in range(T)}},
          os.path.join(args.out, "splits.json"))

    A = {nA: ca.assign_anchors(Z, centroids[nA]) for nA in args.anchors}
    Fp = {}; Fm = {}
    for nA in args.anchors:
        Fp[nA], Fm[nA] = ca.fixed_fingerprints(chain, A[nA])
    np.savez_compressed(
        os.path.join(args.out, f"fingerprints_full_eps{chain.epsilon:g}.npz"),
        **{f"nA{nA}_{d}_{t}": (Fp[nA][t] if d == "plus" else Fm[nA][t])
           for nA in args.anchors for d in ("plus", "minus") for t in range(T)
           if (Fp[nA][t] if d == "plus" else Fm[nA][t]) is not None})

    # ================= A1 informativeness ================================
    if "A1" not in args.skip:
        print("\n[A1] cell-level fingerprint informativeness")
        for nA in args.anchors:
            for t in range(T):
                for lab, F in (("plus", Fp[nA][t]), ("minus", Fm[nA][t])):
                    r = _fp_stats(F, a_full[t], args.n_pairs, rng, lab)
                    if r is None:
                        continue
                    r.update(analysis="A1", n_anchor_setting=nA, t=t,
                             day=float(data.tau[t]))
                    state["interval_rows"].append(r)
            rows = [r for r in state["interval_rows"]
                    if r["analysis"] == "A1" and r["n_anchor_setting"] == nA]
            for lab in ("plus", "minus"):
                v = [r["i_cell_anchor"] for r in rows if r["direction"] == lab]
                print(f"  nA={nA:3d} {lab:5s}: I(cell;anchor) mean {np.mean(v):.4f} "
                      f"min {np.min(v):.4f} max {np.max(v):.4f} (nats, over "
                      f"{len(v)} timepoints)")
        state["analyses"]["A1"] = "done"
        _flush(state, args.out)

    # ================= A2 reproducibility ================================
    if "A2" not in args.skip:
        print("\n[A2] reproducibility in FROZEN anchor coordinates")
        a2 = []

        def _compare(tag, chain_b, idx_b=None, nA=None):
            """JS between the reference and a perturbed fingerprint, per interval.

            ``idx_b`` maps perturbed-set rows back to full-set rows so the two are
            compared on the SAME cells; without it the comparison is meaningless.
            """
            Ab = ([A[nA][t][idx_b[t]] for t in range(T)] if idx_b is not None
                  else A[nA])
            Fpb, Fmb = ca.fixed_fingerprints(chain_b, Ab)
            for t in range(T):
                for lab, Fref, Fb in (("plus", Fp[nA][t], Fpb[t]),
                                      ("minus", Fm[nA][t], Fmb[t])):
                    if Fref is None or Fb is None:
                        continue
                    ref = Fref if idx_b is None else Fref[idx_b[t]]
                    w = uniform_weights(len(ref))
                    js = ca.js_divergence(ref, Fb)
                    a2.append({"analysis": "A2", "perturbation": tag,
                               "n_anchor_setting": nA, "t": t,
                               "day": float(data.tau[t]), "direction": lab,
                               "n_cells_compared": int(len(ref)),
                               "js_mean": float((w * js).sum()),
                               "js_median": float(np.median(js)),
                               "js_p90": float(np.percentile(js, 90)),
                               "i_cell_ref": ca.cell_information(ref, w),
                               "i_cell_perturbed": ca.cell_information(Fb, w)})

        for nA in args.anchors:
            # batch hold-out: rebuild the chain on one batch only
            for b in sorted({str(v) for v in np.asarray(data.replicate[0]).astype(str)}):
                idx, _ = ca.batch_split(data, b)
                Zb = [Z[t][idx[t]] for t in range(T)]
                cb = _chain_for(Zb, data.tau, chain.epsilon, args.support,
                                args.kappa, args.cost_scale_mode, scales, args.verbose)
                _compare(f"batch_{b}", cb, idx, nA)
            # bootstrap
            for r in range(args.n_bootstrap):
                idx = ca.bootstrap_indices(data, args.bootstrap_frac, args.seed + r)
                Zb = [Z[t][idx[t]] for t in range(T)]
                cb = _chain_for(Zb, data.tau, chain.epsilon, args.support,
                                args.kappa, args.cost_scale_mode, scales, args.verbose)
                _compare(f"bootstrap{r}", cb, idx, nA)
            # neighbouring epsilon
            for e in args.epsilon_neighbours:
                cb = _chain_for(Z, data.tau, e, args.support, args.kappa,
                                args.cost_scale_mode, scales, args.verbose)
                _compare(f"eps_{e:g}", cb, None, nA)
            # support perturbation
            for kp in args.kappa_perturb:
                cb = _chain_for(Z, data.tau, chain.epsilon, args.support, kp,
                                args.cost_scale_mode, scales, args.verbose)
                _compare(f"kappa_{kp}", cb, None, nA)
            print(f"  nA={nA}: {len([r for r in a2 if r['n_anchor_setting']==nA])} "
                  f"interval comparisons")

        state["interval_rows"].extend(a2)
        # aggregate per perturbation, but keep the per-interval spread visible
        agg = {}
        for r in a2:
            key = (r["perturbation"], r["n_anchor_setting"], r["direction"])
            agg.setdefault(key, []).append(r["js_mean"])
        state["analyses"]["A2"] = {
            f"{k[0]}|nA{k[1]}|{k[2]}": {"js_mean_mean": float(np.mean(v)),
                                        "js_mean_max": float(np.max(v)),
                                        "n_intervals": len(v)}
            for k, v in sorted(agg.items())}
        for k, v in list(state["analyses"]["A2"].items())[:12]:
            print(f"    {k:34s} JS mean {v['js_mean_mean']:.4f} "
                  f"worst-interval {v['js_mean_max']:.4f}")
        _flush(state, args.out)

    # ================= A3 relationship to expression =====================
    if "A3" not in args.skip:
        print("\n[A3] fingerprint structure vs expression geometry")
        print("     (the old global-I permutation statistic was removed: it responded "
              "to fingerprint diversity, not to expression correspondence)")
        a3 = []
        for nA in args.anchors:
            for t in range(T):
                for lab, F in (("plus", Fp[nA][t]), ("minus", Fm[nA][t])):
                    if F is None:
                        continue
                    n = len(F)
                    i = rng.integers(0, n, size=args.n_pairs)
                    j = rng.integers(0, n, size=args.n_pairs)
                    m = i != j
                    dz = np.sqrt(((Z[t][i[m]] - Z[t][j[m]]) ** 2).sum(1))
                    dfp = ca.js_divergence(F[i[m]], F[j[m]])
                    ok = np.isfinite(dz) & np.isfinite(dfp)
                    # does expression LOCALITY predict fingerprint similarity?
                    conc = ca.expression_neighbour_concordance(
                        Z[t], F, n_neighbors=args.perm_neighbors,
                        n_pairs=args.n_pairs, seed=args.seed + t)
                    # cross-fitted, batch-grouped: what survives an expression model?
                    rep = np.asarray(data.replicate[t]).astype(str)
                    pred = ca.expression_predicts_fingerprint(
                        Z[t], F, n_neighbors=args.perm_neighbors,
                        seed=args.seed + t, groups=rep)
                    a3.append({
                        "analysis": "A3", "n_anchor_setting": nA, "t": t,
                        "day": float(data.tau[t]), "direction": lab,
                        "spearman_dz_vs_dfingerprint": float(_spearman(dz[ok], dfp[ok])),
                        "js_neighbour": conc["js_neighbour"],
                        "js_random": conc["js_random"],
                        "js_neighbour_over_random": conc["ratio"],
                        "kl_to_expression": pred["kl_to_expression"],
                        "kl_to_marginal": pred["kl_to_marginal"],
                        "residual_fraction": pred["residual_fraction"],
                        "prediction_grouped_by_batch": pred["grouped"],
                    })
        state["interval_rows"].extend(a3)
        for nA in args.anchors:
            rr = [r for r in a3 if r["n_anchor_setting"] == nA]
            print(f"  nA={nA:3d}: JS(neighbour)/JS(random) mean "
                  f"{np.nanmean([r['js_neighbour_over_random'] for r in rr]):.3f}  "
                  f"| residual fraction after a cross-fitted expression predictor "
                  f"{np.nanmean([r['residual_fraction'] for r in rr]):.3f}")
        print("     ratio ~1 and residual ~1 -> no expression correspondence; "
              "both <<1 -> the fingerprint is largely an expression readout")
        state["analyses"]["A3"] = "done"
        _flush(state, args.out)

    # ================= A4 geometry =======================================
    if "A4" not in args.skip:
        print("\n[A4] fingerprint-space geometry")
        a4 = []
        for nA in args.anchors:
            for t in range(T):
                r = {"analysis": "A4", "n_anchor_setting": nA, "t": t,
                     "day": float(data.tau[t])}
                parts = [x for x in (Fm[nA][t], Fp[nA][t]) if x is not None]
                R = np.hstack(parts)
                r["concat_dim"] = int(R.shape[1])
                r.update(_pca_spectrum(R, prefix="concat"))
                r["intrinsic_dim_twonn"] = _twonn(R, rng)
                # Hopkins removed: its box null lies off the simplex and inflated
                # the statistic (0.77-0.84 on UNCLUSTERED Dirichlet samples).
                # keys already read 'bimodality_pcN'; do not re-prefix them
                r.update({k: v for k, v in ca.bimodality_clr(parts, rng=rng).items()
                          if k.startswith("bimodality_pc")})
                for Ks in args.stability_K:
                    st = ca.cluster_stability_resampled(parts, Ks, rng,
                                                        n_rep=args.stability_reps)
                    r[f"partition_ari_K{Ks}"] = st["mean_ari"]
                a4.append(r)
        state["interval_rows"].extend(a4)
        print(f"  intrinsic dim (TwoNN) mean "
              f"{np.nanmean([r['intrinsic_dim_twonn'] for r in a4]):.2f}")
        print(f"  bimodality of leading CLR PC mean "
              f"{np.nanmean([r.get('bimodality_pc1', np.nan) for r in a4]):.3f} "
              f"(>0.555 suggests bimodal; controls 0.32 unclustered / 0.76 clustered)")
        for Ks in args.stability_K:
            v = [r.get(f'partition_ari_K{Ks}', np.nan) for r in a4]
            print(f"  partition reproducibility K={Ks}: mean ARI "
                  f"{np.nanmean(v):.3f} (controls ~0.15-0.21 unclustered / 1.00 "
                  f"for two clear clusters)")
        state["analyses"]["A4"] = "done"
        _flush(state, args.out)

    # ================= A5 predictive comparison ==========================
    #
    # [CRITICAL] The first version of A5 was NOT held-out prediction and its
    # "+0.033 held-out" number must not be cited.  The fingerprints used as
    # clustering FEATURES were built from the full-chain coupling, which involves
    # every cell including the ones being scored, and they were then scored
    # against the SAME fingerprints.  Feature and evaluation target were identical
    # and the coupling had already seen the "held-out" cells.
    #
    # Three tiers are reported here, labelled so they cannot be conflated:
    #
    #   tier 1  DESCRIPTIVE            fit and score on the same cells and the same
    #                                  fingerprints. Says how compressible the
    #                                  fingerprints are, nothing about prediction.
    #   tier 2  OUT_OF_SAMPLE          centroids from batch A, assignment of batch B
    #                                  cells, still scored on the shared full-chain
    #                                  coupling. Tests assignment transfer only.
    #   tier 3  INDEPENDENT_COUPLING   centroids learned on fingerprints from batch
    #                                  A's OWN coupling; batch B cells assigned from
    #                                  fingerprints built from batch B's OWN
    #                                  coupling; scored on batch B's coupling. No
    #                                  quantity in the evaluation was touched by the
    #                                  training batch.
    #   tier 3x CROSS_DIRECTION        states learned from forward fingerprints and
    #                                  scored on BACKWARD CMI (and vice versa), so
    #                                  the feature and the target are different
    #                                  variables even within one coupling.
    #
    # Only tiers 3 and 3x support a claim about prediction.
    if "A5" not in args.skip:
        print("\n[A5] predictive comparison -- three explicitly separated tiers")
        from sklearn.cluster import KMeans
        a5 = []
        nA_eval = max(args.anchors)

        # batch-specific couplings, built ONCE on the shared frozen cost scale so
        # the two batches' fingerprints live in comparable units
        batches = sorted({str(v) for v in np.asarray(data.replicate[0]).astype(str)})
        if len(batches) < 2:
            print("  only one batch present: tiers 2/3 are not available")
        bidx, bchain, bF = {}, {}, {}
        for b in batches:
            idx, _ = ca.batch_split(data, b)
            bidx[b] = idx
            Zb = [Z[t][idx[t]] for t in range(T)]
            bchain[b] = _chain_for(Zb, data.tau, chain.epsilon, args.support,
                                   args.kappa, args.cost_scale_mode, scales,
                                   args.verbose)
            Ab = [A[nA_eval][t][idx[t]] for t in range(T)]
            bF[b] = ca.fixed_fingerprints(bchain[b], Ab)
            print(f"  batch {b}: independent coupling built "
                  f"(feasible={bchain[b].feasible}, "
                  f"{sum(len(i) for i in idx)} cells)")

        def _feat(kind, t, F_src):
            if kind == "expression":
                return Z[t]
            parts = [x for x in (F_src[1][t], F_src[0][t]) if x is not None]
            return np.hstack(parts)

        def _score(tier, name, K, t, M, F, idx, direction, extra=None):
            if F is None or len(idx) < 5:
                return
            w = np.full(len(idx), 1.0 / len(idx))
            c = ca.state_cmi(M, F, w)
            ret, why = ca.retained_information(c["cmi"], c["i_cell_anchor"])
            row = {"analysis": "A5", "tier": tier, "states_from": name, "K": K,
                   "t": t, "day": float(data.tau[t]), "direction": direction,
                   "n_cells": int(len(idx)), **c, "retained": ret,
                   "retained_note": why}
            if extra:
                row.update(extra)
            a5.append(row)

        for K in args.match_K:
            for t in range(T):
                Ffull = (Fp[nA_eval], Fm[nA_eval])
                # ---- tier 1: descriptive (same cells, same fingerprints) ----
                for name in ("fingerprint", "expression"):
                    X = _feat(name, t, Ffull)
                    k = int(min(K, len(X)))
                    km = KMeans(n_clusters=k, n_init=4, random_state=args.seed).fit(X)
                    lab = km.labels_
                    M = np.zeros((len(lab), k)); M[np.arange(len(lab)), lab] = 1.0
                    for d_, F in (("plus", Fp[nA_eval][t]), ("minus", Fm[nA_eval][t])):
                        _score("1_DESCRIPTIVE", name, K, t, M, F,
                               np.arange(data.n_cells[t]), d_)

                if len(batches) < 2:
                    continue
                tr_b, te_b = batches[0], batches[1]

                # ---- tier 2: out-of-sample assignment, shared coupling ------
                for name in ("fingerprint", "expression"):
                    X = _feat(name, t, Ffull)
                    k = int(min(K, len(bidx[tr_b][t])))
                    km = KMeans(n_clusters=k, n_init=4,
                                random_state=args.seed).fit(X[bidx[tr_b][t]])
                    lab = km.predict(X)
                    M = np.zeros((len(lab), k)); M[np.arange(len(lab)), lab] = 1.0
                    for d_, F in (("plus", Fp[nA_eval][t]), ("minus", Fm[nA_eval][t])):
                        _score("2_OUT_OF_SAMPLE", name, K, t,
                               M[bidx[te_b][t]], None if F is None else F[bidx[te_b][t]],
                               bidx[te_b][t], d_,
                               extra={"train_batch": tr_b, "eval_batch": te_b})

                # ---- tier 3: independent couplings both sides --------------
                for name in ("fingerprint", "expression"):
                    Xtr = _feat(name, t, bF[tr_b]) if name == "fingerprint" \
                        else Z[t][bidx[tr_b][t]]
                    Xte = _feat(name, t, bF[te_b]) if name == "fingerprint" \
                        else Z[t][bidx[te_b][t]]
                    if Xtr.shape[1] != Xte.shape[1]:
                        continue
                    k = int(min(K, len(Xtr)))
                    km = KMeans(n_clusters=k, n_init=4, random_state=args.seed).fit(Xtr)
                    lab = km.predict(Xte)
                    M = np.zeros((len(lab), k)); M[np.arange(len(lab)), lab] = 1.0
                    for d_, F in (("plus", bF[te_b][0][t]), ("minus", bF[te_b][1][t])):
                        _score("3_INDEPENDENT_COUPLING", name, K, t, M, F,
                               np.arange(len(lab)), d_,
                               extra={"train_batch": tr_b, "eval_batch": te_b,
                                      "coupling": "batch-specific, both sides"})

                # ---- tier 3x: cross-direction (feature != target) ----------
                for feat_dir, targ_dir in (("plus", "minus"), ("minus", "plus")):
                    Ffeat = Fp[nA_eval][t] if feat_dir == "plus" else Fm[nA_eval][t]
                    Ftarg = Fp[nA_eval][t] if targ_dir == "plus" else Fm[nA_eval][t]
                    if Ffeat is None or Ftarg is None:
                        continue
                    k = int(min(K, len(Ffeat)))
                    km = KMeans(n_clusters=k, n_init=4, random_state=args.seed).fit(Ffeat)
                    lab = km.labels_
                    M = np.zeros((len(lab), k)); M[np.arange(len(lab)), lab] = 1.0
                    _score("3x_CROSS_DIRECTION", f"fingerprint_{feat_dir}", K, t,
                           M, Ftarg, np.arange(len(lab)), targ_dir,
                           extra={"feature_direction": feat_dir})

        state["interval_rows"].extend(a5)
        for tier in ("1_DESCRIPTIVE", "2_OUT_OF_SAMPLE", "3_INDEPENDENT_COUPLING",
                     "3x_CROSS_DIRECTION"):
            rows = [r for r in a5 if r["tier"] == tier and r["direction"] == "plus"]
            if not rows:
                continue
            print(f"  --- {tier} (forward) ---")
            for K in args.match_K:
                rk = [r for r in rows if r["K"] == K]
                if not rk:
                    continue
                fpv = [r["retained"] for r in rk if r["states_from"].startswith("fingerprint")]
                exv = [r["retained"] for r in rk if r["states_from"] == "expression"]
                msg = f"    K={K:3d} retained: fingerprint {np.nanmean(fpv):.4f}"
                if exv:
                    msg += (f" vs expression {np.nanmean(exv):.4f}  "
                            f"(delta {np.nanmean(fpv)-np.nanmean(exv):+.4f})")
                print(msg)
                state["summary_rows"].append(
                    {"analysis": "A5", "tier": tier, "K": K, "direction": "plus",
                     "retained_fingerprint": float(np.nanmean(fpv)),
                     "retained_expression": float(np.nanmean(exv)) if exv else None,
                     "delta": (float(np.nanmean(fpv) - np.nanmean(exv)) if exv else None)})
        print("  ONLY tiers 3 and 3x support a predictive claim. Tier 1 is "
              "descriptive; tier 2 tests assignment transfer on a shared coupling.")
        state["analyses"]["A5"] = "done"
        _flush(state, args.out)

    jdump({"config": vars(args)}, os.path.join(args.out, "config.json"))
    jdump(prov, os.path.join(args.out, "provenance.json"))
    _flush(state, args.out)
    print(f"\n[A] wrote {args.out}")
    print("[A] Reminder: low within-state dispersion is NOT a good result if "
          "I(cell;anchor) is itself near zero -- check A1 before reading A5.")


# --------------------------------------------------------------------------
# small statistics helpers (kept local; none are in the package)
# --------------------------------------------------------------------------

def _spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]


def _pca_spectrum(X, prefix="", n=5):
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    try:
        s = np.linalg.svd(Xc, compute_uv=False)
    except np.linalg.LinAlgError:
        return {}
    var = s ** 2
    tot = max(var.sum(), 1e-300)
    out = {f"{prefix}_var_ratio_{i+1}": float(var[i] / tot)
           for i in range(min(n, len(var)))}
    cum = np.cumsum(var) / tot
    out[f"{prefix}_n_pc_90pct"] = int(np.searchsorted(cum, 0.90) + 1)
    # participation ratio: a scale-free effective dimension
    out[f"{prefix}_participation_ratio"] = float(var.sum() ** 2 / max((var ** 2).sum(), 1e-300))
    return out


def _twonn(X, rng, max_cells=2000):
    """TwoNN intrinsic-dimension estimate (Facco et al. 2017).

    Uses the ratio of second to first nearest-neighbour distances, so it is
    insensitive to density variation -- which matters because fingerprint space
    is expected to be non-uniform.
    """
    X = np.asarray(X, float)
    if len(X) > max_cells:
        X = X[rng.choice(len(X), max_cells, replace=False)]
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    srt = np.sort(d2, axis=1)
    r1 = np.sqrt(srt[:, 0]); r2 = np.sqrt(srt[:, 1])
    ok = r1 > 0
    if ok.sum() < 10:
        return float("nan")
    mu = r2[ok] / r1[ok]
    mu = mu[mu > 1]
    if mu.size < 10:
        return float("nan")
    # MLE slope of log(1-F) vs log(mu)
    return float(mu.size / np.log(mu).sum())


# _hopkins was REMOVED. Its null sampled a rectangular bounding box while
# fingerprints live on probability simplices, so the null points lay off the
# simplex and inflated the statistic: 0.77-0.84 on UNCLUSTERED Dirichlet samples
# against 0.91 for genuinely 2-clustered simplex data. A CLR + covariance-matched
# Gaussian null fixed the negative control but failed the positive one (two
# clusters are largely a second-moment feature). See csa_anchors.cluster_tendency
# for the full record, and use cluster_stability_resampled / bimodality_clr.


if __name__ == "__main__":
    main()
