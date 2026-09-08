#!/usr/bin/env python
"""Stage 2 -- fixed-anchor CMI as the ONE evaluation criterion for every candidate.

Scores any set of candidate state definitions on the same footing:

  * per-timepoint expression clustering
  * fixed-fingerprint clustering
  * the SSE objective            (memberships from 06_expression_ablation.py)
  * the Gaussian-expression objective
  * the no-expression objective

against the SAME frozen anchor system, and reports a
complexity-versus-sufficiency curve rather than a single winner.

[CRITICAL] CMI must never be compared without controlling complexity.  A model can
drive CMI to zero by giving every cell its own state, so every comparison here is
reported against four complexity measures at once:

    K,  effective state number exp(H(g)),  the rate I(cell;Z),  min state mass

and the output is a curve.  A method is only better if it is better AT MATCHED
complexity.  The ``retained`` column is ``1 - CMI/I(cell;anchor)`` and is NaN,
with a reason, whenever the denominator is too small for the ratio to mean
anything -- a low CMI on an uninformative coupling is not a success.

    python 07_cmi_compare.py --chain <stage0>/chain_eps0.2.npz \
        --memberships <expr_ablation>/memberships_*.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import csa_anchors as ca
import csa_wot
from csa_wot import DEFAULT_H5AD, RESULTS_ROOT, jdump, load_serum, make_cfg

from cellstateadj.reference import ReferenceChain


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", default=DEFAULT_H5AD)
    p.add_argument("--chain", required=True)
    p.add_argument("--day-min", type=float, default=8.25)
    p.add_argument("--day-max", type=float, default=18.0)
    p.add_argument("--ablation-dir", default=None,
                   help="an 06_expression_ablation.py output directory. Its "
                        "summary.json is the ONLY accepted source of membership "
                        "files, so fit validity travels with every CMI row and no "
                        "stale file from another configuration can enter.")
    p.add_argument("--memberships", nargs="*", default=[],
                   help="DEPRECATED: raw globs cannot carry convergence or "
                        "upper-bound status and admit stale files. Requires "
                        "--allow-unvalidated-memberships.")
    p.add_argument("--allow-unvalidated-memberships", action="store_true",
                   help="permit --memberships globs; every such row is marked "
                        "validated=False and excluded from primary comparisons")
    p.add_argument("--print-anchor-system", default=None,
                   help="which anchor system to print (default: the first "
                        "requested kmeans resolution)")
    p.add_argument("--anchors", type=int, nargs="+", default=[20, 40, 80])
    p.add_argument("--baseline-K", type=int, nargs="+", default=[4, 6, 8, 12, 16, 20],
                   help="K grid for the expression / fingerprint clustering baselines")
    p.add_argument("--annotation-anchors", default=None,
                   help="also evaluate against this obs column as an anchor system "
                        "(e.g. cell_sets) -- likely too coarse, reported separately")
    p.add_argument("--train-batch", default="1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(RESULTS_ROOT, "cmi_compare"))
    p.add_argument("--overwrite", action="store_true",
                   help="NOT recommended: prefer a fresh --out. Reusing a directory "
                        "can leave stale CSV/NPZ beside new ones.")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def _rate(M, a):
    """``I(cell;Z) = H(g) - sum_i a_i H(M_i)`` -- the IB rate, in nats.

    The complexity measure that matters most here: unlike K it responds to soft
    assignment, and unlike exp(H(g)) it cannot rise while a state is being
    annihilated (which is how the 2026-09-07 gate was fooled).
    """
    M = np.asarray(M, float); a = np.asarray(a, float)
    g = M.T @ a
    H = lambda p, ax: -(p * np.log(np.maximum(p, 1e-30))).sum(axis=ax)
    return float(H(g, 0) - (a * H(M, 1)).sum())


def _score_candidate(name, M_list, Fp, Fm, splits, tau, extra=None):
    """Fixed-anchor CMI + all four complexity measures, per timepoint and split."""
    rows = []
    for t in range(len(M_list)):
        M = np.asarray(M_list[t], float)
        for direction, F in (("plus", Fp[t]), ("minus", Fm[t])):
            if F is None:
                continue
            for split, idx in splits[t].items():
                if len(idx) < 5:
                    continue
                w = np.full(len(idx), 1.0 / len(idx))
                Ms = M[idx]
                Ms = Ms / np.maximum(Ms.sum(1, keepdims=True), 1e-30)
                c = ca.state_cmi(Ms, F[idx], w)
                ret, why = ca.retained_information(c["cmi"], c["i_cell_anchor"])
                row = {"candidate": name, "t": t, "day": float(tau[t]),
                       "direction": direction, "split": split,
                       "n_cells": int(len(idx)),
                       "K": int(M.shape[1]),
                       "k_eff": c["k_eff"],
                       "rate_i_cell_state": _rate(Ms, w),
                       "min_state_mass": c["min_state_mass"],
                       "cmi": c["cmi"],
                       "i_cell_anchor": c["i_cell_anchor"],
                       "i_state_anchor": c["i_state_anchor"],
                       "decomposition_error": c["decomposition_error"],
                       "retained": ret, "retained_note": why}
                if extra:
                    row.update(extra)
                rows.append(row)
    return rows


def main():
    args = parse_args()

    data, Z, obs = load_serum(args.h5ad, day_min=args.day_min, day_max=args.day_max,
                              seed=args.seed, verbose=args.verbose)
    T = data.T
    chain = ReferenceChain.load(args.chain)
    identity = ca.validate_chain_identity(chain, data, Z, chain_path=args.chain)
    ca.check_identity_args(identity, h5ad=args.h5ad, day_min=args.day_min,
                           day_max=args.day_max, stride=1, n_per_timepoint=None,
                           seed=args.seed)
    print(f"[CMI] chain identity verified against the loaded data "
          f"(cell ids + order, tau, representation hash, feasibility)")
    import hashlib
    key = hashlib.sha1(json.dumps({
        "chain": os.path.abspath(args.chain), "anchors": sorted(args.anchors),
        "baseline_K": sorted(args.baseline_K),
        "ablation_dir": os.path.abspath(args.ablation_dir) if args.ablation_dir else None,
        "annotation": args.annotation_anchors, "train_batch": args.train_batch,
    }, sort_keys=True).encode()).hexdigest()[:10]
    args.out = os.path.join(args.out, f"cfg_{key}")
    os.makedirs(args.out, exist_ok=True)
    if os.listdir(args.out) and not args.overwrite:
        raise SystemExit(
            f"{args.out} already contains files -- the same comparison already ran. "
            f"Use a fresh --out or --overwrite.")
    print(f"[CMI] chain eps={chain.epsilon:g}  T={T}  K grid {args.baseline_K}  "
          f"-> cfg_{key}")

    tr_idx, te_idx = ca.batch_split(data, args.train_batch)
    splits = [{"train": tr_idx[t], "heldout": te_idx[t], "all": np.arange(data.n_cells[t])}
              for t in range(T)]

    rows = []
    anchor_systems = {}
    for nA in args.anchors:
        cen = ca.make_anchors([Z[t][tr_idx[t]] for t in range(T)], nA, seed=args.seed)
        A = ca.assign_anchors(Z, cen)
        anchor_systems[f"kmeans{nA}"] = ca.fixed_fingerprints(chain, A)
    if args.annotation_anchors and args.annotation_anchors in obs.columns:
        labs = [np.asarray(obs[args.annotation_anchors]
                           .reindex(data.obs[t]["index"]).astype(str)) for t in range(T)]
        Aann, cats = ca.anchors_from_labels(labs)
        anchor_systems[f"annotation:{args.annotation_anchors}"] = \
            ca.fixed_fingerprints(chain, Aann)
        print(f"[CMI] annotation anchor system '{args.annotation_anchors}' "
              f"({len(cats)} categories) -- reported separately; it may be far too "
              f"coarse to expose transition heterogeneity")

    # ---- membership sources, with fit validity attached ------------------
    membership_sources = []
    if args.ablation_dir:
        sp = os.path.join(args.ablation_dir, "summary.json")
        if not os.path.exists(sp):
            raise SystemExit(f"no summary.json in {args.ablation_dir}")
        abl = json.load(open(sp))
        saved = abl.get("saved_memberships", {})
        if not saved:
            raise SystemExit(
                f"{sp} records no saved_memberships. Either the ablation has not "
                f"produced any fits yet, or it predates the metadata fix.")
        if abl.get("chain") and os.path.abspath(abl["chain"]) != os.path.abspath(args.chain):
            raise SystemExit(
                f"the ablation used chain {abl['chain']} but this job was given "
                f"{os.path.abspath(args.chain)} -- refusing to score memberships "
                f"against a different coupling.")
        for tag, meta in saved.items():
            fpth = os.path.join(args.ablation_dir,
                                meta.get("file", f"memberships_{tag}.npz"))
            if not os.path.exists(fpth):
                print(f"  missing {os.path.basename(fpth)} (recorded but absent)")
                continue
            membership_sources.append((fpth, {**meta, "validated": True}))
        # the variant lambda_pm=0 baselines are legitimate candidates too
        for v, vb in abl.get("variant_baselines", {}).items():
            fpth = os.path.join(args.ablation_dir, f"memberships_{v}_lpm0_baseline.npz")
            if os.path.exists(fpth):
                membership_sources.append((fpth, {
                    **vb, "validated": True, "variant": v, "lambda_pm": 0.0,
                    "direction": "baseline",
                    "beats_upper_bound": True}))   # it IS the bound
        print(f"[CMI] {len(membership_sources)} membership file(s) from "
              f"{args.ablation_dir}, each with fit validity attached")
    if args.memberships:
        if not args.allow_unvalidated_memberships:
            raise SystemExit(
                "--memberships uses raw globs, which cannot carry convergence or "
                "upper-bound status and admit stale files from earlier "
                "configurations. Use --ablation-dir, or pass "
                "--allow-unvalidated-memberships to accept diagnostic-only rows.")
        for pat in args.memberships:
            for fpth in sorted(glob.glob(pat)):
                membership_sources.append((fpth, {"validated": False}))
        print(f"[CMI] WARNING: {len(args.memberships)} unvalidated glob(s) accepted; "
              f"those rows are excluded from primary comparisons")

    from sklearn.cluster import KMeans
    for asys, (Fp, Fm) in anchor_systems.items():
        print(f"\n[CMI] anchor system {asys}")

        # ---- baselines: expression and fingerprint clustering, matched K -----
        for K in args.baseline_K:
            for src in ("expression", "fingerprint"):
                M_list = []
                for t in range(T):
                    if src == "expression":
                        X = Z[t]
                    else:
                        parts = [x for x in (Fm[t], Fp[t]) if x is not None]
                        X = np.hstack(parts)
                    k = int(min(K, len(tr_idx[t])))
                    km = KMeans(n_clusters=k, n_init=4, random_state=args.seed)
                    km.fit(X[tr_idx[t]])
                    lab = km.predict(X)
                    M = np.zeros((len(lab), k)); M[np.arange(len(lab)), lab] = 1.0
                    M_list.append(M)
                rows += _score_candidate(f"{src}_kmeans_K{K}", M_list, Fp, Fm,
                                         splits, data.tau,
                                         extra={"anchor_system": asys,
                                                "family": src, "requested_K": K})

        # ---- learned memberships from Workstream B --------------------------
        for path, meta in membership_sources:
            z = np.load(path)
            keys = sorted(k for k in z.files if k.startswith("M_"))
            if len(keys) != T:
                print(f"  SKIP {os.path.basename(path)}: {len(keys)} timepoints, "
                      f"data has {T}")
                continue
            M_list = [z[f"M_{t}"] for t in range(T)]
            # validate EVERY timepoint, not only the first
            bad = [(t, M_list[t].shape, data.n_cells[t]) for t in range(T)
                   if M_list[t].shape[0] != data.n_cells[t]]
            if bad:
                print(f"  SKIP {os.path.basename(path)}: shape mismatch at "
                      f"{bad[:3]}{' ...' if len(bad) > 3 else ''}")
                continue
            Ks = {M.shape[1] for M in M_list}
            if len(Ks) != 1:
                print(f"  SKIP {os.path.basename(path)}: inconsistent K across "
                      f"timepoints {sorted(Ks)}")
                continue
            name = os.path.basename(path).replace("memberships_", "").replace(".npz", "")
            eligible = bool(meta.get("validated")
                            and meta.get("strict_converged")
                            and meta.get("beats_upper_bound"))
            rows += _score_candidate(
                f"learned_{name}", M_list, Fp, Fm, splits, data.tau,
                extra={"anchor_system": asys, "family": "learned",
                       "validated": bool(meta.get("validated")),
                       "variant": meta.get("variant"),
                       "lambda_x": meta.get("lambda_x"),
                       "lambda_pm": meta.get("lambda_pm"),
                       "direction_path": meta.get("direction"),
                       "fit_status": meta.get("status"),
                       "converged": meta.get("converged"),
                       "strict_converged": meta.get("strict_converged"),
                       "objective": meta.get("objective"),
                       "upper_bound": meta.get("upper_bound"),
                       "beats_upper_bound": meta.get("beats_upper_bound"),
                       "grad_norm": meta.get("grad_norm"),
                       "kkt_residual": meta.get("kkt_residual"),
                       "primary_eligible": eligible})
            print(f"  scored {name}  "
                  f"[{'PRIMARY' if eligible else 'diagnostic only'}]"
                  f"{'' if meta.get('validated') else '  (unvalidated glob)'}")

    # ---- write ----------------------------------------------------------
    jdump({"chain": os.path.abspath(args.chain), "epsilon": float(chain.epsilon),
           "anchor_systems": list(anchor_systems), "config": vars(args),
           "n_rows": len(rows)}, os.path.join(args.out, "config.json"))
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.out, "interval_metrics.csv"), index=False)

        # the complexity-sufficiency curve: aggregate over timepoints, keep splits
        # and directions separate, and never average a NaN retained into a number
        if "primary_eligible" not in df.columns:
            df["primary_eligible"] = True
        df["primary_eligible"] = (df["primary_eligible"]
                                  .infer_objects(copy=False).fillna(True).astype(bool))
        # baselines are always primary; learned fits must pass the audit
        df.loc[df.family != "learned", "primary_eligible"] = True
        df[~df.primary_eligible].to_csv(
            os.path.join(args.out, "diagnostic_excluded.csv"), index=False)
        grp = (df.groupby(["anchor_system", "candidate", "family", "direction", "split"])
                 .agg(K=("K", "max"), k_eff=("k_eff", "mean"),
                      primary_eligible=("primary_eligible", "min"),
                      rate=("rate_i_cell_state", "mean"),
                      min_state_mass=("min_state_mass", "min"),
                      cmi=("cmi", "mean"), i_cell_anchor=("i_cell_anchor", "mean"),
                      retained=("retained", "mean"),
                      n_retained_nan=("retained", lambda s: int(s.isna().sum())),
                      n_intervals=("t", "count"))
                 .reset_index())
        grp.to_csv(os.path.join(args.out, "summary.csv"), index=False)
        jdump(grp.to_dict(orient="records"), os.path.join(args.out, "summary.json"))

        show = args.print_anchor_system or f"kmeans{sorted(args.anchors)[0]}"
        if show not in set(grp.anchor_system):
            show = sorted(set(grp.anchor_system))[0]
        print(f"\n[CMI] forward, held-out, {show} anchors -- "
              f"complexity vs sufficiency (PRIMARY rows only):")
        sel = grp[(grp.direction == "plus") & (grp.split == "heldout")
                  & (grp.anchor_system == show)
                  & (grp.primary_eligible)].sort_values(["family", "rate"])
        cols = ["candidate", "K", "k_eff", "rate", "min_state_mass", "cmi",
                "i_cell_anchor", "retained", "n_retained_nan"]
        print(sel[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}")
              if len(sel) else "  (no primary-eligible rows)")
        n_excl = int((~df.primary_eligible).sum())
        if n_excl:
            print(f"\n[CMI] {n_excl} row(s) excluded from primary comparison "
                  f"(not strictly converged, or failed the upper-bound audit, or "
                  f"unvalidated) -> diagnostic_excluded.csv")
        print("\n[CMI] Compare rows only at MATCHED rate / k_eff. A lower CMI at a "
              "higher rate is not an improvement.")
        if sel["n_retained_nan"].sum():
            print(f"[CMI] {int(sel['n_retained_nan'].sum())} interval(s) had "
                  f"I(cell;anchor) too small for 'retained' to mean anything -- "
                  f"those are excluded from the mean, not counted as successes.")
    except Exception as exc:
        jdump(rows, os.path.join(args.out, "interval_metrics.json"))
        print(f"[CMI] pandas summary skipped: {type(exc).__name__}: {exc}")

    print(f"\n[CMI] wrote {args.out}")


if __name__ == "__main__":
    main()
