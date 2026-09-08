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
    p.add_argument("--memberships", nargs="*", default=[],
                   help="memberships_*.npz from 06; glob patterns accepted")
    p.add_argument("--anchors", type=int, nargs="+", default=[20, 40, 80])
    p.add_argument("--baseline-K", type=int, nargs="+", default=[4, 6, 8, 12, 16, 20],
                   help="K grid for the expression / fingerprint clustering baselines")
    p.add_argument("--annotation-anchors", default=None,
                   help="also evaluate against this obs column as an anchor system "
                        "(e.g. cell_sets) -- likely too coarse, reported separately")
    p.add_argument("--train-batch", default="1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(RESULTS_ROOT, "cmi_compare"))
    p.add_argument("--overwrite", action="store_true")
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
    os.makedirs(args.out, exist_ok=True)
    if os.listdir(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} is not empty; pass --overwrite")

    data, Z, obs = load_serum(args.h5ad, day_min=args.day_min, day_max=args.day_max,
                              seed=args.seed, verbose=args.verbose)
    T = data.T
    chain = ReferenceChain.load(args.chain)
    if chain.n_cells != data.n_cells:
        raise SystemExit(f"chain does not match the loaded cells")
    print(f"[CMI] chain eps={chain.epsilon:g}  T={T}  K grid {args.baseline_K}")

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
        paths = [p for pat in args.memberships for p in sorted(glob.glob(pat))]
        for path in paths:
            z = np.load(path)
            keys = [k for k in z.files if k.startswith("M_")]
            if len(keys) != T:
                print(f"  skipping {os.path.basename(path)}: has {len(keys)} "
                      f"timepoints, data has {T}")
                continue
            M_list = [z[f"M_{t}"] for t in range(T)]
            if M_list[0].shape[0] != data.n_cells[0]:
                print(f"  skipping {os.path.basename(path)}: cell count mismatch")
                continue
            name = os.path.basename(path).replace("memberships_", "").replace(".npz", "")
            rows += _score_candidate(f"learned_{name}", M_list, Fp, Fm, splits,
                                     data.tau,
                                     extra={"anchor_system": asys, "family": "learned"})
            print(f"  scored {name}")

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
        grp = (df.groupby(["anchor_system", "candidate", "family", "direction", "split"])
                 .agg(K=("K", "max"), k_eff=("k_eff", "mean"),
                      rate=("rate_i_cell_state", "mean"),
                      min_state_mass=("min_state_mass", "min"),
                      cmi=("cmi", "mean"), i_cell_anchor=("i_cell_anchor", "mean"),
                      retained=("retained", "mean"),
                      n_retained_nan=("retained", lambda s: int(s.isna().sum())),
                      n_intervals=("t", "count"))
                 .reset_index())
        grp.to_csv(os.path.join(args.out, "summary.csv"), index=False)
        jdump(grp.to_dict(orient="records"), os.path.join(args.out, "summary.json"))

        print("\n[CMI] forward, held-out, kmeans40 anchors -- "
              "complexity vs sufficiency:")
        sel = grp[(grp.direction == "plus") & (grp.split == "heldout")
                  & (grp.anchor_system == "kmeans40")].sort_values(["family", "rate"])
        cols = ["candidate", "K", "k_eff", "rate", "min_state_mass", "cmi",
                "i_cell_anchor", "retained", "n_retained_nan"]
        print(sel[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
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
