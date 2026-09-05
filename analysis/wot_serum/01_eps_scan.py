#!/usr/bin/env python
"""Stage A -- the epsilon-informativeness scan on the WOT serum arm, days 0-18.

Build-order step 1 (PROJECT_HANDOFF.txt s4/s8).  No memberships, no objective, no
state learning: this only asks whether an informative, stable window of epsilon
exists at all on this data at this sampling density.  It never aborts -- an epsilon
that cannot be solved is recorded as ``feasible=0``, not raised.

Support is built DENSE here, unlike the fit.  ``epsilon_scan`` sizes its support once
at ``max(epsilons)`` (informativeness.py:378), which is exactly where a kNN support is
least likely to admit a balanced plan; a dense support never reports infeasible
(reference.py:307), so support sizing cannot confound the feasibility column that the
scan is meant to measure.  At 1000 cells/timepoint a dense solve is ~0.7 s.

The cost scale is computed ONCE on the native series and pinned for every stride, so
stride 1 and stride 2 differ by spacing alone and not by cost normalisation.

    python 01_eps_scan.py --stride 1 --stride 2
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import csa_wot
from csa_wot import DEFAULT_H5AD, RESULTS_ROOT, ensure_outdir, jdump, load_serum, make_cfg

from cellstateadj.config import DEFAULT_EPSILON_GRID
from cellstateadj.cost import adjacent_cost, resolve_cost_scales
from cellstateadj.data import subsample_timepoints
from cellstateadj.informativeness import epsilon_scan
from cellstateadj.reference import _underflow_limit


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", default=DEFAULT_H5AD)
    p.add_argument("--day-min", type=float, default=0.0)
    p.add_argument("--day-max", type=float, default=18.0)
    p.add_argument("--stride", type=int, action="append", default=None,
                   help="repeat for several spacings; default 1 and 2")
    p.add_argument("--epsilons", type=float, nargs="+",
                   default=list(DEFAULT_EPSILON_GRID))
    p.add_argument("--n-per-timepoint", type=int, default=None,
                   help="default: use every cell (1000/day)")
    p.add_argument("--support", default="dense", choices=["dense", "knn"])
    p.add_argument("--kappa", type=int, default=400)
    p.add_argument("--provisional-k", type=int, default=30)
    p.add_argument("--intervals", type=int, nargs="+", default=None,
                   help="restrict to these interval indices (default: all)")
    p.add_argument("--cost-perturbation", type=float, default=0.05)
    p.add_argument("--sinkhorn-max-iter", type=int, default=None,
                   help="override CouplingConfig.max_iter (20000) for the scan; "
                        "lowering it makes hopeless epsilons cheap to reject, but "
                        "can also mis-label a slow-but-solvable one as infeasible")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(RESULTS_ROOT, "eps_scan"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def underflow_report(Z, tau, scale, epsilons, dtype="float64", n_probe=400, seed=0):
    """Which (epsilon, interval) pairs are numerically hopeless before we spend time on them.

    ``exp(-C/eps)`` underflows once ``max(C)/eps`` passes ~708 in float64
    (reference.py:_underflow_limit), and no number of Sinkhorn iterations fixes
    that -- the marginals simply cannot be met.  The scan still runs those cells
    and records ``feasible=0``, which is the honest result, but each one costs a
    full ``max_iter`` solve, so it is worth knowing the shape of the problem up
    front.

    On this series the cost scale is global while the PCA geometry expands by
    ~25x between day 3 and day 17, so the late intervals hit the limit at an
    epsilon where the early ones are still nearly independent couplings.  That
    tension is a property of the data, not of the code.
    """
    limit = _underflow_limit(dtype)
    rng = np.random.default_rng(seed)
    max_cost = []
    for t in range(len(Z) - 1):
        i = rng.choice(len(Z[t]), min(n_probe, len(Z[t])), replace=False)
        j = rng.choice(len(Z[t + 1]), min(n_probe, len(Z[t + 1])), replace=False)
        C = adjacent_cost(Z[t][i], Z[t + 1][j], float(tau[t + 1] - tau[t])) / scale
        max_cost.append(float(C.max()))
    max_cost = np.asarray(max_cost)

    print(f"[scan] float64 underflow limit for max(C)/eps is {limit:g}; "
          f"max normalised cost ranges {max_cost.min():.2f}-{max_cost.max():.2f} "
          f"over {len(max_cost)} intervals")
    rows = {}
    for e in sorted(epsilons):
        ratio = max_cost / e
        n_bad = int((ratio > limit).sum())
        rows[f"{e:g}"] = {"n_intervals_over_underflow_limit": n_bad,
                          "max_ratio": float(ratio.max()),
                          "worst_interval": int(np.argmax(ratio))}
        flag = "  <-- expect feasible=0 on those" if n_bad else ""
        print(f"  eps={e:<7g} max(C)/eps={ratio.max():9.0f}  "
              f"{n_bad:2d}/{len(max_cost)} intervals over the limit{flag}")
    return {"underflow_limit": limit, "dtype": dtype,
            "max_normalised_cost_per_interval": max_cost.tolist(),
            "per_epsilon": rows}


def main():
    args = parse_args()
    strides = args.stride or [1, 2]
    out = ensure_outdir(args.out, overwrite=args.overwrite)

    data, Z, _obs = load_serum(args.h5ad, day_min=args.day_min, day_max=args.day_max,
                               stride=1, n_per_timepoint=args.n_per_timepoint,
                               seed=args.seed, verbose=args.verbose)

    cfg = make_cfg(support=args.support, kappa=args.kappa, device=args.device,
                   seed=args.seed, verbose=args.verbose)
    if args.sinkhorn_max_iter is not None:
        cfg.coupling.max_iter = args.sinkhorn_max_iter

    # One scale from the native series, reused by every stride.
    native_scale = float(resolve_cost_scales(Z, data.tau, cfg.coupling.cost_scale_mode)[0])
    print(f"[scan] native global cost scale = {native_scale:.6g}")
    print(f"[scan] epsilon grid = {args.epsilons}")

    underflow = underflow_report(Z, data.tau, native_scale, args.epsilons,
                                 cfg.coupling.dtype)

    summary = {"h5ad": args.h5ad, "day_range": [args.day_min, args.day_max],
               "epsilons": list(args.epsilons), "support": args.support,
               "kappa": args.kappa, "provisional_K": args.provisional_k,
               "cost_scale": native_scale, "n_cells_native": data.n_cells,
               "shared_representation": "obsm['X_model'] (30 PCs), frozen",
               "underflow_report": underflow,
               "branch_caveat": csa_wot.BRANCH_CAVEAT, "per_stride": {}}

    for stride in strides:
        sub = data if stride == 1 else subsample_timepoints(data, stride=stride)
        Zs = [np.asarray(x, dtype=np.float64) for x in sub.X]
        dtau = np.asarray(sub.dtau, dtype=float)
        print(f"\n[scan] stride={stride}: T={sub.T} intervals={sub.T - 1} "
              f"dtau {dtau.min():g}-{dtau.max():g} d")

        t0 = time.time()
        scan = epsilon_scan(
            Zs, sub.tau,
            epsilons=args.epsilons,
            cfg=cfg.coupling,
            intervals=args.intervals,
            provisional_K=args.provisional_k,
            cost_perturbation=args.cost_perturbation,
            replicate=sub.replicate,
            replicate_paired=True,
            cost_scales=[native_scale] * (sub.T - 1),
            seed=args.seed,
            verbose=args.verbose,
        )
        elapsed = time.time() - t0

        payload = {"epsilons": scan.epsilons,
                   "intervals": np.asarray(scan.intervals, dtype=int)}
        payload.update(scan.metrics)
        np.savez_compressed(os.path.join(out, f"scan_stride{stride}.npz"), **payload)
        try:
            scan.to_frame().to_csv(
                os.path.join(out, f"scan_stride{stride}.csv"), index=False)
        except Exception as exc:
            print(f"  (csv export skipped: {type(exc).__name__}: {exc})")

        rec = scan.recommend()
        print(f"[scan] stride={stride} done in {elapsed / 60:.1f} min")
        print(f"[scan] recommendation: {rec}")
        for name in ("I_cell_normalized", "I_fingerprint_plus", "stability_resample",
                     "stability_cost", "feasible"):
            if name in scan.metrics:
                curve = scan.mean_curve(name)
                print("  {:22s} ".format(name)
                      + " ".join(f"{v:7.4f}" if np.isfinite(v) else "    nan"
                                 for v in curve))

        summary["per_stride"][str(stride)] = {
            "T": sub.T,
            "n_intervals": sub.T - 1,
            "dtau": dtau.tolist(),
            "n_cells": sub.n_cells,
            "recommendation": rec,
            "elapsed_s": elapsed,
            "mean_curves": {k: scan.mean_curve(k).tolist() for k in scan.metrics},
        }

    jdump(summary, os.path.join(out, "summary.json"))
    print(f"\n[scan] wrote {out}")

    chosen = summary["per_stride"].get("1", {}).get("recommendation", {})
    if chosen.get("epsilon_star") is None:
        print("[scan] NOTE: no epsilon satisfied every criterion at stride 1. "
              "Read the curves before choosing one for Stage C -- this is the "
              "result the build order exists to surface, not a crash.")
    else:
        print(f"[scan] epsilon* (stride 1) = {chosen['epsilon_star']}  "
              f"window {chosen['window']}")


if __name__ == "__main__":
    main()
