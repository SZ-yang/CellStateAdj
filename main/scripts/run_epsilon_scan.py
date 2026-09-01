#!/usr/bin/env python
"""STEP 1: the epsilon-informativeness curve.

Run this before writing or running anything else on a new dataset.  It answers
whether an informative, stable window in epsilon exists AT ALL at this sampling
density.  If Ibar collapses to ~0 before epsilon is small enough to be stable,
the fingerprints carry nothing and the method cannot work as designed on this
data -- better to learn that in week 1 than month 3.

Examples
--------
    # simulated data, both spacings
    python scripts/run_epsilon_scan.py --sim branching --stride 1 --stride 4

    # real data from an .h5ad of counts with a time column
    python scripts/run_epsilon_scan.py --h5ad wot_phase1.h5ad \
        --time-key day --replicate-key sample --n-per-timepoint 1500 \
        --stride 1 --stride 4 --out results/eps_scan
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cellstateadj.config import CouplingConfig, PipelineConfig, DEFAULT_EPSILON_GRID
from cellstateadj.cost import resolve_cost_scales
from cellstateadj.data import from_anndata, subsample_cells, subsample_timepoints
from cellstateadj.informativeness import epsilon_scan
from cellstateadj.representation import learn_representation
from cellstateadj import simulate as sim


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5ad", help="AnnData file of raw counts")
    src.add_argument("--sim", help=f"simulated scenario: {sorted(sim.SCENARIOS)}")
    p.add_argument("--time-key", default="day")
    p.add_argument("--replicate-key", default=None)
    p.add_argument("--n-per-timepoint", type=int, default=1500)
    p.add_argument("--stride", type=int, action="append", default=None,
                   help="delta-tau subsampling stride; repeat for several spacings")
    p.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPSILON_GRID))
    p.add_argument("--kappa", type=int, default=50)
    p.add_argument("--dense", action="store_true", help="dense support (small data only)")
    p.add_argument("--provisional-k", type=int, default=30)
    p.add_argument("--n-components", type=int, default=30)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--intervals", type=int, nargs="+", default=None,
                   help="restrict to these interval indices (cheap first pass)")
    p.add_argument("--n-resample", type=int, default=2)
    p.add_argument("--cost-perturbation", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/eps_scan")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args(argv)


def load_data(args):
    if args.sim:
        sc = sim.make(args.sim, seed=args.seed)
        sc.n_sample = args.n_per_timepoint
        res = sim.simulate(sc, verbose=args.verbose)
        print(f"[data] simulated {args.sim}: {res.data}")
        return res.data
    import anndata as ad
    adata = ad.read_h5ad(args.h5ad)
    data = from_anndata(adata, time_key=args.time_key,
                        replicate_key=args.replicate_key)
    print(f"[data] {args.h5ad}: {data}")
    return data


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    data = load_data(args)

    strides = args.stride or [1]

    # [CRITICAL for the delta-tau study] Everything except the time spacing must
    # be held FIXED across strides, otherwise a stride-1 vs stride-4 difference
    # confounds four changes at once: spacing, PCA basis, cost scale, and which
    # cells were drawn.  So we subsample cells ONCE on the native series, fit the
    # representation ONCE, and compute the cost scale ONCE from the native
    # series; each stride then just selects timepoints from that fixed setup.
    base = subsample_cells(data, args.n_per_timepoint, seed=args.seed)
    rep_cfg = PipelineConfig().representation
    rep_cfg.n_components = args.n_components
    rep_cfg.n_hvg = args.n_hvg
    Z_base, info = learn_representation(base, rep_cfg)
    print(f"[rep] frozen PCA d={info['d']} fit once on the native series and "
          f"reused for every stride")

    cfg0 = CouplingConfig(support="dense" if args.dense else "knn",
                          kappa=args.kappa)
    native_scale = resolve_cost_scales(Z_base, base.tau, cfg0.cost_scale_mode)[0]
    print(f"[cost] native-series cost scale {native_scale:.6g} "
          f"(mode={cfg0.cost_scale_mode}) reused for every stride")

    summary = {}
    for stride in strides:
        idx = list(range(0, base.T, stride))
        if len(idx) < 2:
            print(f"[skip] stride {stride}: fewer than two timepoints")
            continue
        d = base.select_timepoints(idx)
        Z = [Z_base[i] for i in idx]
        print(f"\n=== stride {stride}  (dtau = {np.unique(np.round(d.dtau, 6)).tolist()}) ===")

        cfg = CouplingConfig(support="dense" if args.dense else "knn",
                             kappa=args.kappa)
        scan = epsilon_scan(
            Z, d.tau, epsilons=args.epsilons, cfg=cfg,
            intervals=args.intervals, provisional_K=args.provisional_k,
            n_resample=args.n_resample, cost_perturbation=args.cost_perturbation,
            replicate=d.replicate, seed=args.seed, verbose=args.verbose,
            cost_scales=[native_scale] * (len(Z) - 1),
        )
        rec = scan.recommend()
        print(f"\n[stride {stride}] recommendation: {rec}")
        print(f"{'eps':>10} {'Ibar':>10} {'I_fp+':>10} {'eff_tgt':>9} "
              f"{'stab_rs':>8} {'stab_C':>8}")
        for i, e in enumerate(scan.epsilons):
            print(f"{e:10.4g} {scan.mean_curve('I_cell_normalized')[i]:10.4f} "
                  f"{scan.mean_curve('I_fingerprint_plus')[i]:10.4f} "
                  f"{scan.mean_curve('eff_targets')[i]:9.1f} "
                  f"{scan.mean_curve('stability_resample')[i]:8.3f} "
                  f"{scan.mean_curve('stability_cost')[i]:8.3f}")

        np.savez_compressed(os.path.join(args.out, f"scan_stride{stride}.npz"),
                            epsilons=scan.epsilons,
                            **{k: v for k, v in scan.metrics.items()})
        summary[f"stride_{stride}"] = {
            "dtau": np.unique(np.round(d.dtau, 6)).tolist(),
            "n_cells": d.n_cells,
            "recommendation": rec,
            "cost_scale": float(native_scale),
            "shared_representation": True,
        }
        try:
            scan.to_frame().to_csv(
                os.path.join(args.out, f"scan_stride{stride}.csv"), index=False)
        except Exception:
            pass

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\n[done] wrote {args.out}")
    return summary


if __name__ == "__main__":
    main()
