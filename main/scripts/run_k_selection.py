#!/usr/bin/env python
"""STEP 5: choose K by held-out compression, protocol (b).

Holds out whole duplicate samples and scores the TRANSFERRED state map --
see cellstateadj/selection.py for why this protocol and not the alternative.
This is the decided procedure; do not switch it silently.

Training compression falls monotonically with K, so it cannot select anything.
The script prints both curves so the contrast is visible.

    python scripts/run_k_selection.py --sim branching --Ks 2 3 4 6 8 12 \
        --epsilon 0.05 --dense
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cellstateadj.config import PipelineConfig
from cellstateadj.data import from_anndata, subsample_cells, subsample_timepoints
from cellstateadj.selection import select_K
from cellstateadj import simulate as sim


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5ad")
    src.add_argument("--sim")
    p.add_argument("--time-key", default="day")
    p.add_argument("--replicate-key", default=None)
    p.add_argument("--Ks", type=int, nargs="+", default=[2, 3, 4, 6, 8, 12])
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--kappa", type=int, default=50)
    p.add_argument("--dense", action="store_true")
    p.add_argument("--n-per-timepoint", type=int, default=300)
    p.add_argument("--n-genes", type=int, default=200)
    p.add_argument("--stride", type=int, default=1)
    # K selection rejects non-converged fits, so a stingy budget silently
    # eliminates the larger K and collapses the recommendation onto the
    # smallest one. Give it room.
    p.add_argument("--max-iter", type=int, default=800)
    p.add_argument("--n-init", type=int, default=2)
    p.add_argument("--restrict-to-shared-replicates", action="store_true",
                   help="if replicate labels differ across timepoints, drop the "
                        "cells whose replicate is not present at every "
                        "timepoint and split on the shared ones (a paired "
                        "hold-out is otherwise undefined)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/k_selection")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    truth = None
    if args.sim:
        sc = sim.make(args.sim, seed=args.seed)
        sc.n_sample = args.n_per_timepoint
        sc.n_init = max(sc.n_init, 3 * args.n_per_timepoint)
        truth = sim.simulate(sc, n_genes=args.n_genes)
        data = truth.data
        print(f"[data] simulated {args.sim} with {sc.n_states} true states")
    else:
        import anndata as ad
        if args.replicate_key is None:
            raise SystemExit("protocol (b) needs --replicate-key: it holds out "
                             "whole duplicate samples")
        data = from_anndata(ad.read_h5ad(args.h5ad), time_key=args.time_key,
                            replicate_key=args.replicate_key)
    if args.stride > 1:
        data = subsample_timepoints(data, stride=args.stride)
    data = subsample_cells(data, args.n_per_timepoint, seed=args.seed)
    print(f"[data] {data}")

    cfg = PipelineConfig()
    cfg.representation.n_hvg = None if args.sim else 2000
    cfg.representation.n_components = (min(20, data.n_genes - 1) if args.sim else 30)
    cfg.coupling.epsilon = args.epsilon
    cfg.coupling.kappa = args.kappa
    cfg.coupling.support = "dense" if args.dense else "knn"
    cfg.optim.max_iter = args.max_iter
    cfg.optim.seed = args.seed
    cfg.optim.verbose = 0

    res = select_K(data, cfg, args.Ks, seed=args.seed,
                   n_init_for_stability=args.n_init,
                   restrict_to_shared_replicates=args.restrict_to_shared_replicates,
                   verbose=args.verbose)

    print(f"\n{'K':>4} {'heldout':>12} {'train':>12} {'Keff':>7} "
          f"{'min_g':>10} {'init_ARI':>9}")
    for r in res.per_K:
        print(f"{r['K']:4d} {r['heldout_compress']:12.6f} "
              f"{r['train_compress']:12.6f} {r['k_eff']:7.2f} "
              f"{r['min_state_mass']:10.2e} {r['init_ari']:9.3f}")
    rec = res.recommend()
    not_conv = [k for k, why in rec.get("rejected", {}).items()
                if "converge" in why]
    if not_conv:
        print(f"\n[WARNING] K = {not_conv} were rejected because their fits did "
              f"not converge, not because those K are bad. The remedy is a "
              f"larger --max-iter (currently {args.max_iter}), NOT accepting the "
              f"recommendation below -- non-converged fits skew toward small K.")
    print(f"\n[recommendation] {rec}")
    if truth is not None:
        print(f"[truth] the simulator used {truth.scenario.n_states} states")

    with open(os.path.join(args.out, "k_selection.json"), "w") as fh:
        json.dump({"protocol": res.protocol, "per_K": res.per_K,
                   "recommendation": rec, "notes": res.notes},
                  fh, indent=2, default=float)
    print(f"\n[done] wrote {args.out}")
    return res


if __name__ == "__main__":
    main()
