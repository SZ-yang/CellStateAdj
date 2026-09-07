#!/usr/bin/env python
"""Stage B -- held-out K selection, protocol (b), one K per invocation.

Protocol (b) as decided in ``cellstateadj/selection.py``'s module docstring and
PROJECT_HANDOFF.txt s7: hold out an entire culture replicate and score the
*transferred* state map, rather than refitting P^ref on a reduced cell set.  The
procedure is unchanged; only ``learn_representation`` is dropped, because
``obsm['X_model']`` is already the frozen representation (see ``csa_wot.select_K_one``).

lambda_pm is forced to 0 throughout: L_pm systematically favours a coarser
neighbouring state space and is exactly 0 at K=1 (Degeneracy 3), so it must never
enter K selection.

One K per process so the sweep runs as a slurm array; ``02_k_reduce.py`` merges the
per-K JSONs and applies the package's own ``KSelectionResult.recommend()``.

    python 02_k_sweep.py --K 12 --epsilon 0.05
"""

from __future__ import annotations

import argparse
import os
import time

import csa_wot
from csa_wot import (DEFAULT_H5AD, RESULTS_ROOT, jdump, load_serum, make_cfg,
                     provenance_block, run_fingerprint, select_K_one)

DEFAULT_KS = [6, 8, 10, 12, 16, 20, 24, 30]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", default=DEFAULT_H5AD)
    p.add_argument("--day-min", type=float, default=0.0)
    p.add_argument("--day-max", type=float, default=18.0)
    p.add_argument("--K", type=int, default=None,
                   help="single K to evaluate; with --array-index, indexes --Ks instead")
    p.add_argument("--Ks", type=int, nargs="+", default=DEFAULT_KS)
    p.add_argument("--array-index", type=int, default=None,
                   help="SLURM_ARRAY_TASK_ID; picks Ks[array-index]")
    p.add_argument("--epsilon", type=float, required=True,
                   help="epsilon* from Stage A -- frozen, not re-selected here")
    p.add_argument("--kappa", type=int, default=200,
                   help="halves hold ~500 cells/timepoint, so a smaller support suffices")
    p.add_argument("--support", default="knn", choices=["knn", "dense"])
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--n-per-timepoint", type=int, default=None)
    p.add_argument("--max-iter", type=int, default=800)
    p.add_argument("--n-init", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="default: <RESULTS_ROOT>/k_sweep_<config hash>, so sweeps "
                        "with different configurations cannot share a directory")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing K_<K>.json for this K")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    if args.K is None:
        idx = args.array_index
        if idx is None:
            idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
        if not (0 <= idx < len(args.Ks)):
            raise SystemExit(
                f"give --K, or --array-index/SLURM_ARRAY_TASK_ID in [0, {len(args.Ks)}) "
                f"for Ks={args.Ks}")
        K = args.Ks[idx]
    else:
        K = args.K

    data, Z, _obs = load_serum(args.h5ad, day_min=args.day_min, day_max=args.day_max,
                               stride=args.stride, n_per_timepoint=args.n_per_timepoint,
                               seed=args.seed, verbose=args.verbose)

    cfg = make_cfg(epsilon=args.epsilon, K=K, kappa=args.kappa, support=args.support,
                   lambda_plus=0.0, lambda_minus=0.0, max_iter=args.max_iter,
                   n_init=args.n_init, seed=args.seed,
                   device=csa_wot.resolve_device(args.device), verbose=args.verbose)

    # The fingerprint deliberately EXCLUDES K -- that is the swept variable, and
    # the whole point is to compare across it -- but pins everything else, so the
    # reducer can refuse to merge files from different configurations.
    fp = run_fingerprint(Z, cfg, h5ad=args.h5ad, day_min=args.day_min,
                         day_max=args.day_max, stride=args.stride,
                         n_per_timepoint=args.n_per_timepoint,
                         sampling_seed=args.seed)
    prov = provenance_block(fp)

    out = args.out or os.path.join(RESULTS_ROOT,
                                   f"k_sweep_{prov['fingerprint_hash']}")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"K_{K:03d}.json")
    if os.path.exists(path) and not args.overwrite:
        raise SystemExit(f"{path} already exists; pass --overwrite to replace it")

    print(f"[K-sweep] K={K} epsilon={args.epsilon} kappa={args.kappa} "
          f"max_iter={args.max_iter} n_init={args.n_init}")
    print(f"[K-sweep] config hash {prov['fingerprint_hash']} -> {out}")
    t0 = time.time()
    rec = select_K_one(data, cfg, K, seed=args.seed,
                       n_init_for_stability=args.n_init, verbose=args.verbose)
    rec["elapsed_s"] = time.time() - t0
    rec["epsilon"] = args.epsilon
    rec["max_iter"] = args.max_iter
    rec["provenance"] = prov
    rec["Ks_requested"] = list(args.Ks)
    rec["protocol"] = ("b: hold out one duplicate SAMPLE (obs['batch'], a batch-wise "
                       "technical hold-out -- not a tracked culture lineage), transfer "
                       "the state map by nearest expression prototype, score "
                       "compression on the held-out half; lambda_pm = 0 throughout "
                       "(Degeneracy 3)")

    jdump(rec, path)
    print(f"[K-sweep] K={K} heldout_compress={rec['heldout_compress']:.6f} "
          f"train={rec['train_compress']:.6f} min_g={rec['min_state_mass']:.2e} "
          f"Keff={rec['k_eff']:.2f} converged={rec['all_converged']} "
          f"({rec['elapsed_s'] / 60:.1f} min)")
    print(f"[K-sweep] wrote {path}")


if __name__ == "__main__":
    main()
