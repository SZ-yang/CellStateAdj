#!/usr/bin/env python
"""Stage C -- the fit.  Frozen chain at epsilon*, two fits at K*, full artifact dump.

Runs two fits against ONE reference chain (epsilon is frozen, so the chain is
identical and rebuilding it per fit would only cost time):

  fit_lam0   lambda_pm = 0    build-order step-3 internal baseline: this reduces to
                              compression + expression coherence, a well-behaved
                              clustering problem.
  fit_main   lambda_pm > 0    the method -- the headline DAG.

Everything the optimiser produces is written to disk regardless of convergence
status; ``run_fit.py`` discards artifacts on a non-converged fit, which is the
wrong trade when the point of the run is to have the matrices.

    python 03_fit.py --epsilon 0.05 --K 12
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import csa_wot
from csa_wot import (DEFAULT_H5AD, RESULTS_ROOT, jdump, load_serum, make_cfg,
                     mass_conservation_check, save_all)

from cellstateadj.pipeline import run_pipeline
from cellstateadj.reference import build_reference_chain


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", default=DEFAULT_H5AD)
    p.add_argument("--day-min", type=float, default=0.0)
    p.add_argument("--day-max", type=float, default=18.0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--n-per-timepoint", type=int, default=None,
                   help="default: every cell (1000/day)")
    p.add_argument("--epsilon", type=float, required=True,
                   help="epsilon* from Stage A")
    p.add_argument("--K", type=int, required=True, help="K* from Stage B")
    p.add_argument("--kappa", type=int, default=400,
                   help="kappa=50 (the package default) is infeasible at 1000 "
                        "cells/timepoint and costs ~150x in support growth")
    p.add_argument("--support", default="knn", choices=["knn", "dense"])
    p.add_argument("--lambda-compress", type=float, default=1.0)
    p.add_argument("--lambda-x", type=float, default=1.0)
    p.add_argument("--lambda-pm", type=float, default=5.0,
                   help="lambda_plus = lambda_minus for the main fit")
    p.add_argument("--max-iter", type=int, default=1500)
    p.add_argument("--n-init", type=int, default=3)
    p.add_argument("--method", default="full_gradient",
                   choices=["full_gradient", "block_coordinate"])
    p.add_argument("--runs", nargs="+", default=["lam0", "main"],
                   choices=["lam0", "main"])
    p.add_argument("--no-geometric-null", action="store_true",
                   help="skip the O(n^2) cross-fitted kernel null")
    p.add_argument("--min-edge-mass", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=RESULTS_ROOT)
    p.add_argument("--tag", default="", help="suffix for the output subdirectories")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = csa_wot.resolve_device(args.device, verbose=args.verbose)
    os.makedirs(args.out, exist_ok=True)

    data, Z, obs = load_serum(args.h5ad, day_min=args.day_min, day_max=args.day_max,
                              stride=args.stride, n_per_timepoint=args.n_per_timepoint,
                              seed=args.seed, verbose=args.verbose)

    # ------------------------------------------------------------------
    # the frozen reference chain -- built once, shared by every fit
    # ------------------------------------------------------------------
    cfg0 = make_cfg(epsilon=args.epsilon, K=args.K, kappa=args.kappa,
                    support=args.support, device=device, seed=args.seed,
                    verbose=args.verbose)
    print(f"\n[chain] epsilon={args.epsilon} support={args.support} "
          f"kappa={args.kappa} over {data.T - 1} intervals")
    t0 = time.time()
    chain = build_reference_chain(Z, data.tau, cfg0.coupling, verbose=args.verbose)
    print(f"[chain] built in {time.time() - t0:.1f}s")

    csum = chain.summary()
    print(f"[chain] feasible={csum['feasible']}  "
          f"max marginal_error={max(csum['marginal_error']):.3e} "
          f"(tol {csum['feasibility_tol']:.1e})")
    print(f"[chain] kappas={csum['kappas']}")
    print(f"[chain] nnz per interval: min={min(csum['nnz'])} max={max(csum['nnz'])}")
    grown = [t for t, k in enumerate(csum["kappas"])
             if k is not None and k > args.kappa]
    if grown:
        print(f"[chain] NOTE: kappa grew above {args.kappa} at intervals {grown}")
    if not csum["feasible"]:
        raise SystemExit(
            f"reference chain is infeasible at intervals {chain.infeasible_intervals()}; "
            f"A_t would not be row-stochastic and every transition number below it "
            f"would be wrong. Raise --kappa, use --support dense, or raise --epsilon.")

    chain_path = os.path.join(args.out, f"reference_chain_eps{args.epsilon:g}.npz")
    chain.save(chain_path)
    print(f"[chain] saved {chain_path} ({os.path.getsize(chain_path) / 1e6:.0f} MB)")
    jdump({"chain": csum, "epsilon": args.epsilon, "K": args.K,
           "n_cells": data.n_cells, "tau": np.asarray(data.tau).tolist(),
           "dtau": np.asarray(data.dtau, dtype=float).tolist(),
           "branch_caveat": csa_wot.BRANCH_CAVEAT},
          os.path.join(args.out, "chain_summary.json"))

    # ------------------------------------------------------------------
    # the fits
    # ------------------------------------------------------------------
    plan = {"lam0": 0.0, "main": args.lambda_pm}
    manifest = {}
    for name in args.runs:
        lam = plan[name]
        outdir = os.path.join(args.out, f"fit_{name}{args.tag}")
        if os.path.isdir(outdir) and os.listdir(outdir) and not args.overwrite:
            raise SystemExit(f"{outdir} is not empty; pass --overwrite")
        os.makedirs(outdir, exist_ok=True)

        cfg = make_cfg(epsilon=args.epsilon, K=args.K, kappa=args.kappa,
                       support=args.support,
                       lambda_compress=args.lambda_compress, lambda_x=args.lambda_x,
                       lambda_plus=lam, lambda_minus=lam,
                       max_iter=args.max_iter, n_init=args.n_init,
                       seed=args.seed, device=device,
                       geometric_null=not args.no_geometric_null,
                       verbose=args.verbose)
        cfg.optim.method = args.method

        print(f"\n{'=' * 72}\n[fit:{name}] K={args.K} lambda_pm={lam} "
              f"max_iter={args.max_iter} n_init={args.n_init} device={device}\n{'=' * 72}")
        t0 = time.time()
        result = run_pipeline(data, cfg, Z=Z, chain=chain,
                              with_diagnostics=True, check_degeneracy=True,
                              verbose=args.verbose)
        elapsed = time.time() - t0

        mc = mass_conservation_check(result, t=0)
        print(f"[fit:{name}] mass check: P^ref={mc['P_ref_mass']:.12f} "
              f"Phat={mc['Phat_total_mass']:.12f} "
              f"(max deviation {mc['max_abs_deviation_from_1']:.2e})")

        save_all(result, outdir, data, obs=obs,
                 min_edge_mass=args.min_edge_mass, chain_path=chain_path,
                 extra={"run": name, "lambda_pm": lam, "epsilon": args.epsilon,
                        "K": args.K, "kappa": args.kappa, "support": args.support,
                        "device": device, "stride": args.stride,
                        "elapsed_s": elapsed},
                 verbose=args.verbose)

        with open(os.path.join(outdir, "summary.json")) as fh:
            sm = json.load(fh)
        manifest[name] = {k: sm[k] for k in ("status", "converged", "dag",
                                             "mass_conservation")}
        manifest[name].update(outdir=outdir, lambda_pm=lam,
                              elapsed_min=round(elapsed / 60, 1))
        print(f"[fit:{name}] finished in {elapsed / 60:.1f} min -> {outdir}")

    # ------------------------------------------------------------------
    # how the two fits differ -- cheap, and the point of running both
    # ------------------------------------------------------------------
    if set(args.runs) == {"lam0", "main"}:
        try:
            from cellstateadj.diagnostics import membership_sensitivity
            M0 = np.load(os.path.join(args.out, f"fit_lam0{args.tag}", "memberships.npz"))
            M1 = np.load(os.path.join(args.out, f"fit_main{args.tag}", "memberships.npz"))
            a = [M0[f"M_{t}"] for t in range(data.T)]
            b = [M1[f"M_{t}"] for t in range(data.T)]
            sens = membership_sensitivity(a, b)
            manifest["lam0_vs_main"] = sens
            print(f"\n[compare] lambda_pm=0 vs {args.lambda_pm}: "
                  f"mean ARI={sens['mean_ari']:.3f} "
                  f"mean L1 membership change={sens['mean_l1_membership_change']:.3f}")
        except Exception as exc:
            print(f"[compare] skipped: {type(exc).__name__}: {exc}")

    jdump(manifest, os.path.join(args.out, f"manifest{args.tag}.json"))
    print(f"\n[done] {args.out}")
    for name, m in manifest.items():
        if isinstance(m, dict) and "status" in m:
            print(f"  {name:6s} status={m['status']:20s} dag={m['dag']}")


if __name__ == "__main__":
    main()
