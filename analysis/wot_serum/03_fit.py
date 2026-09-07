#!/usr/bin/env python
"""Stage C -- the fit.  Frozen chain at epsilon*, fits at K*, full artifact dump.

Runs one fit per lambda_pm value against ONE reference chain (epsilon is frozen,
so the chain is identical and rebuilding it per fit would only cost time):

  fit_lam0   lambda_pm = 0    build-order step-3 internal baseline: this reduces to
                              compression + expression coherence, a well-behaved
                              clustering problem.
  fit_lam<v> lambda_pm = v    the fingerprint terms switched on.

There is deliberately no "headline" run here.  Which lambda_pm (if any) is
defensible is decided by 04_lambda_sweep.py, which looks for a range where the
fingerprint term moves the memberships WITHOUT collapsing effective state
occupancy (Degeneracy 3).  Until that sweep has been read, treat every DAG
produced here as one point on a sensitivity curve.

Everything the optimiser produces is written to disk regardless of convergence
status; ``run_fit.py`` discards artifacts on a non-converged fit, which is the
wrong trade when the point of the run is to have the matrices.

Output layout -- runs are keyed by a configuration hash, so two different
configurations can never collide on one chain file:

    <out>/run_<hash>/
        reference_chain.npz     chain_summary.json     manifest.json
        fit_lam0/  fit_lam5/  ...

    python 03_fit.py --epsilon 0.05 --K 12
    python 03_fit.py --epsilon 0.05 --K 12 --lambda-pm 0 0.5 2
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import csa_wot
from csa_wot import (DEFAULT_H5AD, RESULTS_ROOT, jdump, load_serum, make_cfg,
                     mass_conservation_check, provenance_block, reserve_destinations,
                     run_fingerprint, save_all)

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
    p.add_argument("--cost-scale-mode", default="global",
                   choices=["global", "per_interval", "none"],
                   help="'global' (default, one scale for the series) leaves phase 1 "
                        "~10x over-smoothed relative to the serum phase at any single "
                        "epsilon; 'per_interval' equalises the effective smoothing but "
                        "divides dtau out (nearly a no-op here: dtau=0.5 on 36/38 "
                        "intervals). Changing it changes the run fingerprint.")
    p.add_argument("--kappa", type=int, default=400,
                   help="kappa=50 (the package default) needs ~150x longer at 1000 "
                        "cells/timepoint because the support has to grow")
    p.add_argument("--support", default="knn", choices=["knn", "dense"])
    p.add_argument("--lambda-compress", type=float, default=1.0)
    p.add_argument("--lambda-x", type=float, default=1.0)
    p.add_argument("--lambda-pm", type=float, nargs="+", default=[0.0, 5.0],
                   help="one fit per value; lambda_plus = lambda_minus. The default "
                        "pair is a baseline plus ONE uncalibrated value -- see "
                        "04_lambda_sweep.py before calling any of them headline.")
    p.add_argument("--max-iter", type=int, default=1500)
    p.add_argument("--n-init", type=int, default=3)
    p.add_argument("--method", default="full_gradient",
                   choices=["full_gradient", "block_coordinate"])
    p.add_argument("--no-geometric-null", action="store_true",
                   help="skip the O(n^2) cross-fitted kernel null")
    p.add_argument("--min-edge-mass", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=RESULTS_ROOT)
    p.add_argument("--tag", default="", help="extra suffix on the run directory")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def _fit_dirname(lam: float) -> str:
    return "fit_lam0" if lam == 0 else f"fit_lam{lam:g}".replace(".", "p")


def main():
    args = parse_args()
    device = csa_wot.resolve_device(args.device, verbose=args.verbose)

    data, Z, obs = load_serum(args.h5ad, day_min=args.day_min, day_max=args.day_max,
                              stride=args.stride, n_per_timepoint=args.n_per_timepoint,
                              seed=args.seed, verbose=args.verbose)

    cfg0 = make_cfg(epsilon=args.epsilon, K=args.K, kappa=args.kappa,
                    support=args.support,
                   cost_scale_mode=args.cost_scale_mode, lambda_compress=args.lambda_compress,
                    lambda_x=args.lambda_x, max_iter=args.max_iter,
                    n_init=args.n_init, device=device, seed=args.seed,
                    verbose=args.verbose)
    cfg0.optim.method = args.method

    fp = run_fingerprint(Z, cfg0, h5ad=args.h5ad, day_min=args.day_min,
                         day_max=args.day_max, stride=args.stride,
                         n_per_timepoint=args.n_per_timepoint,
                         sampling_seed=args.seed)
    prov = provenance_block(fp)
    run_id = f"run_{prov['fingerprint_hash']}_K{args.K}{args.tag}"

    # ------------------------------------------------------------------
    # [CRITICAL] Resolve and reserve EVERY destination before writing anything.
    # A collision check that runs after the chain has been saved cannot stop the
    # chain from being replaced under the previous run's fits.
    # ------------------------------------------------------------------
    run_dir = os.path.join(args.out, run_id)
    chain_path = os.path.join(run_dir, "reference_chain.npz")
    lambdas = sorted(set(float(v) for v in args.lambda_pm))
    fit_dirs = {lam: os.path.join(run_dir, _fit_dirname(lam)) for lam in lambdas}

    destinations = [chain_path,
                    os.path.join(run_dir, "chain_summary.json"),
                    os.path.join(run_dir, "manifest.json"),
                    *fit_dirs.values()]
    reserve_destinations(destinations, overwrite=args.overwrite)

    os.makedirs(run_dir, exist_ok=True)
    print(f"\n[run] {run_id}")
    print(f"[run] config hash {prov['fingerprint_hash']} over "
          f"{len(destinations)} reserved destinations")
    print(f"[run] lambda_pm values: {lambdas}")

    # ------------------------------------------------------------------
    # the frozen reference chain -- built once, shared by every fit
    # ------------------------------------------------------------------
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
            f"reference chain is infeasible at intervals {chain.infeasible_intervals()}: "
            f"the marginals were not met to {csum['feasibility_tol']:.1e}, so A_t is "
            f"not row-stochastic and every transition number below it would be wrong. "
            f"Raise --kappa, use --support dense, or raise --epsilon.")

    chain.save(chain_path)
    print(f"[chain] saved {chain_path} ({os.path.getsize(chain_path) / 1e6:.0f} MB)")
    jdump({"chain": csum, "epsilon": args.epsilon, "K": args.K,
           "n_cells": data.n_cells, "tau": np.asarray(data.tau).tolist(),
           "dtau": np.asarray(data.dtau, dtype=float).tolist(),
           "provenance": prov},
          os.path.join(run_dir, "chain_summary.json"))

    # ------------------------------------------------------------------
    # the fits
    # ------------------------------------------------------------------
    manifest = {"run_id": run_id, "provenance": prov, "fits": {}}
    for lam in lambdas:
        outdir = fit_dirs[lam]
        os.makedirs(outdir, exist_ok=True)

        cfg = make_cfg(epsilon=args.epsilon, K=args.K, kappa=args.kappa,
                       support=args.support,
                   cost_scale_mode=args.cost_scale_mode,
                       lambda_compress=args.lambda_compress, lambda_x=args.lambda_x,
                       lambda_plus=lam, lambda_minus=lam,
                       max_iter=args.max_iter, n_init=args.n_init,
                       seed=args.seed, device=device,
                       geometric_null=not args.no_geometric_null,
                       verbose=args.verbose)
        cfg.optim.method = args.method

        print(f"\n{'=' * 72}\n[fit:lam={lam:g}] K={args.K} "
              f"max_iter={args.max_iter} n_init={args.n_init} device={device}\n{'=' * 72}")
        t0 = time.time()
        result = run_pipeline(data, cfg, Z=Z, chain=chain,
                              with_diagnostics=True, check_degeneracy=True,
                              verbose=args.verbose)
        elapsed = time.time() - t0

        mc = mass_conservation_check(result, t=0)
        print(f"[fit:lam={lam:g}] mass check: P^ref={mc['P_ref_mass']:.12f} "
              f"Phat={mc['Phat_total_mass']:.12f} "
              f"(max deviation {mc['max_abs_deviation_from_1']:.2e})")

        save_all(result, outdir, data, obs=obs,
                 min_edge_mass=args.min_edge_mass, chain_path=chain_path,
                 provenance=prov,
                 extra={"run_id": run_id, "lambda_pm": lam, "epsilon": args.epsilon,
                        "K": args.K, "kappa": args.kappa, "support": args.support,
                        "device": device, "stride": args.stride,
                        "elapsed_s": elapsed},
                 verbose=args.verbose)

        with open(os.path.join(outdir, "summary.json")) as fh:
            sm = json.load(fh)
        manifest["fits"][_fit_dirname(lam)] = {
            **{k: sm[k] for k in ("status", "converged", "dag", "mass_conservation")},
            "outdir": outdir, "lambda_pm": lam,
            "elapsed_min": round(elapsed / 60, 1),
        }
        print(f"[fit:lam={lam:g}] finished in {elapsed / 60:.1f} min -> {outdir}")

    # ------------------------------------------------------------------
    # how each fit differs from the lambda_pm = 0 baseline
    # ------------------------------------------------------------------
    if 0.0 in fit_dirs and len(lambdas) > 1:
        try:
            from cellstateadj.diagnostics import membership_sensitivity
            base = np.load(os.path.join(fit_dirs[0.0], "memberships.npz"))
            a = [base[f"M_{t}"] for t in range(data.T)]
            for lam in lambdas:
                if lam == 0.0:
                    continue
                other = np.load(os.path.join(fit_dirs[lam], "memberships.npz"))
                b = [other[f"M_{t}"] for t in range(data.T)]
                sens = membership_sensitivity(a, b)
                manifest["fits"][_fit_dirname(lam)]["vs_lambda0"] = sens
                print(f"[compare] lambda_pm=0 vs {lam:g}: mean ARI={sens['mean_ari']:.3f} "
                      f"mean L1 membership change={sens['mean_l1_membership_change']:.3f}")
        except Exception as exc:
            print(f"[compare] skipped: {type(exc).__name__}: {exc}")

    jdump(manifest, os.path.join(run_dir, "manifest.json"))
    print(f"\n[done] {run_dir}")
    for name, m in manifest["fits"].items():
        print(f"  {name:12s} status={m['status']:20s} dag={m['dag']}")
    print("\n[note] No run here is a headline result until 04_lambda_sweep.py "
          "identifies a lambda_pm range that moves memberships without collapsing "
          "K_eff.")


if __name__ == "__main__":
    main()
