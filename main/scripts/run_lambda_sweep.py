#!/usr/bin/env python
"""STEP 4/5: sweep the fingerprint weights at fixed epsilon and K.

Nested model selection (spec 1.15): epsilon first, then K (with L_pm excluded),
then and only then lambda_+ / lambda_-.  This script assumes the first two are
already fixed.

WATCH K_eff.  L_pm decreases when the neighbouring state space collapses, so if
K_eff falls as lambda grows you have found the edge of the usable regime.  That
is a reportable result, not merely a bug.

    python scripts/run_lambda_sweep.py --sim similar_expression_different_role \
        --K 6 --epsilon 0.05 --lambdas 0 0.5 2 5 20
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
from cellstateadj.diagnostics import (degeneracy_check, membership_sensitivity,
                                      normalized_transition_ratios)
from cellstateadj.optimize import fit
from cellstateadj.reference import build_reference_chain
from cellstateadj.representation import learn_representation
from cellstateadj import evaluate
from cellstateadj import simulate as sim


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5ad")
    src.add_argument("--sim")
    p.add_argument("--time-key", default="day")
    p.add_argument("--replicate-key", default=None)
    p.add_argument("--n-per-timepoint", type=int, default=300)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--kappa", type=int, default=50)
    p.add_argument("--dense", action="store_true")
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[0.0, 0.5, 2.0, 5.0, 20.0])
    p.add_argument("--direction", default="both", choices=["both", "plus", "minus"])
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--n-genes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/lambda_sweep")
    p.add_argument("--verbose", type=int, default=0)
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
    else:
        import anndata as ad
        data = from_anndata(ad.read_h5ad(args.h5ad), time_key=args.time_key,
                            replicate_key=args.replicate_key)
    if args.stride > 1:
        data = subsample_timepoints(data, stride=args.stride)
    data = subsample_cells(data, args.n_per_timepoint, seed=args.seed)
    print(f"[data] {data}")

    cfg = PipelineConfig()
    cfg.representation.n_hvg = None if args.sim else 2000
    cfg.representation.n_components = min(20, data.n_genes - 1) if args.sim else 30
    cfg.coupling.epsilon = args.epsilon
    cfg.coupling.kappa = args.kappa
    cfg.coupling.support = "dense" if args.dense else "knn"
    cfg.model.K = args.K
    cfg.optim.max_iter = args.max_iter
    cfg.optim.seed = args.seed
    cfg.optim.verbose = args.verbose

    Z, _ = learn_representation(data, cfg.representation)
    chain = build_reference_chain(Z, data.tau, cfg.coupling, verbose=0)
    print(f"[chain] eps={args.epsilon} feasible={chain.feasible}")

    from dataclasses import replace as dc_replace
    rows, M_ref = [], None
    print(f"\n{'lambda':>8} {'L':>13} {'L_+':>11} {'L_-':>11} {'Keff':>7} "
          f"{'minG':>9} {'R+':>7} {'ARI_to_0':>9}" + ("  ARI_truth" if truth else ""))
    for lam in args.lambdas:
        mcfg = dc_replace(
            cfg.model,
            lambda_plus=lam if args.direction in ("both", "plus") else 0.0,
            lambda_minus=lam if args.direction in ("both", "minus") else 0.0,
        )
        res = fit(chain, Z, mcfg, cfg.optim)
        ratios = normalized_transition_ratios(res.model)
        rp = float(np.nanmean([v for v in ratios["R_plus"] if np.isfinite(v)]))
        rec = res.summary()
        rec.update(lam=float(lam), k_eff_per_t=res.terms.k_eff, R_plus=rp)
        if M_ref is None:
            M_ref = res.M
            rec["ari_to_lambda0"] = 1.0
        else:
            rec["ari_to_lambda0"] = membership_sensitivity(M_ref, res.M)["mean_ari"]
        line = (f"{lam:8g} {rec['total']:13.6e} {res.terms.plus:11.4e} "
                f"{res.terms.minus:11.4e} {rec['k_eff_mean']:7.3f} "
                f"{rec['g_min']:9.2e} {rp:7.3f} {rec['ari_to_lambda0']:9.3f}")
        if truth is not None:
            tr = evaluate.state_recovery(res.M, truth.states)
            rec["ari_truth"] = tr["mean_ari"]
            line += f"  {tr['mean_ari']:9.3f}"
        print(line)
        rows.append(rec)

    keff = [r["k_eff_mean"] for r in rows]
    if keff[-1] < 0.8 * keff[0]:
        print(f"\n[WARNING] K_eff fell from {keff[0]:.2f} to {keff[-1]:.2f} across "
              f"the sweep: coarsening pressure (Degeneracy 3) is active. The "
              f"upper end of this lambda range is outside the usable regime -- "
              f"report this, it is a result.")

    deg = degeneracy_check(fit(chain, Z, dc_replace(cfg.model, lambda_plus=0.0,
                                                    lambda_minus=0.0),
                               cfg.optim).model, t=0)
    print(f"\n[degeneracy] {deg['verdict']}")
    print(f"  P^ref within-state spread: {deg['ref_within_state_spread']:.6f}")
    print(f"  Phat within-state spread:  {deg['phat_within_state_spread']:.3e}")

    with open(os.path.join(args.out, "sweep.json"), "w") as fh:
        json.dump({"rows": rows, "degeneracy": deg}, fh, indent=2, default=float)
    print(f"\n[done] wrote {args.out}")
    return rows


if __name__ == "__main__":
    main()
