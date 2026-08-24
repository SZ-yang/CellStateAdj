#!/usr/bin/env python
"""STEP 2/4: run the model across every simulator scenario.

Prints one row per scenario comparing transition-defined states against
expression clustering on the same frozen couplings, plus the non-degeneracy
check.  The two decisive rows are:

* ``similar_expression_different_role`` -- ours should beat clustering;
* ``distinct_expression_same_role``     -- ours should NOT collapse them
  (L_expression is in the objective precisely to prevent that).

    python scripts/run_simulation_study.py --lambda-plus 5 --lambda-minus 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cellstateadj.config import PipelineConfig
from cellstateadj.diagnostics import degeneracy_check
from cellstateadj.pipeline import run_pipeline
from cellstateadj import baselines, evaluate
from cellstateadj import simulate as sim


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", nargs="+", default=sorted(sim.SCENARIOS))
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--n-cells", type=int, default=300)
    p.add_argument("--n-genes", type=int, default=200)
    p.add_argument("--lambda-plus", type=float, default=0.0)
    p.add_argument("--lambda-minus", type=float, default=0.0)
    p.add_argument("--max-iter", type=int, default=150)
    p.add_argument("--dense", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/simulation_study")
    p.add_argument("--verbose", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rows = []

    for name in args.scenarios:
        sc = sim.make(name, seed=args.seed)
        sc.n_sample = args.n_cells
        sc.n_init = max(sc.n_init, 3 * args.n_cells)
        truth = sim.simulate(sc, n_genes=args.n_genes)

        cfg = PipelineConfig()
        cfg.coupling.epsilon = args.epsilon
        cfg.coupling.support = "dense" if args.dense else "knn"
        cfg.model.K = args.K
        cfg.model.lambda_plus = args.lambda_plus
        cfg.model.lambda_minus = args.lambda_minus
        cfg.optim.max_iter = args.max_iter
        cfg.optim.verbose = args.verbose
        cfg.optim.seed = args.seed
        cfg.diagnostics.geometric_null = True
        cfg.representation.n_hvg = None
        cfg.representation.n_components = min(20, args.n_genes - 1)

        res = run_pipeline(truth.data, cfg, verbose=args.verbose)
        ours = evaluate.state_recovery(res.fit.M, truth.states)

        labels = baselines.expression_clustering(res.Z, args.K, seed=args.seed)
        bmodel = baselines.induced_from_labels(res.chain, res.Z, labels, args.K,
                                               cfg.model)
        base = evaluate.state_recovery(bmodel.numpy_memberships(), truth.states)
        _, bterms = bmodel.objective()

        tr = evaluate.transition_recovery(res.diagnostics.A, res.diagnostics.g,
                                          res.fit.M, truth.states, truth.T_true,
                                          sc.n_states)
        deg = degeneracy_check(res.fit.model, t=0)

        row = dict(scenario=name, n_true_states=sc.n_states,
                   ari_ours=ours["mean_ari"], ari_expression=base["mean_ari"],
                   obj_ours=res.fit.objective, obj_expression=bterms.total,
                   transition_l1=tr["mean_l1_error"],
                   k_eff=float(np.mean(res.fit.terms.k_eff)),
                   V_plus=res.fit.terms.mean_V_plus,
                   ref_spread=deg["ref_within_state_spread"],
                   phat_spread=deg["phat_within_state_spread"],
                   monotone=res.fit.monotone, converged=res.fit.converged)
        rows.append(row)
        print(f"{name:42s} ARI ours={row['ari_ours']:.3f} "
              f"expr={row['ari_expression']:.3f}  "
              f"T-L1={row['transition_l1']:.3f}  Keff={row['k_eff']:.2f}  "
              f"V+={row['V_plus']:.4f}")

    with open(os.path.join(args.out, "study.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=float)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "study.csv"), index=False)
    except Exception:
        pass
    print(f"\n[done] wrote {args.out}")
    return rows


if __name__ == "__main__":
    main()
