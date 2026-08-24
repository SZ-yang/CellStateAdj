#!/usr/bin/env python
"""STEP 3/4: fit time-local states at a fixed epsilon and K.

Build order matters (handoff s8).  Start with ``--lambda-plus 0 --lambda-minus 0``:
that reduces the objective to compression + expression coherence, a well-behaved
clustering problem, and it is the internal baseline.  Only then turn the
fingerprint terms on, and check the non-degeneracy conditions
(``--degeneracy-check``, on by default) before trusting anything.

    python scripts/run_fit.py --sim branching --K 6 --epsilon 0.05
    python scripts/run_fit.py --sim similar_expression_different_role \
        --K 6 --lambda-plus 5 --lambda-minus 5 --compare-baseline
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
from cellstateadj.diagnostics import membership_sensitivity
from cellstateadj.pipeline import run_pipeline
from cellstateadj import baselines, evaluate
from cellstateadj import simulate as sim


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5ad")
    src.add_argument("--sim")
    p.add_argument("--time-key", default="day")
    p.add_argument("--replicate-key", default=None)
    p.add_argument("--n-per-timepoint", type=int, default=800)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--kappa", type=int, default=50)
    p.add_argument("--dense", action="store_true")
    p.add_argument("--lambda-compress", type=float, default=1.0)
    p.add_argument("--lambda-x", type=float, default=1.0)
    p.add_argument("--lambda-plus", type=float, default=0.0)
    p.add_argument("--lambda-minus", type=float, default=0.0)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--method", default="full_gradient",
                   choices=["full_gradient", "block_coordinate"])
    p.add_argument("--no-geometric-null", action="store_true")
    p.add_argument("--compare-baseline", action="store_true",
                   help="also fit expression clustering + the same induced map")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results/fit")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    truth = None
    if args.sim:
        sc = sim.make(args.sim, seed=args.seed)
        sc.n_sample = args.n_per_timepoint
        truth = sim.simulate(sc, verbose=args.verbose)
        data = truth.data
        print(f"[data] simulated {args.sim}: {data}")
    else:
        import anndata as ad
        data = from_anndata(ad.read_h5ad(args.h5ad), time_key=args.time_key,
                            replicate_key=args.replicate_key)
        print(f"[data] {args.h5ad}: {data}")

    if args.stride > 1:
        data = subsample_timepoints(data, stride=args.stride)
    data = subsample_cells(data, args.n_per_timepoint, seed=args.seed)

    cfg = PipelineConfig()
    cfg.coupling.epsilon = args.epsilon
    cfg.coupling.kappa = args.kappa
    cfg.coupling.support = "dense" if args.dense else "knn"
    cfg.model.K = args.K
    cfg.model.lambda_compress = args.lambda_compress
    cfg.model.lambda_x = args.lambda_x
    cfg.model.lambda_plus = args.lambda_plus
    cfg.model.lambda_minus = args.lambda_minus
    cfg.model.device = args.device
    cfg.optim.max_iter = args.max_iter
    cfg.optim.n_init = args.n_init
    cfg.optim.method = args.method
    cfg.optim.seed = args.seed
    cfg.optim.verbose = args.verbose
    cfg.diagnostics.geometric_null = not args.no_geometric_null

    result = run_pipeline(data, cfg, verbose=args.verbose)
    out = {"config": json.loads(cfg.to_json()), "summary": result.summary()}

    if truth is not None:
        rec = evaluate.state_recovery(result.fit.M, truth.states)
        gs = [g for g in result.diagnostics.g] if result.diagnostics else None
        out["state_recovery"] = rec
        print(f"\n[truth] mean ARI vs ground-truth states: {rec['mean_ari']:.3f}")
        if result.diagnostics is not None:
            tr = evaluate.transition_recovery(
                result.diagnostics.A, result.diagnostics.g, result.fit.M,
                truth.states, truth.T_true, truth.scenario.n_states)
            out["transition_recovery"] = tr
            print(f"[truth] transition-map L1 error: {tr['mean_l1_error']:.3f} "
                  f"(corr {tr['mean_corr']:.3f})")

    if args.compare_baseline:
        labels = baselines.expression_clustering(result.Z, args.K, method="kmeans",
                                                 seed=args.seed)
        bmodel = baselines.induced_from_labels(result.chain, result.Z, labels,
                                               args.K, cfg.model)
        _, bterms = bmodel.objective()
        out["baseline_expression_clustering"] = bterms.as_dict()
        out["baseline_vs_model"] = membership_sensitivity(
            result.fit.M, bmodel.numpy_memberships())
        print(f"\n[baseline] expression clustering objective: {bterms.total:.6e} "
              f"vs model {result.fit.objective:.6e}")
        print(f"[baseline] ARI(model, expression clustering) = "
              f"{out['baseline_vs_model']['mean_ari']:.3f}")
        if truth is not None:
            brec = evaluate.state_recovery(bmodel.numpy_memberships(), truth.states)
            out["baseline_state_recovery"] = brec
            print(f"[baseline] mean ARI vs truth: {brec['mean_ari']:.3f}")

    np.savez_compressed(os.path.join(args.out, "memberships.npz"),
                        **{f"M_{t}": m for t, m in enumerate(result.fit.M)})
    if result.diagnostics is not None:
        np.savez_compressed(
            os.path.join(args.out, "diagnostics.npz"),
            **{f"T_{t}": T for t, T in enumerate(result.diagnostics.T_mat)},
            **{f"g_{t}": g for t, g in enumerate(result.diagnostics.g)},
            **{f"Vp_{t}": v for t, v in enumerate(result.diagnostics.V_plus)
               if v is not None},
            **{f"Vm_{t}": v for t, v in enumerate(result.diagnostics.V_minus)
               if v is not None},
        )
        try:
            result.diagnostics.event_table().to_csv(
                os.path.join(args.out, "events.csv"), index=False)
        except Exception:
            pass
        with open(os.path.join(args.out, "dag_edges.json"), "w") as fh:
            json.dump(result.diagnostics.dag_edges(), fh, indent=2)

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n[done] wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
