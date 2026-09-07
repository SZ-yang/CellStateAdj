#!/usr/bin/env python
"""Stage D -- lambda_pm sweep at fixed epsilon and K (build-order step 4).

Why this exists
---------------
A zero baseline plus one uncalibrated value (lambda_pm = 5) cannot tell you
whether the fingerprint terms are doing anything defensible.  Two failure modes
sit on either side and both look like "it ran":

* too small -- L_pm does not move the memberships at all, so the result is the
  compression + expression clustering with extra runtime;
* too large -- L_pm is minimised by COARSENING the neighbouring state space
  (Degeneracy 3: at K = 1, L_pm = 0 exactly).  K is fixed, but effective
  occupancy ``K_eff_t = exp(H(g_t))`` can still collapse, and a collapsed fit can
  show an excellent L_pm while measuring nothing.

The usable range, if there is one, is where memberships move (ARI against
lambda = 0 drops below 1) while K_eff holds up.  Finding that range is the point;
PROJECT_HANDOFF.txt s3 says an observed collapse is a reportable result, not a bug.

This reuses the package's own ``pipeline.lambda_sweep`` semantics but records the
full per-lambda diagnostic set the handoff asks to instrument, and shares ONE
frozen reference chain across every lambda.

    python 04_lambda_sweep.py --epsilon 0.05 --K 12
    python 04_lambda_sweep.py --epsilon 0.05 --K 12 --lambdas 0 0.1 0.5 1 2 5
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import csa_wot
from csa_wot import (DEFAULT_H5AD, RESULTS_ROOT, jdump, load_serum, make_cfg,
                     provenance_block, reserve_destinations, run_fingerprint)

from cellstateadj.diagnostics import membership_sensitivity
from cellstateadj.optimize import fit as fit_states
from cellstateadj.reference import build_reference_chain

DEFAULT_LAMBDAS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", default=DEFAULT_H5AD)
    p.add_argument("--day-min", type=float, default=0.0)
    p.add_argument("--day-max", type=float, default=18.0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--n-per-timepoint", type=int, default=None)
    p.add_argument("--epsilon", type=float, required=True, help="epsilon* from Stage A")
    p.add_argument("--K", type=int, required=True, help="K* from Stage B")
    p.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS,
                   help="lambda_plus = lambda_minus values to sweep")
    p.add_argument("--cost-scale-mode", default="global",
                   choices=["global", "per_interval", "none"],
                   help="'global' (default, one scale for the series) leaves phase 1 "
                        "~10x over-smoothed relative to the serum phase at any single "
                        "epsilon; 'per_interval' equalises the effective smoothing but "
                        "divides dtau out (nearly a no-op here: dtau=0.5 on 36/38 "
                        "intervals). Changing it changes the run fingerprint.")
    p.add_argument("--kappa", type=int, default=400)
    p.add_argument("--support", default="knn", choices=["knn", "dense"])
    p.add_argument("--lambda-compress", type=float, default=1.0)
    p.add_argument("--lambda-x", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=1500)
    p.add_argument("--n-init", type=int, default=2,
                   help=">1 gives initialisation stability (pairwise restart ARI)")
    p.add_argument("--collapse-frac", type=float, default=0.8,
                   help="flag K_eff below this fraction of the lambda=0 value")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=RESULTS_ROOT)
    p.add_argument("--tag", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = csa_wot.resolve_device(args.device, verbose=args.verbose)

    data, Z, _obs = load_serum(args.h5ad, day_min=args.day_min, day_max=args.day_max,
                               stride=args.stride, n_per_timepoint=args.n_per_timepoint,
                               seed=args.seed, verbose=args.verbose)

    cfg0 = make_cfg(epsilon=args.epsilon, K=args.K, kappa=args.kappa,
                    support=args.support,
                   cost_scale_mode=args.cost_scale_mode, lambda_compress=args.lambda_compress,
                    lambda_x=args.lambda_x, max_iter=args.max_iter,
                    n_init=args.n_init, device=device, seed=args.seed,
                    verbose=args.verbose)

    fp = run_fingerprint(Z, cfg0, h5ad=args.h5ad, day_min=args.day_min,
                         day_max=args.day_max, stride=args.stride,
                         n_per_timepoint=args.n_per_timepoint,
                         sampling_seed=args.seed)
    prov = provenance_block(fp)
    sweep_dir = os.path.join(
        args.out, f"lambda_sweep_{prov['fingerprint_hash']}_K{args.K}{args.tag}")

    # resolve and reserve before writing anything (same rule as Stage C)
    dest = [os.path.join(sweep_dir, n) for n in
            ("lambda_sweep.json", "lambda_sweep.csv", "reference_chain.npz")]
    reserve_destinations(dest, overwrite=args.overwrite)
    os.makedirs(sweep_dir, exist_ok=True)

    lambdas = sorted(set(float(v) for v in args.lambdas))
    print(f"\n[sweep] {sweep_dir}")
    print(f"[sweep] K={args.K} epsilon={args.epsilon} lambdas={lambdas}")

    chain = build_reference_chain(Z, data.tau, cfg0.coupling, verbose=args.verbose)
    if not chain.feasible:
        raise SystemExit(
            f"reference chain infeasible at {chain.infeasible_intervals()}; "
            f"raise --kappa, use --support dense, or raise --epsilon")
    chain.save(os.path.join(sweep_dir, "reference_chain.npz"))

    rows, M_by_lambda = [], {}
    for lam in lambdas:
        cfg = make_cfg(epsilon=args.epsilon, K=args.K, kappa=args.kappa,
                       support=args.support,
                   cost_scale_mode=args.cost_scale_mode,
                       lambda_compress=args.lambda_compress, lambda_x=args.lambda_x,
                       lambda_plus=lam, lambda_minus=lam,
                       max_iter=args.max_iter, n_init=args.n_init,
                       seed=args.seed, device=device, verbose=max(0, args.verbose - 1))
        print(f"\n[sweep] lambda_pm={lam:g} ...")
        t0 = time.time()
        res = fit_states(chain, Z, cfg.model, cfg.optim)
        elapsed = time.time() - t0
        M_by_lambda[lam] = res.M

        terms = res.terms
        init_ari = (float(np.mean([r.get("mean_ari_to_others", np.nan)
                                   for r in res.restarts]))
                    if len(res.restarts) > 1 else float("nan"))
        rec = {
            "lambda_pm": lam,
            # every objective component, separately
            "total": terms.total, "compress": terms.compress,
            "expression": terms.expression, "plus": terms.plus, "minus": terms.minus,
            "compress_t": terms.compress_t, "expression_t": terms.expression_t,
            "plus_t": terms.plus_t, "minus_t": terms.minus_t,
            # convergence
            "status": res.status, "converged": bool(res.converged),
            "monotone": bool(res.monotone), "n_iter": int(res.n_iter),
            "grad_norm": float(res.grad_norm), "elapsed_s": elapsed,
            # occupancy
            "min_state_mass": float(np.min(terms.g_min)),
            "k_eff_per_timepoint": list(terms.k_eff),
            "k_eff_mean": float(np.mean(terms.k_eff)),
            "k_eff_min": float(np.min(terms.k_eff)),
            "floor_fraction": float(np.max(terms.floor_fraction)) if terms.floor_fraction else 0.0,
            "mean_V_plus": terms.mean_V_plus, "mean_V_minus": terms.mean_V_minus,
            # initialisation stability
            "init_ari": init_ari, "n_init": args.n_init,
        }
        if 0.0 in M_by_lambda:
            sens = membership_sensitivity(M_by_lambda[0.0], res.M)
            rec["ari_vs_lambda0"] = sens["mean_ari"]
            rec["l1_vs_lambda0"] = sens["mean_l1_membership_change"]
        rows.append(rec)
        print(f"[sweep] lambda={lam:g}: status={res.status} "
              f"L={terms.total:.6f} (comp {terms.compress:.4f} / x {terms.expression:.4f} "
              f"/ + {terms.plus:.4f} / - {terms.minus:.4f})  "
              f"Keff_mean={rec['k_eff_mean']:.2f} min_g={rec['min_state_mass']:.2e} "
              f"ARI_vs_0={rec.get('ari_vs_lambda0', float('nan')):.3f} "
              f"({elapsed / 60:.1f} min)")

    # ------------------------------------------------------------------
    # the usable range: memberships move, occupancy does not collapse
    # ------------------------------------------------------------------
    base = next((r for r in rows if r["lambda_pm"] == 0.0), None)
    verdict = {"rule": ("usable = memberships moved from lambda=0 (ARI < 0.99) AND "
                        "K_eff_mean >= collapse_frac * K_eff_mean(lambda=0) AND the "
                        "fit converged"),
               "collapse_frac": args.collapse_frac}
    if base is None:
        verdict["error"] = "no lambda_pm = 0 run in the sweep; cannot judge movement"
    else:
        floor = args.collapse_frac * base["k_eff_mean"]
        usable = [r["lambda_pm"] for r in rows
                  if r["lambda_pm"] > 0
                  and r.get("ari_vs_lambda0", 1.0) < 0.99
                  and r["k_eff_mean"] >= floor
                  and r["converged"]]
        collapsed = [r["lambda_pm"] for r in rows if r["k_eff_mean"] < floor]
        inert = [r["lambda_pm"] for r in rows
                 if r["lambda_pm"] > 0 and r.get("ari_vs_lambda0", 0.0) >= 0.99]
        nonconv = [r["lambda_pm"] for r in rows if not r["converged"]]
        verdict.update(k_eff_at_lambda0=base["k_eff_mean"], k_eff_floor=floor,
                       usable_lambdas=usable, collapsed_lambdas=collapsed,
                       inert_lambdas=inert, nonconverged_lambdas=nonconv)
        verdict["headline_candidate"] = (max(usable) if usable else None)
        if not usable:
            # Name the ACTUAL blocking gate. "No usable range" reads as a
            # statement about the method when the real cause may simply be that
            # nothing converged, which is a run-configuration problem.
            positives = [r for r in rows if r["lambda_pm"] > 0]
            if positives and all(not r["converged"] for r in positives):
                why = ("every lambda_pm > 0 fit stopped before meeting its "
                       "tolerances, so none of them can be judged. Raise "
                       "--max-iter (or check for line_search_failed) and re-run; "
                       "this says nothing about the fingerprint terms yet.")
            elif collapsed and not inert:
                why = (f"effective state occupancy collapsed below "
                       f"{args.collapse_frac:.0%} of its lambda=0 value at "
                       f"{collapsed} -- Degeneracy 3, and a reportable result.")
            elif inert and not collapsed:
                why = (f"the fingerprint terms did not move the memberships at "
                       f"{inert} (ARI vs lambda=0 >= 0.99); try larger values.")
            else:
                why = ("no value both moved the memberships and preserved "
                       "effective state occupancy -- the term goes from inert to "
                       "collapsing with no usable window between.")
            verdict["blocking_reason"] = why
            verdict["conclusion"] = (
                f"No usable lambda_pm: {why} Do NOT designate a headline DAG from "
                f"this sweep; report the curve, which is itself a result "
                f"(PROJECT_HANDOFF.txt s3, Degeneracy 3).")
        else:
            verdict["conclusion"] = (
                f"Usable range {min(usable):g}-{max(usable):g}. Report results "
                f"ACROSS it, not at a single value; the endpoints are where the "
                f"term stops mattering and where occupancy starts to collapse.")

    out = {"provenance": prov, "K": args.K, "epsilon": args.epsilon,
           "lambdas": lambdas, "per_lambda": rows, "verdict": verdict,
           "chain": chain.summary()}
    jdump(out, os.path.join(sweep_dir, "lambda_sweep.json"))
    try:
        import pandas as pd
        flat = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows]
        pd.DataFrame(flat).to_csv(os.path.join(sweep_dir, "lambda_sweep.csv"),
                                  index=False)
    except Exception as exc:
        print(f"[sweep] (csv skipped: {type(exc).__name__}: {exc})")

    print(f"\n[sweep] verdict: {json.dumps(verdict, indent=2, default=str)}")
    print(f"[sweep] wrote {sweep_dir}")


if __name__ == "__main__":
    main()
