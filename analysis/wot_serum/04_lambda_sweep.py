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
    p.add_argument("--lambda-x", type=float, nargs="+", default=[1.0],
                   help="sweepable. [CRITICAL] lambda_x carries a UNIT CONVERSION: "
                        "L_comp and L_pm are in nats (KL / conditional MI) while "
                        "L_expr is squared PCA distance / d, so lambda_x=1 has no "
                        "meaning and measured ~90%% of the objective on this data -- "
                        "which made L_pm inert at every value tried. Sweep this "
                        "alongside --lambdas to find a regime where transport "
                        "actually competes.")
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



def _flush(rows, sweep_dir, prov, args, lambdas, lambda_xs, chain):
    """Persist after EVERY cell.

    The first version of this script wrote its JSON only at the end, so when the
    lambda_pm=5 fit raised a non-finite gradient it destroyed five completed fits
    that had each taken minutes.  Partial results are the normal case for a sweep,
    not an exception.
    """
    payload = {"provenance": prov, "K": args.K, "epsilon": args.epsilon,
               "lambda_compress": args.lambda_compress,
               "lambda_xs": lambda_xs, "lambdas_pm": lambdas,
               "per_cell": rows, "chain": chain.summary(),
               "complete": len(rows) == len(lambda_xs) * len(lambdas)}
    jdump(payload, os.path.join(sweep_dir, "lambda_sweep.json"))
    try:
        import pandas as pd
        flat = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows]
        pd.DataFrame(flat).to_csv(os.path.join(sweep_dir, "lambda_sweep.csv"),
                                  index=False)
    except Exception:
        pass


def _verdict(rows, lambda_xs, lambdas, collapse_frac,
             ari_moved_max=0.99, ari_related_min=0.30,
             lpm_collapse_frac=0.5, ming_collapse_frac=0.5):
    """Which (lambda_x, lambda_pm) cells are usable, and why the others are not.

    [CRITICAL] An earlier version of this gate passed cells that were textbook
    Degeneracy 3 collapses, because it tested only "ARI < 0.99" and "k_eff held".
    Three things were wrong and all three are now checked:

    * **ARI needs a LOWER bound too.**  ARI = -0.009 trivially satisfies
      "< 0.99", but it means the fit moved to an UNRELATED partition, not that
      lambda_pm refined the states.  A usable cell must move AND stay related.
    * **k_eff is not sufficient to detect collapse.**  In the K=20 grid k_eff
      ROSE (16.78 -> 18.07) while a state was annihilated to min_g = 5.8e-09:
      the surviving mass simply became more even, which raises the entropy.
      ``min_state_mass`` catches this; ``k_eff`` alone does not.
    * **L_pm itself must not collapse.**  The signature of Degeneracy 3 is
      L_pm crashing toward zero (3.0 -> 0.0001 observed) while L_compress
      degrades (9.8 -> 27.5).  A near-zero sufficiency loss bought by
      destroying the quantity being measured is not a solution.

    A baseline that did not converge cannot anchor an ARI comparison either, so
    the lambda_pm = 0 cell must have converged for its column to be judged.
    """
    ok = [r for r in rows if r.get("status") != "error"]
    out = {"collapse_frac": collapse_frac,
           "ari_band": [ari_related_min, ari_moved_max],
           "gate_1": "transport share >= expression share (else the fit is k-means)",
           "gate_2": (f"converged (and its lambda_pm=0 baseline converged); "
                      f"{ari_related_min} <= ARI vs baseline < {ari_moved_max} "
                      f"(moved but still related); k_eff >= {collapse_frac} x "
                      f"baseline; min_state_mass >= {ming_collapse_frac} x baseline; "
                      f"L_pm >= {lpm_collapse_frac} x baseline (no Degeneracy-3 "
                      f"collapse)"),
           "n_error_cells": len(rows) - len(ok),
           "error_cells": [{"lambda_x": r["lambda_x"], "lambda_pm": r["lambda_pm"],
                            "error": r.get("error")}
                           for r in rows if r.get("status") == "error"]}

    balanced = [r for r in ok
                if r.get("share_compress", 0) >= r.get("share_expression", 1)]
    out["lambda_x_where_transport_competes"] = sorted({r["lambda_x"] for r in balanced})
    if not balanced:
        out["conclusion"] = (
            "GATE 1 FAILED at every lambda_x: expression still owns more of the "
            "objective than transport, so states are expression-defined and "
            "lambda_pm cannot matter. Lower lambda_x, or renormalise L_expr into "
            "nats -- lambda_x carries a unit conversion.")
        return out

    usable, rejected = [], []
    for lx in out["lambda_x_where_transport_competes"]:
        base = next((r for r in ok if r["lambda_x"] == lx and r["lambda_pm"] == 0.0), None)
        if base is None:
            continue
        base_lpm = base["plus"] + base["minus"]
        for r in ok:
            if r["lambda_x"] != lx or r["lambda_pm"] <= 0:
                continue
            ari = r.get("ari_vs_lambda0", 1.0)
            lpm = r["plus"] + r["minus"]
            why = []
            if not r["converged"]:
                why.append(f"did not converge ({r['status']})")
            if not base["converged"]:
                why.append(f"lambda_pm=0 baseline did not converge ({base['status']})")
            if ari >= ari_moved_max:
                why.append(f"inert (ARI {ari:.3f} >= {ari_moved_max})")
            if ari < ari_related_min:
                why.append(f"UNRELATED partition (ARI {ari:.3f} < {ari_related_min}) "
                           f"-- moved to a different basin, not a refinement")
            if r["k_eff_mean"] < collapse_frac * base["k_eff_mean"]:
                why.append(f"k_eff collapsed ({r['k_eff_mean']:.2f} < "
                           f"{collapse_frac * base['k_eff_mean']:.2f})")
            if r["min_state_mass"] < ming_collapse_frac * base["min_state_mass"]:
                why.append(f"state annihilated (min_g {r['min_state_mass']:.2e} < "
                           f"{ming_collapse_frac * base['min_state_mass']:.2e})")
            if lpm < lpm_collapse_frac * base_lpm:
                why.append(f"DEGENERACY 3: L_pm collapsed ({lpm:.4f} < "
                           f"{lpm_collapse_frac * base_lpm:.4f}) while L_comp went "
                           f"{base['compress']:.2f} -> {r['compress']:.2f}")
            cell = {"lambda_x": lx, "lambda_pm": r["lambda_pm"],
                    "ari_vs_lambda0": ari, "k_eff_mean": r["k_eff_mean"],
                    "min_state_mass": r["min_state_mass"], "L_pm": lpm,
                    "L_compress": r["compress"], "init_ari": r.get("init_ari")}
            (usable if not why else rejected).append(
                cell if not why else {**cell, "rejected_because": why})

    out["usable_cells"] = usable
    out["rejected_cells"] = rejected
    n_degen = sum(1 for r in rejected
                  if any("DEGENERACY 3" in w for w in r["rejected_because"]))
    n_unrel = sum(1 for r in rejected
                  if any("UNRELATED" in w for w in r["rejected_because"]))
    out["n_degeneracy3_collapses"] = n_degen
    out["n_unrelated_partitions"] = n_unrel

    # restart reproducibility is a precondition for interpreting ANY cell
    iar = [r.get("init_ari") for r in ok if r.get("init_ari") is not None]
    iar = [x for x in iar if x == x]
    if iar:
        out["init_ari_max"] = max(iar)
        out["init_ari_median"] = sorted(iar)[len(iar) // 2]
        if max(iar) < 0.5:
            out["reproducibility_warning"] = (
                f"No cell is reproducible across restarts (init_ari max "
                f"{max(iar):.3f}, median {sorted(iar)[len(iar)//2]:.3f}). Different "
                f"seeds give unrelated state maps AT EVERY WEIGHTING TESTED, so no "
                f"cell below should be read as 'the' state map even if it passes "
                f"the gates.")

    out["conclusion"] = (
        f"{len(usable)} usable cell(s): {usable}."
        if usable else
        f"No usable cell. {n_degen} cell(s) were Degeneracy-3 collapses (L_pm driven "
        f"to ~0 while compression degraded), {n_unrel} moved to unrelated partitions, "
        f"the rest were inert or did not converge. See rejected_cells.")
    return out


def main():
    args = parse_args()
    device = csa_wot.resolve_device(args.device, verbose=args.verbose)

    data, Z, _obs = load_serum(args.h5ad, day_min=args.day_min, day_max=args.day_max,
                               stride=args.stride, n_per_timepoint=args.n_per_timepoint,
                               seed=args.seed, verbose=args.verbose)

    lambda_xs = sorted(set(float(v) for v in args.lambda_x), reverse=True)

    # The chain depends only on the coupling config, so lambda_x is irrelevant to
    # it; cfg0 pins the LARGEST lambda_x purely as the fingerprint's reference
    # value.  lambda_x is a swept variable here, exactly as K is in the K sweep, so
    # the full list travels in the output rather than in the hash.
    cfg0 = make_cfg(epsilon=args.epsilon, K=args.K, kappa=args.kappa,
                    support=args.support,
                    cost_scale_mode=args.cost_scale_mode,
                    lambda_compress=args.lambda_compress,
                    lambda_x=lambda_xs[0], max_iter=args.max_iter,
                    n_init=args.n_init, device=device, seed=args.seed,
                    verbose=args.verbose)

    fp = run_fingerprint(Z, cfg0, h5ad=args.h5ad, day_min=args.day_min,
                         day_max=args.day_max, stride=args.stride,
                         n_per_timepoint=args.n_per_timepoint,
                         sampling_seed=args.seed)
    fp["lambda_x"] = f"SWEPT: {lambda_xs}"
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

    grid = [(lx, lp) for lx in lambda_xs for lp in lambdas]
    print(f"[sweep] grid: {len(lambda_xs)} lambda_x x {len(lambdas)} lambda_pm "
          f"= {len(grid)} fits")

    rows, M_by_lambda = [], {}
    for lx, lam in grid:
        cfg = make_cfg(epsilon=args.epsilon, K=args.K, kappa=args.kappa,
                       support=args.support,
                       cost_scale_mode=args.cost_scale_mode,
                       lambda_compress=args.lambda_compress, lambda_x=lx,
                       lambda_plus=lam, lambda_minus=lam,
                       max_iter=args.max_iter, n_init=args.n_init,
                       seed=args.seed, device=device, verbose=max(0, args.verbose - 1))
        print(f"\n[sweep] lambda_x={lx:g} lambda_pm={lam:g} ...", flush=True)
        t0 = time.time()
        # A non-finite gradient at one cell must not destroy the whole sweep: it is
        # itself a result (it happened at K=20, lambda_pm=5) and every other cell is
        # still valid.  optimize.py raises FloatingPointError; anything else from the
        # optimiser is recorded the same way rather than propagating.
        try:
            res = fit_states(chain, Z, cfg.model, cfg.optim)
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            elapsed = time.time() - t0
            print(f"[sweep] lambda_x={lx:g} lambda_pm={lam:g} FAILED: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            rows.append({"lambda_x": lx, "lambda_pm": lam, "status": "error",
                         "error": f"{type(exc).__name__}: {exc}",
                         "converged": False, "elapsed_s": elapsed})
            _flush(rows, sweep_dir, prov, args, lambdas, lambda_xs, chain)
            continue
        elapsed = time.time() - t0
        M_by_lambda[(lx, lam)] = res.M

        terms = res.terms
        init_ari = (float(np.mean([r.get("mean_ari_to_others", np.nan)
                                   for r in res.restarts]))
                    if len(res.restarts) > 1 else float("nan"))
        tot = max(abs(terms.total), 1e-30)
        rec = {
            "lambda_x": lx,
            "lambda_pm": lam,
            # the diagnosis that mattered: which term actually owns the objective
            "share_compress": args.lambda_compress * terms.compress / tot,
            "share_expression": lx * terms.expression / tot,
            "share_plus": lam * terms.plus / tot,
            "share_minus": lam * terms.minus / tot,
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
        # compare against lambda_pm = 0 AT THE SAME lambda_x
        if (lx, 0.0) in M_by_lambda:
            sens = membership_sensitivity(M_by_lambda[(lx, 0.0)], res.M)
            rec["ari_vs_lambda0"] = sens["mean_ari"]
            rec["l1_vs_lambda0"] = sens["mean_l1_membership_change"]
        # and against the lambda_x = 1, lambda_pm = 0 reference, to see whether
        # rebalancing alone changed the states
        ref = (max(lambda_xs), 0.0)
        if ref in M_by_lambda and (lx, lam) != ref:
            rec["ari_vs_reference"] = membership_sensitivity(
                M_by_lambda[ref], res.M)["mean_ari"]
        rows.append(rec)
        print(f"[sweep] lx={lx:g} lpm={lam:g}: status={res.status} L={terms.total:.4f} "
              f"| shares comp {rec['share_compress']:.2f} x {rec['share_expression']:.2f} "
              f"pm {rec['share_plus'] + rec['share_minus']:.2f} "
              f"| Keff={rec['k_eff_mean']:.2f} min_g={rec['min_state_mass']:.1e} "
              f"ARI_vs_0={rec.get('ari_vs_lambda0', float('nan')):.3f} "
              f"({elapsed / 60:.1f} min)", flush=True)
        _flush(rows, sweep_dir, prov, args, lambdas, lambda_xs, chain)

    verdict = _verdict(rows, lambda_xs, lambdas, args.collapse_frac)
    _flush(rows, sweep_dir, prov, args, lambdas, lambda_xs, chain)
    jdump({"provenance": prov, "K": args.K, "epsilon": args.epsilon,
           "lambda_compress": args.lambda_compress, "lambda_xs": lambda_xs,
           "lambdas_pm": lambdas, "per_cell": rows, "verdict": verdict,
           "chain": chain.summary(), "complete": True},
          os.path.join(sweep_dir, "lambda_sweep.json"))

    print(f"\n[sweep] verdict: {json.dumps(verdict, indent=2, default=str)}")
    print(f"[sweep] wrote {sweep_dir}")


if __name__ == "__main__":
    main()
