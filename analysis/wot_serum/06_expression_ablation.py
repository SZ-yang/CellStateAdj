#!/usr/bin/env python
"""Workstream B -- expression-term ablations with lambda_pm continuation.

Three expression variants, one shared Stage-0 chain, identical K / starts /
continuation path / evaluation:

  B0  current SSE control  L_expr = sum a M ||z-mu||^2 / d,  lambda_x as given
  B1  Gaussian likelihood  z|Z=k ~ N(mu_k, sigma^2 I), sigma^2 FROZEN
  B2  no expression term   lambda_x = 0

[KEY STRUCTURAL FACT] B1 needs no change to the model.  As implemented,
``L_expr = SSE/d``, and the Gaussian negative log-likelihood with FIXED sigma^2 is

    L_gauss = SSE/(2 sigma^2) + T*(d/2)*log(2 pi sigma^2)
            = (d / (2 sigma^2)) * L_expr  +  const(M)

The constant does not depend on the memberships, so **B1 is exactly the SSE
objective at lambda_x = d/(2 sigma^2)**, and B2 is exactly lambda_x = 0.  All three
variants are therefore points on the lambda_x axis; the value of B1 is that it
supplies a PRINCIPLED, transferable lambda_x (fixing the unit-conversion defect M1)
rather than an arbitrary one.  It does not create a new optimisation problem, and
it will not by itself escape Degeneracy 3.

Measured on the 2026-09-07 lambda_pm=0 fits, sigma^2 = SSE/(d*T) gives
lambda_x ~ 3.2 at K=8 and ~4.8 at K=20 -- i.e. a properly normalised Gaussian term
asks for MORE expression weight than lambda_x=1, not less.  That is worth knowing
before concluding that the 90% expression share was itself the error.

Because the Gaussian constant shifts the total but not the argmin, term-share
plots across variants are misleading; this script reports objective DIFFERENCES,
the assignment-dependent NLL, gradient norms and held-out quantities instead.

Optimisation requirements implemented here (plan, "Optimisation requirements"):

  1. the lambda_pm=0 membership is scored under every target objective FIRST -- it
     is a known feasible upper bound for a minimisation
  2. a run that fails to beat that bound is flagged as an optimisation failure
  3. every restart's seed, init, status, objective, components, gradient norm,
     membership change, iterations and state masses are saved
  4. restart agreement is computed only among converged runs with comparable
     objectives; fewer than two -> "not assessed"
  5. a plateau is not called convergence without membership-change/gradient checks
  6. warm-start and cold-start results are kept separate
  7. non-finite gradients and mass collapse are recorded, never averaged away

    python 06_expression_ablation.py --chain <stage0>/chain_eps0.2.npz --K 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace as dc_replace

import numpy as np

import csa_anchors as ca
import csa_wot
from csa_wot import (DEFAULT_H5AD, RESULTS_ROOT, jdump, load_serum, make_cfg,
                     provenance_block, run_fingerprint)

from cellstateadj.model import CoarseGrainModel
from cellstateadj.optimize import fit as fit_states, initialize_logits
from cellstateadj.reference import ReferenceChain, build_reference_chain

VARIANTS = ("sse", "gaussian", "none")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", default=DEFAULT_H5AD)
    p.add_argument("--chain", required=True, help="Stage-0 chain .npz -- required")
    p.add_argument("--day-min", type=float, default=8.25)
    p.add_argument("--day-max", type=float, default=18.0)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--lambda-x-sse", type=float, default=1.0,
                   help="B0 control weight. Not a transferable scale (M1)")
    p.add_argument("--sigma2", type=float, default=None,
                   help="B1: freeze this sigma^2. Default: MLE from the lambda_pm=0 "
                        "SSE baseline at this K, i.e. SSE/(d*T)")
    p.add_argument("--lambda-compress", type=float, default=1.0)
    p.add_argument("--continuation", type=float, nargs="+",
                   default=[0.0, 0.25, 1.0, 2.0, 5.0],
                   help="lambda_pm path, warm-started along its order")
    p.add_argument("--reverse", action="store_true",
                   help="also walk the path backwards, to expose hysteresis")
    p.add_argument("--n-cold", type=int, default=3,
                   help="independent cold starts per (variant, lambda_pm)")
    p.add_argument("--max-iter", type=int, default=1500)
    p.add_argument("--objective-tol", type=float, default=1e-6,
                   help="relative slack allowed when checking the upper bound")
    p.add_argument("--anchors", type=int, default=40,
                   help="fixed-anchor resolution for the CMI columns")
    p.add_argument("--train-batch", default="1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(RESULTS_ROOT, "expr_ablation"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def lambda_x_for(variant: str, sigma2: float, d: int, lambda_x_sse: float) -> float:
    """The single lambda_x that realises each variant.  See the module docstring."""
    if variant == "sse":
        return float(lambda_x_sse)
    if variant == "gaussian":
        return float(d / (2.0 * sigma2))
    if variant == "none":
        return 0.0
    raise ValueError(variant)


def gaussian_constant(sigma2: float, d: int, T: int) -> float:
    """``T*(d/2)*log(2 pi sigma^2)`` -- added for reporting only; argmin-irrelevant."""
    return float(T * (d / 2.0) * np.log(2.0 * np.pi * sigma2))


def _score(chain, Z, mcfg, U):
    """Objective and components of a GIVEN membership under a GIVEN objective."""
    import torch
    model = CoarseGrainModel(chain, Z, mcfg, U_init=U)
    with torch.no_grad():
        total, terms = model.objective()
    return float(total), terms, model


def _grad_norm(chain, Z, mcfg, U):
    """Gradient norm at a point -- a plateau is not convergence without this."""
    import torch
    model = CoarseGrainModel(chain, Z, mcfg, U_init=U)
    total, _ = model.objective(with_diagnostics=False)
    total.backward()
    g2 = sum(float((u.grad ** 2).sum()) for u in model.U if u.grad is not None)
    return float(np.sqrt(g2))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if os.listdir(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} is not empty; pass --overwrite")
    device = csa_wot.resolve_device(args.device, verbose=args.verbose)

    data, Z, obs = load_serum(args.h5ad, day_min=args.day_min, day_max=args.day_max,
                              seed=args.seed, verbose=args.verbose)
    T, d = data.T, Z[0].shape[1]
    chain = ReferenceChain.load(args.chain)
    if chain.n_cells != data.n_cells:
        raise SystemExit(
            f"chain {args.chain} does not match the loaded cells "
            f"({chain.n_cells[:4]}... vs {data.n_cells[:4]}...)")
    print(f"[B] chain {os.path.basename(args.chain)} eps={chain.epsilon:g} "
          f"feasible={chain.feasible}  K={args.K}  d={d}  T={T}")

    base_cfg = make_cfg(epsilon=chain.epsilon, K=args.K,
                        lambda_compress=args.lambda_compress,
                        lambda_x=args.lambda_x_sse, lambda_plus=0.0, lambda_minus=0.0,
                        max_iter=args.max_iter, n_init=1, seed=args.seed,
                        device=device, verbose=max(0, args.verbose - 1))
    prov = provenance_block(run_fingerprint(
        Z, base_cfg, h5ad=args.h5ad, day_min=args.day_min, day_max=args.day_max,
        stride=1, n_per_timepoint=None, sampling_seed=args.seed))

    # ---- fixed anchors for the CMI columns (frozen on the training batch) ----
    tr_idx, te_idx = ca.batch_split(data, args.train_batch)
    centroids = ca.make_anchors([Z[t][tr_idx[t]] for t in range(T)],
                                args.anchors, seed=args.seed)
    A = ca.assign_anchors(Z, centroids)
    Fp_fix, Fm_fix = ca.fixed_fingerprints(chain, A)

    # ---- the lambda_pm = 0 SSE baseline: sigma^2 AND the upper bound ---------
    print(f"\n[B] fitting the lambda_pm=0 SSE baseline (defines sigma^2 and the "
          f"feasible upper bound for every variant)")
    t0 = time.time()
    base = fit_states(chain, Z, base_cfg.model,
                      dc_replace(base_cfg.optim, n_init=max(2, args.n_cold), seed=args.seed))
    print(f"[B] baseline: status={base.status} n_iter={base.n_iter} "
          f"L_expr_SSE={base.terms.expression:.4f} ({time.time()-t0:.0f}s)")
    if base.status != "converged":
        print(f"[B] *** baseline did NOT converge ({base.status}); it is still a valid "
              f"feasible upper bound, but ARI comparisons against it are weak ***")

    SSE = base.terms.expression * d           # L_expr as implemented is SSE/d
    sigma2 = float(args.sigma2) if args.sigma2 else float(SSE / (d * T))
    U0 = ca.logits_from_memberships(base.M)
    np.savez_compressed(os.path.join(args.out, "memberships_baseline_lam0_sse.npz"),
                        **{f"M_{t}": np.asarray(m, np.float32) for t, m in enumerate(base.M)})

    lam_x = {v: lambda_x_for(v, sigma2, d, args.lambda_x_sse) for v in args.variants}
    print(f"\n[B] sigma^2 = {sigma2:.6g} (from SSE/(d*T); "
          f"{'given' if args.sigma2 else 'MLE at lambda_pm=0'})")
    for v in args.variants:
        extra = (f"  [= SSE objective at lambda_x={lam_x[v]:.4f}; Gaussian constant "
                 f"{gaussian_constant(sigma2, d, T):.2f} nats, argmin-irrelevant]"
                 if v == "gaussian" else "")
        print(f"[B]   variant {v:9s} -> lambda_x = {lam_x[v]:.4f}{extra}")

    state = {"provenance": prov, "chain": os.path.abspath(args.chain),
             "epsilon": float(chain.epsilon), "K": args.K, "d": d, "T": T,
             "sigma2": sigma2, "sigma2_source": "given" if args.sigma2 else "MLE",
             "lambda_x_by_variant": lam_x,
             "gaussian_constant_nats": gaussian_constant(sigma2, d, T),
             "equivalence_note": (
                 "B1 (Gaussian, fixed sigma^2) == the SSE objective at "
                 "lambda_x = d/(2 sigma^2); B2 == lambda_x = 0. The variants are "
                 "points on one axis, so compare objective DIFFERENCES within a "
                 "variant, never term shares across variants."),
             "baseline": {"status": base.status, "n_iter": int(base.n_iter),
                          "L_expr_SSE": base.terms.expression,
                          "L_compress": base.terms.compress,
                          "converged": bool(base.converged)},
             "runs": [], "upper_bounds": {}}

    def _flush():
        jdump(state, os.path.join(args.out, "summary.json"))
        try:
            import pandas as pd
            flat = [{k: v for k, v in r.items() if not isinstance(v, (list, dict))}
                    for r in state["runs"]]
            if flat:
                pd.DataFrame(flat).to_csv(os.path.join(args.out, "summary.csv"),
                                          index=False)
        except Exception:
            pass

    # ---- the run matrix -----------------------------------------------------
    paths = [("forward", list(args.continuation))]
    if args.reverse:
        paths.append(("reverse", list(reversed(args.continuation))))

    for variant in args.variants:
        lx = lam_x[variant]
        for direction, path in paths:
            warm = U0                     # continuation always starts from lambda_pm=0
            for lam in path:
                mcfg = dc_replace(base_cfg.model, K=args.K, lambda_x=lx,
                                  lambda_plus=lam, lambda_minus=lam)

                # requirement 1: score the lambda_pm=0 membership under THIS objective
                ub, ub_terms, _ = _score(chain, Z, mcfg, U0)
                key = f"{variant}|{lam:g}"
                state["upper_bounds"][key] = {
                    "objective_at_lambda0_membership": ub,
                    "compress": ub_terms.compress, "expression": ub_terms.expression,
                    "plus": ub_terms.plus, "minus": ub_terms.minus}

                starts = [("warm", warm, args.seed)] + [
                    ("cold", None, args.seed + 100 * (r + 1)) for r in range(args.n_cold)]
                results = []
                for start_kind, U_init, seed in starts:
                    ocfg = dc_replace(base_cfg.optim, n_init=1, seed=seed,
                                      max_iter=args.max_iter,
                                      verbose=max(0, args.verbose - 1))
                    if U_init is None:
                        U_init = initialize_logits(Z, args.K, method="kmeans",
                                                   scale=base_cfg.optim.init_logit_scale,
                                                   seed=seed)
                    t0 = time.time()
                    rec = {"variant": variant, "lambda_x": lx, "lambda_pm": lam,
                           "direction": direction, "start": start_kind, "seed": seed,
                           "upper_bound": ub}
                    try:
                        res = fit_states(chain, Z, mcfg, ocfg, U_init=U_init)
                    except (FloatingPointError, RuntimeError, ValueError) as exc:
                        # requirement 7
                        rec.update(status="error", error=f"{type(exc).__name__}: {exc}",
                                   converged=False, elapsed_s=time.time() - t0)
                        state["runs"].append(rec); _flush()
                        print(f"  {variant:9s} lpm={lam:<5g} {direction:7s} "
                              f"{start_kind:4s}: ERROR {type(exc).__name__}")
                        continue
                    el = time.time() - t0
                    tm = res.terms
                    Uf = ca.logits_from_memberships(res.M)
                    gn = _grad_norm(chain, Z, mcfg, Uf)

                    # requirement 2: must beat the known feasible bound
                    improved = res.objective <= ub * (1 + args.objective_tol) if ub > 0 \
                        else res.objective <= ub + abs(ub) * args.objective_tol
                    # fixed-anchor CMI, train and held out by batch
                    cmi_cols = {}
                    for lab, F in (("plus", Fp_fix), ("minus", Fm_fix)):
                        for split, idx in (("train", tr_idx), ("heldout", te_idx)):
                            vals, rets = [], []
                            for t in range(T):
                                if F[t] is None:
                                    continue
                                w = np.full(len(idx[t]), 1.0 / max(len(idx[t]), 1))
                                c = ca.state_cmi(np.asarray(res.M[t])[idx[t]],
                                                 F[t][idx[t]], w)
                                r_, _why = ca.retained_information(
                                    c["cmi"], c["i_cell_anchor"])
                                vals.append(c["cmi"]); rets.append(r_)
                            cmi_cols[f"cmi_{lab}_{split}"] = float(np.mean(vals))
                            cmi_cols[f"retained_{lab}_{split}"] = float(np.nanmean(rets))

                    rec.update(
                        status=res.status, converged=bool(res.converged),
                        monotone=bool(res.monotone), n_iter=int(res.n_iter),
                        objective=float(res.objective),
                        beats_upper_bound=bool(improved),
                        objective_minus_upper_bound=float(res.objective - ub),
                        grad_norm_reported=float(res.grad_norm),
                        grad_norm_at_solution=gn,
                        L_compress=tm.compress, L_expr_SSE=tm.expression,
                        L_plus=tm.plus, L_minus=tm.minus,
                        # the assignment-dependent Gaussian NLL, for B1 reporting
                        expr_nll_assignment_dependent=float(
                            tm.expression * d / (2.0 * sigma2)),
                        min_state_mass=float(np.min(tm.g_min)),
                        k_eff_mean=float(np.mean(tm.k_eff)),
                        k_eff_min=float(np.min(tm.k_eff)),
                        floor_fraction=float(np.max(tm.floor_fraction)) if tm.floor_fraction else 0.0,
                        elapsed_s=el, **cmi_cols)
                    # membership change from the lambda_pm=0 map
                    from cellstateadj.diagnostics import membership_sensitivity
                    ms = membership_sensitivity(base.M, res.M)
                    rec["ari_vs_lambda0"] = ms["mean_ari"]
                    rec["l1_vs_lambda0"] = ms["mean_l1_membership_change"]

                    results.append((rec, res))
                    state["runs"].append(rec)
                    _flush()
                    flag = "" if improved else "  <-- FAILED to beat the lambda_pm=0 bound"
                    print(f"  {variant:9s} lpm={lam:<5g} {direction:7s} {start_kind:4s}: "
                          f"{res.status:10s} L={res.objective:9.4f} (bound {ub:9.4f}) "
                          f"|g|={gn:.2e} ARI0={rec['ari_vs_lambda0']:.3f} "
                          f"min_g={rec['min_state_mass']:.1e} ({el/60:.1f}m){flag}")

                # requirement 4: restart agreement only among comparable converged runs
                conv = [(r, x) for r, x in results if r["converged"]]
                if len(conv) >= 2:
                    objs = np.array([r["objective"] for r, _ in conv])
                    close = [c for c, o in zip(conv, objs)
                             if o <= objs.min() * (1 + 1e-3) or o - objs.min() < 1e-6]
                    if len(close) >= 2:
                        from cellstateadj.diagnostics import membership_sensitivity
                        aris = [membership_sensitivity(close[i][1].M, close[j][1].M)["mean_ari"]
                                for i in range(len(close)) for j in range(i + 1, len(close))]
                        agree = {"n_comparable": len(close),
                                 "mean_pairwise_ari": float(np.mean(aris)),
                                 "objective_spread": float(objs.max() - objs.min())}
                    else:
                        agree = {"n_comparable": len(conv), "mean_pairwise_ari": None,
                                 "note": "converged runs reached different objectives; "
                                         "not comparable"}
                else:
                    agree = {"n_comparable": len(conv), "mean_pairwise_ari": None,
                             "note": "fewer than two converged runs: stability NOT assessed"}
                state.setdefault("restart_agreement", {})[f"{variant}|{lam:g}|{direction}"] = agree
                _flush()

                # Save and warm-start from the best run by OBJECTIVE, converged or
                # not.  [CRITICAL] An earlier version only saved converged runs, so
                # when nothing converged -- which the 2026-09-07 grids showed does
                # happen at low lambda_x -- no memberships were written at all and
                # 07_cmi_compare.py had nothing to score.  Non-convergence is a
                # result to record, not a reason to discard hours of fitting; the
                # status travels with the file and with every row in summary.csv.
                if results:
                    conv = [r for r in results if r[0]["converged"]]
                    pool = conv if conv else results
                    best = min(pool, key=lambda rr: rr[0]["objective"])
                    tag = f"{variant}_lpm{lam:g}_{direction}"
                    if not conv:
                        tag += "_NONCONVERGED"
                        print(f"    (no converged run at {variant} lpm={lam:g} "
                              f"{direction}; saving the best of "
                              f"{len(results)} by objective, flagged NONCONVERGED)")
                    warm = ca.logits_from_memberships(best[1].M)
                    np.savez_compressed(
                        os.path.join(args.out, f"memberships_{tag}.npz"),
                        **{f"M_{t}": np.asarray(m, np.float32)
                           for t, m in enumerate(best[1].M)})
                    state.setdefault("saved_memberships", {})[tag] = {
                        "status": best[0]["status"],
                        "converged": best[0]["converged"],
                        "objective": best[0]["objective"],
                        "start": best[0]["start"], "seed": best[0]["seed"],
                        "beats_upper_bound": best[0].get("beats_upper_bound"),
                    }
                    try:
                        import pandas as pd
                        if best[1].history:
                            pd.DataFrame(best[1].history).to_csv(
                                os.path.join(args.out, f"history_{tag}.csv"),
                                index=False)
                    except Exception:
                        pass

    jdump({"config": vars(args)}, os.path.join(args.out, "config.json"))
    jdump(prov, os.path.join(args.out, "provenance.json"))
    jdump(state.get("restart_agreement", {}),
          os.path.join(args.out, "restart_summaries.json"))
    _flush()

    n_fail = sum(1 for r in state["runs"] if not r.get("beats_upper_bound", True))
    n_err = sum(1 for r in state["runs"] if r.get("status") == "error")
    print(f"\n[B] wrote {args.out}")
    print(f"[B] {len(state['runs'])} runs; {n_err} errored; {n_fail} failed to beat "
          f"the lambda_pm=0 upper bound (optimisation failures, not model results)")
    print("[B] Reminder: the three variants are one lambda_x axis, so do NOT compare "
          "term shares across them -- compare objective differences within a variant "
          "and the held-out CMI columns across them.")


if __name__ == "__main__":
    main()
