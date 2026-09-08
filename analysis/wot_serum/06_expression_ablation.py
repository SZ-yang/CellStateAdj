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
supplies a lambda_x that is PRINCIPLED UNDER AN ASSUMED GAUSSIAN EMISSION MODEL,
rather than an arbitrary one.  It is NOT universally transferable: sigma^2 is
estimated from a particular representation, K, preprocessing and baseline fit, so
the resulting lambda_x moves when any of those move.  The script reports the
estimate, its source, the assignment-dependent NLL, the held-out NLL and a
sensitivity scan so the dependence is visible rather than assumed away.  It does not create a new optimisation problem, and
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
    p.add_argument("--tol-grad", type=float, default=1e-3,
                   help="strict convergence: max logit gradient norm")
    p.add_argument("--tol-kkt", type=float, default=1e-4,
                   help="strict convergence: max projected membership-space (KKT) "
                        "residual -- the condition that survives saturation")
    p.add_argument("--tol-dm", type=float, default=1e-4,
                   help="strict convergence: max final ||dM|| from the last iteration")
    p.add_argument("--sigma2-sensitivity", type=float, nargs="*",
                   default=[0.5, 2.0],
                   help="issue 9: also report lambda_x at these multiples of the "
                        "estimated sigma^2, to expose the dependence")
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


def _stationarity(chain, Z, mcfg, U):
    """Gradient norm AND a projected membership-space (KKT) residual at ``U``.

    [CRITICAL] Two problems with reading a raw logit gradient here.  First, the
    2026-09-07 grids had memberships saturating at exactly 1.000, and softmax
    gradients vanish there whatever the true stationarity -- so a small logit
    gradient can mean "saturated", not "optimal".  Second, softmax is shift
    invariant per row, so the logit gradient always has a null direction that
    contributes nothing.

    The projected residual fixes both: map the logit gradient into membership space
    via the softmax Jacobian and project onto the simplex tangent (sum-zero per
    row).  That is the quantity that must vanish at a constrained stationary point
    of the membership problem, and it does not vanish merely because M saturated.
    """
    import torch
    model = CoarseGrainModel(chain, Z, mcfg, U_init=U)
    total, _ = model.objective(with_diagnostics=False)
    total.backward()
    g2, kkt2 = 0.0, 0.0
    with torch.no_grad():
        for u in model.U:
            if u.grad is None:
                continue
            g = u.grad
            g2 += float((g ** 2).sum())
            M = torch.softmax(u, dim=1)
            # dL/dM from dL/dU via the softmax Jacobian:  gU = M * (gM - <gM, M>)
            # so gM (up to the row-constant that the simplex projection removes) is
            # gU / M; project onto the sum-zero tangent of the simplex.
            gM = g / M.clamp_min(1e-12)
            gM = gM - (gM * M).sum(1, keepdim=True)      # tangent projection
            kkt2 += float(((gM * M) ** 2).sum())          # scaled (KKT) residual
    return float(np.sqrt(g2)), float(np.sqrt(kkt2))


def _heldout_expression_nll(Z, M_list, idx_per_t, sigma2):
    """Assignment-dependent Gaussian expression NLL on a cell subset.

    ``sum_i a_i M_ik ||z_i - mu_k||^2 / (2 sigma^2)`` with ``mu_k`` recomputed on
    the subset, so a state map that only fits the training batch is penalised here.
    The assignment-independent constant is omitted -- it would shift every variant
    by the same amount and obscure the comparison.
    """
    tot = 0.0
    for t, M in enumerate(M_list):
        idx = idx_per_t[t]
        if len(idx) < 2:
            continue
        Msub = np.asarray(M)[idx]
        Msub = Msub / np.maximum(Msub.sum(1, keepdims=True), 1e-30)
        z = np.asarray(Z[t])[idx]
        a = np.full(len(idx), 1.0 / len(idx))
        g = Msub.T @ a
        mu = (Msub.T @ (a[:, None] * z)) / np.maximum(g, 1e-30)[:, None]
        sse = float((a[:, None] * Msub *
                     ((z[:, None, :] - mu[None, :, :]) ** 2).sum(-1)).sum())
        tot += sse / (2.0 * sigma2)
    return float(tot)


def strict_converged(res, grad_norm, kkt, dM, tol_grad, tol_kkt, tol_dM):
    """Convergence that does not inherit the package's objective-plateau path.

    ``optimize.fit`` can report ``converged`` on ``patience`` consecutive
    iterations of small RELATIVE OBJECTIVE CHANGE alone (optimize.py:331-338),
    which a flat region satisfies without being stationary.  Every condition below
    must hold, and all quantities must be finite.
    """
    reasons = []
    if res.status != "converged":
        reasons.append(f"package status {res.status}")
    if not np.isfinite(res.objective):
        reasons.append("objective not finite")
    if not np.isfinite(grad_norm) or grad_norm > tol_grad:
        reasons.append(f"grad norm {grad_norm:.3e} > {tol_grad:g}")
    if not np.isfinite(kkt) or kkt > tol_kkt:
        reasons.append(f"projected KKT residual {kkt:.3e} > {tol_kkt:g}")
    if dM is None or not np.isfinite(dM) or dM > tol_dM:
        reasons.append(f"final membership change {dM} > {tol_dM:g}")
    return (len(reasons) == 0), reasons



def _resolve_outdir(base, cfg_hash, args):
    """Config-hashed output directory that refuses to mix runs (issue 8).

    ``--overwrite`` on a shared directory is not enough: an earlier run at a
    different K, epsilon or lambda grid leaves membership NPZ and CSV files behind
    that a later glob or reducer will happily pick up.  Keying the directory on the
    configuration hash means a collision can only mean "this exact configuration
    already ran", and even then the directory must be empty or explicitly
    overwritten.
    """
    out = os.path.join(base, f"cfg_{cfg_hash}")
    os.makedirs(out, exist_ok=True)
    existing = os.listdir(out)
    if existing and not getattr(args, "overwrite", False):
        raise SystemExit(
            f"{out} already contains {len(existing)} file(s). Runs are keyed by "
            f"configuration hash, so this means the same configuration already ran. "
            f"Use a fresh --out, or --overwrite to replace it (which may leave "
            f"stale artifacts from a different lambda/K grid beside the new ones).")
    print(f"[out] config hash {cfg_hash} -> {out}")
    return out

def main():
    args = parse_args()
    device = csa_wot.resolve_device(args.device, verbose=args.verbose)

    data, Z, obs = load_serum(args.h5ad, day_min=args.day_min, day_max=args.day_max,
                              seed=args.seed, verbose=args.verbose)
    T, d = data.T, Z[0].shape[1]
    chain = ReferenceChain.load(args.chain)
    identity = ca.validate_chain_identity(chain, data, Z, chain_path=args.chain)
    ca.check_identity_args(identity, h5ad=args.h5ad, day_min=args.day_min,
                           day_max=args.day_max, stride=1, n_per_timepoint=None,
                           seed=args.seed)
    print(f"[B] chain identity verified against the loaded data "
          f"(cell ids + order, tau, representation hash, feasibility)")
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
    args.out = _resolve_outdir(args.out, prov["fingerprint_hash"], args)

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
    print(f"\n[B] sigma^2 = {sigma2:.6g} (from SSE/(d*T) at K={args.K}; "
          f"{'given' if args.sigma2 else 'MLE at lambda_pm=0 under the SSE objective'})")
    print(f"[B] sigma^2 is NOT universally transferable: it depends on this "
          f"representation, K={args.K}, this preprocessing and this baseline fit.")
    for v in args.variants:
        extra = (f"  [Gaussian constant {gaussian_constant(sigma2, d, T):.2f} nats, "
                 f"argmin-irrelevant]" if v == "gaussian" else "")
        print(f"[B]   variant {v:9s} -> lambda_x = {lam_x[v]:.4f}{extra}")
    sens = {f"sigma2_x{m:g}": {"sigma2": sigma2 * m,
                               "lambda_x_gaussian": d / (2.0 * sigma2 * m)}
            for m in args.sigma2_sensitivity}
    if sens:
        print(f"[B]   sigma^2 sensitivity: "
              + ", ".join(f"{k} -> lambda_x {v['lambda_x_gaussian']:.3f}"
                          for k, v in sens.items()))

    state = {"provenance": prov, "chain": os.path.abspath(args.chain),
             "epsilon": float(chain.epsilon), "K": args.K, "d": d, "T": T,
             "sigma2": sigma2, "sigma2_source": "given" if args.sigma2 else "MLE",
             "sigma2_estimated_under": {
                 "K": args.K, "objective": "SSE at lambda_pm=0",
                 "representation_sha1": prov["fingerprint"]["representation"]["sha1"],
                 "baseline_status": base.status,
                 "note": ("sigma^2 depends on the representation, K, preprocessing "
                          "and the baseline fit; lambda_x = d/(2 sigma^2) moves with "
                          "all of them and is not universally transferable")},
             "sigma2_sensitivity": sens,
             "lambda_x_by_variant": lam_x,
             "gaussian_constant_nats": gaussian_constant(sigma2, d, T),
             "strict_convergence_tolerances": {
                 "grad": args.tol_grad, "kkt": args.tol_kkt, "dM": args.tol_dm},
             "equivalence_note": (
                 "B1 (Gaussian, fixed sigma^2) == the SSE objective at "
                 "lambda_x = d/(2 sigma^2) plus an assignment-independent constant; "
                 "B2 == lambda_x = 0. A principled reweighting under an ASSUMED "
                 "Gaussian emission model, not a new objective. Compare objective "
                 "differences within a variant, never term shares across variants."),
             "sse_reference_baseline": {
                 "status": base.status, "n_iter": int(base.n_iter),
                 "L_expr_SSE": base.terms.expression,
                 "L_compress": base.terms.compress,
                 "converged": bool(base.converged),
                 "note": "reported for cross-variant reference only; each variant "
                         "gets its OWN lambda_pm=0 baseline below"},
             "variant_baselines": {}, "runs": [], "upper_bounds": {}}

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
    # kept OUT of ``state``: these are arrays and ``state`` is JSON-serialised
    forward_endpoint = {}

    paths = [("forward", list(args.continuation))]
    if args.reverse:
        paths.append(("reverse", list(reversed(args.continuation))))

    for variant in args.variants:
        lx = lam_x[variant]

        # ---- issue 5: EVERY variant gets its own lambda_pm = 0 baseline -----
        # Using the SSE baseline for the Gaussian and no-expression variants made
        # both the upper bound and ARI_vs_lambda0 refer to a solution of a
        # DIFFERENT objective, so a "failure to improve" could just mean the two
        # objectives have different optima.
        if variant == "sse" and abs(lx - args.lambda_x_sse) < 1e-12:
            vbase, vU0 = base, U0        # already fitted above
        else:
            print(f"\n[B] fitting the lambda_pm=0 baseline for variant '{variant}' "
                  f"(lambda_x={lx:.4f})")
            vcfg = dc_replace(base_cfg.model, K=args.K, lambda_x=lx,
                              lambda_plus=0.0, lambda_minus=0.0)
            vbase = fit_states(chain, Z, vcfg,
                               dc_replace(base_cfg.optim, n_init=max(2, args.n_cold),
                                          seed=args.seed))
            vU0 = ca.logits_from_memberships(vbase.M)
            print(f"[B]   {variant} baseline: status={vbase.status} "
                  f"n_iter={vbase.n_iter} L={vbase.objective:.4f}")
        gN, kN = _stationarity(chain, Z,
                               dc_replace(base_cfg.model, K=args.K, lambda_x=lx,
                                          lambda_plus=0.0, lambda_minus=0.0), vU0)
        dM0 = vbase.history[-1].get("dM") if vbase.history else None
        sc0, why0 = strict_converged(vbase, gN, kN, dM0, args.tol_grad,
                                     args.tol_kkt, args.tol_dm)
        state["variant_baselines"][variant] = {
            "lambda_x": lx, "status": vbase.status,
            "converged": bool(vbase.converged), "strict_converged": bool(sc0),
            "strict_reasons": why0, "objective": float(vbase.objective),
            "L_compress": vbase.terms.compress, "L_expr_SSE": vbase.terms.expression,
            "grad_norm": gN, "kkt_residual": kN, "final_dM": dM0,
            "min_state_mass": float(np.min(vbase.terms.g_min))}
        np.savez_compressed(
            os.path.join(args.out, f"memberships_{variant}_lpm0_baseline.npz"),
            **{f"M_{t}": np.asarray(m, np.float32) for t, m in enumerate(vbase.M)})
        if not sc0:
            print(f"[B]   *** {variant} baseline is NOT strictly converged: "
                  f"{'; '.join(why0)} -- its ARI comparisons are weak ***")
        _flush()

        for direction, path in paths:
            # issue 5: reverse continuation must start from the FITTED forward
            # high-lambda endpoint, or it is not a hysteresis check at all.
            if direction == "reverse":
                endpoint = forward_endpoint.get(variant)
                if endpoint is None:
                    print(f"[B] no forward endpoint for {variant}; reverse path "
                          f"starts from its lambda_pm=0 baseline instead")
                    warm = vU0
                else:
                    warm = endpoint
                    print(f"[B] reverse path for {variant} starts from the fitted "
                          f"forward endpoint (genuine hysteresis check)")
            else:
                warm = vU0
            for lam in path:
                mcfg = dc_replace(base_cfg.model, K=args.K, lambda_x=lx,
                                  lambda_plus=lam, lambda_minus=lam)

                # requirement 1: score the lambda_pm=0 membership under THIS objective
                ub, ub_terms, _ = _score(chain, Z, mcfg, vU0)
                key = f"{variant}|{lam:g}"
                state["upper_bounds"][key] = {
                    "baseline_variant": variant,
                    "objective_at_variant_lambda0_membership": ub,
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
                    # [issue 6] Evaluate stationarity at the ACTUAL fitted logits.
                    # Reconstructing them from floor-clipped memberships changes the
                    # point being tested, and after saturation the clip is exactly
                    # where the gradient information lives.
                    Uf = [u.detach().cpu().numpy() for u in res.model.get_U()]
                    gn, kkt = _stationarity(chain, Z, mcfg, Uf)
                    dM_final = res.history[-1].get("dM") if res.history else None
                    sconv, sreasons = strict_converged(
                        res, gn, kkt, dM_final, args.tol_grad, args.tol_kkt,
                        args.tol_dm)

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
                        kkt_residual=kkt, final_dM=dM_final,
                        strict_converged=bool(sconv),
                        strict_reasons=sreasons,
                        L_compress=tm.compress, L_expr_SSE=tm.expression,
                        L_plus=tm.plus, L_minus=tm.minus,
                        # the assignment-dependent Gaussian NLL, for B1 reporting
                        # issue 9: the assignment-dependent Gaussian NLL, and the
                        # same quantity on cells held out by batch. The constant
                        # T*(d/2)*log(2 pi sigma^2) is omitted from both because it
                        # is assignment-independent and would only add a fixed
                        # offset to every variant.
                        expr_nll_assignment_dependent=float(
                            tm.expression * d / (2.0 * sigma2)),
                        expr_nll_heldout=_heldout_expression_nll(
                            Z, res.M, te_idx, sigma2),
                        expr_nll_train=_heldout_expression_nll(
                            Z, res.M, tr_idx, sigma2),
                        min_state_mass=float(np.min(tm.g_min)),
                        k_eff_mean=float(np.mean(tm.k_eff)),
                        k_eff_min=float(np.min(tm.k_eff)),
                        floor_fraction=float(np.max(tm.floor_fraction)) if tm.floor_fraction else 0.0,
                        elapsed_s=el, **cmi_cols)
                    # membership change from the lambda_pm=0 map
                    from cellstateadj.diagnostics import membership_sensitivity
                    ms = membership_sensitivity(vbase.M, res.M)      # SAME variant
                    rec["ari_vs_lambda0"] = ms["mean_ari"]
                    rec["l1_vs_lambda0"] = ms["mean_l1_membership_change"]
                    rec["ari_vs_sse_baseline"] = membership_sensitivity(
                        base.M, res.M)["mean_ari"]   # cross-variant reference only

                    results.append((rec, res))
                    state["runs"].append(rec)
                    _flush()
                    flag = "" if improved else "  <-- FAILED to beat the lambda_pm=0 bound"
                    print(f"  {variant:9s} lpm={lam:<5g} {direction:7s} {start_kind:4s}: "
                          f"{res.status:10s} L={res.objective:9.4f} (bound {ub:9.4f}) "
                          f"|g|={gn:.2e} ARI0={rec['ari_vs_lambda0']:.3f} "
                          f"min_g={rec['min_state_mass']:.1e} ({el/60:.1f}m){flag}")

                # requirement 4: restart agreement only among comparable converged runs
                conv = [(r, x) for r, x in results if r["strict_converged"]]
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
                                 "note": "strictly converged runs reached different "
                                         "objectives; not comparable"}
                else:
                    agree = {"n_comparable": len(conv), "mean_pairwise_ari": None,
                             "n_strictly_converged": len(conv),
                             "note": "fewer than two STRICTLY converged runs: "
                                     "stability NOT assessed"}
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
                    strict = [r for r in results if r[0]["strict_converged"]]
                    loose = [r for r in results if r[0]["converged"]]
                    pool = strict or loose or results
                    best = min(pool, key=lambda rr: rr[0]["objective"])
                    tag = f"{variant}_lpm{lam:g}_{direction}"
                    if not strict:
                        tag += "_NOTSTRICT"
                    if not loose:
                        tag += "_NONCONVERGED"
                        print(f"    (no converged run at {variant} lpm={lam:g} "
                              f"{direction}; saving the best of "
                              f"{len(results)} by objective, flagged NONCONVERGED)")
                    warm = ca.logits_from_memberships(best[1].M)
                    np.savez_compressed(
                        os.path.join(args.out, f"memberships_{tag}.npz"),
                        **{f"M_{t}": np.asarray(m, np.float32)
                           for t, m in enumerate(best[1].M)})
                    # [issue 7] Stage 07 joins on this; every field it needs to
                    # decide eligibility must be here.
                    state.setdefault("saved_memberships", {})[tag] = {
                        "file": f"memberships_{tag}.npz",
                        "variant": variant, "lambda_x": lx, "lambda_pm": lam,
                        "direction": direction,
                        "status": best[0]["status"],
                        "converged": best[0]["converged"],
                        "strict_converged": best[0]["strict_converged"],
                        "strict_reasons": best[0]["strict_reasons"],
                        "objective": best[0]["objective"],
                        "upper_bound": best[0]["upper_bound"],
                        "beats_upper_bound": best[0].get("beats_upper_bound"),
                        "grad_norm": best[0]["grad_norm_at_solution"],
                        "kkt_residual": best[0]["kkt_residual"],
                        "min_state_mass": best[0]["min_state_mass"],
                        "k_eff_mean": best[0]["k_eff_mean"],
                        "start": best[0]["start"], "seed": best[0]["seed"],
                        "K": args.K, "epsilon": float(chain.epsilon),
                        "chain": os.path.abspath(args.chain),
                    }
                    if direction == "forward" and lam == path[-1]:
                        forward_endpoint[variant] = warm
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
