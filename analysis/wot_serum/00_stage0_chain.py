#!/usr/bin/env python
"""Stage 0 -- build ONE canonical frozen reference chain, with full provenance.

Every experiment in ADDITIONAL_EXPERIMENTS_2026-09-08 must use the same chain.
The earlier runs did not: the Stage-A epsilon scan used a DENSE support while the
serum fits used kNN, and the cost scale was computed on the whole 0-18 series
rather than on the serum range, so the epsilon values were not equivalent across
stages.  This script fixes all of that in one place and refuses to let a
downstream job silently rebuild it.

Stage-0 requirements, and where each is met:

1. restrict to the serum arm BEFORE fitting the representation
   -> ``--h5ad`` should point at the output of ``00_serum_only_pca.py``.  This
      script checks ``uns['representation_provenance']`` and warns loudly if the
      file is the cross-arm one, because that is a transductive representation.
2. freeze one shared representation, record its config and seed
   -> recorded via ``csa_wot.representation_fingerprint`` (content hash of X_model)
3. one documented global cost scale across the serum intervals
   -> computed on the serum range only and written to ``chain_provenance.json``
4. the SAME support construction in the scan and all later fits
   -> ``--support`` (default knn) is recorded; ``01_eps_scan.py`` must be run with
      the same value.  See the note on the dense/kNN trade-off below.
5. repeat the serum-only epsilon scan under that exact scale and support
   -> ``--print-scan-command`` emits the exact command line
6. keep an epsilon WINDOW, not one value
   -> ``--epsilons`` builds one chain per epsilon so sensitivity is cheap later
7. save once and reuse
   -> ``chain_eps<e>.npz`` + a manifest; downstream scripts load by path

Note on requirement 4: a dense support can never report infeasible, which makes it
the better instrument for measuring feasibility in isolation; a kNN support is what
the fits actually use.  The plan chooses comparability over that isolation, which
is the right call here -- an epsilon selected under a different support is not the
epsilon the fit sees.  The consequence is that the scan's ``feasible`` column now
mixes support adequacy with solver conditioning, so read it together with
``kappas`` (growth means the support, not epsilon, was binding).

    python 00_stage0_chain.py --epsilons 0.1 0.2 0.5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time

import numpy as np

import csa_anchors as ca
import csa_wot
from csa_wot import (DEFAULT_H5AD, RESULTS_ROOT, jdump, load_serum, make_cfg,
                     provenance_block, reserve_destinations, run_fingerprint)

from cellstateadj.cost import resolve_cost_scales
from cellstateadj.reference import build_reference_chain

STAGE0_DIR = os.path.join(RESULTS_ROOT, "stage0_chain")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", default=DEFAULT_H5AD,
                   help="prefer the serum-only PCA file from 00_serum_only_pca.py")
    p.add_argument("--day-min", type=float, default=8.25)
    p.add_argument("--day-max", type=float, default=18.0)
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.1, 0.2, 0.5],
                   help="one chain per value -- keep the selected epsilon AND its "
                        "neighbours so later sensitivity analysis is free")
    p.add_argument("--support", default="knn", choices=["knn", "dense"],
                   help="MUST match what 01_eps_scan.py is run with (requirement 4)")
    p.add_argument("--kappa", type=int, default=400)
    p.add_argument("--cost-scale-mode", default="global",
                   choices=["global", "per_interval", "none"])
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--n-per-timepoint", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=STAGE0_DIR)
    p.add_argument("--allow-cross-arm-representation", action="store_true",
                   help="proceed even though the representation was fitted across "
                        "arms (Stage-0 requirement 1 asks for the arm restriction to "
                        "precede the representation fit). Every downstream result "
                        "must then be labelled as using a cross-arm basis.")
    p.add_argument("--overwrite", action="store_true",
                   help="NOT recommended: prefer a new --out. Output directories are "
                        "config-hashed, so a collision means the same configuration "
                        "was already built; overwriting can leave stale chains from a "
                        "different epsilon set beside the new ones.")
    p.add_argument("--print-scan-command", action="store_true")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", csa_wot.HERE, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unavailable"


def _representation_warning(h5ad):
    """Warn if the representation was fitted across arms (Stage-0 requirement 1)."""
    try:
        import anndata as ad
        a = ad.read_h5ad(h5ad, backed="r")
        prov = a.uns.get("representation_provenance")
        if a.isbacked:
            a.file.close()
        if prov and "SERUM-ONLY" in str(prov):
            return None
        return (f"{os.path.basename(h5ad)} has no SERUM-ONLY representation marker, "
                f"so its PCA was almost certainly fitted on the balanced union of "
                f"shared + serum + 2i cells (see wot_data_check.ipynb) and is "
                f"TRANSDUCTIVE ACROSS ARMS. Stage-0 requirement 1 asks for the arm "
                f"restriction to precede the representation fit. Run "
                f"00_serum_only_pca.py and point --h5ad at its output, or label every "
                f"downstream result as using a cross-arm basis.")
    except Exception as exc:
        return f"could not check representation provenance: {type(exc).__name__}: {exc}"


def main():
    args = parse_args()
    epsilons = sorted(set(float(e) for e in args.epsilons))

    warn = _representation_warning(args.h5ad)
    if warn and not args.allow_cross_arm_representation:
        raise SystemExit(
            f"\nStage-0 requirement 1 NOT met:\n  {warn}\n\n"
            f"Run 00_serum_only_pca.py and point --h5ad at its output, or pass "
            f"--allow-cross-arm-representation to proceed knowingly (every "
            f"downstream result must then be labelled as cross-arm).")
    if warn:
        print(f"\n[stage0] *** PROCEEDING WITH A CROSS-ARM REPRESENTATION "
              f"(--allow-cross-arm-representation) ***\n[stage0] {warn}\n")

    data, Z, obs = load_serum(args.h5ad, day_min=args.day_min, day_max=args.day_max,
                              stride=args.stride, n_per_timepoint=args.n_per_timepoint,
                              seed=args.seed, verbose=args.verbose)

    # one cost scale, computed on THIS range only, reused by every epsilon
    cfg_probe = make_cfg(support=args.support, kappa=args.kappa,
                         cost_scale_mode=args.cost_scale_mode, seed=args.seed,
                         verbose=0)
    scales = resolve_cost_scales(Z, data.tau, cfg_probe.coupling.cost_scale_mode)
    scale_note = (f"per_interval, {len(scales)} scales, range "
                  f"{min(scales):.4g}-{max(scales):.4g}"
                  if args.cost_scale_mode == "per_interval"
                  else f"{args.cost_scale_mode}, single scale {scales[0]:.6g}")
    print(f"[stage0] cost scale: {scale_note}")

    # identity: what every downstream job must be able to verify
    identity = ca.identity_payload(
        data, Z, h5ad=args.h5ad, day_min=args.day_min, day_max=args.day_max,
        stride=args.stride, n_per_timepoint=args.n_per_timepoint, seed=args.seed,
        arm_policy=csa_wot.CONDITIONING_STATEMENT)

    # config-hashed directory so chains from different configurations cannot mix
    cfg_hash = csa_wot.fingerprint_hash(run_fingerprint(
        Z, cfg_probe, h5ad=args.h5ad, day_min=args.day_min, day_max=args.day_max,
        stride=args.stride, n_per_timepoint=args.n_per_timepoint,
        sampling_seed=args.seed))
    out_dir = os.path.join(args.out, f"cfg_{cfg_hash}")
    dest = [os.path.join(out_dir, f"chain_eps{e:g}.npz") for e in epsilons]
    dest += [os.path.join(out_dir, f"chain_eps{e:g}.npz" + ca.IDENTITY_SUFFIX)
             for e in epsilons]
    dest.append(os.path.join(out_dir, "chain_manifest.json"))
    reserve_destinations(dest, overwrite=args.overwrite)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[stage0] config hash {cfg_hash} -> {out_dir}")
    args.out = out_dir

    manifest = {
        "git_commit": _git_commit(),
        "h5ad": os.path.abspath(args.h5ad),
        "representation_warning": warn,
        "day_range": [args.day_min, args.day_max],
        "stride": args.stride,
        "n_per_timepoint": args.n_per_timepoint,
        "seed": args.seed,
        "support": args.support,
        "kappa": args.kappa,
        "cost_scale_mode": args.cost_scale_mode,
        "cost_scales": [float(x) for x in scales],
        "n_cells": data.n_cells,
        "tau": np.asarray(data.tau).tolist(),
        "dtau": np.asarray(data.dtau, dtype=float).tolist(),
        "identity": identity,
        "cross_arm_representation_allowed": bool(args.allow_cross_arm_representation),
        "batch_labels": {str(float(data.tau[t])): np.asarray(data.replicate[t]).astype(str).tolist()
                         for t in range(data.T)},
        "chains": {},
    }

    for e in epsilons:
        cfg = make_cfg(epsilon=e, support=args.support, kappa=args.kappa,
                       cost_scale_mode=args.cost_scale_mode, seed=args.seed,
                       verbose=args.verbose)
        path = os.path.join(args.out, f"chain_eps{e:g}.npz")
        print(f"\n[stage0] epsilon={e:g} -> {os.path.basename(path)}")
        t0 = time.time()
        chain = build_reference_chain(Z, data.tau, cfg.coupling, verbose=args.verbose)
        el = time.time() - t0
        csum = chain.summary()
        chain.save(path)
        jdump(identity, path + ca.IDENTITY_SUFFIX)

        fp = run_fingerprint(Z, cfg, h5ad=args.h5ad, day_min=args.day_min,
                             day_max=args.day_max, stride=args.stride,
                             n_per_timepoint=args.n_per_timepoint,
                             sampling_seed=args.seed)
        grew = [t for t, k in enumerate(csum["kappas"])
                if k is not None and k > args.kappa]
        manifest["chains"][f"{e:g}"] = {
            "path": path,
            "elapsed_s": el,
            "feasible": bool(csum["feasible"]),
            "marginal_error": csum["marginal_error"],
            "feasibility_tol": csum["feasibility_tol"],
            "kappas": csum["kappas"],
            "kappa_grew_at_intervals": grew,
            "nnz": csum["nnz"],
            "provenance": provenance_block(fp),
        }
        print(f"[stage0]   feasible={csum['feasible']} "
              f"max_marg_err={max(csum['marginal_error']):.3e} "
              f"nnz {min(csum['nnz'])}-{max(csum['nnz'])} ({el:.0f}s)")
        if grew:
            print(f"[stage0]   NOTE kappa grew above {args.kappa} at intervals {grew} "
                  f"-- the SUPPORT was binding here, not epsilon")
        if not csum["feasible"]:
            print(f"[stage0]   *** INFEASIBLE at intervals "
                  f"{chain.infeasible_intervals()} -- A_t will not be row-stochastic; "
                  f"do not use this chain without raising kappa or epsilon")

    jdump(manifest, os.path.join(args.out, "chain_manifest.json"))
    print(f"\n[stage0] wrote {args.out}")
    print("[stage0] chains built:", {k: v["feasible"] for k, v in manifest["chains"].items()})

    if args.print_scan_command:
        print("\n[stage0] Stage-0 requirement 5 -- rerun the epsilon scan under the "
              "SAME scale and support:\n")
        # 01_eps_scan.sbatch already passes --stride 1 --stride 2, and --stride is
        # action="append", so repeating it here would silently add a third stride.
        print(f"  sbatch 01_eps_scan.sbatch \\\n"
              f"      --h5ad {os.path.abspath(args.h5ad)} \\\n"
              f"      --day-min {args.day_min:g} --day-max {args.day_max:g} \\\n"
              f"      --support {args.support} --kappa {args.kappa} \\\n"
              f"      --cost-scale-mode {args.cost_scale_mode} \\\n"
              f"      --out {os.path.join(RESULTS_ROOT, 'eps_scan_stage0_' + cfg_hash)}\n")
        print("  (the sbatch already supplies --stride 1 --stride 2; --stride is "
              "append-mode, so do not repeat it)\n")


if __name__ == "__main__":
    main()
