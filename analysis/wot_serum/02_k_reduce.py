#!/usr/bin/env python
"""Stage B reducer -- merge the per-K JSONs and apply the decided selection rule.

[CRITICAL] Every file must come from the SAME configuration.  Merging K files
produced at different epsilons, day ranges, strides, subsamples, supports or
optimiser settings yields a curve whose shape is driven by the configuration
differences rather than by K, and the recommendation drawn from it is invalid --
silently, because nothing in a per-K record forces the mismatch to surface.  Each
record therefore carries a configuration fingerprint (``csa_wot.run_fingerprint``,
which pins everything EXCEPT K), and this reducer refuses to combine files whose
fingerprints disagree, reporting the offending fields.

The rule itself is the package's ``KSelectionResult.recommend()``: rejections
first (non-convergence, a K whose smallest state is essentially empty), then a
one-SE-STYLE conservative pick on held-out compression.  The spread it uses is
fold-direction variability across A->B and B->A of a BATCH-WISE TECHNICAL split,
NOT a sampling or biological standard error -- effective biological replication
is n = 1.  It is a weak selector; the curve is the result.

    python 02_k_reduce.py --dir <k_sweep_dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from csa_wot import (REPLICATE_STATEMENT, RESULTS_ROOT, assemble_k_result,
                     compare_fingerprints, jdump)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=None,
                   help="k_sweep directory; if omitted and exactly one "
                        "k_sweep_* directory exists under RESULTS_ROOT, that one")
    p.add_argument("--expect-Ks", type=int, nargs="+", default=None,
                   help="the K grid that should be present; default: the "
                        "'Ks_requested' recorded by the sweep itself")
    p.add_argument("--min-state-mass", type=float, default=1e-3)
    p.add_argument("--require-converged", dest="require_converged",
                   action="store_true", default=True)
    p.add_argument("--allow-nonconverged", dest="require_converged",
                   action="store_false")
    p.add_argument("--allow-missing-K", action="store_true",
                   help="proceed even if some expected K values are absent")
    return p.parse_args()


def resolve_dir(arg):
    if arg:
        return arg
    cands = sorted(glob.glob(os.path.join(RESULTS_ROOT, "k_sweep*")))
    cands = [c for c in cands if os.path.isdir(c)]
    if len(cands) == 1:
        return cands[0]
    raise SystemExit(
        f"pass --dir: found {len(cands)} k_sweep directories under {RESULTS_ROOT}"
        + (f" ({[os.path.basename(c) for c in cands]})" if cands else ""))


def load_and_validate(directory):
    """Read every K_*.json and refuse to merge incompatible ones."""
    files = sorted(glob.glob(os.path.join(directory, "K_*.json")))
    if not files:
        raise SystemExit(f"no K_*.json under {directory}")

    rows, missing_fp = [], []
    for f in files:
        with open(f) as fh:
            rec = json.load(fh)
        rec["_file"] = os.path.basename(f)
        if "provenance" not in rec or "fingerprint" not in rec.get("provenance", {}):
            missing_fp.append(rec["_file"])
        rows.append(rec)

    if missing_fp:
        raise SystemExit(
            "these K files carry no configuration fingerprint, so they cannot be "
            "shown to be comparable:\n  " + "\n  ".join(missing_fp) +
            "\n\nThey predate the fingerprinting fix. Re-run the sweep, or move the "
            "old files aside -- do NOT merge them, the resulting K curve would mix "
            "configurations without saying so.")

    # group by fingerprint hash; anything but a single group is a hard error
    groups = {}
    for rec in rows:
        groups.setdefault(rec["provenance"]["fingerprint_hash"], []).append(rec)

    if len(groups) > 1:
        ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        lines = [f"{len(groups)} incompatible configurations found in {directory}:"]
        for h, recs in ordered:
            ks = sorted(r["K"] for r in recs)
            lines.append(f"  {h}: K={ks}  files={[r['_file'] for r in recs]}")
        ref_h, ref_recs = ordered[0]
        lines.append(f"\nDifferences from the largest group ({ref_h}):")
        for h, recs in ordered[1:]:
            diffs = compare_fingerprints(ref_recs[0]["provenance"]["fingerprint"],
                                         recs[0]["provenance"]["fingerprint"])
            lines.append(f"  {h}:")
            lines.extend(f"    {d}" for d in (diffs or ["(no field differs -- "
                                                        "check the hash inputs)"]))
        lines.append("\nRefusing to merge. Sweeps are written to "
                     "k_sweep_<config hash>/ precisely so this cannot happen by "
                     "accident; move the stale files aside and re-run the reducer.")
        raise SystemExit("\n".join(lines))

    # duplicate K within the single group
    seen = {}
    for rec in rows:
        seen.setdefault(rec["K"], []).append(rec["_file"])
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        raise SystemExit(f"duplicate K values in {directory}: {dupes}")

    return rows, next(iter(groups))


def main():
    args = parse_args()
    directory = resolve_dir(args.dir)
    rows, fp_hash = load_and_validate(directory)
    prov = rows[0]["provenance"]
    epsilon = prov["fingerprint"]["epsilon"]

    print(f"[K-reduce] {directory}")
    print(f"[K-reduce] merged {len(rows)} K values, all at config {fp_hash} "
          f"(epsilon={epsilon:g})")

    # -- missing / unexpected K ---------------------------------------------
    have = sorted(r["K"] for r in rows)
    expected = args.expect_Ks or rows[0].get("Ks_requested") or have
    expected = sorted(set(int(k) for k in expected))
    missing = [k for k in expected if k not in have]
    unexpected = [k for k in have if k not in expected]
    if unexpected:
        print(f"[K-reduce] NOTE: K values present but not in the expected grid: "
              f"{unexpected} (expected {expected})")
    if missing:
        msg = (f"missing K values {missing} from the expected grid {expected}; "
               f"present: {have}. An incomplete curve can move the elbow, so the "
               f"recommendation is not trustworthy.")
        if not args.allow_missing_K:
            raise SystemExit(
                "[K-reduce] " + msg +
                "\nCheck for failed array tasks (see logs/k_sweep_*), re-run them, "
                "or pass --allow-missing-K to accept the partial curve.")
        print(f"[K-reduce] WARNING: {msg}")

    res = assemble_k_result(rows, epsilon=epsilon,
                            notes={"fingerprint_hash": fp_hash,
                                   "fingerprint": prov["fingerprint"],
                                   "expected_Ks": expected,
                                   "missing_Ks": missing,
                                   "unexpected_Ks": unexpected,
                                   "replicate_structure": REPLICATE_STATEMENT})

    frame = res.to_frame()
    cols = [c for c in ("K", "heldout_compress", "heldout_se", "train_compress",
                        "heldout_expression", "min_state_mass", "k_eff", "init_ari",
                        "all_converged") if c in frame.columns]
    frame.to_csv(os.path.join(directory, "k_sweep.csv"), index=False)
    print(frame[cols].to_string(index=False))

    tr = np.asarray(res.train_compress, dtype=float)
    if len(tr) > 1 and np.all(np.diff(tr) <= 1e-12):
        print("[K-reduce] training compression is monotone in K, as expected -- "
              "which is why it cannot select K and the held-out curve does the work.")

    rec = res.recommend(min_state_mass=args.min_state_mass,
                        require_converged=args.require_converged)
    print(f"[K-reduce] recommendation: {json.dumps(rec, indent=2, default=str)}")

    jdump({"protocol": res.protocol, "notes": res.notes,
           "provenance": prov, "per_K": res.per_K, "recommendation": rec},
          os.path.join(directory, "k_selection.json"))
    print(f"[K-reduce] wrote {os.path.join(directory, 'k_selection.json')}")

    if rec.get("K") is None:
        print("[K-reduce] every candidate K was rejected -- widen the grid or relax "
              "the criteria before running Stage C.")
    else:
        print(f"[K-reduce] K* = {rec['K']}")
    print(f"[K-reduce] NOTE: {REPLICATE_STATEMENT}")


if __name__ == "__main__":
    main()
