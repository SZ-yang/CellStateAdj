#!/usr/bin/env python
"""Re-derive the lambda-sweep verdict from saved results, without refitting.

Every per-cell number the verdict needs is already in ``lambda_sweep.json``, so a
corrected gate can be applied retrospectively.  That matters because the first
version of the gate passed cells that were textbook Degeneracy-3 collapses; the
fits themselves are fine and cost hours, only the interpretation was wrong.

    python 04_lambda_reduce.py                    # every grid under RESULTS_ROOT
    python 04_lambda_reduce.py --dir <sweep_dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from csa_wot import RESULTS_ROOT, jdump

import importlib.util as _u
_spec = _u.spec_from_file_location(
    "_lam", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_lambda_sweep.py"))
_lam = _u.module_from_spec(_spec)
_spec.loader.exec_module(_lam)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", nargs="*", default=None,
                   help="sweep directories; default: every lambda_sweep_* under RESULTS_ROOT")
    p.add_argument("--collapse-frac", type=float, default=0.8)
    p.add_argument("--ari-related-min", type=float, default=0.30,
                   help="below this the fit moved to an UNRELATED partition")
    p.add_argument("--ari-moved-max", type=float, default=0.99,
                   help="at or above this lambda_pm was inert")
    return p.parse_args()


def main():
    args = parse_args()
    dirs = args.dir or sorted(
        os.path.dirname(f) for f in
        glob.glob(os.path.join(RESULTS_ROOT, "lambda_sweep_*", "lambda_sweep.json")))
    if not dirs:
        raise SystemExit(f"no lambda_sweep_*/lambda_sweep.json under {RESULTS_ROOT}")

    for d in dirs:
        path = os.path.join(d, "lambda_sweep.json")
        payload = json.load(open(path))
        rows = payload["per_cell"]
        v = _lam._verdict(rows, payload.get("lambda_xs", []),
                          payload.get("lambdas_pm", []), args.collapse_frac,
                          ari_moved_max=args.ari_moved_max,
                          ari_related_min=args.ari_related_min)
        payload["verdict"] = v
        payload["verdict_rederived"] = True
        jdump(payload, path)

        print(f"\n{'=' * 78}\nK={payload['K']}  {os.path.basename(d)}\n{'=' * 78}")
        print(f"  transport competes at lambda_x: "
              f"{v.get('lambda_x_where_transport_competes')}")
        print(f"  usable: {len(v.get('usable_cells', []))}   "
              f"degeneracy-3 collapses: {v.get('n_degeneracy3_collapses')}   "
              f"unrelated partitions: {v.get('n_unrelated_partitions')}   "
              f"errors: {v.get('n_error_cells')}")
        if "reproducibility_warning" in v:
            print(f"  !! {v['reproducibility_warning']}")
        print(f"  CONCLUSION: {v['conclusion']}")
        for r in v.get("rejected_cells", []):
            print(f"    rejected lx={r['lambda_x']:g} lpm={r['lambda_pm']:g}: "
                  f"{'; '.join(r['rejected_because'])}")
        print(f"  -> rewrote {path}")


if __name__ == "__main__":
    main()
