#!/usr/bin/env python
"""Stage B reducer -- merge the per-K JSONs and apply the decided selection rule.

The rule itself is the package's ``KSelectionResult.recommend()``: rejections first
(non-convergence, a K whose smallest state is essentially empty), then a
one-SE-STYLE conservative pick on held-out compression.  The spread it uses is
fold-direction variability across A->B and B->A, NOT a sampling standard error --
n=2 cultures from one embryo.  It is a weak selector; the curve is the result.

    python 02_k_reduce.py --epsilon 0.05
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from csa_wot import RESULTS_ROOT, assemble_k_result, jdump


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=os.path.join(RESULTS_ROOT, "k_sweep"))
    p.add_argument("--epsilon", type=float, required=True)
    p.add_argument("--min-state-mass", type=float, default=1e-3)
    p.add_argument("--require-converged", dest="require_converged",
                   action="store_true", default=True)
    p.add_argument("--allow-nonconverged", dest="require_converged",
                   action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "K_*.json")))
    if not files:
        raise SystemExit(f"no K_*.json under {args.dir}")

    rows = []
    for f in files:
        with open(f) as fh:
            rows.append(json.load(fh))
    print(f"[K-reduce] merged {len(rows)} K values from {args.dir}")

    res = assemble_k_result(rows, epsilon=args.epsilon)

    frame = res.to_frame()
    cols = [c for c in ("K", "heldout_compress", "heldout_se", "train_compress",
                        "heldout_expression", "min_state_mass", "k_eff", "init_ari",
                        "all_converged") if c in frame.columns]
    frame.to_csv(os.path.join(args.dir, "k_sweep.csv"), index=False)
    print(frame[cols].to_string(index=False))

    tr = np.asarray(res.train_compress, dtype=float)
    if len(tr) > 1 and np.all(np.diff(tr) <= 1e-12):
        print("[K-reduce] training compression is monotone in K, as expected -- "
              "which is why it cannot select K and the held-out curve does the work.")

    rec = res.recommend(min_state_mass=args.min_state_mass,
                        require_converged=args.require_converged)
    print(f"[K-reduce] recommendation: {json.dumps(rec, indent=2, default=str)}")

    jdump({"protocol": res.protocol, "notes": res.notes,
           "per_K": res.per_K, "recommendation": rec},
          os.path.join(args.dir, "k_selection.json"))
    print(f"[K-reduce] wrote {os.path.join(args.dir, 'k_selection.json')}")

    if rec.get("K") is None:
        print("[K-reduce] every candidate K was rejected -- widen the grid or relax "
              "the criteria before running Stage C.")
    else:
        print(f"[K-reduce] K* = {rec['K']}")


if __name__ == "__main__":
    main()
