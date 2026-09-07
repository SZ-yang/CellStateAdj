#!/usr/bin/env python
"""Build a SERUM-ONLY representation, as the sensitivity check for issue 7.

The shipped ``obsm['X_model']`` is transductive across arms: ``wot_data_check.ipynb``
selects HVGs and fits scaling and PCA on the balanced union of shared + serum + 2i
cells, then extracts the serum subset.  That is retained deliberately in the main run
for cross-arm comparability, but it means the frozen basis saw cells the serum-
conditional analysis excludes, and nothing downstream can detect that on its own.

This script produces the comparison dataset: identical preprocessing, but HVGs,
scaling and PCA fitted on the serum-conditional cells ONLY.  Point any stage at the
result with ``--h5ad``; because the representation content hash differs, it lands in
its own run directory and the K reducer will refuse to mix it with the main sweep --
which is the intended behaviour, not an obstacle.

    python 00_serum_only_pca.py --raw <full h5ad> --out serum_only_pca.h5ad
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from csa_wot import ARM_SPLIT_DAY, SERUM_ARM, SHARED_ARM, serum_arm_mask

DEFAULT_RAW = "/dartfs/rc/lab/C/CxQiu/data/joshua/wot_data/raw/wot_schiebinger2019_full.h5ad"
DEFAULT_OUT = "/dartfs/rc/lab/C/CxQiu/data/joshua/wot_data/wot_serum_only_pca.h5ad"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=DEFAULT_RAW,
                   help="the full Schiebinger AnnData (log1p(TP10K), NOT counts)")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--cells-per-group", type=int, default=500,
                   help="balanced subsample per day x batch x arm, as in wot_data_check")
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-pcs-fit", type=int, default=50)
    p.add_argument("--n-pcs-model", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    import anndata as ad
    import pandas as pd
    import scanpy as sc

    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite")

    adata = ad.read_h5ad(args.raw)
    print(f"[raw] {adata.shape}")

    # X is already log1p(TP10K) -- do NOT normalise or log1p again.
    def to_bool(series):
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False).to_numpy()
        return (series.astype(str).str.strip().str.lower()
                .isin(["true", "1", "yes"]).to_numpy())

    day = pd.to_numeric(adata.obs["day"], errors="raise").to_numpy()
    serum_flag, two_i_flag = to_bool(adata.obs["serum"]), to_bool(adata.obs["2i"])
    arm = np.full(adata.n_obs, "invalid", dtype=object)
    arm[day <= ARM_SPLIT_DAY] = SHARED_ARM
    arm[(day > ARM_SPLIT_DAY) & serum_flag & ~two_i_flag] = SERUM_ARM
    arm[(day > ARM_SPLIT_DAY) & two_i_flag & ~serum_flag] = "2i"
    adata.obs["arm"] = pd.Categorical(arm, categories=[SHARED_ARM, SERUM_ARM, "2i"])
    adata.obs["batch"] = adata.obs["batch"].astype(str).astype("category")

    # [THE POINT] restrict to the serum-conditional trajectory BEFORE any HVG,
    # scaling or PCA is fitted. In wot_data_check.ipynb this restriction happens
    # afterwards, which is exactly the difference being tested.
    keep = serum_arm_mask(day, arm)
    adata = adata[keep].copy()
    print(f"[serum-only] {adata.shape}  arms={adata.obs['arm'].value_counts().to_dict()}")

    rng = np.random.default_rng(args.seed)
    picks = []
    for _, idx in adata.obs.groupby(["day", "batch", "arm"], observed=True).indices.items():
        idx = np.asarray(idx, dtype=int)
        picks.append(rng.choice(idx, size=min(args.cells_per_group, len(idx)),
                                replace=False))
    sel = np.concatenate(picks)
    order = np.lexsort((adata.obs_names[sel].astype(str).to_numpy(),
                        adata.obs["batch"].astype(str).to_numpy()[sel],
                        adata.obs["day"].to_numpy()[sel]))
    adata = adata[sel[order]].copy()
    print(f"[balanced] {adata.shape}")

    mito = adata.var_names.str.upper().str.startswith("MT-")
    adata.var["mitochondrial"] = mito
    hvg_in = adata[:, ~mito].copy()
    sc.pp.highly_variable_genes(hvg_in, flavor="seurat", n_top_genes=args.n_hvg,
                                batch_key="batch", subset=False, inplace=True)
    hvg = hvg_in.var_names[hvg_in.var["highly_variable"]]
    print(f"[hvg] {len(hvg)} genes selected on serum-conditional cells only")

    rep = adata[:, hvg].copy()
    rep.layers["lognorm"] = rep.X.copy()
    sc.pp.scale(rep, zero_center=True, max_value=10)
    sc.pp.pca(rep, n_comps=args.n_pcs_fit, zero_center=True,
              svd_solver="randomized", random_state=args.seed)
    rep.obsm["X_model"] = rep.obsm["X_pca"][:, :args.n_pcs_model].astype(np.float32).copy()

    rep.uns["representation_provenance"] = (
        "SERUM-ONLY sensitivity representation: HVGs, scaling and PCA fitted on "
        "arm=='shared' (day<=8) + arm=='serum' (day>8) cells only. Compare against "
        "wot_serum_first_run_pca.h5ad, whose basis was fitted on the balanced union "
        "including 2i.")
    rep.write_h5ad(args.out, compression="gzip")
    print(f"[done] wrote {args.out}  X_model={rep.obsm['X_model'].shape}")
    print("[done] re-run any stage with --h5ad to get a separate, non-mixable run dir")


if __name__ == "__main__":
    main()
