"""Shared helpers for running CellStateAdj on the WOT (Schiebinger 2019) serum arm.

Why this module exists at all
-----------------------------
The packaged h5ad carries its frozen representation in ``obsm['X_model']`` (30 PCs),
but ``cellstateadj.data.from_anndata`` only ever reads ``adata.X`` -- there is no
``obsm`` support anywhere in the package.  ``adata.X`` in this file is the *z-scored*
HVG matrix, so feeding it to ``learn_representation`` (which library-size-normalises
and log1p's) produces NaN.  ``scripts/run_fit.py --h5ad`` therefore cannot be used on
this dataset.

The supported route is the ``Z=`` escape hatch: ``run_pipeline(data, cfg, Z=...)``
skips representation learning entirely (pipeline.py:75).  We build ``TimeSeriesData``
by hand with ``X = Z = X_model``, which is safe because ``X`` is only touched by
representation learning (skipped) and by the row-indexing helpers.

Everything else -- the reference chain, the optimiser, the diagnostics, the DAG
edges -- comes from the package unchanged.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from dataclasses import replace as dc_replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.abspath(os.path.join(HERE, "..", "..", "main"))
if MAIN not in sys.path:
    sys.path.insert(0, MAIN)

DEFAULT_H5AD = "/dartfs/rc/lab/C/CxQiu/data/joshua/wot_data/wot_serum_first_run_pca.h5ad"
RESULTS_ROOT = "/dartfs/rc/lab/C/CxQiu/data/joshua/CellStateAdj/wot_serum"

# Recorded in every summary.json.
#
# Serum and 2i are experimental INTERVENTIONS applied at day 8, not stochastic
# fates with an observed mass split, so there is no "true" allocation of day-8
# cells between them that balanced transport could get wrong.  The estimand is
# conditional on the intervention.
CONDITIONING_STATEMENT = (
    "This analysis conditions on the serum intervention: shared cells through day 8 "
    "are followed only by serum-cultured cells after day 8. It estimates a "
    "serum-conditional trajectory and does not estimate allocation or transitions "
    "between the serum and 2i interventions. The day-8 -> day-8.25 interval spans "
    "the point of intervention, so edges across it describe the serum-conditional "
    "continuation, not a developmental bifurcation."
)

# The real limitation of the balanced formulation, stated separately because it is
# a different claim from the conditioning above.
BALANCED_PILOT_STATEMENT = (
    "BALANCED PILOT -- not a reproduction of Waddington-OT. Marginals are uniform "
    "over sampled cells at each timepoint and proliferation and death are NOT "
    "modelled; obs['cell_growth_rate'] is present in the data but deliberately "
    "unused. Schiebinger et al. (2019) used growth-aware unbalanced transport, so "
    "these couplings are not comparable to theirs and transported mass must not be "
    "read as a population abundance. Unbalanced transport is a documented later "
    "extension (PROJECT_HANDOFF.txt s11), gated on validating abundance assumptions."
)

REPLICATE_STATEMENT = (
    "obs['batch'] (1/2) is a duplicate SAMPLE collected at each timepoint, used here "
    "as a batch-wise technical hold-out. Sampling is destructive, so batch 1 is not "
    "established to be one longitudinal culture lineage followed across days; the "
    "WOT paper reports duplicate samples per timepoint, not tracked parallel "
    "cultures. Spread across the two split directions is fold-direction variability "
    "of a technical split, NOT a biological standard error. Effective biological "
    "replication is n = 1 (a single female embryo)."
)

REPRESENTATION_STATEMENT = (
    "The frozen representation obsm['X_model'] is TRANSDUCTIVE ACROSS ARMS: "
    "analysis/wot_data_check.ipynb selects HVGs and fits scaling and PCA on the "
    "balanced union of shared + serum + 2i cells, and extracts the serum subset "
    "afterwards. It is therefore not a serum-only basis. This is retained "
    "deliberately for cross-arm comparability, but it means the representation saw "
    "cells this run excludes. A serum-only PCA sensitivity run is the check; see "
    "the README. The package does NOT learn a representation during these fits -- "
    "X_model is passed in precomputed via the Z= argument of run_pipeline, so "
    "config.representation in any saved config.json is INERT and describes nothing "
    "that ran."
)

# Kept as a single name for the run summaries.
BRANCH_CAVEAT = CONDITIONING_STATEMENT

from cellstateadj.config import PipelineConfig  # noqa: E402
from cellstateadj.cost import resolve_cost_scales  # noqa: E402
from cellstateadj.data import (  # noqa: E402
    TimeSeriesData,
    split_half_by_replicate,
    subsample_cells,
    subsample_timepoints,
)
from cellstateadj.model import CoarseGrainModel  # noqa: E402
from cellstateadj.optimize import fit as fit_states  # noqa: E402
from cellstateadj.reference import build_reference_chain  # noqa: E402
from cellstateadj.selection import KSelectionResult, transfer_memberships  # noqa: E402


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

ARM_SPLIT_DAY = 8.0
SHARED_ARM = "shared"
SERUM_ARM = "serum"
EXCLUDED_ARM = "2i"


def serum_arm_mask(days: np.ndarray, arm: np.ndarray,
                   split_day: float = ARM_SPLIT_DAY) -> np.ndarray:
    """Cells belonging to the serum-conditional trajectory.

    ``arm == 'shared'`` at day <= split_day, ``arm == 'serum'`` after it.  Any
    other combination -- notably every ``2i`` cell -- is dropped.

    Day filtering alone is NOT sufficient: the CLI accepts ``--h5ad``, and the
    parent file (``wot_balanced_shared_pca.h5ad``) carries both arms over the
    same day range, so a day-only filter would silently pull 2i cells into a
    trajectory that claims to condition on the serum intervention.
    """
    days = np.asarray(days, dtype=float)
    arm = np.asarray(arm).astype(str)
    pre = (days <= split_day) & (arm == SHARED_ARM)
    post = (days > split_day) & (arm == SERUM_ARM)
    return pre | post


def arm_composition(data: TimeSeriesData, obs, arm_key: str = "arm"):
    """day x batch x arm counts of the cells actually in ``data``.

    Recomputed from ``data`` rather than cached on it: ``select_cells`` and
    ``select_timepoints`` build fresh ``TimeSeriesData`` objects, so an attribute
    stashed at load time would silently describe the pre-subsampling series.
    """
    import pandas as pd

    rows = []
    for t in range(data.T):
        idx = np.asarray(data.obs[t]["index"]) if data.obs else None
        arm = (np.asarray(obs[arm_key].reindex(idx)).astype(str) if idx is not None
               else np.full(data.n_cells[t], "unknown"))
        rep = (np.asarray(data.replicate[t]) if data.replicate is not None
               else np.full(data.n_cells[t], "NA"))
        rows.append(pd.DataFrame({"day": data.tau[t], "batch": rep, "arm": arm}))
    kept = pd.concat(rows, ignore_index=True)
    return (kept.groupby(["day", "batch", "arm"], observed=True)
            .size().rename("n_cells").reset_index())


def load_serum(
    h5ad: str = DEFAULT_H5AD,
    day_min: float = 0.0,
    day_max: float = 18.0,
    stride: int = 1,
    n_per_timepoint: Optional[int] = None,
    obsm_key: str = "X_model",
    time_key: str = "day",
    replicate_key: str = "batch",
    arm_key: str = "arm",
    seed: int = 0,
    verbose: int = 1,
):
    """Build ``TimeSeriesData`` whose ``X`` IS the frozen representation.

    Retains ONLY the serum-conditional trajectory -- ``arm == 'shared'`` through
    day 8 and ``arm == 'serum'`` after it -- and asserts that no ``2i`` cell
    survives.  See :func:`serum_arm_mask` for why the day filter is not enough.

    Returns ``(data, Z, obs)`` where ``obs`` is the surviving ``adata.obs`` frame,
    indexed by cell name, for later annotation joins.  ``Z`` is ``data.X`` cast to
    float64 -- the same arrays, kept separate only because ``run_pipeline`` wants
    them passed explicitly.
    """
    import anndata as ad
    import pandas as pd

    # Backed: only obs/obsm are read.  The expression matrix and the 'lognorm'
    # layer are ~310 MB each and are never used -- the representation is frozen
    # in obsm[obsm_key] and nothing downstream touches genes.
    adata = ad.read_h5ad(h5ad, backed="r")
    obs_all = adata.obs.copy()
    if arm_key not in obs_all.columns:
        raise ValueError(
            f"{h5ad} has no obs['{arm_key}'] column, so the serum arm cannot be "
            f"isolated. This loader refuses to fall back to a day-only filter: on "
            f"the parent AnnData that would silently include 2i cells after day "
            f"{ARM_SPLIT_DAY:g}. Available obs columns: {list(obs_all.columns)[:12]}...")
    days_all = np.asarray(obs_all[time_key]).astype(float)
    arm_all = np.asarray(obs_all[arm_key]).astype(str)
    Xm_all = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    names_all = np.asarray(adata.obs_names)
    reps_all = np.asarray(obs_all[replicate_key]).astype(str)
    if adata.isbacked:
        adata.file.close()

    in_days = (days_all >= day_min) & (days_all <= day_max)
    in_arm = serum_arm_mask(days_all, arm_all)
    keep = in_days & in_arm

    n_dropped_arm = int((in_days & ~in_arm).sum())
    if verbose and n_dropped_arm:
        dropped = pd.Series(arm_all[in_days & ~in_arm]).value_counts().to_dict()
        print(f"[load] arm filter dropped {n_dropped_arm} cells in the day range: "
              f"{dropped}")

    days = days_all[keep]
    Xm = Xm_all[keep]
    names = names_all[keep]
    reps = reps_all[keep]
    arms = arm_all[keep]
    obs_all = obs_all.loc[keep]

    # Hard post-condition: the estimand is serum-conditional, so a 2i cell here
    # is a silent scientific error, not a tolerable impurity.
    n_2i = int((arms == EXCLUDED_ARM).sum())
    if n_2i:
        raise AssertionError(
            f"{n_2i} '{EXCLUDED_ARM}' cells survived the serum arm filter -- refusing "
            f"to build a serum-conditional trajectory that contains them.")
    bad_pre = int(((days <= ARM_SPLIT_DAY) & (arms != SHARED_ARM)).sum())
    bad_post = int(((days > ARM_SPLIT_DAY) & (arms != SERUM_ARM)).sum())
    if bad_pre or bad_post:
        raise AssertionError(
            f"serum arm filter left {bad_pre} non-'{SHARED_ARM}' cells at day "
            f"<= {ARM_SPLIT_DAY:g} and {bad_post} non-'{SERUM_ARM}' cells after it.")

    tau = np.unique(days)
    if len(tau) < 2:
        raise ValueError(f"need >= 2 timepoints in [{day_min}, {day_max}], got {len(tau)}")

    X, replicate, obs = [], [], []
    for d in tau:
        m = days == d
        X.append(Xm[m])
        replicate.append(reps[m])
        obs.append({"index": names[m]})

    data = TimeSeriesData(X=X, tau=tau, replicate=replicate, obs=obs,
                          gene_names=np.array([f"{obsm_key}_{j}" for j in range(Xm.shape[1])]))

    if stride > 1:
        data = subsample_timepoints(data, stride=stride)
    if n_per_timepoint is not None:
        data = subsample_cells(data, n_per_timepoint, seed=seed, stratify_by_replicate=True)

    Z = [np.asarray(x, dtype=np.float64) for x in data.X]

    composition = arm_composition(data, obs_all, arm_key=arm_key)

    if verbose:
        print(f"[load] {h5ad}")
        print(f"[load] days {data.tau[0]:g}-{data.tau[-1]:g}  T={data.T}  "
              f"cells={sum(data.n_cells)}  d={Z[0].shape[1]}  stride={stride}")
        print(f"[load] dtau unique: {sorted(set(np.round(data.dtau, 4).tolist()))}")
        print(f"[load] arms retained: "
              f"{composition.groupby('arm', observed=True)['n_cells'].sum().to_dict()}")
        if verbose > 1:
            print(composition.to_string(index=False))

    return data, Z, obs_all


def make_cfg(
    epsilon: float = 0.05,
    K: int = 20,
    kappa: int = 400,
    support: str = "knn",
    cost_scale_mode: str = "global",
    lambda_compress: float = 1.0,
    lambda_x: float = 1.0,
    lambda_plus: float = 0.0,
    lambda_minus: float = 0.0,
    max_iter: int = 1500,
    n_init: int = 1,
    seed: int = 0,
    device: str = "cpu",
    geometric_null: bool = True,
    verbose: int = 1,
) -> PipelineConfig:
    """One place that sets the non-default knobs this dataset needs.

    ``kappa=400`` rather than the package default 50: at 1000 cells/timepoint a
    kappa-50 support admits no balanced plan, so ``solve_interval`` grows
    50 -> 100 -> 200 -> 400 (reference.py:319), burning a full 20k-iteration
    Sinkhorn solve at each failed step.  Measured cost of that on this data is
    147 s/interval versus 1.0 s/interval when kappa starts at 400 -- identical
    answer, 150x the time.

    ``cost_scale_mode`` defaults to ``'global'``, the package default.  The stated
    reason to keep it is that a per-interval scale divides dtau out of the cost.
    On THIS series that argument is weak: dtau is 0.5 d for 36 of 38 intervals
    (only days 8-9 differ, at 0.25 d), so the cancellation is nearly a no-op --
    while the cost of keeping it is large, because the typical transport cost is
    ~10x higher in the serum phase than in phase 1 (measured), so a single epsilon
    cannot sit in the informative regime for both.  ``'per_interval'`` is therefore
    a legitimate thing to test here; it changes the fingerprint, so it lands in its
    own run directory.

    Always builds a fresh config: ``PipelineConfig()`` is used as a shared mutable
    default argument at pipeline.py:46,63.
    """
    cfg = PipelineConfig()
    cfg.coupling.epsilon = float(epsilon)
    cfg.coupling.support = support
    cfg.coupling.kappa = int(kappa)
    cfg.coupling.kappa_max = 1000
    cfg.coupling.cost_scale_mode = cost_scale_mode
    cfg.coupling.device = device

    cfg.model.K = int(K)
    cfg.model.lambda_compress = float(lambda_compress)
    cfg.model.lambda_x = float(lambda_x)
    cfg.model.lambda_plus = float(lambda_plus)
    cfg.model.lambda_minus = float(lambda_minus)
    cfg.model.device = device

    cfg.optim.max_iter = int(max_iter)
    cfg.optim.n_init = int(n_init)
    cfg.optim.seed = int(seed)
    cfg.optim.verbose = int(verbose)

    cfg.diagnostics.geometric_null = bool(geometric_null)
    cfg.diagnostics.seed = int(seed)
    return cfg


def resolve_device(requested: str, verbose: int = 1) -> str:
    """'auto' -> cuda when actually available.  Never silently claims a GPU."""
    if requested != "auto":
        return requested
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"[device] auto -> {dev}"
              + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    return dev


# --------------------------------------------------------------------------
# run identity: what makes two runs comparable
# --------------------------------------------------------------------------

# Every field that changes the numbers. Two artifacts may only be combined or
# compared if their fingerprints agree on ALL of these. Kept as an explicit tuple
# so that adding a knob to the CLI without deciding whether it affects
# comparability is a visible omission rather than a silent one.
FINGERPRINT_FIELDS = (
    "h5ad",                 # which AnnData
    "representation",       # obsm key, shape, and a content hash of X_model
    "arm_policy",           # how the serum trajectory was isolated
    "day_min", "day_max",   # day range
    "stride",               # timepoint subsampling
    "n_per_timepoint",      # cell subsampling ...
    "sampling_seed",        # ... and the seed that drew it
    "epsilon",              # coupling
    "support", "kappa", "kappa_max", "cost_scale_mode", "coupling_dtype",
    "lambda_compress", "lambda_x",   # objective terms shared by every stage
    "optim",                # optimiser settings that affect the answer
    "seed_policy",
)


def representation_fingerprint(Z: Sequence[np.ndarray], obsm_key: str = "X_model") -> dict:
    """Content hash of the frozen representation, not just its name.

    Two runs that both say ``X_model`` but read different files, different day
    ranges or different subsamples are NOT comparable, and a name alone cannot
    tell them apart. Hashing the actual float32 bytes can.
    """
    import hashlib

    h = hashlib.sha1()
    for z in Z:
        h.update(np.ascontiguousarray(np.asarray(z, dtype=np.float32)).tobytes())
    return {"obsm_key": obsm_key,
            "n_timepoints": len(Z),
            "n_cells": [int(len(z)) for z in Z],
            "n_dims": int(Z[0].shape[1]) if len(Z) else 0,
            "sha1": h.hexdigest()[:16],
            "provenance": REPRESENTATION_STATEMENT}


def run_fingerprint(Z, cfg: PipelineConfig, *, h5ad: str, day_min: float,
                    day_max: float, stride: int, n_per_timepoint: Optional[int],
                    sampling_seed: int, obsm_key: str = "X_model") -> dict:
    """The canonical, comparable description of a run configuration."""
    return {
        "h5ad": os.path.abspath(h5ad),
        "representation": representation_fingerprint(Z, obsm_key),
        "arm_policy": (f"arm=='{SHARED_ARM}' for day<={ARM_SPLIT_DAY:g}, "
                       f"arm=='{SERUM_ARM}' after; '{EXCLUDED_ARM}' excluded"),
        "day_min": float(day_min),
        "day_max": float(day_max),
        "stride": int(stride),
        "n_per_timepoint": (None if n_per_timepoint is None else int(n_per_timepoint)),
        "sampling_seed": int(sampling_seed),
        "epsilon": float(cfg.coupling.epsilon),
        "support": cfg.coupling.support,
        "kappa": int(cfg.coupling.kappa),
        "kappa_max": int(cfg.coupling.kappa_max),
        "cost_scale_mode": cfg.coupling.cost_scale_mode,
        "coupling_dtype": cfg.coupling.dtype,
        "lambda_compress": float(cfg.model.lambda_compress),
        "lambda_x": float(cfg.model.lambda_x),
        "optim": {"method": cfg.optim.method, "direction": cfg.optim.direction,
                  "max_iter": int(cfg.optim.max_iter),
                  "n_init": int(cfg.optim.n_init),
                  "tol_objective": float(cfg.optim.tol_objective),
                  "tol_membership": float(cfg.optim.tol_membership)},
        "seed_policy": f"cfg.optim.seed={int(cfg.optim.seed)}; "
                       f"restart r uses seed+r (optimize.py:471)",
    }


def fingerprint_hash(fp: dict, n: int = 10) -> str:
    """Short stable id for a fingerprint. Same config -> same directory name."""
    import hashlib

    payload = json.dumps({k: fp[k] for k in FINGERPRINT_FIELDS if k in fp},
                         sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:n]


def compare_fingerprints(a: dict, b: dict) -> List[str]:
    """Field-by-field mismatches between two fingerprints, as readable strings."""
    out = []
    for k in FINGERPRINT_FIELDS:
        va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
        if k == "representation" and isinstance(va, dict) and isinstance(vb, dict):
            va = {kk: va.get(kk) for kk in ("obsm_key", "sha1", "n_cells", "n_dims")}
            vb = {kk: vb.get(kk) for kk in ("obsm_key", "sha1", "n_cells", "n_dims")}
        if va != vb:
            sa, sb = str(va), str(vb)
            if len(sa) > 120:
                sa = sa[:117] + "..."
            if len(sb) > 120:
                sb = sb[:117] + "..."
            out.append(f"{k}: {sa}  !=  {sb}")
    return out


def provenance_block(fp: dict) -> dict:
    """The statements every artifact must carry, in one place."""
    return {
        "fingerprint": fp,
        "fingerprint_hash": fingerprint_hash(fp),
        "conditioning": CONDITIONING_STATEMENT,
        "balanced_pilot": BALANCED_PILOT_STATEMENT,
        "replicate_structure": REPLICATE_STATEMENT,
        "representation": REPRESENTATION_STATEMENT,
    }


# --------------------------------------------------------------------------
# JSON / IO helpers
# --------------------------------------------------------------------------

def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_jsonable)


def ensure_outdir(path: str, overwrite: bool = False) -> str:
    os.makedirs(path, exist_ok=True)
    if os.listdir(path) and not overwrite:
        raise SystemExit(
            f"output directory {path} is not empty; pass --overwrite to replace it")
    return path


def reserve_destinations(paths: Sequence[str], overwrite: bool = False) -> None:
    """Check EVERY output destination before any of them is written.

    [CRITICAL] Ordering, not politeness.  Writing the reference chain and then
    discovering that ``fit_main/`` already exists leaves the previous run's fits
    pointing at a chain that has been silently replaced -- a different day range,
    stride, subsample or support all produce a different chain under the same
    name, and the old fits still load it without complaint.  A collision check
    that runs after a write cannot protect anything, so every destination is
    resolved and validated up front and this function must not create or modify
    anything.

    A path is "occupied" if it is a file that exists, or a directory that exists
    and is non-empty.
    """
    occupied = []
    for path in paths:
        if os.path.isdir(path):
            if os.listdir(path):
                occupied.append(f"{path}  (directory, {len(os.listdir(path))} entries)")
        elif os.path.exists(path):
            occupied.append(f"{path}  ({os.path.getsize(path) / 1e6:.0f} MB file)")
    if occupied and not overwrite:
        raise SystemExit(
            "refusing to start: these destinations already exist and nothing has "
            "been written yet --\n  " + "\n  ".join(occupied) +
            "\n\nPass --overwrite to replace them, or change --out / --tag. "
            "Runs are keyed by a configuration hash, so a collision here means "
            "the SAME configuration was already run; a different day range, "
            "stride, subsample, support or epsilon would have produced a "
            "different run directory.")


# --------------------------------------------------------------------------
# the DAG
# --------------------------------------------------------------------------

def build_dag(diag, tau: Sequence[float], min_mass: float = 1e-4,
              labels: Optional[Dict[Tuple[int, int], str]] = None):
    """Time-layered DAG: nodes ``(t, k)`` above the mass floor, edges from ``T_t``.

    Acyclic by construction -- every edge goes from layer ``t`` to layer ``t+1``.
    Edges are transport-implied developmental compatibility, never observed lineage.
    """
    import networkx as nx

    G = nx.DiGraph()
    active = diag.active if diag.active else [np.ones_like(g, dtype=bool) for g in diag.g]
    for t, g in enumerate(diag.g):
        for k in range(len(g)):
            if not active[t][k]:
                continue
            attrs = dict(t=int(t), state=int(k), day=float(tau[t]),
                         mass=float(g[k]),
                         n_child=float(diag.n_child[t][k]) if (
                             diag.n_child and diag.n_child[t] is not None) else float("nan"),
                         n_parent=float(diag.n_parent[t][k]) if (
                             diag.n_parent and diag.n_parent[t] is not None) else float("nan"))
            if labels is not None:
                attrs["annotation"] = labels.get((t, k), "")
            G.add_node(f"t{t}_k{k}", **attrs)

    for e in diag.dag_edges(min_mass=min_mass):
        u, v = f"t{e['t']}_k{e['source']}", f"t{e['t'] + 1}_k{e['target']}"
        if u in G and v in G:
            G.add_edge(u, v, mass=e["mass"], forward=e["forward"], reverse=e["reverse"])
    return G


# --------------------------------------------------------------------------
# the full artifact dump
# --------------------------------------------------------------------------

def save_all(result, outdir: str, data: TimeSeriesData, obs=None,
             min_edge_mass: float = 1e-4, extra: Optional[dict] = None,
             chain_path: Optional[str] = None, provenance: Optional[dict] = None,
             verbose: int = 1) -> None:
    """Persist EVERY optimised matrix, converged or not.

    ``scripts/run_fit.py`` saves only M, T, g, V+- and the DAG edges, and on a
    non-converged fit it discards even those (run_fit.py:185).  Since the point of
    this run is to have the matrices on disk for later analysis, nothing is dropped
    here and the convergence status simply travels with the files.

    ``chain_path`` points at an already-written ``reference_chain.npz`` shared by
    several fits (epsilon is frozen, so the chain is identical); the path is then
    recorded in ``summary.json`` instead of the file being duplicated.
    """
    import pandas as pd
    import torch

    os.makedirs(outdir, exist_ok=True)
    res_fit = result.fit
    model = res_fit.model
    diag = result.diagnostics
    tau = np.asarray(data.tau, dtype=float)

    # -- the frozen reference chain (P^ref sparse triplets, marginals, Z) -----
    if chain_path is None:
        chain_path = os.path.join(outdir, "reference_chain.npz")
        result.chain.save(chain_path)

    # -- memberships, hard labels, expression prototypes ---------------------
    mem = {}
    for t, M in enumerate(res_fit.M):
        mem[f"M_{t}"] = np.asarray(M, dtype=np.float32)
        mem[f"labels_{t}"] = np.asarray(M).argmax(1).astype(np.int32)
    if diag is not None:
        for t, mu in enumerate(diag.mu):
            mem[f"mu_{t}"] = np.asarray(mu, dtype=np.float32)
    mem["tau"] = tau
    np.savez_compressed(os.path.join(outdir, "memberships.npz"), **mem)

    # -- induced transitions -------------------------------------------------
    with torch.no_grad():
        M_t = model.memberships()
        Ts, As, Bs, gs = model.induced_transitions(M_t)
        Fp, Fm = model.fingerprints(M_t)
        phip, phim = [], []
        for t in range(model.T):
            phip.append(None if Fp[t] is None else
                        model.prototypes(M_t, Fp[t], gs, t).cpu().numpy())
            phim.append(None if Fm[t] is None else
                        model.prototypes(M_t, Fm[t], gs, t).cpu().numpy())

    trans = {"tau": tau}
    for t in range(len(Ts)):
        trans[f"T_{t}"] = Ts[t].cpu().numpy().astype(np.float64)
        trans[f"A_{t}"] = As[t].cpu().numpy().astype(np.float64)
        trans[f"B_{t}"] = Bs[t].cpu().numpy().astype(np.float64)
    for t in range(len(gs)):
        trans[f"g_{t}"] = gs[t].cpu().numpy().astype(np.float64)
    np.savez_compressed(os.path.join(outdir, "transitions.npz"), **trans)

    # -- per-cell fingerprints and their KL-barycentre prototypes ------------
    fp = {"tau": tau}
    for t in range(model.T):
        if Fp[t] is not None:
            fp[f"f_plus_{t}"] = Fp[t].cpu().numpy().astype(np.float32)
            fp[f"phi_plus_{t}"] = np.asarray(phip[t], dtype=np.float64)
        if Fm[t] is not None:
            fp[f"f_minus_{t}"] = Fm[t].cpu().numpy().astype(np.float32)
            fp[f"phi_minus_{t}"] = np.asarray(phim[t], dtype=np.float64)
    np.savez_compressed(os.path.join(outdir, "fingerprints.npz"), **fp)

    # -- diagnostics ---------------------------------------------------------
    if diag is not None:
        dg = {"tau": tau, "k_eff": np.asarray(diag.k_eff, dtype=float)}
        for t in range(len(diag.g)):
            dg[f"g_{t}"] = diag.g[t]
            dg[f"active_{t}"] = diag.active[t] if diag.active else np.ones_like(diag.g[t], bool)
            for name, seq in (("V_plus", diag.V_plus), ("V_minus", diag.V_minus),
                              ("G_plus", diag.G_plus), ("G_minus", diag.G_minus),
                              ("n_child", diag.n_child), ("n_parent", diag.n_parent)):
                v = seq[t] if (seq and t < len(seq)) else None
                if v is not None:
                    dg[f"{name}_{t}"] = np.asarray(v, dtype=float)
        np.savez_compressed(os.path.join(outdir, "diagnostics.npz"), **dg)

        edges = diag.dag_edges(min_mass=min_edge_mass)
        jdump(edges, os.path.join(outdir, "dag_edges.json"))

        ev = diag.event_table()
        if hasattr(ev, "to_csv"):
            ev.insert(1, "day", [tau[int(t)] for t in ev["t"]])
            ev.to_csv(os.path.join(outdir, "dag_nodes.csv"), index=False)
        else:
            jdump(ev, os.path.join(outdir, "dag_nodes.json"))

        try:
            import networkx as nx
            G = build_dag(diag, tau, min_mass=min_edge_mass)
            nx.write_graphml(G, os.path.join(outdir, "dag.graphml"))
            dag_info = {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
                        "is_dag": bool(nx.is_directed_acyclic_graph(G)),
                        "min_edge_mass": min_edge_mass}
        except Exception as exc:  # networkx is optional
            dag_info = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        dag_info = {"note": "diagnostics were not computed"}

    # -- optimisation history ------------------------------------------------
    if res_fit.history:
        pd.DataFrame(res_fit.history).to_csv(
            os.path.join(outdir, "history.csv"), index=False)

    # -- one row per cell: the join key for every later biological question --
    rows = []
    for t in range(data.T):
        idx = np.asarray(data.obs[t]["index"]) if data.obs else np.arange(data.n_cells[t])
        M = np.asarray(res_fit.M[t])
        rows.append(pd.DataFrame({
            "cell": idx,
            "t": t,
            "day": tau[t],
            "replicate": (np.asarray(data.replicate[t]) if data.replicate is not None
                          else np.full(len(idx), "NA")),
            "state": M.argmax(1),
            "membership_max": M.max(1),
        }))
    cells = pd.concat(rows, ignore_index=True)
    if obs is not None:
        for col in ("arm", "cell_sets", "major_cell_sets", "cell_growth_rate"):
            if col in obs.columns:
                cells[col] = obs[col].reindex(cells["cell"]).to_numpy()
    with gzip.open(os.path.join(outdir, "cell_table.csv.gz"), "wt") as fh:
        cells.to_csv(fh, index=False)

    # -- arm composition of the cells actually fitted ------------------------
    if obs is not None and data.obs is not None:
        try:
            arm_composition(data, obs).to_csv(
                os.path.join(outdir, "cell_composition.csv"), index=False)
        except Exception as exc:
            print(f"[save]   (composition skipped: {type(exc).__name__}: {exc})")

    # -- config + summary ----------------------------------------------------
    if result.config is not None:
        cfg_json = json.loads(result.config.to_json())
        # [CRITICAL] The package did NOT learn a representation for this run:
        # X_model was passed in through run_pipeline(Z=...), which skips
        # learn_representation entirely (pipeline.py:75).  Leaving the default
        # RepresentationConfig in place unannotated makes config.json look like a
        # record of a PCA that never ran.
        cfg_json["representation"] = {
            "_inert": True,
            "_note": ("NOT USED. The representation was precomputed and passed via "
                      "run_pipeline(Z=...); learn_representation was never called. "
                      "The fields below are dataclass defaults and describe nothing "
                      "that ran."),
            "_actual": (provenance or {}).get("fingerprint", {}).get("representation"),
            "_defaults": cfg_json.get("representation"),
        }
        with open(os.path.join(outdir, "config.json"), "w") as fh:
            json.dump(cfg_json, fh, indent=2)

    summary = {
        "status": res_fit.status,
        "converged": bool(res_fit.converged),
        "monotone": bool(res_fit.monotone),
        "n_iter": int(res_fit.n_iter),
        "wall_time_s": float(res_fit.wall_time),
        "objective": float(res_fit.objective),
        "grad_norm": float(res_fit.grad_norm),
        "seed": int(res_fit.seed),
        "restarts": res_fit.restarts,
        "terms": res_fit.terms.as_dict(),
        "pipeline": result.summary(),
        "degeneracy": result.degeneracy,
        "dag": dag_info,
        "n_cells": data.n_cells,
        "tau": tau.tolist(),
        "dtau": np.asarray(data.dtau, dtype=float).tolist(),
        "reference_chain": chain_path,
        "mass_conservation": mass_conservation_check(result, t=0),
        "provenance": provenance or provenance_block({}),
        "conditioning": CONDITIONING_STATEMENT,
        "balanced_pilot": BALANCED_PILOT_STATEMENT,
        "replicate_structure": REPLICATE_STATEMENT,
        "representation": REPRESENTATION_STATEMENT,
    }
    if extra:
        summary.update(extra)
    jdump(summary, os.path.join(outdir, "summary.json"))

    if verbose:
        print(f"[save] {outdir}")
        print(f"[save]   status={res_fit.status} n_iter={res_fit.n_iter} "
              f"objective={res_fit.objective:.6f}")
        print(f"[save]   dag: {dag_info}")


def mass_conservation_check(result, t: int = 0) -> dict:
    """P^ref and Phat both carry total mass 1; the KL then reduces to sum p log(p/q).

    Verifying it numerically is the cheap upstream check the handoff asks for
    (PROJECT_HANDOFF.txt s7 "KL SIMPLIFICATION").
    """
    import torch

    model = result.fit.model
    with torch.no_grad():
        M = model.memberships()
        g = model.state_masses(M)
        Ts, _, _, _ = model.induced_transitions(M)
        p_mass = float(model.tt["values"][t].sum())
        t_mass = float(Ts[t].sum())
        # Phat total mass over the FULL grid equals sum_ij a_i b_j M_i W M_j = 1
        W = Ts[t] / (g[t].clamp_min(1e-30)[:, None] * g[t + 1].clamp_min(1e-30)[None, :])
        qa = (model.a[t][:, None] * M[t]).sum(0)
        qb = (model.a[t + 1][:, None] * M[t + 1]).sum(0)
        phat_mass = float(qa @ W @ qb)
    return {"interval": t, "P_ref_mass": p_mass, "T_mass": t_mass,
            "Phat_total_mass": phat_mass,
            "max_abs_deviation_from_1": max(abs(p_mass - 1.0), abs(phat_mass - 1.0))}


# --------------------------------------------------------------------------
# K selection, protocol (b), with a precomputed representation
# --------------------------------------------------------------------------

def select_K_one(
    data: TimeSeriesData,
    cfg: PipelineConfig,
    K: int,
    seed: int = 0,
    n_init_for_stability: int = 2,
    cost_scale: Optional[float] = None,
    verbose: int = 1,
) -> dict:
    """One K of ``selection.select_K``'s protocol (b), for Z-given data.

    ``selection.select_K`` cannot be used here: it calls ``learn_representation``
    and hard-requires ``cfg.representation.method == 'pca'`` (selection.py:233-236),
    which on this dataset is the NaN path described in the module docstring.  This
    is the same procedure with that one step removed -- ``data.X`` already IS the
    frozen representation, so no basis has to be re-applied per half.

    Everything protocol-critical is unchanged and reuses the package:
      * ``split_half_by_replicate(paired=True)`` -- a batch-wise technical
        hold-out: the same ``obs['batch']`` label is held out at every timepoint.
        Note the package docstring calls this "one culture lineage"; the WOT paper
        reports duplicate SAMPLES per timepoint and sampling is destructive, so a
        longitudinal lineage interpretation is not established. The split is still
        the right technical hold-out, but its spread is not a biological SE;
      * lambda_pm forced to 0 -- L_pm must never enter K selection (Degeneracy 3);
      * ``transfer_memberships`` -- assign, never refit, on the held-out half;
      * one global cost scale from the FULL data, reused by both halves.

    Split as a single K so the sweep can run as a slurm array.
    """
    if data.replicate is None:
        raise ValueError("protocol (b) needs replicate labels")

    Z_full = [np.asarray(x, dtype=np.float64) for x in data.X]
    if cost_scale is None:
        cost_scale = float(resolve_cost_scales(
            Z_full, data.tau, cfg.coupling.cost_scale_mode)[0])

    halves = split_half_by_replicate(data, seed=seed, paired=True)
    Zs, chains = [], []
    for h, half in enumerate(halves):
        Zh = [np.asarray(x, dtype=np.float64) for x in half.X]
        Zs.append(Zh)
        chains.append(build_reference_chain(
            Zh, half.tau, cfg.coupling,
            cost_scales=[cost_scale] * (len(Zh) - 1), verbose=max(0, verbose - 1)))
        if verbose:
            print(f"[K={K}] half {h}: n={[len(z) for z in Zh]} "
                  f"feasible={chains[h].feasible} kappas={chains[h].kappas}")

    mcfg = dc_replace(cfg.model, K=int(K), lambda_plus=0.0, lambda_minus=0.0)
    ocfg = dc_replace(cfg.optim, n_init=n_init_for_stability, seed=seed,
                      verbose=max(0, verbose - 1))

    ho_c, ho_x, tr_c, tr_x, gmins, keffs, aris, statuses = [], [], [], [], [], [], [], []
    for src in (0, 1):
        tgt = 1 - src
        res = fit_states(chains[src], Zs[src], mcfg, ocfg)
        statuses.append(res.status)
        tr_c.append(res.terms.compress)
        tr_x.append(res.terms.expression)
        gmins.append(float(np.min(res.terms.g_min)))
        keffs.append(float(np.mean(res.terms.k_eff)))
        aris.append(float(np.mean([r.get("mean_ari_to_others", np.nan)
                                   for r in res.restarts]))
                    if len(res.restarts) > 1 else float("nan"))

        import torch
        with torch.no_grad():
            M = res.model.memberships()
            g = res.model.state_masses(M)
            mu = [x.cpu().numpy() for x in res.model.expression_prototypes(M, g)]
        U_tgt = transfer_memberships(Zs[tgt], mu)
        ev_cfg = dc_replace(cfg.model, K=int(K), lambda_plus=0.0, lambda_minus=0.0)
        model_B = CoarseGrainModel(chains[tgt], Zs[tgt], ev_cfg, U_init=U_tgt)
        _, terms = model_B.objective()
        ho_c.append(terms.compress)
        ho_x.append(terms.expression)
        if verbose:
            print(f"[K={K}] {src}->{tgt}: train_compress={res.terms.compress:.6f} "
                  f"heldout_compress={terms.compress:.6f} status={res.status}")

    sd = float(np.std(ho_c, ddof=1)) if len(ho_c) > 1 else float("nan")
    se = sd / np.sqrt(len(ho_c)) if np.isfinite(sd) else float("nan")
    with np.errstate(invalid="ignore"):
        init_ari = (float(np.nanmean(aris))
                    if any(np.isfinite(x) for x in aris) else float("nan"))
    return dict(
        K=int(K),
        heldout_compress=float(np.mean(ho_c)),
        heldout_expression=float(np.mean(ho_x)),
        train_compress=float(np.mean(tr_c)),
        train_expression=float(np.mean(tr_x)),
        heldout_compress_sd=sd,
        heldout_se=se,
        min_state_mass=float(np.min(gmins)),
        k_eff=float(np.mean(keffs)),
        init_ari=init_ari,
        statuses=statuses,
        all_converged=all(st == "converged" for st in statuses),
        cost_scale=cost_scale,
        half_sizes=[[len(z) for z in Zh] for Zh in Zs],
        chains_feasible=[bool(c.feasible) for c in chains],
    )


def assemble_k_result(rows: List[dict], epsilon: float,
                      notes: Optional[dict] = None) -> KSelectionResult:
    """Wrap per-K records into the package's ``KSelectionResult`` so ``recommend()``
    -- the decided one-SE-style rule with its rejections -- is reused verbatim."""
    rows = sorted(rows, key=lambda r: r["K"])
    # keep per_K tabular: provenance lives once in the JSON, not once per row
    rows = [{k: v for k, v in r.items() if k not in ("provenance", "_file")}
            for r in rows]
    out = KSelectionResult(
        Ks=[r["K"] for r in rows],
        heldout_compress=[r["heldout_compress"] for r in rows],
        heldout_expression=[r["heldout_expression"] for r in rows],
        train_compress=[r["train_compress"] for r in rows],
        train_expression=[r["train_expression"] for r in rows],
        min_state_mass=[r["min_state_mass"] for r in rows],
        k_eff=[r["k_eff"] for r in rows],
        init_ari=[r["init_ari"] for r in rows],
        heldout_se=[r["heldout_se"] for r in rows],
        all_converged=[r["all_converged"] for r in rows],
        statuses=[r["statuses"] for r in rows],
        per_K=rows,
    )
    out.notes = {
        "epsilon": epsilon,
        "lambda_pm": "0 (excluded from K selection by construction, Degeneracy 3)",
        "representation": "precomputed obsm['X_model'] (30 PCs), never refit",
        "split": ("paired split_half_by_replicate on obs['batch'] -- a BATCH-WISE "
                  "TECHNICAL hold-out of duplicate samples, not a tracked culture "
                  "lineage"),
        "replicate_structure": REPLICATE_STATEMENT,
        "heldout_se_meaning": ("fold-direction spread across the two split "
                               "directions of a technical batch split; NOT a "
                               "sampling or biological standard error"),
    }
    if notes:
        out.notes.update(notes)
    return out
