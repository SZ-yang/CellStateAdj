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

# Recorded in every summary.json: the serum arm is a single unbranched series only
# because the 2i cells were dropped.  Balanced OT therefore pushes ALL day-8 mass
# onto serum day-8.25 cells even though roughly half the true descendants went to 2i.
BRANCH_CAVEAT = (
    "Serum arm only: 'shared' cells (day <= 8) plus 'serum' cells (day > 8); the 2i "
    "arm is excluded so the series stays unbranched (PROJECT_HANDOFF.txt s11). "
    "Consequence: transport is balanced, so all day-8 mass is forced onto serum "
    "day-8.25 cells although roughly half the real descendants entered 2i. The "
    "day-8 -> day-8.25 interval is an experimental split, not a developmental one."
)

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

def load_serum(
    h5ad: str = DEFAULT_H5AD,
    day_min: float = 0.0,
    day_max: float = 18.0,
    stride: int = 1,
    n_per_timepoint: Optional[int] = None,
    obsm_key: str = "X_model",
    time_key: str = "day",
    replicate_key: str = "batch",
    seed: int = 0,
    verbose: int = 1,
):
    """Build ``TimeSeriesData`` whose ``X`` IS the frozen representation.

    Returns ``(data, Z, obs)`` where ``obs`` is the full ``adata.obs`` frame,
    indexed by cell name, for later annotation joins.  ``Z`` is ``data.X`` cast to
    float64 -- the same arrays, kept separate only because ``run_pipeline`` wants
    them passed explicitly.
    """
    import anndata as ad

    # Backed: only obs/obsm are read.  The expression matrix and the 'lognorm'
    # layer are ~310 MB each and are never used -- the representation is frozen
    # in obsm[obsm_key] and nothing downstream touches genes.
    adata = ad.read_h5ad(h5ad, backed="r")
    obs_all = adata.obs.copy()
    days_all = np.asarray(obs_all[time_key]).astype(float)
    Xm_all = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    names_all = np.asarray(adata.obs_names)
    reps_all = np.asarray(obs_all[replicate_key]).astype(str)
    if adata.isbacked:
        adata.file.close()

    keep = (days_all >= day_min) & (days_all <= day_max)
    days = days_all[keep]
    Xm = Xm_all[keep]
    names = names_all[keep]
    reps = reps_all[keep]
    obs_all = obs_all.loc[keep]

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
    if verbose:
        print(f"[load] {h5ad}")
        print(f"[load] days {data.tau[0]:g}-{data.tau[-1]:g}  T={data.T}  "
              f"cells={sum(data.n_cells)}  d={Z[0].shape[1]}  stride={stride}")
        print(f"[load] dtau unique: {sorted(set(np.round(data.dtau, 4).tolist()))}")
    return data, Z, obs_all


def make_cfg(
    epsilon: float = 0.05,
    K: int = 20,
    kappa: int = 400,
    support: str = "knn",
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

    ``cost_scale_mode='global'`` is the package default and must stay: the serum
    series has unequal spacing (0.5 d, then 0.25 d over days 8-9, then 0.5 d) and a
    per-interval scale would cancel dtau out of the cost.

    Always builds a fresh config: ``PipelineConfig()`` is used as a shared mutable
    default argument at pipeline.py:46,63.
    """
    cfg = PipelineConfig()
    cfg.coupling.epsilon = float(epsilon)
    cfg.coupling.support = support
    cfg.coupling.kappa = int(kappa)
    cfg.coupling.kappa_max = 1000
    cfg.coupling.cost_scale_mode = "global"
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
             chain_path: Optional[str] = None, verbose: int = 1) -> None:
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

    # -- config + summary ----------------------------------------------------
    if result.config is not None:
        with open(os.path.join(outdir, "config.json"), "w") as fh:
            fh.write(result.config.to_json())

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
        "branch_caveat": BRANCH_CAVEAT,
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
      * ``split_half_by_replicate(paired=True)`` -- one culture lineage per half;
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
        "split": "paired split_half_by_replicate on obs['batch'] (culture C1/C2)",
    }
    if notes:
        out.notes.update(notes)
    return out
