# CellStateAdj on the WOT serum arm, days 0–18

First real-data run of the method. Produces a time-layered DAG of transition-defined
cell states plus every optimised matrix, saved for later analysis.

Judging whether the method *works* is deliberately out of scope here — V±, G±, the
degeneracy verdict and the baselines are computed and saved, but interpreting them is
a separate pass.

## Data

`/dartfs/rc/lab/C/CxQiu/data/joshua/wot_data/wot_serum_first_run_pca.h5ad`

38,817 cells × 2,000 HVGs; 39 timepoints, days 0.0–18.0; 1,000 cells/day (817 at
day 1.5). The frozen representation is `obsm['X_model']` (30 PCs).
`obs['batch']` ∈ {1,2} is the culture replicate; `obs['arm']` ∈ {shared, serum}.

Two properties shape the code:

- **Irregular Δτ** — 0.5 d through day 8, then 0.25 d over days 8→9, then 0.5 d to
  day 18. The cost is `‖z_i−z_j‖²/Δτ_t`, so real day values are used and
  `cost_scale_mode="global"` is kept (a per-interval scale would cancel Δτ out).
- **Serum-only is a branch decision.** `PROJECT_HANDOFF.txt:504` forbids feeding both
  arms as one series. Keeping shared + serum makes it unbranched. The cost: balanced
  OT forces *all* day-8 mass onto serum day-8.25 cells although roughly half the real
  descendants entered 2i. This caveat is written into every `summary.json`.

## Why these scripts exist rather than `main/scripts/run_fit.py`

`cellstateadj.data.from_anndata` reads `adata.X`, and the package has no `obsm`
support anywhere. `adata.X` in this file is the **z-scored** HVG matrix, so
`learn_representation` (which library-size-normalises and `log1p`s) turns it into NaN.
`run_fit.py --h5ad` therefore cannot be used on this dataset.

The supported route is the `Z=` escape hatch: `run_pipeline(data, cfg, Z=...)` skips
representation learning entirely (`pipeline.py:75`). `csa_wot.load_serum` builds
`TimeSeriesData` with `X = Z = X_model`. Everything downstream — the reference chain,
the optimiser, the diagnostics, the DAG edges — is the package, unchanged.

The one other fork is `csa_wot.select_K_one`: `selection.select_K` hard-requires
`cfg.representation.method == "pca"` and refits PCA from `data.X` (`selection.py:233`),
which is the NaN path again. The protocol itself (hold out a whole culture replicate,
transfer the state map, score compression on the held-out half, λ± = 0 throughout) is
identical — only the representation step is dropped, because `X` already *is* the
representation.

## Compute notes

Measured on 3 real intervals at 1,000 cells/timepoint, K=20, λ±=5:

| support | chain build | fit |
|---|---|---|
| `kappa=50` (package default) | **147 s/interval** | — |
| `kappa=400` | 1.0 s/interval | 0.118 s / interval / iteration |
| `dense` | 0.7 s/interval | 0.315 s / interval / iteration |

`kappa=50` is a trap on this data: it admits no balanced plan at n=1000, so
`solve_interval` grows 50→100→200→400 (`reference.py:319`), burning a full
20,000-iteration Sinkhorn solve at each failed step — 150× the time for an identical
answer. `make_cfg` therefore starts at `kappa=400` (`kappa_max=1000`).

Stage A uses `support="dense"` instead: `epsilon_scan` sizes its support once at
`max(epsilons)`, exactly where a kNN support is least likely to be feasible, and dense
never reports infeasible — so support sizing cannot confound the feasibility column
the scan exists to measure.

### The epsilon range is bounded from below by float64, not by taste

Measured on the full serum series: the global cost scale is **581.4**, and the
normalised max cost per interval ranges 1.02 (day ~3) to 32.5 (day ~17) — the PCA
geometry expands ~25x across the time course while the cost scale is a single global
number, deliberately, so that Δτ survives.

`exp(-C/eps)` underflows once `max(C)/eps` passes ~708 in float64
(`reference.py:_underflow_limit`), and no number of Sinkhorn iterations fixes that:

| eps | max(C)/eps | intervals over the limit |
|---|---|---|
| 0.002 | 16262 | 35 / 38 |
| 0.005 | 6505 | 26 / 38 |
| 0.01 | 3252 | 18 / 38 |
| 0.02 | 1626 | 13 / 38 |
| **0.05** | 650 | **0 / 38** |
| 0.1 | 325 | 0 / 38 |
| ≥ 0.2 | ≤ 163 | 0 / 38 |

So any informative window on this series sits at **eps ≥ 0.05**. The scan still runs
the smaller values and records `feasible=0` — that is the honest result, and
`recommend()` counts an unevaluated criterion as failed — but each such cell costs a
full 20,000-iteration solve, which is why Stage A is budgeted 2 days.
`01_eps_scan.py` prints this table before scanning; it is also saved in
`eps_scan/summary.json` under `underflow_report`.

Sinkhorn iteration counts vary enormously with where an interval sits relative to the
scale: at eps=0.05 some intervals converge in ~170 iterations and others need the full
20,000. Do not extrapolate the runtime from one interval.

CPU is sufficient throughout (`standard` partition). For a GPU run pass
`--device auto`; `csa_wot.resolve_device` sets **both** `cfg.coupling.device` and
`cfg.model.device` — `run_fit.py` only sets the latter, silently leaving Sinkhorn on
the CPU.

### Slurm: the account is `qiulab`, and it owns no nodes

Verified with `sacctmgr` / `scontrol`:

- The canonical account string is lowercase **`qiulab`** (`sacctmgr show account`).
  `QiuLab` is also accepted — Slurm matches account names case-insensitively, and
  `sbatch --test-only` does reject genuinely invalid accounts — but `qiulab` is what
  the accounting database stores, so that is what the scripts use.
- **`qiulab` has no dedicated CPU partition.** `standard` is `AllowAccounts=ALL`,
  45 nodes / 2880 cores, shared campus-wide. Several other labs *do* own partitions
  (`hautier_high`, `preempt_lsong`, `l40s_indrani`, `preempt_wenlin`, `v100_vaickus`);
  there is nothing qiu-specific. What the account provides is a fairshare allocation:
  `NormShares` 0.167 with `FairShare` 1.000 (no usage yet), i.e. currently top
  priority — but the queue still schedules against everyone else. `--test-only`
  estimated a ~1 day wait for a 32-core job, so submit early.
- This user's other associations are `frostlab`, `lsonglab` (with the
  `lsonglab_gpu` QoS), `nccc`, and `free`.

### Mail

All three jobs carry `--mail-type=END,FAIL,TIME_LIMIT` to
`shizhao.yang.gr@dartmouth.edu`. For the Stage B array this sends **one** mail for the
array as a whole; add `ARRAY_TASKS` to `--mail-type` if you want one per K (8 mails).

`logs/` must exist **before** submission — Slurm opens the `--output` file before the
batch script runs, so the `mkdir -p logs` inside `_common.sh` is only a backstop. The
directory is kept in the tree with a `.gitkeep`.

Submit from *this* directory: the `--output` paths and the `csa_wot` import are both
relative to `$SLURM_SUBMIT_DIR`. `_common.sh` checks for `csa_wot.py` and exits 2 with
an explanation if you submit from somewhere else.

## Running it

Results go to `/dartfs/rc/lab/C/CxQiu/data/joshua/CellStateAdj/wot_serum/`.

```bash
cd CellStateAdj/analysis/wot_serum

# Stage A -- epsilon informativeness (build-order step 1). ~20 min.
sbatch 01_eps_scan.sbatch
#   -> eps_scan/scan_stride{1,2}.{npz,csv}, summary.json  ==> epsilon*

# Stage B -- held-out K selection, protocol (b). Array over K.
sbatch 02_k_sweep.sbatch --epsilon <eps*>
python 02_k_reduce.py    --epsilon <eps*>
#   -> k_sweep/K_*.json, k_sweep.csv, k_selection.json    ==> K*

# Stage C -- the fit. Both lambda_pm = 0 and lambda_pm = 5 against one frozen chain.
sbatch 03_fit.sbatch --epsilon <eps*> --K <K*>
#   -> fit_lam0/, fit_main/, reference_chain_eps<e>.npz, manifest.json

# then open 90_inspect.ipynb
```

Smoke run first (a few minutes, 9 timepoints):

```bash
python 03_fit.py --day-max 4 --K 6 --epsilon 0.05 --max-iter 20 --n-init 1 \
    --out /tmp/smoke --device cpu
```

## What gets saved

`run_fit.py` saves only `M`, `T`, `g`, `V±` and the DAG edges, and discards even those
when a fit does not converge (`run_fit.py:185`). Since the point of this run is to
have the matrices, `csa_wot.save_all` writes everything, converged or not, and the
status travels with the files.

Per fit directory:

| file | contents |
|---|---|
| `memberships.npz` | `M_t`, hard `labels_t`, expression prototypes `mu_t` |
| `transitions.npz` | `T_t = M_tᵀ P^ref_t M_{t+1}`, `A_t`, `B_t`, `g_t` |
| `fingerprints.npz` | per-cell `f⁺_t`, `f⁻_t` and their KL-barycentre prototypes `φ±` |
| `diagnostics.npz` | `V±`, `G±`, `n_child`, `n_parent`, `active` |
| `dag_edges.json` / `dag_nodes.csv` / `dag.graphml` | the DAG |
| `history.csv` | per-iteration objective, components, `K_eff`, `g` range, floor fraction, `‖dM‖` |
| `cell_table.csv.gz` | one row per cell: name, day, replicate, state, membership, published annotation |
| `summary.json` / `config.json` | status, terms, degeneracy check, mass conservation, the branch caveat |

The reference chain (`P^ref` sparse triplets, marginals, `Z`) is written once at the
top level as `reference_chain_eps<e>.npz` and shared by both fits — ε is frozen, so
the chain is identical.

## Reading the DAG

Nodes are `(t, k)` with `g_tk > 1e-3`; edges carry `T_t[k,l]` with `A_t[k,l]` forward
and `B_t[l,k]` reverse. Time-layered, hence acyclic by construction.

State index `k` is **time-local** — `k` at `t` has no relation to `k` at `t+1`; the
only linkage is transported mass. There is no cross-timepoint family merging in the
codebase (deliberately unimplemented, `main/README.md:377`).

Edges are transport-implied developmental compatibility, never observed lineage.
