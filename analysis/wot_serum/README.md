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
`obs['batch']` ∈ {1,2} is the duplicate sample; `obs['arm']` ∈ {shared, serum}.

Two properties shape the code:

- **Irregular Δτ** — 0.5 d through day 8, then 0.25 d over days 8→9, then 0.5 d to
  day 18. The cost is `‖z_i−z_j‖²/Δτ_t`, so real day values are used and
  `cost_scale_mode="global"` is kept (a per-interval scale would cancel Δτ out).
- **The serum arm is enforced, not assumed.** `load_serum()` keeps `arm == "shared"`
  through day 8 and `arm == "serum"` after it, asserts that no `2i` cell survives, and
  refuses a file with no `arm` column. A day-only filter is not enough: the CLI accepts
  `--h5ad`, and the parent file (`wot_balanced_shared_pca.h5ad`) carries both arms over
  the same day range. The surviving day × batch × arm counts are printed and saved as
  `cell_composition.csv`.

## What this run does and does not estimate

Four statements are written into every `summary.json`. They are different claims.

**Conditioning on the intervention.** This analysis conditions on the serum
intervention: shared cells through day 8 are followed only by serum-cultured cells
after day 8. It estimates a *serum-conditional trajectory* and does not estimate
allocation or transitions between the serum and 2i interventions. Serum and 2i are
experimental interventions applied at day 8, not stochastic fates with an observed mass
split — so there is no "true" day-8 allocation that balanced transport could be getting
wrong. Edges across day 8 → 8.25 describe the serum-conditional continuation, not a
developmental bifurcation.

**This is a balanced pilot, not a WOT reproduction.** Marginals are uniform over
sampled cells at each timepoint; proliferation and death are **not** modelled, and
`obs['cell_growth_rate']` is present but deliberately unused. Schiebinger et al. (2019)
used growth-aware *unbalanced* transport, so these couplings are not comparable to
theirs and transported mass must not be read as a population abundance. This — not the
serum/2i split — is the genuine limitation of the balanced formulation. Unbalanced
transport is a documented later extension, gated on validating abundance assumptions.

**The representation is transductive across arms.** `analysis/wot_data_check.ipynb`
selects HVGs and fits scaling and PCA on the balanced union of shared + serum + 2i
cells, then extracts the serum subset. `obsm['X_model']` is therefore **not** a
serum-only basis: it saw cells this run excludes. That is retained deliberately for
cross-arm comparability, but it must be stated, and a serum-only PCA sensitivity run is
the check (see *Sensitivity runs*). Note also that the package does **not** learn a
representation during these fits — `X_model` is passed in precomputed via
`run_pipeline(Z=...)` — so `config.json` marks its `representation` block `_inert` and
records the actual representation fingerprint next to it.

**`batch` 1/2 is a technical hold-out, not a biological replicate.** The WOT paper
reports duplicate *samples* collected at each timepoint; sampling is destructive, so
batch 1 is not established to be one longitudinal culture lineage followed across days.
The split is still the right batch-wise technical hold-out for K selection, but the
spread across its two directions is fold-direction variability of a technical split,
**not** a sampling or biological standard error. Effective biological replication is
n = 1 (a single female embryo).

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

### Epsilon conditioning (a warning, NOT a lower bound)

Measured on the full serum series: the global cost scale is **581.4**, and the
normalised max cost per interval ranges 1.02 (day ~3) to 32.5 (day ~17) — the PCA
geometry expands ~25x across the time course while the cost scale is a single global
number, deliberately, so that Δτ survives.

`max(C)/eps` past ~708 in float64 means the highest-cost entries of `exp(-C/eps)` fall
below the smallest representable normal and carry no weight. **That does not make the
balanced problem infeasible, and it does not put a floor under epsilon:**

- the Sinkhorn here is **log-domain** (`sinkhorn.py` says so in its first line), so the
  plan is never formed by evaluating `exp(-C/eps)` directly;
- feasibility is a property of the *surviving* support, not of the cost range. A
  near-diagonal problem stays exactly feasible however extreme the ratio gets — a 2x2
  diagonal cost is solvable at any ratio — because the entries that survive still admit
  the row and column sums.

So `01_eps_scan.py` prints a **conditioning report**, which flags where to expect
ill-conditioning and slow convergence, and additionally tracks the *nearest-neighbour*
cost/eps ratio — the case where marginal-essential support really could be lost.
Whether an epsilon is usable is decided by the measured `marginal_error`, by whether the
solver had stopped improving (`SinkhornResult.stalled` separates "support admits no
balanced plan" from "ran out of iterations"), and by the `feasible` column the scan
records directly. The report is saved under `eps_scan/summary.json` as
`conditioning_report`.

Sinkhorn iteration counts vary enormously with where an interval sits relative to the
scale: at eps=0.05 some intervals converge in ~170 iterations and others need the full
20,000. Do not extrapolate the runtime from one interval, and do not lower
`--sinkhorn-max-iter` casually — it makes the `feasible` column incomparable across
epsilons.

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

# Stage A -- epsilon informativeness (build-order step 1).
sbatch 01_eps_scan.sbatch
#   -> eps_scan/scan_stride{1,2}.{npz,csv}, summary.json  ==> epsilon*

# Stage B -- held-out K selection, protocol (b). Array over K.
sbatch 02_k_sweep.sbatch --epsilon <eps*>
python 02_k_reduce.py                      # auto-finds the k_sweep_<hash> dir
#   -> k_sweep_<hash>/K_*.json, k_sweep.csv, k_selection.json   ==> K*

# Stage D -- lambda_pm sweep at fixed epsilon and K (build-order step 4).
#   RUN THIS BEFORE CALLING ANY DAG A RESULT.
sbatch 04_lambda_sweep.sbatch --epsilon <eps*> --K <K*>
#   -> lambda_sweep_<hash>_K<K>/lambda_sweep.{json,csv}  ==> usable lambda range

# Stage C -- fits at the lambdas the sweep supports.
sbatch 03_fit.sbatch --epsilon <eps*> --K <K*> --lambda-pm 0 <lam...>
#   -> run_<hash>_K<K>/{reference_chain.npz, fit_lam0/, fit_lam<v>/, manifest.json}

# then open 90_inspect.ipynb
```

### Output layout and rerun safety

Every stage writes into a directory keyed by a **configuration hash** covering the
h5ad, a content hash of the representation, the arm policy, day range, stride, cell
subsampling and its seed, epsilon, support/kappa, the shared lambdas, and the optimiser
settings (`csa_wot.FINGERPRINT_FIELDS`). A different configuration therefore lands in a
different directory and cannot overwrite an earlier one.

Within a stage, **every destination is resolved and checked before anything is
written** (`csa_wot.reserve_destinations`). This ordering is the point: writing the
reference chain and only then discovering that `fit_lam0/` exists would leave the
previous run's fits pointing at a chain that had been silently replaced. A failed
collision check cannot modify an existing run — it creates nothing.

### Sensitivity runs

- **serum-only PCA** — the shipped `X_model` is transductive across arms (see above).
  `00_serum_only_pca.py` builds the comparison dataset: identical preprocessing, but
  HVGs, scaling and PCA fitted on the serum-conditional cells **only** (the restriction
  happens *before* the basis is fitted, which is the whole difference from
  `wot_data_check.ipynb`). Then re-run any stage with `--h5ad <that file>`. The
  representation content hash differs, so results land in a separate run directory and
  the reducer refuses to mix them with the main sweep — which is what should happen.
- **Δτ** — Stage A already runs stride 1 and 2; `--stride` on the other stages extends it.

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
| `cell_table.csv.gz` | one row per cell: name, day, batch, state, membership, published annotation |
| `cell_composition.csv` | day x batch x arm counts of the cells actually fitted |
| `summary.json` / `config.json` | status, terms, degeneracy check, mass conservation, the full provenance block (fingerprint + the four statements above); `config.json` marks its `representation` block `_inert` |

The reference chain (`P^ref` sparse triplets, marginals, `Z`) is written once per run
directory as `reference_chain.npz` and shared by every fit in it — ε is frozen, so the
chain is identical across lambdas.

Stage D writes `lambda_sweep.json` / `.csv` with, per lambda: every objective component
(total, compress, expression, plus, minus, and their per-timepoint breakdowns),
convergence status, minimum state mass, `K_eff` per timepoint, initialisation stability
(pairwise restart ARI), and the membership difference from λ=0 (ARI and mean L1). Its
`verdict` block reports the usable range — where memberships move without effective
occupancy collapsing — or says plainly that no such range exists.

## Reading the DAG

Nodes are `(t, k)` with `g_tk > 1e-3`; edges carry `T_t[k,l]` with `A_t[k,l]` forward
and `B_t[l,k]` reverse. Time-layered, hence acyclic by construction.

State index `k` is **time-local** — `k` at `t` has no relation to `k` at `t+1`; the
only linkage is transported mass. There is no cross-timepoint family merging in the
codebase (deliberately unimplemented, `main/README.md:377`).

Edges are transport-implied developmental compatibility, never observed lineage.
