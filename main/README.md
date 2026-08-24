# cellstateadj — transition-defined cell states from time-resolved scRNA-seq

Reference implementation of the bidirectional temporal coarse-graining method in
`../plan_8-23.pdf`. Design rationale, dataset facts, and the reasoning behind
several non-obvious choices are in `../PROJECT_HANDOFF.txt`; this file covers
what the code does and how to run it.

## What the method is

Given count matrices `X^(1)…X^(T)` at ordered times `τ_1 < … < τ_T`
(destructive sampling — different cells at each time), build a time-layered DAG
of cell states. A state is defined by its role in the transition system: a group
of cells with similar probable predecessors and similar probable descendants,
not merely similar expression.

Two explicit stages:

1. **Build a fixed transport chain.** Learn a shared expression representation
   and freeze it; form adjacent-time costs `C_ij = ‖z_i − z_j‖² / Δτ`; solve
   entropic OT; select `ε*` from the informativeness criterion; freeze
   `P^ref_t`.
2. **Jointly coarse-grain the whole chain.** Learn soft memberships
   `M_t = softmax(U_t)` minimising

   ```
   L = λ_compress·L_compress + λ_x·L_expression + λ_−·L_− + λ_+·L_+
   ```

   with the state transition map *induced*, never fitted: `T_t = M_tᵀ P^ref_t M_{t+1}`.

The key identity, exact and unit-tested (`tests/test_identities.py`):

```
L_+,t = I(I_t ; Z_{t+1} | Z_t)        L_−,t = I(I_t ; Z_{t−1} | Z_t)
```

After conditioning on the state, individual cell identity should carry little
extra information about predecessor or descendant state. This holds *because*
the prototypes are KL barycentres — changing the prototype definition breaks it.

## Install

Needs `numpy`, `torch`, `scipy`, `scikit-learn`; optional `scanpy`/`anndata`
(for `.h5ad` input and the Leiden baseline) and `pandas` (for CSV export).
Development was done in the `LCL` conda env:

```bash
conda run -n LCL python -m pytest tests -q      # 59 tests
```

## Build order

Do **not** start at the top of Algorithm 1. Build what can cheaply kill the
project first.

**Step 1 — the ε-informativeness curve.** No memberships, no objective, no
state learning. Answers whether an informative *and* stable window in ε exists
at all at this sampling density.

```bash
python scripts/run_epsilon_scan.py --sim branching --stride 1 --stride 2 --dense
python scripts/run_epsilon_scan.py --h5ad wot_phase1.h5ad --time-key day \
    --replicate-key sample --n-per-timepoint 1500 --stride 1 --stride 4
```

Run it at both 12h and 48h spacing — that doubles as the first data point of
the Δτ study.

**Step 2 — the simulator.** The only place ground truth exists. All eleven
required scenarios are in `cellstateadj/simulate.py`.

**Step 3 — the optimiser with `λ_± = 0`.** Compression + expression coherence
only: a well-behaved clustering problem and the internal baseline.

```bash
python scripts/run_fit.py --sim branching --K 6 --epsilon 0.05 --dense \
    --compare-baseline
```

**Step 4 — turn on the fingerprint terms** and run the non-degeneracy check
(on by default in `run_fit.py`, and printed by `run_lambda_sweep.py`). Do not
proceed if it fails.

```bash
python scripts/run_simulation_study.py --lambda-plus 5 --lambda-minus 5
python scripts/run_lambda_sweep.py --sim similar_expression_different_role \
    --K 6 --epsilon 0.05 --dense --lambdas 0 1 5 20 100
```

The sweep prints `K_eff` at every λ and warns when it collapses — that is
Degeneracy 3 becoming active and marks the edge of the usable regime.

Steps 5–9 (K selection, diagnostics, stability, Δτ study) use
`pipeline.k_selection`, `pipeline.lambda_sweep`,
`diagnostics.compute_diagnostics`, `stability.split_half_stability`,
`stability.epsilon_sensitivity`, and `data.subsample_timepoints`.

## Module map

| Module | Spec section | Contents |
|---|---|---|
| `config.py` | — | every hyperparameter, serialisable |
| `data.py` | — | `TimeSeriesData`, Δτ subsampling, replicate split-half |
| `representation.py` | 1.2 | frozen shared PCA; time is *not* regressed out |
| `cost.py` | 1.2, 1.5 | Eq. 1 costs, symmetrised kNN support |
| `sinkhorn.py` | 1.3 | log-domain Sinkhorn, dense and sparse |
| `reference.py` | 1.3–1.5 | the frozen `P^ref` chain, κ-growth on infeasibility |
| `informativeness.py` | 1.4 | **Step 1**: `I_cell`, fingerprint spread, stability |
| `model.py` | 1.6–1.14 | memberships, induced `T_t`, `P̂`, fingerprints, all four losses |
| `optimize.py` | 1.19 | L-BFGS/steepest + backtracking; block-coordinate + damping |
| `diagnostics.py` | 1.16–1.18 | `V±`, geometric null `G±`, branching/merging, degeneracy checks |
| `simulate.py` | — | birth–death SDE + NB observation, eleven scenarios |
| `stability.py` | 1.21 | split-half refits, state alignment, edge support (Eq. 36) |
| `baselines.py` | — | expression clustering → same induced map; spectral stand-in |
| `evaluate.py` | — | ARI, matched transition-map recovery |
| `pipeline.py` | Alg. 1 | glue, λ sweep, K selection |

## Design choices that must not be "simplified"

These are load-bearing. Each has a test that fails if it is undone.

- **`P^ref` is estimated independently and frozen.** If fingerprints came from
  the model's own `P̂`, then `f+_ti = M_t(i,:) B_t` with `B_t` independent of
  `i` — the `a_ti` in row `i` of `Q_t` cancels exactly on division. Cells
  hard-assigned to the same state would have *identical* fingerprints for any
  assignment, and `L_±` would measure nothing.
  `diagnostics.degeneracy_check` demonstrates this empirically; the ablation is
  a reportable finding, not just a bug fix. **Report the right number**: the
  sharp statement is `phat_rank_one_residual ≈ 0` (the fingerprints factor
  exactly through `M`). The `phat_within_state_spread` summary is nonzero for
  *soft* memberships, because `f+ = M_t(i,:) B_t` still varies with `i` through
  the membership row — it vanishes only in the hard limit. Quote the ratio
  against `ref_within_state_spread`, not "zero".
- **`P^ref` never depends on `M`** in any way.
- **Prototypes stay weighted means / KL barycentres.** The CMI identity dies
  otherwise.
- **`T_t` is induced, not a free parameter** with an alignment loss.
- **`L_±` is excluded from K selection.** It systematically favours coarser
  neighbouring state spaces and is exactly zero at `K = 1` (Degeneracy 3).
- **No ℓ1 penalty on `Q_t` or `T_t`** — their total mass is fixed, so their ℓ1
  norms are constant.
- **No outgoing-entropy penalty** to encourage sparse branching: a clean
  bifurcation has `H_out = log 2` and would be penalised, suppressing exactly
  the events we want to find.
- **No mode counting on `A_t(k,:)`** — state indices carry no topology. Use
  `N_child = exp(H(A_t(k,·)))`.
- **Balanced OT by default.** Uncalibrated cell counts are not population
  abundances.

## Instrumented every iteration

`K_eff_t = exp(H(g_t))`, min/max `g_t,k`, the fraction of `P̂` entries at the
numerical floor, mean `V±`, membership change, and each loss component
separately. If `K_eff` collapses as `λ_+` grows you have found the edge of the
usable regime — report it, it is a result.

## Implementation notes

- `L_compress` is evaluated only on `supp(P^ref)`. Both `P^ref` and `P̂` have
  total mass 1, so the generalised-KL `−p+q` terms cancel and KL reduces to
  `Σ p log(p/q)`. There is a unit test asserting the masses match to ~1e-10.
- `P̂` on the support is computed as
  `a_i b_j · (M_t W)[i] · M_{t+1}[j]` with `W = diag(g_t)⁻¹ T_t diag(g_{t+1})⁻¹`,
  costing `n_t K² + |S| K` rather than the naive `|S| K²`.
- `L_+` uses `Σ_k g_k H(φ_k) − Σ_i a_i H(f_i)`, exactly equal to the literal
  double sum (tested) and far cheaper.
- The objective is **self-referential** — `L_+` at `t` depends on `M_{t+1}` —
  so simultaneous updates can increase it. Every proposal must pass an Armijo
  test on the *complete* objective. Report convergence as "to a fixed point"
  unless the line search actually guaranteed descent; `FitResult.monotone`
  records which happened.
- A row-wise kNN graph does not guarantee a feasible balanced coupling. The
  support unions source→target and target→source neighbours, and κ grows until
  Sinkhorn converges. Report κ sensitivity alongside ε sensitivity.

## Interpretive limits (must appear in any writeup)

- Snapshot data do not uniquely identify individual trajectories. OT picks a
  coupling by minimising a chosen cost; that assumption adds no information.
- Cells with identical measured profiles cannot be assigned distinct futures.
- Distinguishing genuine fate priming from transport geometry is not possible
  from expression snapshots alone — hence `G±`, and hence the "necessary but
  not sufficient" language.
- Inferred merging is population-level convergence, not proof that individual
  cells from distinct lineages physically merged.
- Transient states occurring entirely between sampled times cannot be recovered.
- Edges are "transport-implied developmental compatibility", never "observed
  lineage".
- For the WOT data, replicates are culture/well-level from one embryo. Label
  stability results "culture-replicate stability", never "biological
  uncertainty", and do not build a large-R biological bootstrap.

## Open items carried from the handoff

1. **Held-out compression protocol for K selection is undecided** —
   (a) refit `P^ref` on a reduced cell set and evaluate on held-out pairs, or
   (b) hold out whole duplicate samples and evaluate the transferred state map.
   `pipeline.k_selection` currently reports *training* compression only and
   says so in its docstring. Pick one, document it, don't switch silently.
2. Whether raw reads exist for WOT (spliced/unspliced → velocity-informed cost).
3. Whether the informative ε window exists at 12h spacing on the real data.
4. Whether the day 5–8 population separates under transition-defined states —
   the central hypothesis test.

## Not implemented on purpose

Amortised assignment, jointly-learned embeddings, unbalanced transport,
continuous dynamics, time-varying `K_t`, split-merge proposals, and the optional
global-family postprocessing (spec 1.22). All are documented later extensions.

## Scripts

| Script | Step | What it does |
|---|---|---|
| `run_epsilon_scan.py` | 1 | ε-informativeness curve at one or more Δτ strides |
| `run_simulation_study.py` | 2, 4 | all eleven scenarios vs expression clustering |
| `run_fit.py` | 3, 4 | fit states, diagnostics, DAG export, optional baseline |
| `run_lambda_sweep.py` | 4, 5 | λ± sweep with K_eff coarsening watch |

## Status of this version

Written and tested; 66 unit tests pass (`python -m pytest -q`). Verified
end-to-end on simulated data only — the ε-scan reproduces both degeneracies,
the CMI identity matches a brute-force computation of `I(I_t;Z_{t+1}|Z_t)`,
Degeneracy 1 is demonstrated numerically, and the line search is monotone with
the fingerprint terms on. It has **not** been run on the WOT data.
