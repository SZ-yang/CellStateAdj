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
| `selection.py` | 1.20 | **K selection by held-out compression, protocol (b)** |
| `stability.py` | 1.21 | split-half refits, joint alignment, node/edge support |
| `diagnostics.py` | 1.16–1.18 | `V±`, geometric null `G±`, branching/merging, degeneracy checks |
| `simulate.py` | — | birth–death SDE + NB observation, eleven scenarios |
| `baselines.py` | — | expression clustering → same induced map; spectral stand-in |
| `evaluate.py` | — | ARI, matched transition-map recovery |
| `pipeline.py` | Alg. 1 | glue, λ sweep, K selection |

## Two invariants that are easy to break

**The cost scale must be shared across intervals.** Eq. 1 is
`C = ‖Δz‖² / Δτ`. Normalizing each interval by *its own* median gives
`‖Δz‖² / median(‖Δz‖²)` — Δτ cancels **exactly**, and the relative weighting
between a short and a long gap is gone. That silently disables the Δτ study and
mishandles WOT's uneven spacing (6h between days 8–9 against 12h elsewhere).
`CouplingConfig.cost_scale_mode` defaults to `"global"`; `"per_interval"` exists
only as a documented sensitivity analysis and is not valid for Δτ work.

**An infeasible coupling must never reach the model.** With a restricted
support a balanced plan need not exist. If the marginals do not hold then
`T_t 1 = M_tᵀ·rowsums(P) ≠ M_tᵀ a_t = g_t`, so `A_t` stops being row-stochastic
and every transition number — `A`, `B`, `N_child`, `N_parent`, the DAG edges —
is wrong with no other symptom. Worse, it is **invisible at initialization**:
with uniform memberships both sides collapse to `1/K`, so a smoke test will not
catch it. `on_infeasible` defaults to `"raise"`, `CoarseGrainModel` refuses such
a chain, and `feasibility_tol` is the single threshold used everywhere.

Two failures are distinguished, because they need opposite fixes:
`InfeasibleCouplingError` (the support admits no balanced plan — grow κ or go
dense) versus `SinkhornConvergenceError` (the support is fine — raise
`max_iter`, or raise ε). A **dense** support is never reported as infeasible:
it always admits a balanced plan, so a miss there is convergence or precision.

`SinkhornConvergenceError` also diagnoses ε being too small for an interval's
cost *range*: once `max(C)/ε` approaches ~708 (float64), `exp(-C/ε)` underflows
and the marginals cannot be met at any iteration count. A shared cost scale
makes genuinely wider-transport intervals more expensive — correct behaviour,
and such an interval is telling you it needs more entropy. This surfaces on the
`missing_timepoints` scenario at ε=0.05 (`max(C)/ε ≈ 570`), which converges
normally at ε=0.1. `run_simulation_study.py` and the ε-scan record such a
failure per scenario / per ε and carry on rather than aborting the sweep.

Tolerances are calibrated to the harm, not to a round number. The marginal
error propagates to the transition map as
`|A_t(k,·).sum() − 1| ≤ marginal_error / g_tk`, so it is **amplified for
low-mass states**. `feasibility_tol` defaults to `1e-5` (≈1e-3 row-sum error
for a state of mass 1e-2); the genuinely broken case that motivates the guard
had a marginal error of `0.5`. Sinkhorn stops at
`min(tol, feasibility_tol/10)` rather than chasing digits that buy nothing.

## Decided protocols

**K selection — protocol (b).** Hold out whole duplicate samples and score the
*transferred* state map (`selection.select_K`). Training compression falls
monotonically with K, so it cannot select anything; the held-out curve does the
work and does turn over. `pipeline.k_diagnostic_sweep` reports the training
curve only and refuses to recommend a K. Do not switch protocols silently.

Spec 1.20 lists five criteria, not one, so `recommend()` applies the others as
**rejections** before scoring: a fit that stopped at `max_iter` or
`line_search_failed` never met its tolerances and cannot win, a K whose smallest
state is empty is not really that K, and initialisation stability can set a
floor. The threshold comes from the measured spread, not a fixed percentage —
but call it a **one-SE-*style* conservative rule**, not a one-SE rule. The
spread is taken across the two split directions A→B and B→A, which are two
complementary evaluations of *the same two cultures*, not two independent
replicate estimates; the number is real fold-direction variability and is the
right thing to be conservative about, but it carries no sampling
interpretation. Give it one only with more independent replicate folds. Note
that a stingy `--max-iter` will reject the larger K for non-convergence and
collapse the recommendation onto the smallest one — the remedy is more
iterations, and the script warns when this happens.

**Replicate pairing.** Paired by default, in two places that must agree:
`epsilon_scan(replicate_paired=True)` holds the same group out at both ends of
an interval, and `split_half_by_replicate(paired=True)` chooses the subset
**once** and reuses it at every timepoint. A per-timepoint draw would put
replicate 0 in half A early and replicate 1 in half A later — the same
experimental unit then appears in both halves, which is not a held-out sample
and silently inflates both the held-out K curve and the replicate support. Set
`paired=False` only if the labels really are independent wells harvested per
timepoint.

Choosing the subset once is only enough when **every timepoint carries the same
label set**. With `t0: A,B,C`, `t1: A,B,D`, `t2: A,B,C`, the half "everything
not in {A,B}" is culture C at t0 and culture D at t1 — one label set on paper,
two different cultures in fact, and neither half is empty, so an emptiness check
waves it through. `split_half_by_replicate(paired=True)` therefore requires the
label sets to agree and raises otherwise; `restrict_to_shared=True`
(`--restrict-to-shared-replicates` on `run_k_selection.py`,
`restrict_to_shared_replicates=` on `select_K` and `split_half_stability`) drops
the non-shared cells and splits on the intersection. That discards data, so it
is opt-in rather than a silent repair. The restriction runs **before** the
representation is learned — `data.restrict_to_shared_replicates` is applied
first, then PCA and the global cost scale — because both are fit on whatever
cells they are handed, and restricting afterwards would leave the discarded
replicates shaping the space the halves are then compared in. The scan applies the same rule per
interval: groups are drawn from the labels shared by both ends, and if fewer
than two are shared it leaves `stability_resample` unevaluated instead of
falling back to an unrelated target group.

**Replicate subsets are enumerated, not drawn.** When the resampling unit is
the replicate, the scan uses each replicate group exactly once — `n_resample`
does not apply, and the number of subsets is the number of groups. Drawing
groups at random duplicates them: with two replicates, two independent draws of
one group coincide about half the time, the two "resampled" datasets are then
bit-identical, and the agreement is exactly 1.0 for no reason at all. That
happened on 7 of 12 seeds of the same data. Replicate subsets are also disjoint,
so there is no shared cell for a pairwise comparison; each subset is instead
scored against the full coupling on its own cells and the scores averaged over
**all** subsets — the previous fallback scored the first subset alone and
discarded the rest. `stability_is_vs_full` records which comparison ran (1.0 =
against the full coupling, 0.0 = pairwise between overlapping subsets) and
`n_resample_subsets` how many were evaluated. Random cell subsets
(`resample_fraction`, no replicate labels) overlap by construction and still use
the stronger pairwise route.

**Every resampled and cost-perturbed coupling is feasibility-checked.** Each
resampled support grows its own κ (at the largest ε, since feasibility is
combinatorial), and every resampled and perturbed plan must clear the same
`feasibility_tol` as the main one. A subset has fewer cells, so the κ that made
the full support feasible can leave it with no balanced plan — and an infeasible
plan still yields fingerprints, which still agree with each other, which still
scores as high stability. Ungated, a pair of completely invalid plans scored
`stability_cost = 0.9998` and could push an ε into the admissible window; they
now yield `NaN`, which fails admissibility. `n_resample_subsets`,
`n_resample_feasible` and `marginal_error_perturbed` record what happened.

**A run directory is never reused silently.** `run_fit.py` refuses a non-empty
`--out` unless `--overwrite` is given, and `--overwrite` clears the artifacts it
knows how to write before starting. Without this the non-convergence gate is
defeated by history: it writes only `summary.json`, so a reused directory keeps
the *previous* run's `memberships.npz` and `dag_edges.json` sitting there
unmarked and looking valid while the console says only `summary.json` was
written. In the other direction, a converged rerun would inherit a stale
`NONCONVERGED.txt` and disown a perfectly good fit.

**Convergence reaches the export paths.** `run_fit.py` refuses to write
memberships, diagnostics, `events.csv` or `dag_edges.json` from a fit whose
status is not `converged`; it writes `summary.json` with the status and exits
non-zero. `--allow-nonconverged` writes everything but stamps the directory with
`NONCONVERGED.txt`. `split_half_stability` raises if any half fails to converge
(`require_converged=False` to inspect, which marks the report `[UNRELIABLE]`),
and `edge_support` flags non-converged fits in its notes — support counts votes
across fits, so a fit stopped at `max_iter` votes from wherever the optimiser
happened to be.

With `--stride`, `run_fit.py` no longer scores a strided fit against unstrided
ground truth. State labels are strided alongside the data; transition recovery
is skipped, because the lineage flow across a skipped timepoint is not a product
of the recorded per-interval joints.

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
- **No `V − G` "excess" column.** Eqs. 29 and 31 compare the fingerprints
  against *different* references (the KL barycentre `φ_k` versus the smooth
  predictor `f̂(z)`), so there is no `V = V_geometry + V_residual` decomposition
  and the difference has no guaranteed sign — in practice it is routinely
  negative. Report the (V, G) **quadrant** instead: `geometry_plus` /
  `geometry_minus` in the event table.
- **State alignment must use transition role, not expression alone.** Two
  states can share an expression prototype and differ entirely in temporal
  role — the exact case this method exists to resolve — so
  `stability.align_states_joint` is the default for stability reporting.
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
- `FitResult.status` is `converged` / `line_search_failed` / `max_iter`. A
  stuck line search is **not** convergence: the tolerances were not met, the
  iterate simply could not move. `grad_norm` is recorded so a stuck fit can be
  told from a flat one. In the block-coordinate optimiser the failure check runs
  **before** the convergence check and compares against the blocks actually
  attempted — when every block refuses every step the objective and memberships
  are exactly unchanged, so a convergence-first test would report a completely
  stuck optimiser as converged.
- The ε-scan sizes its support at the **largest** ε on the grid. Whether a
  support admits a balanced plan is combinatorial and ε-independent; what varies
  with ε is conditioning. Sizing at the smallest ε conflates the two, so a
  numerical failure there would abandon κ growth and leave every ε scored on an
  under-sized support — reporting "no informative window" when a good one
  exists.
- `EpsilonScanResult.recommend()` splits the admissible set into **contiguous
  runs** and picks ε\* from inside the best run. Treating (first, last) as one
  window can return a value that failed every criterion.
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
| `run_k_selection.py` | 5 | held-out K selection, protocol (b) |

For the Δτ study, `run_epsilon_scan.py` subsamples cells once, fits the PCA
once, and computes the cost scale once — all on the **native** series — then
each stride merely selects timepoints from that fixed setup. Recomputing any of
them per stride would make a stride-1 vs stride-4 comparison confound four
changes at once (spacing, PCA basis, cost normalisation, and which cells were
drawn) instead of isolating spacing.

## Status of this version

133 unit tests pass (`python -m pytest -q`), including `tests/test_regressions.py`,
which covers four rounds of review findings (nineteen in total, plus two issues
found while fixing them) that the original suite missed because it only ever
exercised feasible, balanced couplings on equally-spaced timepoints with
converged fits, complete replicate labels, more than two resampling units, and
clean output directories.

Verified end-to-end on simulated data only: the ε-scan reproduces both
degeneracies, the CMI identity matches a brute-force `I(I_t;Z_{t+1}|Z_t)`,
Degeneracy 1 is demonstrated numerically, the λ sweep reproduces Degeneracy 3
(K_eff collapses as λ grows), and held-out compression turns over in K while
training compression does not. It has **not** been run on the WOT data.

Any Δτ-sensitive result generated before the shared-cost-scale fix is invalid
and must be regenerated. The identity tests are unaffected — they do not depend
on the cost scale.
