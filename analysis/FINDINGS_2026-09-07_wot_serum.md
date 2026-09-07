# CellStateAdj on the WOT serum arm — results, problems, and fixes

**Date:** 2026-09-07
**Data:** `/dartfs/rc/lab/C/CxQiu/data/joshua/wot_data/wot_serum_first_run_pca.h5ad`
**Runners:** `analysis/wot_serum/` · **Results:** `/dartfs/rc/lab/C/CxQiu/data/joshua/CellStateAdj/wot_serum/`

First real-data run of the method. Prior to this, the implementation had only ever
run on simulated data.

---

## TL;DR

1. **Stage A (ε) succeeded and produced a real scientific finding.** The serum phase
   (days 8.25–18) has an informative *and* stable window at **ε ≈ 0.2–0.5**. Phase 1
   (days 0–8) has **no** such window at 12 h or 24 h spacing, and that is not fixable
   by rescaling — see [Finding A2](#a2-phase-1-has-no-informative-stable-window).
2. **Stages B–D were first run in a mis-configured objective.** `λ_x = 1` made
   expression coherence **90 %** of the objective, reducing the model to k-means and
   making the fingerprint terms provably inert.
3. **Rebalancing (λ_x × λ_pm grids, 24 fits) fixed the inertness but not the
   method.** With transport at 76–79 % of the objective, `λ_pm` becomes active — and
   goes straight from inert to **degenerate**, with no usable window at either K=8 or
   K=20. See [the grid results](#stage-d--the-λ_x--λ_pm-grids-the-decisive-test).
4. **Degeneracy 3 is now empirically demonstrated**, not just predicted: `L±` is
   driven from ~3.0 to ~0.0001 while compression degrades 3× and states are
   annihilated (`min_g` → 5.8e-09).
5. **Fits are irreproducible at every weighting tested.** `init_ari` across all 20
   converged cells: **max 0.249, median 0.000**. Different seeds give unrelated state
   maps regardless of how the terms are balanced. **This kills my earlier hypothesis
   that the k-means dominance explained the instability** — it does not.
6. **Conclusion: the current objective does not work on this data, and the problem is
   not tuning.** See [alternative approaches](#alternative-approaches-worth-considering).

## What was run

| Stage | Job | Result | Wall |
|---|---|---|---|
| A — ε scan, days 0–18, strides 1 & 2 | 9370477 | COMPLETED | 5 h 27 |
| B — held-out K selection, serum only, 8 K values | 9373598_[0-7] | all COMPLETED | ~10 min each |
| C — fit K=20, λ_pm ∈ {0,5} | 9375095 | **FAILED** at λ_pm=5 | 1 h 15 |
| C — fit K=8, λ_pm ∈ {0,5} | 9375096 | COMPLETED | 42 min |
| D — λ_pm sweep, K=20 | 9375094 | **FAILED** at λ_pm=5 | 1 h 40 |
| D — (λ_x × λ_pm) grid, K=8 | 9377375 | COMPLETED, 12/12 cells | 4 h 24 |
| D — (λ_x × λ_pm) grid, K=20 | 9377376 | COMPLETED, 8/12 cells (4 crashed) | 3 h 01 |

Scope decision taken during the run: **serum phase only** (days 8.25–18, 22
timepoints, 22 000 cells, all `arm == serum`), on the strength of Finding A2.

---

## Stage A — the ε scan

Global cost scale **581.45**. Every one of the 418 (ε, interval) couplings was
**feasible** (max marginal error 9.8e-06). No ε satisfied all five criteria, because
`stability_resample` peaks at **0.730** against the default 0.8 threshold.

### A1 — the window exists, but only in the serum phase

Splitting the 38 intervals at day 8:

| ε | days 0–8: I_cell_n / stab / n≥0.8 | days 8.25–18: I_cell_n / stab / n≥0.8 |
|---|---|---|
| 0.02 | 0.317 / 0.635 / 8-of-16 | 0.747 / 0.653 / 7-of-22 |
| 0.05 | 0.143 / 0.588 / 7-of-16 | 0.565 / 0.822 / 15-of-22 |
| 0.10 | 0.068 / 0.504 / 5-of-16 | 0.416 / 0.895 / 19-of-22 |
| **0.20** | 0.027 / 0.307 / 2-of-16 | **0.287 / 0.930 / 21-of-22** |
| 0.50 | 0.006 / 0.077 / 0-of-16 | 0.157 / 0.941 / 22-of-22 |

The serum phase shows a textbook informativeness/stability trade-off with a genuine
plateau. **ε\* = 0.2** was chosen from it.

### A2 — phase 1 has no informative, stable window

No ε makes days 0–8 both informative and reproducible. Mechanism, measured: typical
transport cost is **~10× larger late than early** (median normalised cost 0.16 at day
4.5 vs 3.5 at day 17), so under one global cost scale a single ε cannot sit in the
informative regime for both phases.

Two candidate fixes were tested on 5 phase-1 intervals:

| variant | I_cell_n (ε=0.02→0.2) | stability |
|---|---|---|
| global scale, whole series | 0.32 → 0.03 | 0.64 → 0.31 |
| **per-interval** cost scale | 0.68 → 0.13 | **0.44 → 0.47** |
| **phase-1-only** global scale | 0.66 → 0.13 | **0.47 → 0.47** |

Both rescalings **fixed informativeness** (0.14 → 0.42 at ε=0.05) and **left
stability unchanged at ~0.5**. So phase-1 instability is *intrinsic*, not a
normalisation artifact. Stride 2 (24 h) makes it worse (0.576 max), so wider spacing
does not rescue it either.

> **This is a publishable negative result about entropic OT on densely-sampled
> snapshots, not about our coarse-graining objective.** At 12 h spacing in a
> slowly-changing population the coupling is noise-dominated at small ε and
> independence-dominated at large ε, with no reproducible regime between. Nobody in
> the OT-trajectory literature reports this. It is the empirical first data point for
> the "when are transition-defined states recoverable" study in `PROJECT_HANDOFF.txt`
> §9.

---

## Stage B — K selection (results VOID, see M0)

Protocol (b), λ_pm = 0, batch-wise technical hold-out. All 8 tasks converged.
`recommend()` returned **K\* = 20** by the one-SE-style rule.

| K | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 30 |
|---|---|---|---|---|---|---|---|---|
| held-out compress | 14.18 | 13.29 | 12.82 | 12.64 | 12.45 | **12.38** | 12.41 | 12.35 |
| train compress | 12.72 | 11.33 | 10.40 | 9.76 | 8.67 | 7.98 | 7.48 | 6.94 |
| min state mass | 0.014 | 0.012 | 0.008 | 0.004 | 0.002 | 0.002 | 0.002 | 0.002 |
| k_eff | 5.16 | 6.93 | 8.59 | 10.29 | 13.52 | 16.75 | 19.75 | 24.69 |
| **init ARI** | 0.177 | 0.070 | 0.131 | 0.038 | 0.022 | **0.011** | 0.007 | 0.004 |

Three red flags:

- **No interior optimum.** Held-out compression declines monotonically to the grid
  ceiling (argmin = K=30). K=20 was picked only by backing off from K=30 by one
  fold-direction spread. Total decline K=12→30 is 2.3 %.
- **Fits are not reproducible.** `init_ari` at K=20 is 0.011 — two *converged* fits
  from different seeds agree at chance level. Note `min_init_ari` is an **optional**
  rejection in `recommend()`, defaults to `None`, and therefore never fired.
- Training compression is monotone in K, as the protocol expects — the hold-out was
  supposed to do the work, and didn't.

---

## Interlude — membership structure probe

To test the hypothesis "if true K ≈ 6–10, then at K=20/30 the surplus columns should
be near-empty", fits were run at K ∈ {6,10,20,30} and the state masses inspected.

| K | sorted mean column mass | cols < 0.01 | min/max | mean max-membership |
|---|---|---|---|---|
| 6 | 0.210 … 0.146 | 0 / 6 | 0.69 | 0.988 |
| 10 | 0.130 … 0.082 | 0 / 10 | 0.63 | 0.998 |
| 20 | 0.065 … 0.033 | 0 / 20 | 0.51 | **1.000** |
| 30 | 0.046 … 0.025 | 0 / 30 | 0.54 | **1.000** |

**Not one near-empty column at any K.** At K=30 the smallest of thirty states still
holds 2.5 % of the mass. `k_eff/K` holds a near-constant 0.82–0.87 from K=6 to K=30
— it never plateaus. Mass splits almost evenly however many states are offered.

Also: **memberships saturate to exactly 1.000** at K ≥ 20. The design relies on
softmax keeping `M` strictly positive so `Phat > 0` and the CMI identity holds; in
float it saturates anyway.

---

## The diagnosis — why B, C and D were void

### M0 — `λ_x = 1` made the model k-means

Measured objective composition at λ_pm = 0:

| run | total | compress | expression | L₊ | L₋ |
|---|---|---|---|---|---|
| K=8 | 115.2 | 11.1 (9.6 %) | **104.1 (90.4 %)** | 3.06 | 2.86 |
| K=20 | 77.4 | 8.0 (10.3 %) | **69.4 (89.7 %)** | 2.77 | 2.58 |

Expression coherence owns ~90 % of the objective. Transport is a 10 % correction and
the fingerprint terms are ~0.7 % at λ_pm = 0.1. The λ_pm sweep confirms the
consequence exactly:

```
λ=0.1  comp 7.9984  x 69.4207  + 2.7655  - 2.5798   ARI_vs_0 = 1.000
λ=0.5  comp 7.9984  x 69.4207  + 2.7655  - 2.5798   ARI_vs_0 = 1.000
λ=1    comp 7.9984  x 69.4207  + 2.7655  - 2.5798   ARI_vs_0 = 1.000
λ=2    comp 8.0095  x 69.4633  + 2.7704  - 2.5839   ARI_vs_0 = 0.998
```

Identical to four decimals; the fitted memberships are the *same solution*. The only
thing changing in `total` is arithmetic applied to a fixed answer.

**So the model was doing k-means in 30-dim PCA space, with transport as a 10 %
correction and sufficiency contributing nothing.** That single fact explains every
Stage-B/C oddity:

| observation | k-means explanation |
|---|---|
| memberships saturate at 1.000 | k-means gives hard assignments |
| flat, even mass profiles, no empty columns | Voronoi cells on a continuum |
| `init_ari` ≈ 0.01 | k-means on a continuum is seed-dependent |
| held-out compression monotone, no U-shape | it was never really being optimised |
| DAG 58 % dense (4874 of 8400 edges at K=20) | even tessellation of a continuum |

**These are not properties of the method. They are properties of a k-means that the
transport terms were too weak to influence.** The method's distinctive machinery was
switched off for the entire test.

### The rebalanced smoke test (K=5, days 8.25–14, 25 iters)

| λ_x | λ_pm | comp share | expr share | pm share | ARI vs λ_pm=0 | K_eff |
|---|---|---|---|---|---|---|
| 1.0 | 0 | 0.14 | **0.86** | 0.00 | 1.000 | 4.56 |
| 1.0 | 3 | 0.12 | 0.75 | 0.13 | 0.911 | 4.55 |
| 0.05 | 0 | **0.74** | 0.26 | 0.00 | 1.000 | 4.60 |
| 0.05 | 3 | 0.38 | 0.12 | **0.50** | **0.840** | **4.70** |

At λ_x = 0.05 the shares invert, and λ_pm = 3 **moves the memberships while
occupancy holds** (K_eff 4.60 → 4.70, i.e. it *rises*). That is the regime the whole
protocol was looking for and never found.

**Caveat:** these ran at `max_iter = 25`; none converged, so the ARI is not yet a
property of the objective. The full grids (`max_iter = 1500`) settle it.

---

---

## Stage D — the (λ_x × λ_pm) grids: the decisive test

Two 4×3 grids at ε=0.2 on the serum range, `max_iter=1500`, `n_init=2`.
λ_x ∈ {1, 0.3, 0.1, 0.03} × λ_pm ∈ {0, 1, 5}. **24 fits, ~7.5 h total.**

### K=20 (8 of 12 cells; all four λ_pm=5 cells crashed)

| λ_x | λ_pm | L₊ | L₋ | L_comp | min_g | K_eff | ARI vs λ_pm=0 | init_ari |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 2.766 | 2.578 | 8.00 | 2.0e-03 | 16.78 | 1.000 | 0.005 |
| 1 | 1 | 2.766 | 2.580 | 8.00 | 2.0e-03 | 16.78 | 1.000 | 0.001 |
| 0.3 | 1 | 2.774 | 2.583 | 8.02 | 2.0e-03 | 16.78 | 0.999 | 0.000 |
| 0.1 | 1 | 2.650 | 2.079 | **22.28** | 3.3e-04 | 17.43 | **0.052** | 0.052 |
| 0.03 | 1 | **0.024** | **0.020** | **27.53** | **5.8e-09** | 18.07 | **−0.009** | −0.009 |
| any | 5 | — | — | — | — | — | **ERROR** | — |

### K=8 (12 of 12 cells, no crashes)

| λ_x | λ_pm | L₊ | L₋ | L_comp | min_g | K_eff | ARI vs λ_pm=0 | init_ari |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 3.062 | 2.858 | 11.10 | 1.5e-02 | 7.05 | 1.000 | 0.045 |
| 1 | 1 | 3.001 | 2.803 | 10.98 | 1.6e-02 | 7.04 | 0.981 | 0.029 |
| 1 | 5 | 3.099 | 2.888 | 11.42 | 1.5e-02 | 7.04 | 0.973 | 0.183 |
| 0.3 | 1 | 3.002 | 2.820 | 11.07 | 1.5e-02 | 7.05 | 0.948 | **0.249** |
| 0.3 | 5 | 3.538 | 2.991 | 12.49 | **4.5e-27** | 6.57 | 0.822 | −0.001 |
| 0.1 | 0 | 2.920 | 2.755 | 9.82 | 1.8e-02 | 7.09 | 1.000 | −0.000 *(max_iter)* |
| 0.1 | 1 | 3.038 | 2.845 | 11.26 | 1.6e-02 | 7.05 | 0.862 | 0.061 |
| 0.1 | 5 | **0.007** | **0.006** | **27.53** | 6.8e-02 | 7.84 | **−0.005** | −0.005 |
| 0.03 | 0 | 2.880 | 2.754 | 9.59 | 1.9e-02 | 7.10 | 1.000 | −0.000 *(max_iter)* |
| 0.03 | 1 | 3.006 | 2.811 | 11.22 | 1.6e-02 | 7.06 | 0.813 | −0.005 |
| 0.03 | 5 | **0.0001** | **0.0001** | **27.53** | 1.1e-01 | 7.99 | **−0.002** | −0.002 |

### D1 — Degeneracy 3, demonstrated

In every collapse cell, `L±` falls from ~3.0 to ~0.0001 (a 30 000× drop) **while
`L_comp` degrades from ~9.8 to 27.53**. The optimiser buys a near-zero sufficiency
loss by destroying the quantity being measured, sacrificing compression to do it.
At K=20/λ_x=0.03 a state is annihilated outright (`min_g` = 5.8e-09).

`L_comp = 27.53` appears in **all** collapse cells at **both** K=8 and K=20 — a
K-independent degenerate attractor. Tested hypothesis: uniform memberships, which
would make `Phat = a bᵀ` and hence `L_comp = Σ I_cell = 42.98`. **Rejected** —
27.53 ≠ 42.98. The attractor is real and reproducible but not yet identified.

### D2 — no usable window at either K

After correcting the gate (see [T9](#technical--fixed)), **0 usable cells** in both
grids. The failure modes, by cell:

- λ_x ≥ 0.3: `λ_pm` **inert** (ARI 0.948–1.000)
- λ_x ≤ 0.1 at K=20: jumps straight to an **unrelated** partition (ARI 0.05, −0.01)
- λ_pm = 5 at λ_x ≤ 0.1: **Degeneracy 3 collapse**
- λ_pm = 5 at K=20: **numerical failure**, all four cells
- The two best-looking K=8 cells (λ_pm=1, ARI 0.81–0.86) have **λ_pm=0 baselines that
  did not converge**, so their ARI has no valid reference

There is no middle ground between inert and degenerate.

### D3 — irreproducibility survives rebalancing ⚠️

`init_ari` (agreement between restarts) across **all 20 converged cells in both
grids**: **max 0.249, median 0.000, none above 0.25.**

At λ_x = 0.03 the transport share is 76–79 % — the regime I predicted would fix
this — and `init_ari` is −0.005.

> **This refutes the hypothesis stated earlier in this document** that the Stage-B
> ARI collapse was "a property of a k-means that the transport terms were too weak to
> influence". Rebalancing made `λ_pm` active but left reproducibility untouched. The
> non-identifiability is **independent of the weighting**, so M3 (continuum) stands
> while my explanation for it does not.

## Problems found

### Technical — FIXED

| # | Problem | Fix |
|---|---|---|
| T1 | `recommend()` averaged interval-level NaNs away via `nanmean`, so an ε could pass with stability measured on only some intervals | interval-level completeness gate + `unevaluated_intervals()` reporting which ε and which intervals |
| T2 | `03_fit.py` wrote the reference chain *before* checking whether fit dirs existed — a rerun could replace the chain under previous fits, then abort | `reserve_destinations()` validates **every** destination before any write; runs keyed by config hash |
| T3 | `02_k_reduce.py` merged any `K_*.json` in a directory without checking they shared a configuration | config fingerprint on every record; reducer refuses mixed groups and names the offending fields; detects missing/duplicate K |
| T4 | `load_serum()` filtered by day only — the parent h5ad would have silently contributed 2i cells | explicit arm mask, assertion that no 2i survives, refusal if no `arm` column. Verified: 22 000 2i cells dropped from the parent file |
| T5 | λ sweep wrote its JSON only at the end — the λ_pm=5 crash destroyed 5 completed fits | `_flush()` after every cell |
| T6 | A non-finite gradient killed the whole sweep | per-cell `try/except`, recorded as an `error` row, sweep continues |
| T7 | `--cost-scale-mode per_interval` was silently collapsed to a single scale (`[native_scale] * n`) | per-stride per-interval scales; stride comparability explicitly given up, as is inherent |
| T8 | Verdict messages blamed the fingerprint terms when the real blocker was non-convergence | `blocking_reason` names the actual gate (twice — I reintroduced this bug in the grid rewrite) |
| T9 | **The λ-grid verdict passed Degeneracy-3 collapses as "usable"** — it tested only `ARI < 0.99` (so ARI = −0.009 passed), gated on `k_eff` but not `min_state_mass` (k_eff *rose* to 18.07 while a state died at 5.8e-09), and never checked whether `L±` itself collapsed | rewritten with an ARI **band** (0.30–0.99), a `min_state_mass` gate, an `L±`-collapse gate, and a requirement that the λ_pm=0 baseline converged. `04_lambda_reduce.py` re-derives verdicts from saved JSONs without refitting |

### Technical — OPEN

- **`FloatingPointError: non-finite gradient` at K=20, λ_pm=5.** K=8 handles λ_pm=5
  fine, so it is K-dependent. Likely cause: `fingerprint_floor = 1e-30` gives
  `log ≈ −69`, and with `min_state_mass = 0.002` the fingerprint gradient has both a
  near-zero denominator (`1/g`) and a huge log to differentiate. **Proposed fix:**
  mass floor `M ← (1−η)M + η/K` (η ≈ 1e-3) — Dirichlet smoothing that bounds every
  `1/g` division and prevents saturation. Not yet implemented.
- **The degeneracy check degrades when memberships saturate.**
  `spread_ratio_ref_over_phat` reads 56.3 at K=8/λ=0 (sane), 1.6e5 at K=8/λ=5, and
  **9.1e28** at K=20/λ=0. As `phat_within_state_spread → 0` the ratio explodes and
  the diagnostic stops meaning anything. It should be reported as a difference or
  with an explicit floor.
- **`qiulab` has a hard 80-core cap** (`GrpTRES cpu=80`, shared across all account
  users). Two 32-core jobs nearly exhaust it. Sweeps should be **many narrow tasks**
  (e.g. 6 × 12 cores) rather than a few wide ones. Not yet applied to the `.sbatch`
  files.
- `.sbatch` files invoke the `.py` from `$SLURM_SUBMIT_DIR` **at run time**, so
  editing a script while a job is queued silently changes what runs. Cancel and
  resubmit rather than edit in place.

### Model-innate — NEED THINKING

#### M1 — the objective is dimensionally inconsistent

```
L = λ_comp·L_comp + λ_x·L_expr + λ₊·L₊ + λ₋·L₋
      [nats]         [PCA²/d]     [nats]  [nats]
```

`L_comp` is a KL divergence, `L₊`/`L₋` are conditional mutual informations — all in
nats. `L_expr = Σ aᵢ‖zᵢ − μ‖²/d` is squared PCA distance. **`λ_x` is silently
carrying a unit conversion**, which is why `λ_x = 1` has no meaning and why it
happened to land at 90 %.

This is not fixable by sweeping: `λ_x = 0.05` is just a different arbitrary point on
an unnormalised axis, and the value would not transfer to another dataset (different
HVG count or normalisation ⇒ different PCA scale ⇒ different "right" λ_x).

Candidate repairs, most to least principled:

1. **Gaussian likelihood.** Model `z | Z=k ~ N(μ_k, σ²I)`; then `L_expr/(2σ²)` is a
   negative log-likelihood in nats and the weights become interpretable ratios. σ² is
   *estimable* from within-state residual variance — turning a free hyperparameter
   into an estimate.
2. **Unitless.** Divide by within-timepoint total variance → a `1 − R²` quantity in
   [0,1]. Cheaper, bounded, honest, still not nats.
3. **Drop `L_expr`.** Expression already enters through the cost that built `P^ref`,
   so the separate term double-counts geometry. States would then be transport-defined
   in the strict sense — closer to the paper's actual claim. Risk: transport-sensible
   but expression-incoherent states, which is a testable outcome rather than an
   assumption.

#### M2 — the soft-membership design does not hold

Mean max-membership reaches exactly **1.000** at K ≥ 20. The spec relies on softmax
positivity for `Phat > 0` and for the CMI identity. Needs an explicit membership
entropy term, a temperature floor, or the mass floor above.

#### M3 — K may be the wrong abstraction for this data

Flat mass profiles at every K with no redundant columns is what tessellating a
**continuum** looks like. If the serum trajectory is continuous then no K-state model
has a defensible K, and Degeneracy 3 guarantees `L_pm` cannot be used to pick one.

Escape hatch: **`f±` are defined per cell without any `M`** (`fingerprints()` needs
only `P^ref` and the neighbour memberships). One could characterise the geometry of
fingerprint space directly — intrinsic dimension, clustered vs continuous, relation
to expression geometry — and report states only where they demonstrably reproduce.
That sidesteps both the K problem and the circularity of tuning on `L_pm`.

⚠️ **Update after the grids:** this caveat has been resolved, and not in the
direction I expected. M2 and M3 were measured in the λ_x = 1 regime, but the grids
show **`init_ari` stays ≈ 0 at every λ_x down to 0.03** (transport share 79 %). The
irreproducibility is *not* an artifact of expression dominance. M3 stands; M2 should
be re-measured but the mechanism I proposed for it is disproven.

#### M4 — nested selection cannot see interactions

K is selected at λ_pm = 0 (necessarily — Degeneracy 3), then frozen. If the
fingerprint terms would prefer a different granularity, the protocol cannot discover
it. The honest framing of any final result is *"best λ_pm behaviour at the K that
compression selects"*, not a jointly optimal (K, λ_pm).

#### M5 — a possible reformulation: pure information bottleneck

```
min  I(Iₜ ; Zₜ)  −  β·[ I(Zₜ ; Zₜ₊₁) + I(Zₜ ; Zₜ₋₁) ]
      ↑ rate            ↑ predictive relevance
```

Everything in nats, **one** tradeoff parameter instead of four weights. It would:

- make expression coherence implicit (`P^ref` already encodes expression geometry),
  removing the term that dominated;
- penalise hard assignment automatically — rate is *maximised* by deterministic
  membership, so M2 is fixed by construction;
- preserve the CMI identity, since
  `I(Iₜ;Zₜ₊₁) = I(Zₜ;Zₜ₊₁) + I(Iₜ;Zₜ₊₁|Zₜ)` — the existing `L_pm` is a
  decomposition of the relevance term, not something to discard;
- oppose the coarsening pressure of Degeneracy 3, because rate grows with effective
  state count;
- **make the deliverable a rate–relevance curve rather than a point estimate**, so
  M3 stops being a blocker and becomes the finding (a continuum shows up as a curve
  with no knee).

Cost: a real rewrite of `model.objective()`. Only worth it if rebalancing turns out
not to rescue the current form.

---

---

## Alternative approaches worth considering

The grids say the problem is not tuning. Three facts now constrain any redesign:

- **(F1)** The fits are irreproducible at every weighting — the objective has many
  near-equal optima and the seed picks one.
- **(F2)** `L±` has a degenerate global optimum that the optimiser *finds* once the
  term has enough weight to matter. It is not a theoretical worry.
- **(F3)** State masses stay flat with no redundant columns at every K from 6 to 30 —
  the data behaves like a continuum, not a set of attractors.

Ordered by how directly each addresses those.

### 1. Test whether discrete states exist at all — spectral gap (CHEAP, DO FIRST)

Before redesigning anything, ask whether the transport operator has metastable
structure. Build the row-normalised operator from the frozen `P^ref` and look at its
eigenvalue spectrum: **a spectral gap after m eigenvalues means m metastable sets; no
gap means a continuum.** This is CellRank/GPCCA's criterion and it is a *property of
the operator*, not of an objective with weights — so it cannot be confounded the way
our compression criterion was.

- If a gap exists at some m, that m is a principled K, and F1/F3 become an
  optimisation problem rather than a modelling one.
- If there is no gap, F3 is confirmed independently and **discrete states are the
  wrong abstraction for this data** — which redirects the whole project.

Cost: hours. Uses `reference_chain.npz`, already on disk. **Highest information per
unit effort of anything on this list.**

### 2. Information bottleneck (M5) — addresses F2 directly

```
min  I(Iₜ ; Zₜ)  −  β·[ I(Zₜ ; Zₜ₊₁) + I(Zₜ ; Zₜ₋₁) ]
```

One tradeoff parameter, everything in nats. Crucially the **rate term
`I(Iₜ;Zₜ)` penalises exactly the collapse we observed**: annihilating states and
driving `L±` to zero lowers the rate too, so the degenerate solution is no longer
free. It also makes hard assignment costly, addressing M2.

Deliverable becomes a **rate–relevance curve** rather than a point estimate, so F3
stops being a blocker: a continuum shows up as a curve with no knee, which is a
reportable result.

Does **not** obviously fix F1 — a non-convex objective over a continuum can still
have many optima.

### 3. Consensus / ensemble states — addresses F1 head-on

Accept non-identifiability instead of fighting it. Run many restarts, build the
cell×cell co-assignment matrix, and report only structure that reproduces. States
become *distributions over partitions*, and edge/node stability is measured rather
than assumed.

`stability.edge_support()` already exists in the package and is unused. This is the
cheapest way to turn F1 from a fatal flaw into a quantified uncertainty — and it is
compatible with keeping the current objective.

### 4. Skip states entirely — cell-level fingerprint geometry

`f±` are defined **per cell without any `M`** (`fingerprints()` needs only `P^ref`
and the neighbour memberships, which can be a fixed fine clustering). So one can
characterise the geometry of fingerprint space directly: intrinsic dimension, whether
it is clustered or continuous, how it relates to expression geometry, where cells
with similar expression have divergent futures.

This sidesteps K, Degeneracy 3, and the circularity of tuning on `L±` at once, and it
still answers the paper's actual question ("do cells that look alike have different
futures?"). Given F3, this may be the most honest framing available.

### 5. Low-rank OT — factorise instead of freeze-then-coarsen

HM-OT / FRLC learn latent states and couplings jointly via low-rank factorisation,
where the **rank is the state count** and is part of the optimisation rather than a
nested selection step. That removes the M4 problem (K chosen by a criterion that
doesn't reflect the final objective) structurally.

Caveat from `PROJECT_HANDOFF.txt` §9: HM-OT already does joint state+transition
learning and a forward–backward pass, so this direction has a **novelty problem** —
it would need to be framed as comparison, not contribution. Also §8: reproduce HM-OT
on its own published data before porting to WOT.

### 6. Continuous latent dynamics — if F3 is confirmed

If the spectral test says "no gap", the principled response is to stop discretising:
model the trajectory as a continuous latent process (neural ODE / Schrödinger bridge
/ flow matching between timepoints). Much larger scope, different literature, and it
abandons the "states" framing entirely — but it is the honest answer to a continuum.

### What I would do, concretely

1. **Spectral gap test on the existing chain** (hours). It determines whether this is
   an optimisation problem or a modelling one, and everything else branches on it.
2. **Consensus states** (days) — reuses existing machinery, converts F1 into a
   measurement, and is publishable regardless of which way (1) goes.
3. Then IB (2) *or* fingerprint geometry (4), depending on (1).

Do **not** implement the mass floor / more λ sweeps as a way to rescue the current
objective. The grids show that path is exhausted: the term is either inert or
degenerate, and reproducibility does not respond to weighting.

## Code changes made

Package (`main/cellstateadj/`) — behaviour-affecting:

- `informativeness.py` — interval-level NaN gate in `recommend()`; new
  `unevaluated_intervals()`; `mean_curve()` documented as descriptive-only and no
  longer warns on all-NaN rows.
- `reference.py` — corrected the `_underflow_limit` docstring and the
  `SinkhornConvergenceError` message. **The previous claim that `max(C)/ε > 708`
  makes the problem infeasible was wrong**: the solver is log-domain and feasibility
  depends on the surviving support, not the cost range. Now framed as a conditioning
  warning.
- `data.py` — docstring only. `split_half_by_replicate` claimed a replicate label
  "tracks one culture across time"; the WOT paper reports duplicate *samples* per
  timepoint and sampling is destructive, so that is not established. Relabelled a
  batch-wise technical hold-out; behaviour unchanged.

Runners (`analysis/wot_serum/`):

- `csa_wot.py` — arm enforcement (`serum_arm_mask`, `arm_composition`); the four
  provenance statements (conditioning / balanced pilot / representation /
  replicate structure); config fingerprint machinery (`run_fingerprint`,
  `fingerprint_hash`, `compare_fingerprints`, `provenance_block`);
  `reserve_destinations`; `save_all` records provenance and marks `config.json`'s
  representation block `_inert`; `cost_scale_mode` parameter.
- `01_eps_scan.py` — `conditioning_report` replaces the incorrect underflow report;
  `--cost-scale-mode`; `--sinkhorn-max-iter`.
- `02_k_sweep.py` / `02_k_reduce.py` — fingerprints, config-hashed directories,
  mixed-file rejection, missing/duplicate K detection.
- `03_fit.py` — destination reservation before any write; `run_<hash>_K<K>/` layout;
  `--lambda-pm` takes a list; no run is labelled "headline".
- `04_lambda_sweep.py` — **new**; a (λ_x × λ_pm) grid with incremental writes,
  per-cell error capture, term-share reporting, and a two-gate verdict. Gate 2 was
  rewritten after it passed Degeneracy-3 collapses (see T9): it now requires an ARI
  *band*, non-collapsing `min_state_mass`, non-collapsing `L±`, and a converged
  λ_pm=0 baseline.
- `04_lambda_reduce.py` — **new**; re-derives verdicts from saved `lambda_sweep.json`
  without refitting, so a corrected gate can be applied to hours-old results.
- `00_serum_only_pca.py` — **new**; builds the serum-only PCA sensitivity dataset
  (restriction applied *before* the basis is fitted, unlike `wot_data_check.ipynb`).
- `90_inspect.ipynb` — reads artifacts back, draws the DAG, plots the λ grid.
- `_common.sh` + `*.sbatch` — mail on END/FAIL/TIME_LIMIT, submit-dir guard,
  thread pinning, `srun` dropped.

Tests: **138 package tests pass** (was 133) — added 4 for the interval-NaN fix and 1
counterexample showing a 2×2 diagonal problem stays exactly feasible at a cost ratio
of 2000. Two pre-existing tests were updated where they asserted the old report
format and the now-corrected underflow claim. Plus **17 new runner tests**
(`analysis/wot_serum/test_wot_serum.py`) covering arm enforcement, destination
reservation, K-file compatibility, and the fingerprint.

### Two errors of mine, for the record

1. **The ε floor.** I claimed `ε ≥ 0.05` was *required* because `exp(−C/ε)`
   underflows. Wrong — log-domain solver, and feasibility is about the surviving
   support. Corrected in code, docs, and memory; a regression test now encodes the
   counterexample.
2. **`λ_x`.** I ran the entire λ_pm sweep without checking that the step-3 baseline
   had its two remaining terms on comparable scales. That voided Stages B–D. The
   term-share columns now make this visible on the face of every sweep output.

3. **The λ-grid verdict.** I wrote a gate that reported Degeneracy-3 collapses as
   "usable cells" — cells with ARI = −0.009 and a state annihilated to `min_g` =
   5.8e-09. Had that gone unchecked it would have produced a headline DAG from a
   degenerate fit. Fixed (T9) and re-derived.
4. **The k-means hypothesis.** I claimed the Stage-B restart collapse was explained
   by expression dominance and would resolve once the terms were balanced. The grids
   show `init_ari` stays ≈ 0 at every λ_x. The explanation was wrong; the observation
   was real.

Errors 1–2 were the same failure mode: asserting a limit that was an artifact of my
own configuration. Errors 3–4 were the opposite: over-optimistic readings of a
partial result. Both directions argue for the same discipline — check the diagnostic
that would falsify the claim before reporting it. The term-share columns, the ARI
band, the `min_g` and `L±` collapse gates and the `init_ari` reporting all exist now
because each was missing when it mattered.

---

## Open decisions

Superseded by the grid results — reweighting the current objective is exhausted:

1. ~~`λ_x` normalisation~~ — still a genuine formulation defect (M1), but fixing it
   does not rescue the method; the grids swept λ_x across 33× and found no usable
   window. Worth fixing for correctness if the objective survives at all.
2. ~~Mass floor~~ — would fix the λ_pm=5 crash and M2, but the crash is now a
   *symptom* rather than the blocker. Only worth doing if we keep this objective.
3. **Spectral-gap test** — the one thing I would do next. Determines whether
   discrete states exist in this data at all. See
   [alternatives §1](#1-test-whether-discrete-states-exist-at-all--spectral-gap-cheap-do-first).
4. **Consensus states** — converts the irreproducibility (F1) from a fatal flaw into
   a measured quantity, using machinery that already exists.
5. **Whether to keep the transductive cross-arm PCA.** `00_serum_only_pca.py` exists
   but has not been run (needs the 1.5 GB raw file).
6. **Phase 1.** Report A2 as a negative result, or attempt longer-range couplings
   (`t → t+2`).
7. **Framing.** A2 (OT couplings not reproducible at dense Δτ) and D3
   (coarse-graining not reproducible at any weighting) are both negative results
   about *reproducibility* at different levels of the pipeline. Together they may be
   a stronger and more honest paper than the original sufficiency claim — see
   `PROJECT_HANDOFF.txt` §9's "fallback / possible first paper".

## Where things live

```
analysis/wot_serum/            runners, sbatch, notebook, tests, logs/
/dartfs/rc/lab/C/CxQiu/data/joshua/CellStateAdj/wot_serum/
    eps_scan/                  Stage A: scan_stride{1,2}.{npz,csv}, summary.json
    k_sweep_529e4e2127/        Stage B: K_*.json, k_sweep.csv, k_selection.json
    run_58854e35be_K20/        Stage C: fit_lam0 complete; fit_lam5 crashed
    run_58854e35be_K8_K8ref/   Stage C: both fits complete
    lambda_sweep_*_K{8,20}_grid/   Stage D grids (running)
```

Every artifact directory is keyed by a configuration hash covering the h5ad, a
content hash of the representation, the arm policy, day range, stride, cell
subsampling and seed, ε, support/kappa, the shared λs, and the optimiser settings.
A different configuration cannot overwrite an earlier run.
