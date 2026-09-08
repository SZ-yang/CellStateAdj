# Additional WOT serum experiments — exploratory plan

**Date:** 2026-09-08  
**Status:** planned, not yet run  
**Data:** WOT serum arm, days 8.25–18  
**Relationship to prior memo:** follows
`FINDINGS_2026-09-07_wot_serum.md`; it does not replace or strengthen that
memo's conclusions.

---

## Purpose

The first WOT runs mixed three questions:

1. Does the frozen OT chain contain reproducible past/future structure beyond
   ordinary expression similarity?
2. Does the current four-term objective encode that structure without being
   dominated by expression or driven into a degenerate solution?
3. Can the current non-convex optimiser reliably find a good solution?

These experiments separate those questions. They are exploratory. Their purpose is
to identify which parts of the current approach are supported by the WOT data and
which parts need reformulation, not to select a final method or make a biological
conclusion.

The plan has two parallel workstreams after constructing one corrected frozen
reference chain:

- **A — fixed-anchor cell-level fingerprint geometry:** test whether stable
  transition-defined structure exists without optimising the current objective.
- **B — expression-term ablations:** compare the current expression loss, a
  Gaussian-likelihood expression term, and no explicit expression term.

Every candidate state assignment will then be evaluated with the same fixed-anchor
conditional-mutual-information (CMI) criteria.

---

## Questions this plan should answer

### Q1 — Is there reproducible transition information in the reference chain?

Do individual cells have stable distributions over probable past and future
destinations when cells, technical batches, epsilon, and OT support are perturbed?

### Q2 — Is that information different from expression clustering?

Among cells that are near-neighbours in expression space, are there reproducible
differences in their past/future fingerprints? Can transition fingerprints improve
held-out prediction relative to an expression-only partition at matched complexity?

### Q3 — Are discrete states an adequate summary?

Does fingerprint space contain stable groups, a hierarchy, or mainly continuous
variation? A single automatic K is not assumed to exist.

### Q4 — What role is the explicit expression term playing?

Does a properly scaled Gaussian expression likelihood produce a better
compression–prediction–stability trade-off than the raw PCA-distance loss? What
happens when the explicit expression loss is removed entirely?

### Q5 — Can the current optimiser be trusted for these comparisons?

Do independent converged starts reach comparable objective values and state maps?
Does continuation improve upon the previous lambda=0 solution under each target
objective?

---

## Stage 0 — construct one corrected frozen reference chain

All subsequent experiments must use the same chain. The earlier epsilon scan and
serum fits used different cost scales and supports, so their epsilon values were not
directly equivalent.

### Requirements

1. Restrict to the serum arm before fitting the representation.
2. Freeze one shared expression representation and record its preprocessing,
   feature set, dimensionality, and random seed.
3. Use one documented global cost scale across the serum intervals.
4. Use the same sparse-support construction in the epsilon scan and all later fits.
5. Repeat the serum-only epsilon informativeness/stability scan under that exact
   scale and support.
6. Select an exploratory epsilon window rather than treating one value as a
   biological estimate. Preserve at least the selected value and its nearest
   reasonable neighbours for sensitivity analysis.
7. Save the chain once and reuse it. Do not silently rebuild it inside individual
   ablation jobs.

### Required provenance

- input file hash and arm/time filters;
- git commit;
- representation configuration;
- cost scale and interval spacing;
- epsilon and support parameters;
- marginal errors and solver status for every interval;
- cell identifiers and technical-batch labels.

No downstream result is interpretable if it was produced from a different chain
without being explicitly labelled as a sensitivity analysis.

---

## Workstream A — fixed-anchor cell-level fingerprint geometry

### Motivation

The current model defines a cell state by past and future role, but its learned
fingerprints depend on neighbouring memberships. This creates circularity: changing
`M[t+1]` changes the future fingerprints at `t`, which changes the objective used to
learn all memberships.

This workstream replaces learned neighbouring states with a fixed, independent,
fine-grained coordinate system. It asks whether the transition signal exists before
asking the joint objective to discover it.

### Fixed anchors

For every timepoint, construct a fine set of expression anchors `A[t]`. In the first
pass, use hard microcluster assignments, with anchor counts such as 20, 40, and 80 as
a resolution sensitivity analysis.

Anchors are not proposed biological states. They are bins used to describe probable
origins and destinations. To make resampling comparisons valid:

1. learn anchor centroids on a designated training split;
2. freeze those centroids;
3. assign validation/bootstrap cells to the frozen centroids;
4. preserve anchor identities across all comparisons using the same centroids.

Known annotations may be evaluated as an additional anchor system, but should not
be the only system because they may be too coarse to reveal transition
heterogeneity.

### Fingerprints

Let `P[t]` be the frozen coupling from time `t` to `t+1`, and let `A[t]` be the
cell-by-anchor assignment matrix. Define

```text
f_plus[t]  = P[t] A[t+1] / row_mass(P[t])
f_minus[t] = P[t-1]^T A[t-1] / col_mass(P[t-1])
r[t]       = [f_minus[t], f_plus[t]]
```

The first and last timepoints have only one direction. Every fingerprint is a
probability distribution over fixed neighbouring anchors.

### Analyses

#### A1 — informativeness

- total cell-level information in `f_plus` and `f_minus`;
- fingerprint entropy and effective destination/origin counts;
- pairwise Jensen–Shannon or Hellinger distances;
- comparison with an independence-coupling null.

Do not interpret low dispersion as a good state result if the fingerprints
themselves are uninformative.

#### A2 — reproducibility

Recompute the OT chain or fingerprints under:

- technical-batch split;
- bootstrap/subsampled cells;
- neighbouring epsilon values;
- reasonable support-size perturbations;
- anchor resolutions 20, 40, and 80.

Compare fingerprints only in the same fixed anchor coordinate system. Report
interval-specific results as well as an aggregate; do not average away failed or
unevaluated intervals.

#### A3 — relationship to expression

- correlate expression-space distance with fingerprint distance;
- within local expression neighbourhoods, test whether fingerprint divergence
  exceeds a neighbourhood-preserving permutation null;
- identify cells that are close in expression but have reproducibly different
  past/future fingerprints;
- measure how well expression alone predicts each fingerprint on held-out cells.

This is the direct test of whether transition-defined structure contributes
something beyond ordinary expression clustering on WOT.

#### A4 — geometry

- low-dimensional embeddings of forward, backward, and concatenated fingerprints;
- local intrinsic-dimension estimates;
- cluster-tendency and multiresolution stability analyses;
- comparison of clustered, hierarchical, and continuous descriptions.

A spectral gap or a clustering score may be reported as evidence, but neither
should be treated alone as proof of a true biological K.

#### A5 — held-out predictive comparison

Construct candidate partitions from training fingerprints, assign held-out cells,
and evaluate their forward/backward predictive sufficiency on held-out couplings or
technical batches. Compare them with expression clustering at matched K and matched
effective state complexity.

### Interpretation

- **Stable fingerprints with structure distinct from expression:** supports the
  scientific premise; failure of the current joint fit would point toward its
  objective or optimiser.
- **Stable but continuous fingerprint geometry:** transition information exists,
  but a single discrete K may be an inappropriate summary.
- **Stable fingerprints explained almost entirely by expression:** limited evidence
  that transition-defined states add information on this dataset.
- **Unstable or uninformative fingerprints:** the present OT chain/data do not
  support reliable cell-level transition roles, regardless of the downstream
  objective.

---

## Workstream B — expression-term ablations

All variants use the identical Stage-0 reference chain, K, initialisations, lambda
continuation path, and evaluation procedure.

### B0 — current raw expression loss (control)

```text
L = L_compress + lambda_x L_expr_SSE + lambda_pm (L_plus + L_minus)
```

`L_expr_SSE` is squared PCA distance divided by latent dimension. It is retained as
the control needed to interpret the two ablations; it should not be assumed to have
a transferable unit scale.

### B1 — Gaussian-likelihood expression term

Assume a shared isotropic emission model

```text
z[t,i] | Z[t]=k ~ Normal(mu[t,k], sigma^2 I).
```

The expression negative log-likelihood is

```text
L_expr_Gaussian = sum(t,i,k) a[t,i] M[t,i,k] *
  ( ||z[t,i]-mu[t,k]||^2 / (2 sigma^2)
    + d/2 * log(2 pi sigma^2) ).
```

For the first ablation:

1. estimate `sigma^2` from a designated expression-only/lambda-pm-zero training
   baseline at the chosen K;
2. freeze `sigma^2` before the lambda-pm continuation;
3. reuse it for every restart and lambda-pm value at that K;
4. report the estimate and its sensitivity;
5. do not re-estimate it independently at every lambda-pm value.

This places the assignment-dependent expression cost on a likelihood scale, but it
does not remove non-convexity or the self-referential fingerprint construction.
Because the Gaussian normalising constant does not affect memberships when
`sigma^2` is fixed, absolute term-share plots can still be misleading. Report
objective differences, assignment-dependent NLL, gradient norms, and held-out NLL
in addition to the total loss.

### B2 — no explicit expression term

```text
L = L_compress + lambda_pm (L_plus + L_minus).
```

This removes explicit double-counting of expression geometry. The model is not
expression-free: `P_ref` was constructed from expression. This ablation tests what
the frozen transport chain alone supports during coarse-graining.

### Initial run matrix

Use a diagnostic matrix before launching another large sweep:

- primary K: 8;
- secondary stress test: K=20 only after inspecting K=8, except for a small
  numerical check;
- expression variants: current SSE, Gaussian likelihood, none;
- lambda-pm continuation: `0 -> 0.25 -> 1 -> 2 -> 5`;
- reverse continuation for promising or hysteretic paths;
- at least three valid starts per key setting, including the previous-lambda warm
  start and independent starts.

The exact lambda grid may be refined after the first continuation path. Its purpose
is to trace a trade-off, not to declare one universal lambda.

### Optimisation requirements

1. Score the lambda-pm-zero membership under every target objective before fitting.
   It is a known feasible upper bound for a minimisation run.
2. A target run must improve upon that score within numerical tolerance; otherwise
   flag it as an optimisation failure or poor basin.
3. Save each restart's seed, initialisation, status, final objective, component
   losses, gradient norm, membership change, number of iterations, and state masses.
4. Compute restart agreement only among converged solutions with comparable
   objectives. If fewer than two exist, report stability as not assessed.
5. Do not label an objective plateau as convergence without a membership-change and
   gradient/stationarity check.
6. Keep warm-start and cold-start results separate.
7. Record non-finite gradients and state-mass collapse as results; do not average
   them away.

### Comparison outputs

For each valid fit report:

- transport reconstruction on training and held-out data;
- raw and weighted objective components;
- forward/backward fixed-anchor CMI;
- retained predictive information;
- expression NLL or distortion on held-out cells;
- effective state complexity and minimum state mass;
- agreement among comparable converged restarts;
- agreement with the lambda-pm-zero state map;
- sensitivity to epsilon and OT support for shortlisted settings.

No variant should be declared better solely because one component occupies a desired
percentage of the total objective.

---

## Stage 2 — CMI as the common evaluation criterion

### Meaning

Conditional mutual information asks how much individual cell identity still tells
us about the past or future after its proposed state is known.

For fixed future anchors `A[t+1]` and a candidate current state `Z[t]`,

```text
CMI_plus[t] = I(cell_identity[t] ; A[t+1] | Z[t]).
```

For the past,

```text
CMI_minus[t] = I(cell_identity[t] ; A[t-1] | Z[t]).
```

Equivalently, forward CMI is the membership-weighted within-state KL divergence
between each cell's fixed future fingerprint and its state's mean fingerprint:

```text
CMI_plus[t] = sum(i,k) a[t,i] M[t,i,k]
              KL(f_plus[t,i] || phi_plus[t,k]).
```

Low CMI means cells within a proposed state have similar transition roles. High CMI
means the state has grouped cells whose origins or destinations remain different.

### Why fixed anchors matter

In the current training loss, the future variable is the learned `Z[t+1]`. The
optimiser can make CMI small by collapsing or redefining `Z[t+1]`. With fixed
anchors, the target alphabet cannot be changed by the candidate state learner, so
low held-out CMI requires genuine predictive grouping.

### Retained predictive information

For fixed anchors,

```text
I(cell_identity ; A_next)
  = I(Z ; A_next) + I(cell_identity ; A_next | Z).
```

When the denominator is demonstrably informative, report

```text
retained_forward = 1 - CMI_plus / I(cell_identity ; A_next)
retained_backward = 1 - CMI_minus / I(cell_identity ; A_previous).
```

These quantities estimate the fraction of available cell-level transition
information preserved by the proposed states. If total cell-level information is
near zero, the ratio is unstable and must be marked uninformative rather than
reported as a successful low CMI.

### Complexity matching

CMI must never be compared without controlling representation complexity. A model
can reduce information loss by assigning every cell to its own state. Compare
candidate methods at matched:

- K;
- effective state number;
- state entropy or rate `I(cell_identity; Z)`;
- minimum state mass.

Prefer a rate–relevance or complexity–sufficiency curve over a single automatically
selected K.

### Candidate state definitions

The same evaluation can be applied to:

- per-timepoint expression clustering;
- fixed-fingerprint clustering;
- the current SSE objective;
- the Gaussian-expression objective;
- the no-expression objective;
- later, CellRank/GPCCA and HM-OT outputs at comparable resolutions.

For the immediate run, the first five are sufficient. External comparators can be
added after their published pipelines have been independently reproduced.

---

## Stage 3 — decision logic

This experiment is intended to narrow the modeling space, not force a final
conclusion.

### Outcome 1 — transition fingerprints are stable and one objective learns stable states

Continue developing that objective, validate on simulations with known truth, and
then broaden the real-data comparison.

### Outcome 2 — fingerprints are stable, but all joint objectives are unstable

Keep CMI as the state definition/evaluation criterion and consider a simpler learner
using fixed fingerprints, a constrained formulation, or a spectral relaxation. Do
not infer that the scientific premise failed.

### Outcome 3 — fingerprints are stable but continuous

Treat K as a display resolution or report continuous predictive coordinates and
multiresolution structure. Do not force a unique biological state count.

### Outcome 4 — fingerprints are stable but add little beyond expression

The WOT serum data provide limited evidence for a distinct transition-defined state
partition under the current expression-derived OT assumptions.

### Outcome 5 — fingerprints are unstable or uninformative

Prioritise the representation, OT assumptions, sampling interval, or a dataset with
additional lineage information before redesigning the downstream state objective.

---

## Recommended scripts and artifacts

Suggested new runners under `analysis/wot_serum/`:

```text
05_fixed_anchor_fingerprints.py
06_expression_ablation.py
07_cmi_compare.py
05_fixed_anchor_fingerprints.sbatch
06_expression_ablation.sbatch
07_cmi_compare.sbatch
```

Every analysis should write incrementally and include:

```text
config.json
provenance.json
summary.json
summary.csv
interval_metrics.csv
restart_summaries.json
history_<setting>_<restart>.csv
memberships_<setting>.npz
fingerprints_<split>_<anchor_resolution>.npz
plots/
```

The fixed-anchor analysis should additionally save anchor centroids, assignments,
split definitions, and cell identifiers so that fingerprints can be aligned and
recomputed locally.

### Minimum plots

1. fingerprint informativeness and stability by interval and epsilon;
2. expression distance versus fingerprint distance;
3. fingerprint embeddings coloured by time, expression annotation, and technical
   batch;
4. fixed-anchor CMI versus effective state complexity;
5. retained forward/backward information versus complexity;
6. compression versus CMI Pareto plots for the three expression variants;
7. restart objective versus pairwise state agreement;
8. state-mass and non-finite-gradient diagnostics along continuation paths.

---

## HPC execution order

1. Pull the commit containing this plan and record the commit hash.
2. Generate or verify the corrected Stage-0 chain.
3. Run the fixed-anchor analysis and K=8 expression ablations in parallel.
4. Inspect informativeness, convergence, upper-bound checks, and incremental
   summaries before launching K=20 or a denser lambda grid.
5. Run the common fixed-anchor CMI comparison.
6. Pull all JSON/CSV/NPZ artifacts and logs back locally for audit.
7. Update the findings memo only after distinguishing data-signal, model, and
   optimisation outcomes.

---

## Non-goals of this experiment

- It will not prove a global optimum for the current joint objective.
- It will not establish one true biological K.
- A low training CMI alone will not be treated as success.
- A spectral gap or visually separated embedding alone will not establish discrete
  biological states.
- A negative WOT result will not be generalized to all time-resolved single-cell
  datasets.
- No conclusion about replacing the current method with HM-OT, a neural ODE, or a
  continuous model will be made before the fixed-fingerprint signal test.

The immediate goal is narrower: determine whether reproducible transition-role
information exists, whether it differs from expression clustering, and which parts
of the current formulation can use that information without collapsing or becoming
optimisation-dependent.
