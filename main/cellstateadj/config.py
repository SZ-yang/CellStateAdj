"""Configuration dataclasses.

Every knob the method exposes lives here so that a run is fully described by a
serialisable object.  Defaults are the development-scale settings (a few
thousand cells per timepoint), not the full-scale ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence
import json


@dataclass
class RepresentationConfig:
    """Frozen expression representation z = f_psi(x)  (spec 1.2).

    Learned once, before transport construction, and never updated.  Timepoint
    is deliberately *not* regressed out as if it were a batch effect.
    """

    method: str = "pca"           # 'pca' | 'precomputed'
    n_components: int = 30
    n_hvg: Optional[int] = 2000   # None -> use all genes
    target_sum: float = 1e4       # library-size normalisation
    log1p: bool = True
    zero_center: bool = True
    random_state: int = 0


@dataclass
class CouplingConfig:
    """Cell-level reference couplings (spec 1.3-1.5)."""

    epsilon: float = 0.05
    # Cost is C_ij = ||z_i - z_j||^2 / dtau (Eq. 1), then divided by a scale so
    # that one epsilon is portable across datasets.
    #
    # [CRITICAL] The scale MUST be shared across intervals.  Dividing each
    # interval by its own median gives
    #     (||dz||^2 / dtau) / median(||dz||^2 / dtau) = ||dz||^2 / median(||dz||^2)
    # in which dtau cancels *exactly* -- Eq. 1 is silently reduced to a plain
    # squared distance and the relative weighting between a short and a long
    # interval is destroyed.  That breaks unequal spacing (WOT has 6h gaps
    # between days 8-9 against 12h elsewhere) and it invalidates the delta-tau
    # study, which is the whole point of varying the spacing.
    #   'global'       -- one scale for all intervals (default; keeps dtau)
    #   'per_interval' -- the cancelling behaviour, kept ONLY as a documented
    #                     sensitivity analysis.  Do not use for delta-tau work.
    #   'none'         -- raw costs in z-units; epsilon is then dataset-specific
    cost_scale_mode: str = "global"
    support: str = "knn"          # 'dense' | 'knn'
    kappa: int = 50               # neighbours per cell, symmetrised both ways
    kappa_max: int = 400          # grow until a balanced plan is feasible
    kappa_growth: float = 2.0
    # Sinkhorn iterations.  A shared cost scale means some intervals run at a
    # smaller *effective* epsilon than a per-interval scale gave them, and those
    # need noticeably more iterations -- 3000 was too few even for n=40 dense.
    max_iter: int = 20000
    # Sinkhorn stopping target on the marginal L1 error.  Chasing 1e-9 wastes
    # iterations: the solver only has to be comfortably better than
    # feasibility_tol, and the effective target is min(tol, feasibility_tol/10).
    tol: float = 1e-7
    # A restricted support need not admit ANY balanced plan.  This is the single
    # threshold that decides feasibility everywhere; Sinkhorn reports the raw
    # marginal error and callers judge against this.
    #
    # Calibration: the marginal error propagates to the induced transition map as
    #     |A_t(k,:).sum() - 1|  =  |M_t^T(rowsums(P) - a_t)|_k / g_tk
    #                           <=  marginal_error / g_tk
    # so it is AMPLIFIED for low-mass states.  At 1e-5 with a state of mass 1e-2
    # the row sums are off by ~1e-3, which is negligible; the genuinely broken
    # case that motivates the guard had a marginal error of 0.5.  Tighten this
    # if you care about very low-mass states.
    feasibility_tol: float = 1e-5
    # What to do when kappa_max is reached and the plan is still infeasible.
    # An infeasible coupling breaks T_t 1 = g_t, so A_t stops being stochastic
    # and every downstream transition number is wrong -- default to refusing.
    on_infeasible: str = "raise"  # 'raise' | 'dense' | 'warn'
    dtype: str = "float64"
    device: str = "cpu"


@dataclass
class ModelConfig:
    """Coarse-graining model and objective (spec 1.6-1.14)."""

    K: int = 20
    lambda_compress: float = 1.0
    lambda_x: float = 1.0
    lambda_minus: float = 0.0     # turned on only after step 3 baseline works
    lambda_plus: float = 0.0
    delta_floor: float = 1e-12    # numerical floor inside log(Phat + delta)
    fingerprint_floor: float = 1e-30
    g_min: float = 1e-3           # visualisation-only activity threshold
    # Scale L_expression by 1/d so lambda_x is comparable across latent dims.
    scale_expression_by_dim: bool = True
    dtype: str = "float32"
    device: str = "cpu"


@dataclass
class OptimConfig:
    """Optimiser (spec 1.19).

    Default is full-gradient descent with backtracking line search: a step is
    accepted only when the *complete* objective decreases.  The objective is
    self-referential (L_+ at t depends on M_{t+1}), so simultaneous updates are
    not monotone without the line search.
    """

    method: str = "full_gradient"   # 'full_gradient' | 'block_coordinate'
    # Search DIRECTION only; the acceptance rule is the same either way, so the
    # monotone guarantee the spec asks for is preserved.  Steepest descent
    # crawls once the memberships start to saturate, so L-BFGS is the default.
    direction: str = "lbfgs"        # 'lbfgs' | 'steepest'
    lbfgs_memory: int = 10
    max_iter: int = 300
    tol_objective: float = 1e-7     # relative change
    tol_membership: float = 1e-5    # max_t ||M_t^new - M_t^old||_F / sqrt(n_t)
    # Logits can keep drifting along near-flat directions long after the
    # objective has settled, so a stalled objective also counts as a fixed
    # point: `patience` consecutive iterations below tol_objective.
    patience: int = 8
    step_init: float = 1.0
    step_grow: float = 2.0
    step_shrink: float = 0.5
    armijo_c: float = 1e-4
    max_backtrack: int = 30
    damping: float = 1.0            # gamma in Eq. 35, block-coordinate only
    n_sweeps_per_iter: int = 1      # block-coordinate forward+backward sweeps
    init: str = "kmeans"            # 'kmeans' | 'random' | 'given'
    init_logit_scale: float = 3.0
    n_init: int = 1
    seed: int = 0
    verbose: int = 1
    log_every: int = 1


@dataclass
class DiagnosticsConfig:
    """Post-fit diagnostics (spec 1.16-1.18)."""

    geometric_null: bool = True
    null_method: str = "nadaraya_watson"
    null_n_folds: int = 5
    null_bandwidth: Optional[float] = None   # None -> median-distance heuristic
    null_bandwidth_scale: float = 1.0
    null_max_cells: int = 4000               # subsample for the O(n^2) kernel
    seed: int = 0


@dataclass
class PipelineConfig:
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    def to_json(self, path: Optional[str] = None) -> str:
        s = json.dumps(asdict(self), indent=2, sort_keys=True)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(s)
        return s

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        return cls(
            representation=RepresentationConfig(**d.get("representation", {})),
            coupling=CouplingConfig(**d.get("coupling", {})),
            model=ModelConfig(**d.get("model", {})),
            optim=OptimConfig(**d.get("optim", {})),
            diagnostics=DiagnosticsConfig(**d.get("diagnostics", {})),
        )


DEFAULT_EPSILON_GRID: Sequence[float] = (
    0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
)
