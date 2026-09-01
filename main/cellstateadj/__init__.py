"""cellstateadj -- transition-defined cell states from time-resolved scRNA-seq.

Reference implementation of the bidirectional temporal coarse-graining method
described in ``plan_8-23.pdf`` (see ``../PROJECT_HANDOFF.txt`` for rationale).

Two explicit stages:

1. build a chain of adjacent-time entropic OT couplings from a *frozen*
   expression representation, and freeze them (``reference.build_reference_chain``);
2. jointly coarse-grain the whole chain into time-local states
   (``model.CoarseGrainModel`` + ``optimize``).

The reference couplings never depend on the memberships.  That is not an
implementation detail, it is what keeps the fingerprint losses from being
degenerate; see ``docs`` in :mod:`cellstateadj.model`.
"""

from .config import (
    RepresentationConfig,
    CouplingConfig,
    ModelConfig,
    OptimConfig,
    DiagnosticsConfig,
    PipelineConfig,
)
from .data import TimeSeriesData, subsample_cells, subsample_timepoints
from .representation import learn_representation
from .cost import adjacent_cost, build_support
from .sinkhorn import sinkhorn_dense, sinkhorn_sparse
from .reference import (ReferenceChain, build_reference_chain,
                        InfeasibleCouplingError, SinkhornConvergenceError)
from .informativeness import epsilon_scan, EpsilonScanResult
from .model import CoarseGrainModel, ObjectiveTerms
from .optimize import fit, FitResult
from .pipeline import (PipelineResult, run_epsilon_scan, run_pipeline,
                       lambda_sweep, k_diagnostic_sweep)
from .selection import KSelectionResult, select_K
from . import diagnostics, simulate, baselines, evaluate, stability, selection

__all__ = [
    "RepresentationConfig",
    "CouplingConfig",
    "ModelConfig",
    "OptimConfig",
    "DiagnosticsConfig",
    "PipelineConfig",
    "TimeSeriesData",
    "subsample_cells",
    "subsample_timepoints",
    "learn_representation",
    "adjacent_cost",
    "build_support",
    "sinkhorn_dense",
    "sinkhorn_sparse",
    "ReferenceChain",
    "build_reference_chain",
    "InfeasibleCouplingError",
    "SinkhornConvergenceError",
    "epsilon_scan",
    "EpsilonScanResult",
    "CoarseGrainModel",
    "ObjectiveTerms",
    "fit",
    "FitResult",
    "diagnostics",
    "simulate",
    "baselines",
    "evaluate",
    "stability",
    "PipelineResult",
    "run_epsilon_scan",
    "run_pipeline",
    "lambda_sweep",
    "k_diagnostic_sweep",
    "select_K",
    "KSelectionResult",
    "selection",
]

__version__ = "0.1.0"
