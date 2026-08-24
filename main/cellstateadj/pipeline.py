"""End-to-end glue: representation -> reference chain -> states -> diagnostics.

Follows Algorithm 1, but the build order is the handoff's, not the paper's:
the epsilon scan (``run_epsilon_scan``) is meant to run *first*, on its own,
before any state learning.  ``run_pipeline`` assumes epsilon and K are already
chosen and recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import numpy as np

from .config import PipelineConfig, DEFAULT_EPSILON_GRID
from .data import TimeSeriesData
from .diagnostics import Diagnostics, compute_diagnostics, degeneracy_check
from .informativeness import EpsilonScanResult, epsilon_scan
from .model import CoarseGrainModel
from .optimize import FitResult, fit
from .reference import ReferenceChain, build_reference_chain
from .representation import learn_representation


@dataclass
class PipelineResult:
    Z: List[np.ndarray]
    rep_info: dict
    chain: ReferenceChain
    fit: FitResult
    diagnostics: Optional[Diagnostics] = None
    degeneracy: Optional[dict] = None
    config: Optional[PipelineConfig] = None

    def summary(self) -> dict:
        d = {"chain": self.chain.summary(), "fit": self.fit.summary()}
        if self.diagnostics is not None:
            d["k_eff"] = self.diagnostics.k_eff
        if self.degeneracy is not None:
            d["degeneracy"] = self.degeneracy
        return d


def run_epsilon_scan(
    data: TimeSeriesData,
    cfg: PipelineConfig = PipelineConfig(),
    epsilons: Sequence[float] = DEFAULT_EPSILON_GRID,
    Z: Optional[Sequence[np.ndarray]] = None,
    **scan_kw,
) -> tuple:
    """STEP 1.  Returns ``(scan_result, Z, rep_info)``."""
    if Z is None:
        Z, info = learn_representation(data, cfg.representation)
    else:
        Z, info = [np.asarray(z) for z in Z], {"method": "given"}
    scan = epsilon_scan(Z, data.tau, epsilons=epsilons, cfg=cfg.coupling,
                        replicate=data.replicate, **scan_kw)
    return scan, Z, info


def run_pipeline(
    data: TimeSeriesData,
    cfg: PipelineConfig = PipelineConfig(),
    Z: Optional[Sequence[np.ndarray]] = None,
    chain: Optional[ReferenceChain] = None,
    with_diagnostics: bool = True,
    check_degeneracy: bool = True,
    verbose: int = 1,
) -> PipelineResult:
    """Fit states at a fixed epsilon and K, then run the diagnostics."""
    if Z is None:
        if verbose:
            print("[1/4] learning frozen representation")
        Z, info = learn_representation(data, cfg.representation)
    else:
        Z, info = [np.asarray(z) for z in Z], {"method": "given"}

    if chain is None:
        if verbose:
            print(f"[2/4] building reference chain at eps={cfg.coupling.epsilon}")
        chain = build_reference_chain(Z, data.tau, cfg.coupling, verbose=verbose)
        if not chain.feasible and verbose:
            print("  WARNING: at least one interval did not reach a feasible "
                  "balanced plan; report kappa sensitivity")

    if verbose:
        print(f"[3/4] fitting K={cfg.model.K} states "
              f"(lambda: compress={cfg.model.lambda_compress}, x={cfg.model.lambda_x}, "
              f"+={cfg.model.lambda_plus}, -={cfg.model.lambda_minus})")
    res = fit(chain, Z, cfg.model, cfg.optim)

    diags = None
    deg = None
    if with_diagnostics:
        if verbose:
            print("[4/4] diagnostics")
        diags = compute_diagnostics(res.model, cfg.diagnostics,
                                    replicate=data.replicate, Z=Z)
    if check_degeneracy and chain.T > 1:
        deg = degeneracy_check(res.model, t=0)
        if verbose:
            print(f"  degeneracy check: {deg['verdict']}")

    return PipelineResult(Z=list(Z), rep_info=info, chain=chain, fit=res,
                          diagnostics=diags, degeneracy=deg, config=cfg)


def lambda_sweep(
    chain: ReferenceChain,
    Z: Sequence[np.ndarray],
    cfg: PipelineConfig,
    lambdas: Sequence[float],
    direction: str = "both",
    verbose: int = 1,
) -> List[dict]:
    """Sweep lambda_plus / lambda_minus at fixed epsilon and K (spec 1.15 step 4).

    Watch K_eff: L_pm decreases when the neighbouring state space collapses, so
    a shrinking K_eff as lambda grows marks the edge of the usable regime.  That
    is a reportable result, not just a bug.
    """
    from dataclasses import replace as dc_replace
    out = []
    for lam in lambdas:
        mcfg = dc_replace(
            cfg.model,
            lambda_plus=lam if direction in ("both", "plus") else cfg.model.lambda_plus,
            lambda_minus=lam if direction in ("both", "minus") else cfg.model.lambda_minus,
        )
        if verbose:
            print(f"[lambda sweep] lambda={lam}")
        res = fit(chain, Z, mcfg, cfg.optim)
        rec = res.summary()
        rec["lambda"] = lam
        rec["k_eff_per_t"] = res.terms.k_eff
        out.append(rec)
    return out


def k_selection(
    chain: ReferenceChain,
    Z: Sequence[np.ndarray],
    cfg: PipelineConfig,
    Ks: Sequence[int],
    verbose: int = 1,
) -> List[dict]:
    """Sweep K with the fingerprint terms OFF (spec 1.15, 1.20).

    L_pm must NOT enter K selection: it systematically favours coarser
    neighbouring state spaces and is exactly zero at K = 1.

    NOTE (unresolved spec item, handoff s7/s10): the held-out compression
    protocol is not decided yet.  Option (a) refits P^ref on a reduced cell set
    and evaluates on held-out pairs; option (b) holds out whole duplicate
    samples and evaluates the transferred state map.  This function reports
    training compression only.  Pick a protocol, document it, and do not switch
    silently.
    """
    from dataclasses import replace as dc_replace
    out = []
    for K in Ks:
        mcfg = dc_replace(cfg.model, K=int(K), lambda_plus=0.0, lambda_minus=0.0)
        if verbose:
            print(f"[K selection] K={K} (L_pm excluded by construction)")
        res = fit(chain, Z, mcfg, cfg.optim)
        rec = res.summary()
        rec.update(K=int(K), min_state_mass=float(np.min(res.terms.g_min)),
                   k_eff_mean=float(np.mean(res.terms.k_eff)))
        out.append(rec)
    return out
