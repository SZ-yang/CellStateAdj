"""State-number selection by held-out compression (spec 1.20).

THE PROTOCOL -- decided, documented, and not to be switched silently
--------------------------------------------------------------------
The spec calls for "held-out or cross-fitted compression loss" as the primary
criterion for K but does not say which.  Two procedures were on the table
(handoff s7, s10 item 2):

  (a) refit P^ref on a reduced cell set and evaluate on held-out pairs;
  (b) hold out entire duplicate samples and evaluate the transferred state map.

**This project uses (b).**  Rationale: it uses the replicate structure the WOT
data actually has, it never refits P^ref (whose marginals would change with the
cell set, altering the very coupling being scored), and it scores the thing we
care about -- whether a state map learned on one culture still compresses
transport in another.

Why a held-out criterion is required at all: training compression decreases
monotonically in K, so a training sweep cannot select K.  It can only describe.
The held-out protocol is doing all the work.

Procedure, per candidate K and per direction of the split:

1. build ``P^ref_A`` on half A's cells only, at the fixed epsilon and the
   **global cost scale taken from the full data**, so both halves live on the
   same cost scale and their compression numbers are comparable;
2. fit ``M_A`` on A with ``lambda_pm = 0``.  L_pm must never enter K selection:
   it systematically favours coarser neighbouring state spaces and is exactly 0
   at K = 1 (Degeneracy 3);
3. **transfer, do not refit** -- assign B's cells to A's states by nearest
   expression prototype ``mu_A`` in the frozen z space.  This is the step that
   makes the estimate held-out; fitting anything on B would destroy it;
4. build ``P^ref_B`` on B's cells and evaluate ``L_compress`` and
   ``L_expression`` there under the transferred map.

Both directions are averaged.  Reported alongside, per spec 1.20: replicate
stability, initialisation spread, and minimum active state mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dc_replace
from typing import Dict, List, Optional, Sequence
import numpy as np

from .config import PipelineConfig
from .cost import resolve_cost_scales
from .data import (TimeSeriesData, split_half_by_replicate,
                   restrict_to_shared_replicates as restrict_shared)
from .model import CoarseGrainModel
from .optimize import fit as fit_states
from .reference import build_reference_chain
from .representation import apply_representation, learn_representation

_EPS = 1e-30


def transfer_memberships(
    Z_target: Sequence[np.ndarray],
    mu_source: Sequence[np.ndarray],
    logit_scale: float = 8.0,
) -> List[np.ndarray]:
    """Carry a fitted state map onto unseen cells WITHOUT refitting.

    Each held-out cell is assigned to the nearest source-fit expression
    prototype in the frozen z space.  Deliberately the simplest possible
    transfer: anything fitted on the held-out half would leak and the
    "held-out" compression would stop being held out.
    """
    out = []
    for t, z in enumerate(Z_target):
        mu = np.asarray(mu_source[t])
        d2 = ((np.asarray(z)[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
        out.append(-logit_scale * d2 / max(float(d2.mean()), _EPS))
    return out


@dataclass
class KSelectionResult:
    Ks: List[int]
    heldout_compress: List[float]
    heldout_expression: List[float]
    train_compress: List[float]
    train_expression: List[float]
    min_state_mass: List[float]
    k_eff: List[float]
    init_ari: List[float]
    # fold-direction spread (sd/sqrt(2) over the two split directions), NOT a
    # sampling standard error -- see recommend()
    heldout_se: List[float] = field(default_factory=list)
    all_converged: List[bool] = field(default_factory=list)
    statuses: List[List[str]] = field(default_factory=list)
    per_K: List[dict] = field(default_factory=list)
    protocol: str = "b: held-out duplicate samples, transferred state map"
    notes: Dict[str, object] = field(default_factory=dict)

    def recommend(
        self,
        min_state_mass: float = 1e-3,
        min_init_ari: Optional[float] = None,
        require_converged: bool = True,
        fallback_tolerance: float = 0.01,
    ) -> dict:
        """Select K by a one-SE-STYLE conservative rule on held-out compression.

        Spec 1.20 lists five criteria, not one.  Held-out compression picks the
        candidate; the others act as REJECTIONS applied first:

        * ``require_converged`` -- a fit that stopped at ``max_iter`` or
          ``line_search_failed`` never met its tolerances, so its held-out score
          is not a property of the objective at that K and must not be allowed
          to win;
        * ``min_state_mass`` -- a K whose smallest state is essentially empty is
          not really that K;
        * ``min_init_ari`` -- optional floor on initialisation stability.

        The rule has the FORM of a one-SE rule -- smallest K whose held-out
        score is within one spread of the best -- but the spread is not a
        statistically genuine standard error and must not be reported as one.
        It is computed across the two split DIRECTIONS, A->B and B->A, which are
        two complementary evaluations of the same pair of cultures, not two
        independent replicate estimates.  The quantity is real (fold-direction
        variability, sd/sqrt(2)) and it is the right thing to be conservative
        about; it just does not carry a sampling interpretation, and with n=2
        it is itself very noisy.  Call it a one-SE-style conservative rule.
        Give it a sampling interpretation only with more independent replicate
        folds.  ``fallback_tolerance`` (a relative tolerance) is used when the
        spread is unavailable or zero.  Either way this is a weak selector --
        report the curve.
        """
        v = np.asarray(self.heldout_compress, dtype=float)
        n = len(self.Ks)
        rejected: Dict[int, str] = {}
        eligible = []
        for i in range(n):
            if not np.isfinite(v[i]):
                rejected[int(self.Ks[i])] = "non-finite held-out compression"
                continue
            if require_converged and self.all_converged and not self.all_converged[i]:
                st = self.statuses[i] if self.statuses else []
                rejected[int(self.Ks[i])] = f"fit did not converge ({'/'.join(st)})"
                continue
            if self.min_state_mass and self.min_state_mass[i] < min_state_mass:
                rejected[int(self.Ks[i])] = (
                    f"minimum state mass {self.min_state_mass[i]:.2e} < "
                    f"{min_state_mass:.1e}")
                continue
            if (min_init_ari is not None and self.init_ari
                    and np.isfinite(self.init_ari[i])
                    and self.init_ari[i] < min_init_ari):
                rejected[int(self.Ks[i])] = (
                    f"initialisation ARI {self.init_ari[i]:.3f} < {min_init_ari}")
                continue
            eligible.append(i)

        if not eligible:
            return {"K": None, "rejected": rejected,
                    "reason": "every candidate K was rejected; widen the grid "
                              "or relax the criteria"}

        best_i = min(eligible, key=lambda i: v[i])
        se = (self.heldout_se[best_i]
              if self.heldout_se and np.isfinite(self.heldout_se[best_i]) else 0.0)
        if se > 0:
            thresh = v[best_i] + se
            rule = ("one-SE-STYLE: smallest K within one fold-direction spread "
                    "(sd/sqrt(2) over A->B and B->A) of the best; the spread is "
                    "not a sampling standard error")
        else:
            thresh = v[best_i] + abs(v[best_i]) * fallback_tolerance
            rule = (f"smallest K within {fallback_tolerance:.0%} of the best "
                    f"(SE unavailable)")

        order = sorted(eligible, key=lambda i: self.Ks[i])
        i = next((j for j in order if v[j] <= thresh), best_i)
        return {"K": int(self.Ks[i]), "heldout_compress": float(v[i]),
                "heldout_se": float(se),
                "heldout_se_interpretation":
                    "fold-direction spread over A->B and B->A, NOT a sampling "
                    "standard error: the two directions evaluate the same two "
                    "cultures",
                "best_heldout_compress": float(v[best_i]),
                "best_K": int(self.Ks[best_i]),
                "rule": rule,
                "rejected": rejected,
                "caveat": "n=2 culture replicates, so the spread is itself "
                          "very noisy; this is a weak selector. Report the "
                          "curve, not just the argmin."}

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.per_K)


def _evaluate_transferred(
    chain_B, Z_B, M_logits_B, K: int, cfg: PipelineConfig,
) -> Dict[str, float]:
    """Objective components on the held-out half under the transferred map."""
    mcfg = dc_replace(cfg.model, K=K, lambda_plus=0.0, lambda_minus=0.0)
    model = CoarseGrainModel(chain_B, Z_B, mcfg, U_init=M_logits_B)
    _, terms = model.objective()
    return {"compress": terms.compress, "expression": terms.expression}


def select_K(
    data: TimeSeriesData,
    cfg: PipelineConfig,
    Ks: Sequence[int],
    seed: int = 0,
    n_init_for_stability: int = 2,
    restrict_to_shared_replicates: bool = False,
    verbose: int = 1,
) -> KSelectionResult:
    """Protocol (b) K selection.  Requires replicate labels."""
    if data.replicate is None:
        raise ValueError(
            "protocol (b) needs replicate labels: it holds out whole duplicate "
            "samples. Provide TimeSeriesData.replicate, or choose protocol (a) "
            "and implement it explicitly rather than silently substituting."
        )

    # [CRITICAL] Restrict FIRST.  The PCA basis and the global cost scale below
    # are fit on whatever cells they are given, so discarding the non-shared
    # replicates afterwards would leave them influencing both -- the halves
    # would then be compared on a basis built partly from cells neither half
    # contains.
    if restrict_to_shared_replicates:
        data = restrict_shared(data, verbose=verbose)

    # The representation is frozen by construction; fitting it once on all cells
    # and re-applying it is correct here.  Refitting per half would confound
    # representation variability with state variability.
    Z_full, rep_info = learn_representation(data, cfg.representation)
    if rep_info.get("method") != "pca":
        raise ValueError("K selection needs a re-applicable representation "
                         "(cfg.representation.method='pca')")

    # One cost scale from the FULL data, reused by both halves, so the two
    # compression numbers are on the same scale and dtau survives (see
    # CouplingConfig.cost_scale_mode).
    scale = resolve_cost_scales(Z_full, data.tau, cfg.coupling.cost_scale_mode)[0]

    halves = split_half_by_replicate(data, seed=seed)   # already restricted
    Zs, chains = [], []
    for h, half in enumerate(halves):
        Zh = [apply_representation(x, rep_info) for x in half.X]
        Zs.append(Zh)
        chains.append(build_reference_chain(
            Zh, half.tau, cfg.coupling,
            cost_scales=[scale] * (len(Zh) - 1), verbose=max(0, verbose - 1)))
        if verbose:
            print(f"[K-selection] half {h}: n={[len(z) for z in Zh]} "
                  f"feasible={chains[h].feasible}")

    rows: List[dict] = []
    for K in Ks:
        mcfg = dc_replace(cfg.model, K=int(K), lambda_plus=0.0, lambda_minus=0.0)
        ocfg = dc_replace(cfg.optim, n_init=n_init_for_stability, seed=seed,
                          verbose=max(0, verbose - 1))
        ho_c, ho_x, tr_c, tr_x, gmins, keffs, aris = [], [], [], [], [], [], []
        statuses: List[str] = []

        for src in (0, 1):
            tgt = 1 - src
            res = fit_states(chains[src], Zs[src], mcfg, ocfg)
            # A fit that stopped at max_iter or line_search_failed never met its
            # tolerances; its held-out score says nothing about the objective at
            # this K, so the status has to travel with the number.
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
                mu = [x.cpu().numpy()
                      for x in res.model.expression_prototypes(M, g)]
            U_tgt = transfer_memberships(Zs[tgt], mu)
            ev = _evaluate_transferred(chains[tgt], Zs[tgt], U_tgt, int(K), cfg)
            ho_c.append(ev["compress"])
            ho_x.append(ev["expression"])

        # ddof=1 across the two split directions, then SE = sd / sqrt(n_folds).
        sd = float(np.std(ho_c, ddof=1)) if len(ho_c) > 1 else float("nan")
        se = sd / np.sqrt(len(ho_c)) if np.isfinite(sd) else float("nan")
        all_conv = all(st == "converged" for st in statuses)
        with np.errstate(invalid="ignore"):
            init_ari = (float(np.nanmean(aris))
                        if any(np.isfinite(x) for x in aris) else float("nan"))
        rec = dict(K=int(K),
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
                   all_converged=all_conv)
        rows.append(rec)
        if verbose:
            flag = "" if all_conv else f"  [NOT CONVERGED: {'/'.join(statuses)}]"
            print(f"[K-selection] K={K:3d}  heldout_compress="
                  f"{rec['heldout_compress']:.6f} (se {se:.2e})"
                  f"  train={rec['train_compress']:.6f}"
                  f"  Keff={rec['k_eff']:.2f}  min_g={rec['min_state_mass']:.2e}"
                  f"  init_ARI={rec['init_ari']:.3f}{flag}")

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
        notes={"epsilon": cfg.coupling.epsilon,
               "cost_scale_mode": cfg.coupling.cost_scale_mode,
               "cost_scale": scale,
               "lambda_pm": "0 (excluded from K selection by construction)",
               "n_replicate_groups": int(len(np.unique(data.replicate[0])))},
    )
    if verbose:
        tr = np.asarray(out.train_compress)
        if len(tr) > 1 and np.all(np.diff(tr) <= 1e-12):
            print("[K-selection] training compression is monotone in K, as "
                  "expected -- this is why it cannot select K and the held-out "
                  "curve does the work.")
        print(f"[K-selection] {out.recommend()}")
    return out
