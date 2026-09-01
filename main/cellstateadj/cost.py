"""Adjacent-time transport costs and sparse support (spec 1.2, 1.5).

C^(t)_ij = ||z_i^(t) - z_j^(t+1)||^2 / dtau_t          (Eq. 1)

Support: a row-wise kNN graph alone does NOT guarantee that a balanced
coupling with the required column marginals exists.  We therefore take the
union of source->target and target->source neighbourhoods, and grow kappa until
Sinkhorn reaches a feasible plan (handled in :mod:`cellstateadj.reference`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class Support:
    """Sparse bipartite support S subset of {1..n}x{1..m} with its costs."""

    rows: np.ndarray      # (nnz,) int64 index into source cells
    cols: np.ndarray      # (nnz,) int64 index into target cells
    cost: np.ndarray      # (nnz,) float64 cost values
    shape: Tuple[int, int]
    kappa: Optional[int] = None
    dense: bool = False

    @property
    def nnz(self) -> int:
        return len(self.rows)

    @property
    def density(self) -> float:
        return self.nnz / (self.shape[0] * self.shape[1])

    def min_row_degree(self) -> int:
        return int(np.bincount(self.rows, minlength=self.shape[0]).min())

    def min_col_degree(self) -> int:
        return int(np.bincount(self.cols, minlength=self.shape[1]).min())


def squared_distances(Za: np.ndarray, Zb: np.ndarray) -> np.ndarray:
    """(n, m) matrix of squared Euclidean distances, clipped at zero."""
    aa = (Za ** 2).sum(1)[:, None]
    bb = (Zb ** 2).sum(1)[None, :]
    D = aa + bb - 2.0 * (Za @ Zb.T)
    np.maximum(D, 0.0, out=D)
    return D


def adjacent_cost(Za: np.ndarray, Zb: np.ndarray, dtau: float) -> np.ndarray:
    """Dense cost matrix of Eq. 1."""
    if dtau <= 0:
        raise ValueError("dtau must be positive")
    return squared_distances(Za, Zb) / dtau


def _knn_from_distance(D: np.ndarray, k: int, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of the k smallest entries along ``axis``."""
    k = int(min(k, D.shape[axis]))
    if axis == 1:
        idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(D.shape[0]), k)
        cols = idx.ravel()
    else:
        idx = np.argpartition(D, kth=k - 1, axis=0)[:k, :]
        cols = np.tile(np.arange(D.shape[1]), k)
        rows = idx.ravel()
    return rows, cols


def interval_cost_median(Za: np.ndarray, Zb: np.ndarray, dtau: float,
                         max_cells: int = 2000, seed: int = 0) -> float:
    """Median of the Eq. 1 cost for one interval, on a capped subsample.

    Used to build the SHARED cost scale; subsampled because only the scale is
    needed, not the full matrix.
    """
    rng = np.random.default_rng(seed)
    ia = (np.sort(rng.choice(Za.shape[0], max_cells, replace=False))
          if Za.shape[0] > max_cells else slice(None))
    ib = (np.sort(rng.choice(Zb.shape[0], max_cells, replace=False))
          if Zb.shape[0] > max_cells else slice(None))
    med = float(np.median(adjacent_cost(Za[ia], Zb[ib], dtau)))
    return med if np.isfinite(med) and med > 0 else 1.0


def global_cost_scale(Z: Sequence[np.ndarray], tau: np.ndarray,
                      max_cells: int = 2000, seed: int = 0) -> float:
    """One cost scale shared by every interval.

    [CRITICAL] This is what keeps Delta-tau alive.  Scaling each interval by its
    own median cancels the ``/ dtau`` of Eq. 1 exactly; scaling them all by one
    common constant preserves the ratios between intervals, so a long gap really
    does cost differently from a short one.
    """
    meds = [interval_cost_median(Z[t], Z[t + 1], float(tau[t + 1] - tau[t]),
                                 max_cells=max_cells, seed=seed)
            for t in range(len(Z) - 1)]
    s = float(np.median(meds)) if meds else 1.0
    return s if np.isfinite(s) and s > 0 else 1.0


def resolve_cost_scales(Z: Sequence[np.ndarray], tau: np.ndarray, mode: str,
                        max_cells: int = 2000, seed: int = 0) -> List[float]:
    """Per-interval scales implied by ``mode``; see :class:`CouplingConfig`."""
    n_int = len(Z) - 1
    if mode == "global":
        return [global_cost_scale(Z, tau, max_cells, seed)] * n_int
    if mode == "none":
        return [1.0] * n_int
    if mode == "per_interval":
        return [interval_cost_median(Z[t], Z[t + 1], float(tau[t + 1] - tau[t]),
                                     max_cells, seed) for t in range(n_int)]
    raise ValueError(f"unknown cost_scale_mode {mode!r}")


def build_support(
    Za: np.ndarray,
    Zb: np.ndarray,
    dtau: float,
    kappa: Optional[int] = None,
    dense: bool = False,
    cost_scale: Optional[float] = None,
) -> Tuple[Support, float]:
    """Build the (possibly sparse) support and its costs.

    Returns ``(support, cost_scale)``.  Costs are divided by ``cost_scale``.
    Pass the SHARED scale from :func:`resolve_cost_scales`; leaving it ``None``
    falls back to this interval's own median, which cancels dtau and is only
    correct for a single isolated interval.

    Symmetrisation matters: we take the union of the row-kNN and the column-kNN
    so that every source cell has candidate targets AND every target cell has
    candidate sources.  Without the second half the column marginal constraint
    can be infeasible.
    """
    D = adjacent_cost(Za, Zb, dtau)
    if cost_scale is None:
        cost_scale = float(np.median(D))
        if not np.isfinite(cost_scale) or cost_scale <= 0:
            cost_scale = 1.0
    if cost_scale != 1.0:
        D = D / cost_scale

    n, m = D.shape
    if dense or kappa is None or kappa >= min(n, m):
        rows = np.repeat(np.arange(n), m)
        cols = np.tile(np.arange(m), n)
        return Support(rows, cols, D.ravel().copy(), (n, m), kappa=None, dense=True), cost_scale

    r1, c1 = _knn_from_distance(D, kappa, axis=1)   # source -> target
    r2, c2 = _knn_from_distance(D, kappa, axis=0)   # target -> source
    rows = np.concatenate([r1, r2])
    cols = np.concatenate([c1, c2])
    # deduplicate the union
    lin = rows.astype(np.int64) * m + cols.astype(np.int64)
    lin = np.unique(lin)
    rows = (lin // m).astype(np.int64)
    cols = (lin % m).astype(np.int64)
    return Support(rows, cols, D[rows, cols].copy(), (n, m), kappa=kappa, dense=False), cost_scale


def perturb_cost(support: Support, rel_sigma: float, rng: np.random.Generator) -> Support:
    """Multiplicative log-normal jitter on the costs -- the cost-perturbation
    stability probe used in epsilon selection (spec 1.4, criterion 3)."""
    factor = np.exp(rng.normal(0.0, rel_sigma, size=support.nnz))
    return Support(support.rows, support.cols, support.cost * factor,
                   support.shape, support.kappa, support.dense)
