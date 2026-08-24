"""Time-series container and subsampling utilities.

The delta-tau subsampling utility (``subsample_timepoints``) is deliberately
built early: the WOT main experiment has ~12h spacing, so taking every 2nd /
4th / 8th timepoint gives 24h / 48h / 96h series for free, and the
delta-tau study is one of the strongest planned contributions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence
import numpy as np

try:  # scipy is optional at import time so the module can be introspected
    from scipy import sparse as sp
except Exception:  # pragma: no cover
    sp = None


@dataclass
class TimeSeriesData:
    """Count matrices at ordered timepoints, plus optional metadata.

    Attributes
    ----------
    X : list of (n_t, G) count matrices (dense ndarray or scipy sparse).
    tau : (T,) strictly increasing real observation times.
    replicate : optional list of (n_t,) labels identifying the *culture /
        sample* a cell came from.  Used for split-half stability and for
        cross-fitting the geometric null (a cell's replicate must be held out
        with it).  These are NOT independent biological replicates for the WOT
        data -- label any resulting numbers "culture-replicate stability".
    obs : optional list of dicts of per-cell arrays (e.g. ground-truth state).
    gene_names : optional (G,) array.
    """

    X: List[object]
    tau: np.ndarray
    replicate: Optional[List[np.ndarray]] = None
    obs: Optional[List[dict]] = None
    gene_names: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.tau = np.asarray(self.tau, dtype=float)
        if self.tau.ndim != 1 or len(self.tau) != len(self.X):
            raise ValueError("tau must be 1-D with one entry per timepoint")
        if np.any(np.diff(self.tau) <= 0):
            raise ValueError("tau must be strictly increasing")
        if self.replicate is not None:
            if len(self.replicate) != len(self.X):
                raise ValueError("replicate must have one array per timepoint")
            self.replicate = [np.asarray(r) for r in self.replicate]
        if self.obs is not None and len(self.obs) != len(self.X):
            raise ValueError("obs must have one dict per timepoint")

    # -- basic properties ---------------------------------------------------
    @property
    def T(self) -> int:
        return len(self.X)

    @property
    def n_cells(self) -> List[int]:
        return [x.shape[0] for x in self.X]

    @property
    def n_genes(self) -> int:
        return int(self.X[0].shape[1])

    @property
    def dtau(self) -> np.ndarray:
        """(T-1,) intervals between adjacent timepoints."""
        return np.diff(self.tau)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TimeSeriesData(T={self.T}, n_cells={self.n_cells}, "
            f"G={self.n_genes}, tau={np.round(self.tau, 3).tolist()})"
        )

    # -- selection ----------------------------------------------------------
    def select_timepoints(self, idx: Sequence[int]) -> "TimeSeriesData":
        idx = list(idx)
        return TimeSeriesData(
            X=[self.X[i] for i in idx],
            tau=self.tau[idx],
            replicate=None if self.replicate is None else [self.replicate[i] for i in idx],
            obs=None if self.obs is None else [self.obs[i] for i in idx],
            gene_names=self.gene_names,
        )

    def select_cells(self, per_t: Sequence[np.ndarray]) -> "TimeSeriesData":
        if len(per_t) != self.T:
            raise ValueError("need one index array per timepoint")
        return TimeSeriesData(
            X=[self.X[t][np.asarray(per_t[t])] for t in range(self.T)],
            tau=self.tau.copy(),
            replicate=(
                None if self.replicate is None
                else [self.replicate[t][np.asarray(per_t[t])] for t in range(self.T)]
            ),
            obs=(
                None if self.obs is None
                else [{k: np.asarray(v)[np.asarray(per_t[t])] for k, v in self.obs[t].items()}
                      for t in range(self.T)]
            ),
            gene_names=self.gene_names,
        )


def subsample_timepoints(
    data: TimeSeriesData,
    stride: int = 1,
    start: int = 0,
    stop: Optional[int] = None,
) -> TimeSeriesData:
    """Every ``stride``-th timepoint -- the delta-tau study knob.

    ``stride=1`` is the native spacing; ``stride=4`` on a 12h series gives 48h.
    Intervals stay unequal where the original series was unequal, which is
    intentional: the cost divides by dtau (Eq. 1).
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    idx = list(range(start, data.T if stop is None else stop, stride))
    if len(idx) < 2:
        raise ValueError("need at least two timepoints after subsampling")
    return data.select_timepoints(idx)


def subsample_cells(
    data: TimeSeriesData,
    n_per_timepoint: int,
    seed: int = 0,
    stratify_by_replicate: bool = True,
) -> TimeSeriesData:
    """Cap the number of cells per timepoint (development-scale runs).

    When replicate labels are present, sampling is stratified so both
    culture replicates survive -- split-half stability needs them.
    """
    rng = np.random.default_rng(seed)
    picks: List[np.ndarray] = []
    for t in range(data.T):
        n = data.X[t].shape[0]
        if n <= n_per_timepoint:
            picks.append(np.arange(n))
            continue
        if stratify_by_replicate and data.replicate is not None:
            groups = np.unique(data.replicate[t])
            per = max(1, n_per_timepoint // len(groups))
            chosen = []
            for gval in groups:
                pool = np.flatnonzero(data.replicate[t] == gval)
                take = min(per, len(pool))
                chosen.append(rng.choice(pool, size=take, replace=False))
            chosen = np.concatenate(chosen)
            if len(chosen) < n_per_timepoint:  # top up from the remainder
                rest = np.setdiff1d(np.arange(n), chosen)
                extra = rng.choice(rest, size=min(n_per_timepoint - len(chosen), len(rest)),
                                   replace=False)
                chosen = np.concatenate([chosen, extra])
            picks.append(np.sort(chosen))
        else:
            picks.append(np.sort(rng.choice(n, size=n_per_timepoint, replace=False)))
    return data.select_cells(picks)


def split_half_by_replicate(data: TimeSeriesData, seed: int = 0):
    """Split into two series by culture replicate, for stability checks.

    Returns ``(half_a, half_b)``.  With exactly two replicates per timepoint
    this is the duplicate-sample split; with more it is a random 50/50 split of
    replicate labels at each timepoint.
    """
    if data.replicate is None:
        raise ValueError("split_half_by_replicate needs replicate labels")
    rng = np.random.default_rng(seed)
    idx_a, idx_b = [], []
    for t in range(data.T):
        groups = np.unique(data.replicate[t])
        perm = rng.permutation(len(groups))
        half = max(1, len(groups) // 2)
        ga = set(groups[perm[:half]].tolist())
        mask = np.array([r in ga for r in data.replicate[t]])
        idx_a.append(np.flatnonzero(mask))
        idx_b.append(np.flatnonzero(~mask))
    return data.select_cells(idx_a), data.select_cells(idx_b)


def from_anndata(adata, time_key: str, replicate_key: Optional[str] = None,
                 layer: Optional[str] = None) -> TimeSeriesData:
    """Split a single AnnData of counts into a :class:`TimeSeriesData`."""
    times = np.asarray(adata.obs[time_key]).astype(float)
    order = np.unique(times)
    mat = adata.layers[layer] if layer is not None else adata.X
    X, reps, obs = [], [], []
    for tval in order:
        m = times == tval
        X.append(mat[m])
        if replicate_key is not None:
            reps.append(np.asarray(adata.obs[replicate_key])[m])
        obs.append({"index": np.asarray(adata.obs_names)[m]})
    return TimeSeriesData(
        X=X,
        tau=order,
        replicate=reps if replicate_key is not None else None,
        obs=obs,
        gene_names=np.asarray(adata.var_names),
    )
