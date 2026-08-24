"""Frozen expression representation (spec 1.2).

z_i^(t) = f_psi(x_i^(t)) is learned ONCE across all timepoints and then never
touched again.  Two rules that matter:

* the representation is shared across timepoints (one PCA basis, not one per
  timepoint) -- otherwise the adjacent-time cost in Eq. 1 compares coordinates
  in different bases;
* timepoint is not removed as though it were a batch effect.  The developmental
  signal is the time-varying part; regressing it out would delete the thing we
  are trying to model.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from .config import RepresentationConfig
from .data import TimeSeriesData

try:
    from scipy import sparse as sp
except Exception:  # pragma: no cover
    sp = None


def _to_dense(x) -> np.ndarray:
    if sp is not None and sp.issparse(x):
        return np.asarray(x.todense())
    return np.asarray(x)


def _normalize_counts(X, target_sum: float, log1p: bool) -> np.ndarray:
    """Library-size normalise then log1p.  Works for dense or sparse input."""
    Xd = _to_dense(X).astype(np.float64)
    lib = Xd.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    Xd = Xd / lib * target_sum
    if log1p:
        Xd = np.log1p(Xd)
    return Xd


def select_hvg(Xs: List[np.ndarray], n_hvg: Optional[int]) -> np.ndarray:
    """Highly-variable genes by pooled variance of the normalised matrix.

    Pooled across all timepoints so the gene set -- like the basis -- is shared.
    """
    G = Xs[0].shape[1]
    if n_hvg is None or n_hvg >= G:
        return np.arange(G)
    n_tot = sum(x.shape[0] for x in Xs)
    s1 = np.zeros(G)
    s2 = np.zeros(G)
    for x in Xs:
        s1 += x.sum(axis=0)
        s2 += (x ** 2).sum(axis=0)
    mean = s1 / n_tot
    var = np.maximum(s2 / n_tot - mean ** 2, 0.0)
    # normalised dispersion: variance relative to the mean trend
    disp = var / np.maximum(mean, 1e-12)
    return np.sort(np.argsort(-disp)[:n_hvg])


def learn_representation(
    data: TimeSeriesData,
    cfg: RepresentationConfig = RepresentationConfig(),
    Z_precomputed: Optional[List[np.ndarray]] = None,
) -> Tuple[List[np.ndarray], dict]:
    """Return ``(Z, info)`` where ``Z[t]`` is (n_t, d) and is FROZEN.

    ``info`` records everything needed to reproduce or reapply the map
    (gene subset, mean, loadings), so a held-out replicate or a second
    experiment can be pushed through the same fixed representation.
    """
    if cfg.method == "precomputed":
        if Z_precomputed is None:
            raise ValueError("method='precomputed' requires Z_precomputed")
        Z = [np.asarray(z, dtype=np.float64) for z in Z_precomputed]
        if [z.shape[0] for z in Z] != data.n_cells:
            raise ValueError("Z_precomputed shapes do not match the data")
        return Z, {"method": "precomputed", "d": Z[0].shape[1]}

    if cfg.method != "pca":
        raise ValueError(f"unknown representation method {cfg.method!r}")

    Xs = [_normalize_counts(x, cfg.target_sum, cfg.log1p) for x in data.X]
    hvg = select_hvg(Xs, cfg.n_hvg)
    Xs = [x[:, hvg] for x in Xs]

    Xall = np.concatenate(Xs, axis=0)
    mean = Xall.mean(axis=0) if cfg.zero_center else np.zeros(Xall.shape[1])
    Xall = Xall - mean

    d = int(min(cfg.n_components, min(Xall.shape) - 1))
    # Randomised SVD via numpy: for development sizes a full SVD of the
    # (n x hvg) matrix is fine; fall back to the Gram trick when n < genes.
    rng = np.random.default_rng(cfg.random_state)
    q = min(Xall.shape[1], d + 10)
    Omega = rng.standard_normal((Xall.shape[1], q))
    Y = Xall @ Omega
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ Xall
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    components = Vt[:d]                     # (d, n_hvg)
    explained = (S[:d] ** 2) / max(Xall.shape[0] - 1, 1)

    Z = []
    off = 0
    for t in range(data.T):
        n = data.X[t].shape[0]
        Z.append((Xall[off:off + n] @ components.T).astype(np.float64))
        off += n

    info = {
        "method": "pca",
        "d": d,
        "hvg": hvg,
        "mean": mean,
        "components": components,
        "explained_variance": explained,
        "target_sum": cfg.target_sum,
        "log1p": cfg.log1p,
    }
    return Z, info


def apply_representation(X, info: dict) -> np.ndarray:
    """Push new counts through an already-frozen representation."""
    if info["method"] == "precomputed":
        raise ValueError("cannot re-apply a precomputed representation")
    Xd = _normalize_counts(X, info["target_sum"], info["log1p"])[:, info["hvg"]]
    return (Xd - info["mean"]) @ info["components"].T
