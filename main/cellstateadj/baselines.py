"""Comparator methods, run on the same cost specification as the model.

The honest framing (handoff s9): our method is structurally two-stage
(build coupling -> coarse-grain), the same shape as CellRank.  The claim is a
better coarse-graining *objective* applied jointly across the whole series,
not "transition-defined vs expression-defined".  These baselines exist so that
claim can be tested rather than asserted.

Implemented here:

* ``expression_clustering`` -- Leiden (if scanpy is available) or k-means per
  timepoint, then the transition map induced from the SAME frozen P^ref.  This
  is the WOT/moscot-style "cluster first, transport after" baseline.
* ``spectral_coarse_grain`` -- a GPCCA-flavoured stand-in: cluster the rows of
  the cell-level forward transition matrix.  Transition-defined in a meaningful
  sense, per-interval rather than joint, which is the actual contrast with our
  method.

CellRank 2 and HM-OT are external comparators; reproduce HM-OT on its own
published data before porting it (handoff s8).
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np

from .model import CoarseGrainModel
from .reference import ReferenceChain
from .config import ModelConfig
from .optimize import _kmeans
from .utils import onehot_logits


def leiden_labels(Z: np.ndarray, resolution: float = 1.0, seed: int = 0,
                  n_neighbors: int = 15) -> np.ndarray:
    """Leiden clustering of one timepoint (requires scanpy)."""
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(np.asarray(Z, dtype=np.float32))
    a.obsm["X_rep"] = np.asarray(Z, dtype=np.float32)
    sc.pp.neighbors(a, use_rep="X_rep", n_neighbors=n_neighbors, random_state=seed)
    sc.tl.leiden(a, resolution=resolution, random_state=seed,
                 key_added="leiden", flavor="igraph", n_iterations=2,
                 directed=False)
    return a.obs["leiden"].to_numpy().astype(int)


def expression_clustering(
    Z: Sequence[np.ndarray],
    K: int,
    method: str = "kmeans",
    seed: int = 0,
    resolution: float = 1.0,
) -> List[np.ndarray]:
    """Per-timepoint expression clustering into at most K groups."""
    out = []
    for t, z in enumerate(Z):
        z = np.asarray(z)
        if method == "kmeans":
            out.append(_kmeans(z, int(min(K, z.shape[0])), seed=seed + 1000 * t))
        elif method == "leiden":
            lab = leiden_labels(z, resolution=resolution, seed=seed + t)
            if lab.max() + 1 > K:   # merge the smallest clusters into K groups
                lab = _merge_to_k(z, lab, K)
            out.append(lab)
        else:
            raise ValueError(f"unknown clustering method {method!r}")
    return out


def _merge_to_k(Z: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    """Agglomerate cluster centroids down to K groups."""
    uniq = np.unique(labels)
    cent = np.array([Z[labels == u].mean(0) for u in uniq])
    new = _kmeans(cent, K, seed=0)
    remap = {int(u): int(new[i]) for i, u in enumerate(uniq)}
    return np.array([remap[int(l)] for l in labels])


def membership_from_labels(labels: np.ndarray, K: int,
                           smoothing: float = 1e-4) -> np.ndarray:
    """Near-hard membership matrix; smoothing keeps every entry strictly positive."""
    n = len(labels)
    M = np.full((n, K), smoothing / K)
    M[np.arange(n), np.asarray(labels).astype(int) % K] += 1.0 - smoothing
    return M / M.sum(1, keepdims=True)


def logits_from_labels(labels: np.ndarray, K: int, scale: float = 8.0) -> np.ndarray:
    return onehot_logits(labels, K, scale=scale)


def induced_from_labels(
    chain: ReferenceChain,
    Z: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    K: int,
    cfg: Optional[ModelConfig] = None,
) -> CoarseGrainModel:
    """Push fixed labels through the model machinery.

    Gives the baseline the identical induced T_t = M^T P^ref M, the identical
    diagnostics, and the identical objective value -- so the comparison is
    about the coarse-graining, not about differing conventions.
    """
    cfg = cfg or ModelConfig(K=K)
    U = [logits_from_labels(l, K) for l in labels]
    return CoarseGrainModel(chain, Z, cfg, U_init=U)


def spectral_coarse_grain(
    chain: ReferenceChain,
    K: int,
    seed: int = 0,
) -> List[np.ndarray]:
    """Cluster cells by their cell-level forward transition rows.

    A cheap stand-in for GPCCA-style macrostates: per interval, embed each cell
    by its outgoing distribution over the *next* timepoint's cells (reduced by
    SVD) and cluster.  Per-interval and greedy -- deliberately, because that is
    the comparator our joint objective is supposed to beat.
    """
    labels: List[np.ndarray] = []
    for t in range(chain.T):
        if t < chain.T - 1:
            c = chain.couplings[t]
            n, m = c.shape
            rs = np.bincount(c.rows, weights=c.values, minlength=n)
            rs[rs <= 0] = 1.0
            # random projection of the sparse row-stochastic matrix
            rng = np.random.default_rng(seed + t)
            R = rng.standard_normal((m, min(50, m)))
            E = np.zeros((n, R.shape[1]))
            np.add.at(E, c.rows, (c.values / rs[c.rows])[:, None] * R[c.cols])
        else:
            c = chain.couplings[t - 1]
            n, m = c.shape
            cs = np.bincount(c.cols, weights=c.values, minlength=m)
            cs[cs <= 0] = 1.0
            rng = np.random.default_rng(seed + t)
            R = rng.standard_normal((n, min(50, n)))
            E = np.zeros((m, R.shape[1]))
            np.add.at(E, c.cols, (c.values / cs[c.cols])[:, None] * R[c.rows])
        labels.append(_kmeans(E, int(min(K, E.shape[0])), seed=seed + t))
    return labels
