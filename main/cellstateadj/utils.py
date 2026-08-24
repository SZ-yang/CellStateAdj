"""Small numerical helpers shared across the package."""

from __future__ import annotations

from typing import Optional
import numpy as np
import torch


# --------------------------------------------------------------------------
# entropies / divergences
# --------------------------------------------------------------------------

def entropy(p, axis: int = -1, eps: float = 1e-30):
    """Shannon entropy (nats) of a distribution along ``axis``."""
    if isinstance(p, torch.Tensor):
        return -(p * torch.log(p.clamp_min(eps))).sum(dim=axis)
    p = np.asarray(p)
    return -(p * np.log(np.maximum(p, eps))).sum(axis=axis)


def effective_number(p, axis: int = -1):
    """exp(H(p)) -- the perplexity / effective support size."""
    if isinstance(p, torch.Tensor):
        return torch.exp(entropy(p, axis=axis))
    return np.exp(entropy(p, axis=axis))


def kl_rows(p, q, eps: float = 1e-30):
    """Row-wise KL(p || q) for distributions stored as rows."""
    if isinstance(p, torch.Tensor):
        return (p * (torch.log(p.clamp_min(eps)) - torch.log(q.clamp_min(eps)))).sum(-1)
    p = np.asarray(p)
    q = np.asarray(q)
    return (p * (np.log(np.maximum(p, eps)) - np.log(np.maximum(q, eps)))).sum(-1)


# --------------------------------------------------------------------------
# segment reductions on sparse triplets
# --------------------------------------------------------------------------

def segment_logsumexp(values: torch.Tensor, index: torch.Tensor, n_segments: int) -> torch.Tensor:
    """log sum exp of ``values`` grouped by ``index``; empty groups give -inf."""
    neg_inf = torch.finfo(values.dtype).min
    m = torch.full((n_segments,), neg_inf, dtype=values.dtype, device=values.device)
    m = m.scatter_reduce(0, index, values, reduce="amax", include_self=True)
    m_safe = torch.where(m > neg_inf / 2, m, torch.zeros_like(m))
    s = torch.zeros(n_segments, dtype=values.dtype, device=values.device)
    s = s.scatter_add(0, index, torch.exp(values - m_safe[index]))
    out = m_safe + torch.log(s.clamp_min(torch.finfo(values.dtype).tiny))
    return torch.where(s > 0, out, torch.full_like(out, float("-inf")))


def segment_sum(values: torch.Tensor, index: torch.Tensor, n_segments: int) -> torch.Tensor:
    out = torch.zeros(n_segments, dtype=values.dtype, device=values.device)
    return out.scatter_add(0, index, values)


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------

def as_tensor(x, dtype: torch.dtype, device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def uniform_weights(n: int, dtype=np.float64) -> np.ndarray:
    """Uniform cell weights a_t in the simplex.

    Balanced by default: uncalibrated cell counts are not population
    abundances (handoff s9), so we do not read growth from sample sizes.
    """
    return np.full(n, 1.0 / n, dtype=dtype)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def median_nonzero(x: np.ndarray) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size == 0:
        return 1.0
    return float(np.median(x))


def hard_labels(M) -> np.ndarray:
    """argmax assignment of a soft membership matrix."""
    if isinstance(M, torch.Tensor):
        M = M.detach().cpu().numpy()
    return np.asarray(M).argmax(axis=1)


def onehot_logits(labels: np.ndarray, K: int, scale: float = 3.0) -> np.ndarray:
    """Logits whose softmax is a smoothed one-hot at ``labels``."""
    U = np.zeros((len(labels), K), dtype=np.float64)
    U[np.arange(len(labels)), np.asarray(labels).astype(int)] = scale
    return U
