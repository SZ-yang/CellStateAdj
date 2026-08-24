"""Log-domain Sinkhorn for the entropic couplings of Eq. 2.

    P^eps = argmin_{P in Pi(a,b)}  <C, P> - eps H(P),      H(P) = -sum P log P

i.e. the standard entropic OT problem with kernel exp(-C/eps).  Everything is
done in the log domain so that small eps does not underflow, and the sparse
variant operates on an explicit support (spec 1.5).

Both routines return a :class:`SinkhornResult` carrying the potentials, so an
epsilon scan can warm-start the next value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch

from .cost import Support
from .utils import segment_logsumexp, segment_sum


@dataclass
class SinkhornResult:
    """A solved coupling on a fixed support.

    ``values`` are the plan entries on ``(rows, cols)``; for the dense solver
    the support is the full grid.
    """

    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray
    shape: Tuple[int, int]
    f: np.ndarray                 # potentials, in cost units
    g: np.ndarray
    epsilon: float
    n_iter: int
    marginal_error: float         # max of the two L1 marginal errors
    converged: bool

    @property
    def nnz(self) -> int:
        return len(self.values)

    def to_dense(self) -> np.ndarray:
        P = np.zeros(self.shape, dtype=self.values.dtype)
        P[self.rows, self.cols] = self.values
        return P

    def row_sums(self) -> np.ndarray:
        return np.bincount(self.rows, weights=self.values, minlength=self.shape[0])

    def col_sums(self) -> np.ndarray:
        return np.bincount(self.cols, weights=self.values, minlength=self.shape[1])


def sinkhorn_sparse(
    support: Support,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    max_iter: int = 3000,
    tol: float = 1e-9,
    f_init: Optional[np.ndarray] = None,
    g_init: Optional[np.ndarray] = None,
    device: str = "cpu",
    dtype: str = "float64",
    check_every: int = 10,
) -> SinkhornResult:
    """Sinkhorn restricted to ``support``.

    A balanced plan need not exist on an arbitrary support.  Non-convergence of
    the marginal error is the signal that the support is infeasible; the caller
    (``reference.build_reference_chain``) grows kappa in response rather than
    silently returning an unbalanced plan.
    """
    td = torch.float64 if dtype == "float64" else torch.float32
    dev = torch.device(device)
    n, m = support.shape

    C = torch.as_tensor(support.cost, dtype=td, device=dev)
    rows = torch.as_tensor(support.rows, dtype=torch.long, device=dev)
    cols = torch.as_tensor(support.cols, dtype=torch.long, device=dev)
    la = torch.log(torch.as_tensor(a, dtype=td, device=dev))
    lb = torch.log(torch.as_tensor(b, dtype=td, device=dev))

    f = (torch.zeros(n, dtype=td, device=dev) if f_init is None
         else torch.as_tensor(f_init, dtype=td, device=dev).clone())
    g = (torch.zeros(m, dtype=td, device=dev) if g_init is None
         else torch.as_tensor(g_init, dtype=td, device=dev).clone())

    err = float("inf")
    it = 0
    for it in range(1, max_iter + 1):
        f = epsilon * (la - segment_logsumexp((-C + g[cols]) / epsilon, rows, n))
        g = epsilon * (lb - segment_logsumexp((-C + f[rows]) / epsilon, cols, m))
        if it % check_every == 0 or it == max_iter:
            logP = (f[rows] + g[cols] - C) / epsilon
            P = torch.exp(logP)
            rs = segment_sum(P, rows, n)
            cs = segment_sum(P, cols, m)
            err = float(max((rs - torch.as_tensor(a, dtype=td, device=dev)).abs().sum(),
                            (cs - torch.as_tensor(b, dtype=td, device=dev)).abs().sum()))
            if err < tol:
                break

    logP = (f[rows] + g[cols] - C) / epsilon
    P = torch.exp(logP)
    values = P.cpu().numpy().astype(np.float64)
    rs = np.bincount(support.rows, weights=values, minlength=n)
    cs = np.bincount(support.cols, weights=values, minlength=m)
    err = float(max(np.abs(rs - a).sum(), np.abs(cs - b).sum()))
    return SinkhornResult(
        rows=support.rows, cols=support.cols, values=values, shape=support.shape,
        f=f.cpu().numpy(), g=g.cpu().numpy(), epsilon=epsilon, n_iter=it,
        marginal_error=err, converged=bool(err < max(tol, 1e-12) * 10 or err < 1e-8),
    )


def sinkhorn_dense(
    C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    max_iter: int = 3000,
    tol: float = 1e-9,
    f_init: Optional[np.ndarray] = None,
    g_init: Optional[np.ndarray] = None,
    device: str = "cpu",
    dtype: str = "float64",
    check_every: int = 10,
) -> SinkhornResult:
    """Dense log-domain Sinkhorn (development sizes and unit tests)."""
    td = torch.float64 if dtype == "float64" else torch.float32
    dev = torch.device(device)
    Ct = torch.as_tensor(C, dtype=td, device=dev)
    at = torch.as_tensor(a, dtype=td, device=dev)
    bt = torch.as_tensor(b, dtype=td, device=dev)
    n, m = Ct.shape
    f = (torch.zeros(n, dtype=td, device=dev) if f_init is None
         else torch.as_tensor(f_init, dtype=td, device=dev).clone())
    g = (torch.zeros(m, dtype=td, device=dev) if g_init is None
         else torch.as_tensor(g_init, dtype=td, device=dev).clone())

    err = float("inf")
    it = 0
    for it in range(1, max_iter + 1):
        f = epsilon * (torch.log(at) - torch.logsumexp((-Ct + g[None, :]) / epsilon, dim=1))
        g = epsilon * (torch.log(bt) - torch.logsumexp((-Ct + f[:, None]) / epsilon, dim=0))
        if it % check_every == 0 or it == max_iter:
            P = torch.exp((f[:, None] + g[None, :] - Ct) / epsilon)
            err = float(max((P.sum(1) - at).abs().sum(), (P.sum(0) - bt).abs().sum()))
            if err < tol:
                break

    P = torch.exp((f[:, None] + g[None, :] - Ct) / epsilon).cpu().numpy().astype(np.float64)
    rows = np.repeat(np.arange(n), m)
    cols = np.tile(np.arange(m), n)
    rs, cs = P.sum(1), P.sum(0)
    err = float(max(np.abs(rs - a).sum(), np.abs(cs - b).sum()))
    return SinkhornResult(
        rows=rows, cols=cols, values=P.ravel().copy(), shape=(n, m),
        f=f.cpu().numpy(), g=g.cpu().numpy(), epsilon=epsilon, n_iter=it,
        marginal_error=err, converged=bool(err < 1e-8),
    )
