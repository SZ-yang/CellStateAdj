"""The frozen reference chain P^ref_1 ... P^ref_{T-1} (spec 1.3-1.5).

[CRITICAL] These couplings are estimated once, from the frozen expression
representation only, and are then held fixed for the whole of state learning.
They must never become a function of the memberships M.

Why: if the fingerprints were computed from the model's own low-rank coupling
Phat_t = Q_t diag(g_t)^-1 T_t diag(g_{t+1})^-1 Q_{t+1}^T, then

    f+_ti = Phat_t(i,:) M_{t+1} / Phat_t(i,:) 1 = M_t(i,:) B_t

with B_t independent of i (the a_ti in row i of Q_t cancels exactly on
division).  Cells hard-assigned to the same state would then have *identical*
fingerprints for ANY assignment, the two-sided loss would be driven to zero by
phi+_tk = B_t(k,:), and L_pm would measure nothing.  See
``diagnostics.degeneracy_check`` for the empirical demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np
import torch

from .config import CouplingConfig
from .cost import Support, build_support
from .sinkhorn import SinkhornResult, sinkhorn_sparse, sinkhorn_dense
from .utils import torch_dtype, uniform_weights


@dataclass
class ReferenceChain:
    """A frozen chain of adjacent-time couplings plus the tensors it needs.

    ``P[t]`` couples timepoint ``t`` to ``t+1`` and has total mass 1 with
    marginals ``a[t]`` and ``a[t+1]``.
    """

    couplings: List[SinkhornResult]
    a: List[np.ndarray]
    tau: np.ndarray
    epsilon: float
    cost_scales: List[float]
    kappas: List[Optional[int]]
    Z: Optional[List[np.ndarray]] = None

    # lazily-built torch views
    _torch: Optional[dict] = None

    @property
    def T(self) -> int:
        return len(self.a)

    @property
    def n_cells(self) -> List[int]:
        return [len(x) for x in self.a]

    @property
    def feasible(self) -> bool:
        return all(c.converged for c in self.couplings)

    def marginal_errors(self) -> List[float]:
        return [c.marginal_error for c in self.couplings]

    def summary(self) -> dict:
        return {
            "T": self.T,
            "n_cells": self.n_cells,
            "epsilon": self.epsilon,
            "kappas": self.kappas,
            "nnz": [c.nnz for c in self.couplings],
            "marginal_error": self.marginal_errors(),
            "feasible": self.feasible,
        }

    # ------------------------------------------------------------------
    def to_torch(self, device: str = "cpu", dtype: str = "float32") -> dict:
        """Build (and cache) the tensors the model consumes.

        For every interval we keep both the CSR coupling and its transpose:
        ``P @ M_{t+1}`` gives the outgoing fingerprint numerators and the
        induced T_t, while ``P^T @ M_t`` gives the incoming ones.  Sparse
        matmul against a dense M is autograd-friendly and avoids ever
        materialising an (nnz x K) intermediate.
        """
        key = (device, dtype)
        if self._torch is not None and self._torch.get("_key") == key:
            return self._torch

        td = torch_dtype(dtype)
        dev = torch.device(device)
        out: dict = {"_key": key}
        out["a"] = [torch.as_tensor(x, dtype=td, device=dev) for x in self.a]

        P_csr, Pt_csr, rows, cols, vals, rsum, csum = [], [], [], [], [], [], []
        for c in self.couplings:
            n, m = c.shape
            r = torch.as_tensor(c.rows, dtype=torch.long, device=dev)
            cc = torch.as_tensor(c.cols, dtype=torch.long, device=dev)
            v = torch.as_tensor(c.values, dtype=td, device=dev)
            coo = torch.sparse_coo_tensor(torch.stack([r, cc]), v, (n, m)).coalesce()
            coo_t = torch.sparse_coo_tensor(torch.stack([cc, r]), v, (m, n)).coalesce()
            P_csr.append(coo.to_sparse_csr())
            Pt_csr.append(coo_t.to_sparse_csr())
            rows.append(r)
            cols.append(cc)
            vals.append(v)
            rsum.append(torch.as_tensor(c.row_sums(), dtype=td, device=dev))
            csum.append(torch.as_tensor(c.col_sums(), dtype=td, device=dev))
        out.update(P=P_csr, Pt=Pt_csr, rows=rows, cols=cols, values=vals,
                   row_sums=rsum, col_sums=csum)
        self._torch = out
        return out

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        d = {
            "tau": self.tau,
            "epsilon": np.array(self.epsilon),
            "cost_scales": np.array(self.cost_scales),
            "kappas": np.array([-1 if k is None else k for k in self.kappas]),
            "T": np.array(self.T),
        }
        for t, x in enumerate(self.a):
            d[f"a_{t}"] = x
        for t, c in enumerate(self.couplings):
            d[f"rows_{t}"] = c.rows
            d[f"cols_{t}"] = c.cols
            d[f"vals_{t}"] = c.values
            d[f"shape_{t}"] = np.array(c.shape)
            d[f"err_{t}"] = np.array(c.marginal_error)
        if self.Z is not None:
            for t, z in enumerate(self.Z):
                d[f"Z_{t}"] = z
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path: str) -> "ReferenceChain":
        d = np.load(path, allow_pickle=False)
        T = int(d["T"])
        a = [d[f"a_{t}"] for t in range(T)]
        couplings = []
        for t in range(T - 1):
            shape = tuple(int(v) for v in d[f"shape_{t}"])
            couplings.append(SinkhornResult(
                rows=d[f"rows_{t}"], cols=d[f"cols_{t}"], values=d[f"vals_{t}"],
                shape=shape, f=np.zeros(shape[0]), g=np.zeros(shape[1]),
                epsilon=float(d["epsilon"]), n_iter=-1,
                marginal_error=float(d[f"err_{t}"]),
                converged=bool(float(d[f"err_{t}"]) < 1e-6),
            ))
        Z = [d[f"Z_{t}"] for t in range(T)] if f"Z_0" in d else None
        kap = [None if k < 0 else int(k) for k in d["kappas"]]
        return cls(couplings=couplings, a=a, tau=d["tau"], epsilon=float(d["epsilon"]),
                   cost_scales=list(d["cost_scales"]), kappas=kap, Z=Z)


def build_reference_chain(
    Z: Sequence[np.ndarray],
    tau: np.ndarray,
    cfg: CouplingConfig = CouplingConfig(),
    a: Optional[Sequence[np.ndarray]] = None,
    supports: Optional[Sequence[Support]] = None,
    cost_scales: Optional[Sequence[float]] = None,
    verbose: int = 1,
) -> ReferenceChain:
    """Fit and freeze the chain of couplings at ``cfg.epsilon``.

    Feasibility: with a restricted support a balanced plan need not exist.  We
    detect this as a Sinkhorn marginal error that will not fall below
    ``cfg.feasibility_tol`` and respond by growing kappa (up to
    ``cfg.kappa_max``), exactly as the spec prescribes.  The kappa actually used
    is recorded per interval and should be reported alongside epsilon.
    """
    Z = [np.asarray(z, dtype=np.float64) for z in Z]
    tau = np.asarray(tau, dtype=float)
    T = len(Z)
    if a is None:
        a = [uniform_weights(z.shape[0]) for z in Z]
    else:
        a = [np.asarray(x, dtype=np.float64) for x in a]

    couplings: List[SinkhornResult] = []
    used_scales: List[float] = []
    used_kappa: List[Optional[int]] = []

    for t in range(T - 1):
        dtau = float(tau[t + 1] - tau[t])
        kappa = None if cfg.support == "dense" else int(cfg.kappa)
        scale = None if cost_scales is None else float(cost_scales[t])
        res = None
        while True:
            if supports is not None:
                sup = supports[t]
                scale_used = 1.0 if scale is None else scale
            else:
                sup, scale_used = build_support(
                    Z[t], Z[t + 1], dtau, kappa=kappa,
                    dense=(cfg.support == "dense"),
                    normalize=cfg.normalize_cost, cost_scale=scale,
                )
            if sup.dense:
                Cd = np.zeros(sup.shape)
                Cd[sup.rows, sup.cols] = sup.cost
                res = sinkhorn_dense(Cd, a[t], a[t + 1], cfg.epsilon,
                                     max_iter=cfg.max_iter, tol=cfg.tol,
                                     device=cfg.device, dtype=cfg.dtype)
            else:
                res = sinkhorn_sparse(sup, a[t], a[t + 1], cfg.epsilon,
                                      max_iter=cfg.max_iter, tol=cfg.tol,
                                      device=cfg.device, dtype=cfg.dtype)
            feasible = res.marginal_error < cfg.feasibility_tol
            if feasible or sup.dense or supports is not None or kappa is None:
                break
            if kappa >= cfg.kappa_max or kappa >= min(sup.shape):
                if verbose:
                    print(f"  [interval {t}] WARNING: marginal error "
                          f"{res.marginal_error:.2e} at kappa={kappa} (max reached); "
                          f"support may be infeasible")
                break
            kappa = int(min(cfg.kappa_max, np.ceil(kappa * cfg.kappa_growth)))
            if verbose:
                print(f"  [interval {t}] marginal error {res.marginal_error:.2e} "
                      f"-> growing kappa to {kappa}")
            scale = scale_used  # keep the cost scale fixed while growing kappa

        couplings.append(res)
        used_scales.append(scale_used)
        used_kappa.append(sup.kappa)
        if verbose:
            print(f"  [interval {t}] dtau={dtau:g} nnz={res.nnz} kappa={sup.kappa} "
                  f"iters={res.n_iter} marg_err={res.marginal_error:.2e}")

    return ReferenceChain(couplings=couplings, a=list(a), tau=tau,
                          epsilon=cfg.epsilon, cost_scales=used_scales,
                          kappas=used_kappa, Z=list(Z))
