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
from .cost import Support, build_support, resolve_cost_scales
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
    # The ONE feasibility standard.  Kept on the chain so that `feasible` and
    # the kappa-growth decision that produced it cannot drift apart.
    feasibility_tol: float = 1e-6
    cost_scale_mode: str = "global"

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
        return all(e < self.feasibility_tol for e in self.marginal_errors())

    def marginal_errors(self) -> List[float]:
        return [c.marginal_error for c in self.couplings]

    def infeasible_intervals(self) -> List[int]:
        return [t for t, e in enumerate(self.marginal_errors())
                if e >= self.feasibility_tol]

    def summary(self) -> dict:
        return {
            "T": self.T,
            "n_cells": self.n_cells,
            "epsilon": self.epsilon,
            "kappas": self.kappas,
            "cost_scale_mode": self.cost_scale_mode,
            "cost_scales": list(self.cost_scales),
            "nnz": [c.nnz for c in self.couplings],
            "marginal_error": self.marginal_errors(),
            "feasibility_tol": self.feasibility_tol,
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
            "feasibility_tol": np.array(self.feasibility_tol),
            "cost_scale_mode": np.array(self.cost_scale_mode),
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
            ))
        Z = [d[f"Z_{t}"] for t in range(T)] if f"Z_0" in d else None
        kap = [None if k < 0 else int(k) for k in d["kappas"]]
        return cls(couplings=couplings, a=a, tau=d["tau"], epsilon=float(d["epsilon"]),
                   cost_scales=list(d["cost_scales"]), kappas=kap, Z=Z,
                   feasibility_tol=float(d["feasibility_tol"])
                   if "feasibility_tol" in d else 1e-6,
                   cost_scale_mode=str(d["cost_scale_mode"])
                   if "cost_scale_mode" in d else "global")


def _underflow_limit(dtype: str) -> float:
    """|log(smallest positive normal)| for the working dtype.

    exp(-C/eps) below this underflows, so an interval whose max(cost)/epsilon
    approaches it cannot have its marginals met however long Sinkhorn runs.
    """
    return 708.4 if dtype == "float64" else 88.7


class SinkhornConvergenceError(RuntimeError):
    """Sinkhorn ran out of iterations while still making progress.

    Distinct from :class:`InfeasibleCouplingError` and fixed the opposite way:
    the support is fine, there were simply not enough iterations.  Small
    epsilon and a large cost scale both slow convergence, so this shows up most
    at the sharp end of an epsilon scan.
    """

    def __init__(self, interval: int, n_iter: int, marginal_error: float,
                 tol: float, epsilon: float, cost_ratio: Optional[float] = None,
                 underflow_limit: Optional[float] = None):
        self.interval = interval
        self.marginal_error = marginal_error
        self.cost_ratio = cost_ratio
        msg = (f"interval {interval}: Sinkhorn stopped after {n_iter} iterations "
               f"with marginal error {marginal_error:.3e} > feasibility_tol "
               f"{tol:.1e} at epsilon={epsilon:g}. The support is not the "
               f"problem -- growing kappa will not help.")
        if (cost_ratio is not None and underflow_limit is not None
                and cost_ratio > 0.3 * underflow_limit):
            msg += (
                f" max(cost)/epsilon = {cost_ratio:.0f}, against a "
                f"{underflow_limit:.0f} underflow limit for this dtype: epsilon "
                f"is too small for this interval's cost RANGE, so exp(-C/eps) "
                f"loses precision and the marginals cannot be met at any "
                f"iteration count. Use a larger epsilon (or dtype='float64' if "
                f"not already). Note a shared cost scale makes intervals with "
                f"genuinely wider transport more expensive -- that is correct "
                f"behaviour and this interval is telling you it needs more "
                f"entropy."
            )
        else:
            msg += f" Raise CouplingConfig.max_iter, or use a larger epsilon."
        super().__init__(msg)


class InfeasibleCouplingError(RuntimeError):
    """No balanced plan exists on the support that was reachable.

    Raised rather than returned, because an unbalanced P^ref silently breaks
    ``T_t 1 = g_t``: ``T_t 1 = M_t^T rowsums(P)``, which equals
    ``M_t^T a_t = g_t`` only when the marginals hold.  Every downstream
    transition number -- A_t, B_t, N_child, N_parent, the DAG edges -- is then
    wrong, with no other symptom.  Note the failure is invisible at
    initialisation: with uniform memberships both sides collapse to 1/K.
    """

    def __init__(self, interval: int, kappa, marginal_error: float, tol: float):
        self.interval = interval
        self.kappa = kappa
        self.marginal_error = marginal_error
        super().__init__(
            f"interval {interval}: Sinkhorn marginal error {marginal_error:.3e} "
            f"exceeds feasibility_tol {tol:.1e} at kappa={kappa}. The restricted "
            f"support admits no balanced coupling. Raise kappa/kappa_max, set "
            f"on_infeasible='dense' to fall back to a dense support, or set "
            f"support='dense'. Do NOT proceed with an unbalanced coupling: it "
            f"breaks T_t 1 = g_t and A_t stops being row-stochastic."
        )


def solve_interval(
    Za: np.ndarray,
    Zb: np.ndarray,
    dtau: float,
    a: np.ndarray,
    b: np.ndarray,
    cfg: CouplingConfig,
    cost_scale: Optional[float] = None,
    interval: int = 0,
    support: Optional[Support] = None,
    verbose: int = 1,
):
    """Solve one adjacent-time coupling, growing kappa until feasible.

    Shared by :func:`build_reference_chain` and the epsilon scan so that both
    apply the same feasibility standard -- otherwise the epsilon chosen in
    Step 1 would be validated under weaker conditions than the chain later
    built at it.

    Returns ``(SinkhornResult, Support, cost_scale)``.
    """
    # Stop as soon as the plan is comfortably feasible rather than chasing an
    # arbitrarily tight target: the extra digits buy nothing downstream.
    solve_tol = min(cfg.tol, cfg.feasibility_tol / 10.0)

    def _solve(sup: Support) -> SinkhornResult:
        if sup.dense:
            Cd = np.zeros(sup.shape)
            Cd[sup.rows, sup.cols] = sup.cost
            return sinkhorn_dense(Cd, a, b, cfg.epsilon, max_iter=cfg.max_iter,
                                  tol=solve_tol, device=cfg.device,
                                  dtype=cfg.dtype)
        return sinkhorn_sparse(sup, a, b, cfg.epsilon, max_iter=cfg.max_iter,
                               tol=solve_tol, device=cfg.device, dtype=cfg.dtype)

    # a caller-supplied support is used as given -- no growth, no fallback
    if support is not None:
        res = _solve(support)
        scale = 1.0 if cost_scale is None else cost_scale
        if res.marginal_error >= cfg.feasibility_tol and cfg.on_infeasible == "raise":
            raise InfeasibleCouplingError(interval, support.kappa,
                                          res.marginal_error, cfg.feasibility_tol)
        return res, support, scale

    kappa = None if cfg.support == "dense" else int(cfg.kappa)
    scale = cost_scale
    while True:
        sup, scale = build_support(Za, Zb, dtau, kappa=kappa,
                                   dense=(cfg.support == "dense"),
                                   cost_scale=scale)
        res = _solve(sup)
        if res.marginal_error < cfg.feasibility_tol:
            return res, sup, scale

        ratio = float(sup.cost.max()) / max(cfg.epsilon, 1e-300)
        limit = _underflow_limit(cfg.dtype)

        # A DENSE support always admits a balanced plan for positive marginals,
        # so a miss there is never infeasibility -- it is convergence or
        # precision.  Misreporting it as infeasible sends the user to grow kappa,
        # which cannot possibly help.
        if sup.dense or kappa is None:
            raise SinkhornConvergenceError(interval, res.n_iter,
                                           res.marginal_error,
                                           cfg.feasibility_tol, cfg.epsilon,
                                           ratio, limit)
        # Sparse and still improving: out of iterations, not out of support.
        if res.hit_max_iter and not res.stalled:
            raise SinkhornConvergenceError(interval, res.n_iter,
                                           res.marginal_error,
                                           cfg.feasibility_tol, cfg.epsilon,
                                           ratio, limit)

        if kappa < cfg.kappa_max and kappa < min(sup.shape):
            kappa = int(min(cfg.kappa_max, np.ceil(kappa * cfg.kappa_growth)))
            if verbose:
                print(f"  [interval {interval}] marginal error "
                      f"{res.marginal_error:.2e} -> growing kappa to {kappa}")
            continue

        # kappa exhausted -- escalate
        if cfg.on_infeasible == "dense":
            if verbose:
                print(f"  [interval {interval}] marginal error "
                      f"{res.marginal_error:.2e} at kappa={kappa} (max); "
                      f"falling back to a dense support")
            sup, scale = build_support(Za, Zb, dtau, dense=True, cost_scale=scale)
            res = _solve(sup)
            if res.marginal_error >= cfg.feasibility_tol:
                raise InfeasibleCouplingError(interval, "dense",
                                              res.marginal_error,
                                              cfg.feasibility_tol)
            return res, sup, scale
        if cfg.on_infeasible == "warn":
            if verbose:
                print(f"  [interval {interval}] WARNING: marginal error "
                      f"{res.marginal_error:.2e} at kappa={kappa} exceeds "
                      f"feasibility_tol {cfg.feasibility_tol:.1e}. The coupling "
                      f"is UNBALANCED; A_t will not be row-stochastic and every "
                      f"transition number derived from it is invalid.")
            return res, sup, scale
        raise InfeasibleCouplingError(interval, kappa, res.marginal_error,
                                      cfg.feasibility_tol)


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

    Two things this function is responsible for getting right:

    * the cost scale is SHARED across intervals (``cfg.cost_scale_mode``), so
      the ``/ dtau`` of Eq. 1 survives normalisation;
    * a restricted support need not admit any balanced plan, so kappa is grown
      and, failing that, ``cfg.on_infeasible`` decides whether to fall back to a
      dense support, warn, or raise.  The default is to raise.

    The kappa and scale actually used are recorded per interval and should be
    reported alongside epsilon.
    """
    Z = [np.asarray(z, dtype=np.float64) for z in Z]
    tau = np.asarray(tau, dtype=float)
    T = len(Z)
    if a is None:
        a = [uniform_weights(z.shape[0]) for z in Z]
    else:
        a = [np.asarray(x, dtype=np.float64) for x in a]

    if cost_scales is None:
        scales = resolve_cost_scales(Z, tau, cfg.cost_scale_mode)
    else:
        scales = [float(s) for s in cost_scales]
    if verbose:
        print(f"  cost_scale_mode={cfg.cost_scale_mode} scale={scales[0]:.4g}"
              + ("" if cfg.cost_scale_mode != "per_interval"
                 else "  [WARNING: per-interval scaling cancels dtau]"))

    couplings: List[SinkhornResult] = []
    used_scales: List[float] = []
    used_kappa: List[Optional[int]] = []

    for t in range(T - 1):
        dtau = float(tau[t + 1] - tau[t])
        res, sup, scale_used = solve_interval(
            Z[t], Z[t + 1], dtau, a[t], a[t + 1], cfg,
            cost_scale=scales[t], interval=t,
            support=None if supports is None else supports[t], verbose=verbose,
        )
        couplings.append(res)
        used_scales.append(scale_used)
        used_kappa.append(sup.kappa)
        if verbose:
            print(f"  [interval {t}] dtau={dtau:g} nnz={res.nnz} kappa={sup.kappa} "
                  f"iters={res.n_iter} marg_err={res.marginal_error:.2e}")

    return ReferenceChain(couplings=couplings, a=list(a), tau=tau,
                          epsilon=cfg.epsilon, cost_scales=used_scales,
                          kappas=used_kappa, Z=list(Z),
                          feasibility_tol=cfg.feasibility_tol,
                          cost_scale_mode=cfg.cost_scale_mode)
