"""Bidirectional temporal coarse-graining model (spec 1.6-1.14).

Parameters are the logits U_1..U_T; everything else -- memberships, state
masses, the induced transition map, the low-rank reconstruction, the
fingerprints and their prototypes -- is a deterministic function of them and is
differentiated through.

Identities used for efficiency (each is exact and is unit-tested):

* ``L_+ = sum_k g_k H(phi+_k) - sum_i a_i H(f+_i)``
  which is literally H(Z_{t+1}|Z_t) - H(Z_{t+1}|I_t) = I(I_t; Z_{t+1} | Z_t).
* ``L_expression = sum_i a_i ||z_i||^2 - sum_k g_k ||mu_k||^2``
  (mu is the a-weighted mean, so the cross term collapses).
* ``Phat_ij = a_i b_j * M_t(i,:) W_t M_{t+1}(j,:)^T`` with
  ``W_t = diag(g_t)^-1 T_t diag(g_{t+1})^-1``, so the support evaluation costs
  ``n_t K^2 + |S| K`` rather than ``|S| K^2``.
* ``phi+_tk = A_t(k,:)`` and ``phi-_{t+1,l} = B_t(l,:)`` whenever the coupling
  marginals are exact.  We still compute the prototypes from their definition
  (Eqs. 17-18) so the KL-barycenter property -- and hence the CMI identity --
  survives small marginal error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import numpy as np
import torch

from .config import ModelConfig
from .reference import ReferenceChain
from .utils import entropy, torch_dtype

_LOG_FLOOR = 1e-30


@dataclass
class ObjectiveTerms:
    """Objective value split into its parts, plus per-iteration instrumentation."""

    total: float = 0.0
    compress: float = 0.0
    expression: float = 0.0
    plus: float = 0.0
    minus: float = 0.0
    # per-interval / per-timepoint detail
    compress_t: List[float] = field(default_factory=list)
    plus_t: List[float] = field(default_factory=list)
    minus_t: List[float] = field(default_factory=list)
    expression_t: List[float] = field(default_factory=list)
    # instrumentation (handoff s7 "what to instrument every iteration")
    k_eff: List[float] = field(default_factory=list)
    g_min: List[float] = field(default_factory=list)
    g_max: List[float] = field(default_factory=list)
    floor_fraction: List[float] = field(default_factory=list)
    mean_V_plus: float = float("nan")
    mean_V_minus: float = float("nan")

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "compress": self.compress,
            "expression": self.expression,
            "plus": self.plus,
            "minus": self.minus,
            "k_eff_mean": float(np.mean(self.k_eff)) if self.k_eff else float("nan"),
            "k_eff_min": float(np.min(self.k_eff)) if self.k_eff else float("nan"),
            "g_min": float(np.min(self.g_min)) if self.g_min else float("nan"),
            "floor_fraction": float(np.max(self.floor_fraction)) if self.floor_fraction else 0.0,
            "mean_V_plus": self.mean_V_plus,
            "mean_V_minus": self.mean_V_minus,
        }


class CoarseGrainModel:
    """Holds the frozen chain + the learnable logits, and evaluates Eq. 26."""

    def __init__(
        self,
        chain: ReferenceChain,
        Z: Sequence[np.ndarray],
        cfg: ModelConfig = ModelConfig(),
        U_init: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        self.chain = chain
        self.cfg = cfg
        self.dtype = torch_dtype(cfg.dtype)
        self.device = torch.device(cfg.device)
        self.K = int(cfg.K)
        self.T = chain.T

        self.tt = chain.to_torch(device=cfg.device, dtype=cfg.dtype)
        self.a = self.tt["a"]
        self.Z = [torch.as_tensor(np.asarray(z), dtype=self.dtype, device=self.device)
                  for z in Z]
        if [z.shape[0] for z in self.Z] != chain.n_cells:
            raise ValueError("Z shapes do not match the reference chain")
        self.d = self.Z[0].shape[1]

        # constant part of L_expression: sum_i a_i ||z_i||^2
        self._z_sq = [float((self.a[t] * (self.Z[t] ** 2).sum(1)).sum())
                      for t in range(self.T)]
        self._az = [self.a[t][:, None] * self.Z[t] for t in range(self.T)]

        if U_init is None:
            U_init = [np.zeros((n, self.K)) for n in chain.n_cells]
        self.U = [torch.as_tensor(np.asarray(u), dtype=self.dtype,
                                  device=self.device).clone().requires_grad_(True)
                  for u in U_init]
        for t, u in enumerate(self.U):
            if u.shape != (chain.n_cells[t], self.K):
                raise ValueError(f"U[{t}] has shape {tuple(u.shape)}, expected "
                                 f"{(chain.n_cells[t], self.K)}")

    # ------------------------------------------------------------------
    # basic quantities
    # ------------------------------------------------------------------
    def memberships(self, U: Optional[Sequence[torch.Tensor]] = None) -> List[torch.Tensor]:
        """M_t = softmax(U_t), Eq. 6 -- strictly positive, rows sum to 1."""
        U = self.U if U is None else U
        return [torch.softmax(u, dim=1) for u in U]

    def state_masses(self, M: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """g_t = M_t^T a_t, Eq. 8."""
        return [M[t].transpose(0, 1) @ self.a[t] for t in range(self.T)]

    def induced_transitions(self, M: Optional[Sequence[torch.Tensor]] = None):
        """Return ``(T_list, A_list, B_list, g_list)`` -- Eqs. 9-11.

        T_t is induced from the memberships, never a free parameter.
        """
        M = self.memberships() if M is None else M
        g = self.state_masses(M)
        Ts, As, Bs = [], [], []
        for t in range(self.T - 1):
            PM = torch.sparse.mm(self.tt["P"][t], M[t + 1])       # (n_t, K)
            Tt = M[t].transpose(0, 1) @ PM                        # (K, K)
            Ts.append(Tt)
            As.append(Tt / g[t].clamp_min(_LOG_FLOOR)[:, None])
            Bs.append(Tt.transpose(0, 1) / g[t + 1].clamp_min(_LOG_FLOOR)[:, None])
        return Ts, As, Bs, g

    def fingerprints(self, M: Optional[Sequence[torch.Tensor]] = None):
        """Cell-level fingerprints, Eqs. 15-16, computed from P^ref only.

        Returns ``(F_plus, F_minus)`` with ``F_plus[t]`` of shape (n_t, K) for
        t < T-1 (None at the last timepoint) and ``F_minus[t]`` for t > 0
        (None at the first).
        """
        M = self.memberships() if M is None else M
        Fp: List[Optional[torch.Tensor]] = [None] * self.T
        Fm: List[Optional[torch.Tensor]] = [None] * self.T
        for t in range(self.T - 1):
            PM = torch.sparse.mm(self.tt["P"][t], M[t + 1])
            Fp[t] = PM / self.tt["row_sums"][t].clamp_min(_LOG_FLOOR)[:, None]
            PtM = torch.sparse.mm(self.tt["Pt"][t], M[t])
            Fm[t + 1] = PtM / self.tt["col_sums"][t].clamp_min(_LOG_FLOOR)[:, None]
        return Fp, Fm

    def prototypes(self, M, F, g, t: int) -> torch.Tensor:
        """phi_tk = sum_i a_ti M_t,ik f_ti / g_tk  (Eqs. 17-18).

        This weighted mean is the KL barycentre of the fingerprints assigned to
        state k.  Changing it breaks the CMI identity of Eqs. 22-23.
        """
        num = M[t].transpose(0, 1) @ (self.a[t][:, None] * F)     # (K, K)
        return num / g[t].clamp_min(_LOG_FLOOR)[:, None]

    def expression_prototypes(self, M, g):
        """mu_tk, Eq. 24."""
        return [(M[t].transpose(0, 1) @ self._az[t]) / g[t].clamp_min(_LOG_FLOOR)[:, None]
                for t in range(self.T)]

    # ------------------------------------------------------------------
    # loss pieces
    # ------------------------------------------------------------------
    def _compress_interval(self, M, g, t: int, chunk: Optional[int] = None,
                           PM: Optional[torch.Tensor] = None):
        """KL(P^ref_t || Phat_t) on supp(P^ref_t), Eqs. 12-14.

        Both matrices have total mass 1, so the generalised-KL "-p+q" terms
        cancel and only sum p log(p/q) over the support is needed.

        ``PM = P^ref_t M_{t+1}`` is shared with the fingerprint computation --
        it is the single most expensive op per interval, so it is passed in
        rather than recomputed.
        """
        if PM is None:
            PM = torch.sparse.mm(self.tt["P"][t], M[t + 1])
        Tt = M[t].transpose(0, 1) @ PM
        W = Tt / (g[t].clamp_min(_LOG_FLOOR)[:, None]
                  * g[t + 1].clamp_min(_LOG_FLOOR)[None, :])
        R = M[t] @ W                                              # (n_t, K)

        rows, cols = self.tt["rows"][t], self.tt["cols"][t]
        p = self.tt["values"][t]
        a_i, b_j = self.a[t], self.a[t + 1]
        delta = self.cfg.delta_floor

        nnz = rows.numel()
        step = nnz if chunk is None else int(chunk)
        loss = torch.zeros((), dtype=self.dtype, device=self.device)
        n_floor = 0
        for s0 in range(0, nnz, step):
            sl = slice(s0, min(s0 + step, nnz))
            r, c = rows[sl], cols[sl]
            phat = a_i[r] * b_j[c] * (R[r] * M[t + 1][c]).sum(1)
            ps = p[sl]
            loss = loss + (ps * (torch.log(ps.clamp_min(_LOG_FLOOR))
                                 - torch.log(phat + delta))).sum()
            n_floor += int((phat < delta).sum())
        return loss, n_floor / max(nnz, 1), Tt

    def _fingerprint_loss(self, F: torch.Tensor, phi: torch.Tensor,
                          a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """sum_{i,k} a_i M_ik KL(f_i || phi_k) via the entropy decomposition."""
        H_phi = entropy(phi, axis=1, eps=self.cfg.fingerprint_floor)
        H_f = entropy(F, axis=1, eps=self.cfg.fingerprint_floor)
        return (g * H_phi).sum() - (a * H_f).sum()

    def fingerprint_loss_direct(self, F, phi, M, a) -> torch.Tensor:
        """Literal Eq. 19/20 -- O(nK^2); used by the tests to check the fast form."""
        eps = self.cfg.fingerprint_floor
        logF = torch.log(F.clamp_min(eps))
        logphi = torch.log(phi.clamp_min(eps))
        # KL(f_i || phi_k) for every (i, k)
        KL = (F * logF).sum(1)[:, None] - F @ logphi.transpose(0, 1)
        return (a[:, None] * M * KL).sum()

    # ------------------------------------------------------------------
    def objective(
        self,
        U: Optional[Sequence[torch.Tensor]] = None,
        chunk: Optional[int] = None,
        with_diagnostics: bool = True,
    ):
        """Evaluate Eq. 26.  Returns ``(loss_tensor, ObjectiveTerms)``."""
        cfg = self.cfg
        M = self.memberships(U)
        g = self.state_masses(M)

        terms = ObjectiveTerms()
        zero = torch.zeros((), dtype=self.dtype, device=self.device)
        L_comp, L_plus, L_minus, L_expr = zero, zero.clone(), zero.clone(), zero.clone()

        # -- expression coherence (Eq. 25) --------------------------------
        mus = self.expression_prototypes(M, g)
        scale_x = 1.0 / self.d if cfg.scale_expression_by_dim else 1.0
        for t in range(self.T):
            val = (self._z_sq[t] - (g[t] * (mus[t] ** 2).sum(1)).sum()) * scale_x
            L_expr = L_expr + val
            if with_diagnostics:
                terms.expression_t.append(float(val.detach()))

        # -- fingerprints, prototypes, and the two CMI losses -------------
        need_fp = cfg.lambda_plus != 0.0 or cfg.lambda_minus != 0.0 or with_diagnostics
        Fp: List[Optional[torch.Tensor]] = [None] * self.T
        Fm: List[Optional[torch.Tensor]] = [None] * self.T
        phip: List[Optional[torch.Tensor]] = [None] * self.T
        phim: List[Optional[torch.Tensor]] = [None] * self.T

        for t in range(self.T - 1):
            PM = torch.sparse.mm(self.tt["P"][t], M[t + 1])

            # -- compression (always needed) ------------------------------
            val, frac, _ = self._compress_interval(M, g, t, chunk=chunk, PM=PM)
            L_comp = L_comp + val
            if with_diagnostics:
                terms.compress_t.append(float(val.detach()))
                terms.floor_fraction.append(frac)

            if need_fp:
                Fp[t] = PM / self.tt["row_sums"][t].clamp_min(_LOG_FLOOR)[:, None]
                phip[t] = self.prototypes(M, Fp[t], g, t)
                lp = self._fingerprint_loss(Fp[t], phip[t], self.a[t], g[t])
                L_plus = L_plus + lp
                if with_diagnostics:
                    terms.plus_t.append(float(lp.detach()))

                PtM = torch.sparse.mm(self.tt["Pt"][t], M[t])
                Fm[t + 1] = PtM / self.tt["col_sums"][t].clamp_min(_LOG_FLOOR)[:, None]
                phim[t + 1] = self.prototypes(M, Fm[t + 1], g, t + 1)
                lm = self._fingerprint_loss(Fm[t + 1], phim[t + 1],
                                            self.a[t + 1], g[t + 1])
                L_minus = L_minus + lm
                if with_diagnostics:
                    terms.minus_t.append(float(lm.detach()))

        total = (cfg.lambda_compress * L_comp + cfg.lambda_x * L_expr
                 + cfg.lambda_plus * L_plus + cfg.lambda_minus * L_minus)

        if with_diagnostics:
            terms.total = float(total.detach())
            terms.compress = float(L_comp.detach())
            terms.expression = float(L_expr.detach())
            terms.plus = float(L_plus.detach())
            terms.minus = float(L_minus.detach())
            with torch.no_grad():
                for t in range(self.T):
                    gt = g[t]
                    terms.k_eff.append(float(torch.exp(entropy(gt, axis=0))))
                    terms.g_min.append(float(gt.min()))
                    terms.g_max.append(float(gt.max()))
                vp, vm = [], []
                for t in range(self.T):
                    if Fp[t] is not None:
                        vp.append(float(self._mean_dispersion(Fp[t], phip[t], M, g, t)))
                    if Fm[t] is not None:
                        vm.append(float(self._mean_dispersion(Fm[t], phim[t], M, g, t)))
                terms.mean_V_plus = float(np.mean(vp)) if vp else float("nan")
                terms.mean_V_minus = float(np.mean(vm)) if vm else float("nan")
        return total, terms

    def _mean_dispersion(self, F, phi, M, g, t) -> torch.Tensor:
        """g-weighted mean of V_tk (Eqs. 29-30); equals L_pm,t here."""
        V = self.within_state_dispersion(F, phi, M, g, t)
        return (g[t] * V).sum() / g[t].sum().clamp_min(_LOG_FLOOR)

    def within_state_dispersion(self, F, phi, M, g, t) -> torch.Tensor:
        """V_tk = (1/g_tk) sum_i a_ti M_t,ik KL(f_ti || phi_tk), Eqs. 29-30."""
        eps = self.cfg.fingerprint_floor
        logF = torch.log(F.clamp_min(eps))
        logphi = torch.log(phi.clamp_min(eps))
        KL = (F * logF).sum(1)[:, None] - F @ logphi.transpose(0, 1)   # (n, K)
        num = (self.a[t][:, None] * M[t] * KL).sum(0)
        return num / g[t].clamp_min(_LOG_FLOOR)

    # ------------------------------------------------------------------
    # parameter plumbing for the optimiser
    # ------------------------------------------------------------------
    def get_U(self) -> List[torch.Tensor]:
        return self.U

    def set_U(self, U: Sequence[torch.Tensor]) -> None:
        with torch.no_grad():
            for a, b in zip(self.U, U):
                a.copy_(b)

    def clone_U(self) -> List[torch.Tensor]:
        return [u.detach().clone() for u in self.U]

    def zero_grad(self) -> None:
        for u in self.U:
            if u.grad is not None:
                u.grad = None

    def numpy_memberships(self) -> List[np.ndarray]:
        return [m.detach().cpu().numpy() for m in self.memberships()]

    def active_states(self, g_min: Optional[float] = None) -> List[np.ndarray]:
        """Boolean mask of states above the visualisation mass threshold.

        Visualisation only -- this never alters the optimisation (spec 1.6).
        """
        gm = self.cfg.g_min if g_min is None else g_min
        with torch.no_grad():
            g = self.state_masses(self.memberships())
        return [(gt.cpu().numpy() > gm) for gt in g]
