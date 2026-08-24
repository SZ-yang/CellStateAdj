"""Optimisation of the reduced objective L(U_1..U_T)  (spec 1.19).

The objective is *self-referential*: L_+ at t depends on M_{t+1} and L_- at
t+1 depends on M_t, so the feature space being clustered moves with the labels.
Simultaneous updates of all M_t can therefore increase the objective.  The
default optimiser is full-gradient descent with an Armijo backtracking line
search that accepts a step only when the complete objective decreases; the
block-coordinate alternative updates one timepoint at a time with neighbours
frozen and damping (Eq. 35), and likewise only accepts decreasing steps.

Convergence should be reported as "to a fixed point", not as monotone descent,
unless the line search actually guaranteed it -- ``FitResult.monotone`` records
which happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import time
import numpy as np
import torch

from .config import ModelConfig, OptimConfig
from .model import CoarseGrainModel, ObjectiveTerms
from .reference import ReferenceChain
from .utils import hard_labels, onehot_logits, set_seed


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------

def _kmeans(Z: np.ndarray, K: int, seed: int, n_iter: int = 50) -> np.ndarray:
    """k-means++ then Lloyd; falls back to sklearn when available."""
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=4, random_state=seed).fit(Z)
        return km.labels_
    except Exception:
        pass
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    centers = [Z[rng.integers(n)]]
    d2 = ((Z - centers[0]) ** 2).sum(1)
    for _ in range(1, K):
        probs = d2 / max(d2.sum(), 1e-30)
        centers.append(Z[rng.choice(n, p=probs)])
        d2 = np.minimum(d2, ((Z - centers[-1]) ** 2).sum(1))
    C = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        D = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new = D.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for k in range(K):
            m = labels == k
            if m.any():
                C[k] = Z[m].mean(0)
    return labels


def initialize_logits(
    Z: Sequence[np.ndarray],
    K: int,
    method: str = "kmeans",
    scale: float = 3.0,
    seed: int = 0,
) -> List[np.ndarray]:
    """Initial U_t.

    ``kmeans``: fine over-clustering per timepoint (spec Alg. 1 line 7),
    smoothed to strictly positive memberships.  ``random``: small Gaussian
    logits, used for the initialisation-spread study.
    """
    rng = np.random.default_rng(seed)
    Us = []
    for t, z in enumerate(Z):
        z = np.asarray(z)
        if method == "kmeans":
            k = int(min(K, z.shape[0]))
            lab = _kmeans(z, k, seed=seed + 1000 * t)
            U = onehot_logits(lab, K, scale=scale)
            U += 0.01 * rng.standard_normal(U.shape)
        elif method == "random":
            U = scale * rng.standard_normal((z.shape[0], K)) / np.sqrt(K)
        else:
            raise ValueError(f"unknown init method {method!r}")
        Us.append(U)
    return Us


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    model: CoarseGrainModel
    M: List[np.ndarray]
    terms: ObjectiveTerms
    history: List[dict] = field(default_factory=list)
    converged: bool = False
    monotone: bool = True
    n_iter: int = 0
    seed: int = 0
    wall_time: float = 0.0
    restarts: List[dict] = field(default_factory=list)

    @property
    def objective(self) -> float:
        return self.terms.total

    @property
    def labels(self) -> List[np.ndarray]:
        return [m.argmax(1) for m in self.M]

    def history_array(self, key: str) -> np.ndarray:
        return np.array([h.get(key, np.nan) for h in self.history])

    def summary(self) -> dict:
        d = self.terms.as_dict()
        d.update(converged=self.converged, monotone=self.monotone,
                 n_iter=self.n_iter, wall_time=self.wall_time, seed=self.seed)
        return d


# ---------------------------------------------------------------------------
# core loops
# ---------------------------------------------------------------------------

def _flat_grad_norm_sq(grads) -> float:
    return float(sum((g ** 2).sum() for g in grads))


def _flatten(tensors) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _unflatten(vec: torch.Tensor, like) -> List[torch.Tensor]:
    out, off = [], 0
    for t in like:
        n = t.numel()
        out.append(vec[off:off + n].view_as(t))
        off += n
    return out


class _LBFGS:
    """Two-loop recursion over the flattened logits.

    Only the search direction changes; every proposal still has to pass the
    Armijo test on the *complete* objective, so the "accept only when the
    objective decreases" guarantee is untouched.  A curvature pair is stored
    only when s.y > 0, and a non-descent direction falls back to -grad.
    """

    def __init__(self, memory: int = 10) -> None:
        self.memory = memory
        self.s: List[torch.Tensor] = []
        self.y: List[torch.Tensor] = []

    def push(self, s: torch.Tensor, y: torch.Tensor) -> None:
        sy = float(s @ y)
        if not np.isfinite(sy) or sy <= 1e-12 * float(s @ s) ** 0.5 * float(y @ y) ** 0.5:
            return
        self.s.append(s)
        self.y.append(y)
        if len(self.s) > self.memory:
            self.s.pop(0)
            self.y.pop(0)

    def direction(self, grad: torch.Tensor) -> torch.Tensor:
        if not self.s:
            return -grad
        q = grad.clone()
        alphas = []
        for s, y in zip(reversed(self.s), reversed(self.y)):
            rho = 1.0 / float(y @ s)
            a = rho * float(s @ q)
            alphas.append(a)
            q -= a * y
        s_last, y_last = self.s[-1], self.y[-1]
        gamma = float(s_last @ y_last) / max(float(y_last @ y_last), 1e-30)
        r = gamma * q
        for (s, y), a in zip(zip(self.s, self.y), reversed(alphas)):
            rho = 1.0 / float(y @ s)
            beta = rho * float(y @ r)
            r += s * (a - beta)
        d = -r
        if float(grad @ d) >= 0:      # not a descent direction
            return -grad
        return d


def _membership_change(M_old, M_new) -> float:
    out = 0.0
    for a, b in zip(M_old, M_new):
        out = max(out, float(torch.linalg.norm(a - b) / np.sqrt(a.shape[0])))
    return out


def _log_line(it: int, terms: ObjectiveTerms, extra: dict) -> str:
    return (f"  it {it:4d}  L={terms.total: .6e}  "
            f"comp={terms.compress: .4e} expr={terms.expression: .4e} "
            f"L+={terms.plus: .4e} L-={terms.minus: .4e}  "
            f"Keff={np.mean(terms.k_eff):5.2f}(min {np.min(terms.k_eff):5.2f})  "
            f"gmin={np.min(terms.g_min):.2e}  floor={np.max(terms.floor_fraction) if terms.floor_fraction else 0:.1e}  "
            f"|dM|={extra.get('dM', float('nan')):.2e}  step={extra.get('step', float('nan')):.2e}")


def _run_full_gradient(model: CoarseGrainModel, opt: OptimConfig,
                       chunk: Optional[int]) -> FitResult:
    t0 = time.time()
    history: List[dict] = []
    step = opt.step_init
    monotone = True
    converged = False
    lbfgs = _LBFGS(opt.lbfgs_memory) if opt.direction == "lbfgs" else None

    loss, terms = model.objective(chunk=chunk)
    prev_obj = float(loss.detach())
    prev_flat_U: Optional[torch.Tensor] = None
    prev_flat_g: Optional[torch.Tensor] = None
    stalled = 0
    it = 0

    for it in range(1, opt.max_iter + 1):
        model.zero_grad()
        loss, _ = model.objective(chunk=chunk, with_diagnostics=False)
        loss.backward()
        grads = [u.grad.detach().clone() for u in model.U]
        gnorm2 = _flat_grad_norm_sq(grads)
        if not np.isfinite(gnorm2):
            raise FloatingPointError("non-finite gradient")
        if gnorm2 == 0.0:
            converged = True
            break

        U_old = model.clone_U()
        M_old = model.memberships(U_old)
        flat_U = _flatten(U_old)
        flat_g = _flatten(grads)

        if lbfgs is not None:
            if prev_flat_U is not None:
                lbfgs.push(flat_U - prev_flat_U, flat_g - prev_flat_g)
            d_flat = lbfgs.direction(flat_g)
            # L-BFGS is scaled to take a unit step; steepest descent is not
            s = 1.0 if lbfgs.s else opt.step_init
        else:
            d_flat = -flat_g
            s = step
        directional = float(flat_g @ d_flat)      # < 0 for a descent direction
        direction = _unflatten(d_flat, U_old)

        base = float(loss.detach())
        accepted = False
        for _ in range(opt.max_backtrack):
            trial = [u + s * dd for u, dd in zip(U_old, direction)]
            with torch.no_grad():
                new_loss, _ = model.objective(U=trial, chunk=chunk,
                                              with_diagnostics=False)
            if float(new_loss.detach()) <= base + opt.armijo_c * s * directional:
                model.set_U(trial)
                accepted = True
                break
            s *= opt.step_shrink
        if not accepted:
            # no progress along this direction: reset the curvature memory once
            # and retry with plain steepest descent before giving up
            if lbfgs is not None and lbfgs.s:
                lbfgs = _LBFGS(opt.lbfgs_memory)
                prev_flat_U = prev_flat_g = None
                continue
            converged = True
            step = s
            break

        prev_flat_U, prev_flat_g = flat_U, flat_g
        if lbfgs is None:
            step = s * opt.step_grow
        with torch.no_grad():
            M_new = model.memberships()
            dM = _membership_change(M_old, M_new)
        obj = float(new_loss.detach())
        rel = abs(prev_obj - obj) / max(abs(prev_obj), 1e-12)
        if obj > prev_obj + 1e-12:
            monotone = False
        prev_obj = obj
        with torch.no_grad():                      # one diagnostic pass per step
            _, terms = model.objective(chunk=chunk)

        rec = dict(iter=it, step=s, dM=dM, rel_change=rel, **terms.as_dict())
        history.append(rec)
        if opt.verbose and (it % opt.log_every == 0):
            print(_log_line(it, terms, rec))

        if rel < opt.tol_objective:
            stalled += 1
        else:
            stalled = 0
        if (rel < opt.tol_objective and dM < opt.tol_membership) or \
                stalled >= opt.patience:
            converged = True
            break

    loss, terms = model.objective(chunk=chunk)
    return FitResult(model=model, M=model.numpy_memberships(), terms=terms,
                     history=history, converged=converged, monotone=monotone,
                     n_iter=it, seed=opt.seed, wall_time=time.time() - t0)


def _run_block_coordinate(model: CoarseGrainModel, opt: OptimConfig,
                          chunk: Optional[int]) -> FitResult:
    """Forward-backward sweeps: update one U_t with all others frozen.

    Each block proposal is damped (Eq. 35) and accepted only when the
    *complete* objective decreases -- the neighbours' losses depend on U_t, so a
    locally good move can still be globally bad.
    """
    t0 = time.time()
    history: List[dict] = []
    monotone = True
    converged = False
    step = {t: opt.step_init for t in range(model.T)}

    loss, terms = model.objective(chunk=chunk)
    prev_obj = float(loss.detach())
    stalled = 0
    it = 0
    for it in range(1, opt.max_iter + 1):
        M_before = [m.detach().clone() for m in model.memberships()]
        order = list(range(model.T)) + list(range(model.T - 2, -1, -1))
        for _sweep in range(opt.n_sweeps_per_iter):
            for t in order:
                model.zero_grad()
                loss, _ = model.objective(chunk=chunk)
                loss.backward()
                gt = model.U[t].grad
                if gt is None:
                    continue
                gt = gt.detach().clone()
                gnorm2 = float((gt ** 2).sum())
                if gnorm2 == 0.0:
                    continue
                base = float(loss.detach())
                U_t_old = model.U[t].detach().clone()
                s = step[t]
                accepted = False
                for _ in range(opt.max_backtrack):
                    prop = U_t_old - s * gt
                    if opt.damping < 1.0:
                        # damping in membership space, then back to logits
                        with torch.no_grad():
                            M_prop = torch.softmax(prop, 1)
                            M_old_t = torch.softmax(U_t_old, 1)
                            M_mix = (1 - opt.damping) * M_old_t + opt.damping * M_prop
                            prop = torch.log(M_mix.clamp_min(1e-30))
                    trial = list(model.clone_U())
                    trial[t] = prop
                    with torch.no_grad():
                        new_loss, _ = model.objective(U=trial, chunk=chunk)
                    if float(new_loss.detach()) <= base - opt.armijo_c * s * gnorm2 * opt.damping:
                        model.set_U(trial)
                        accepted = True
                        break
                    s *= opt.step_shrink
                step[t] = s * opt.step_grow if accepted else s

        loss, terms = model.objective(chunk=chunk)
        obj = float(loss.detach())
        with torch.no_grad():
            dM = _membership_change(M_before, model.memberships())
        rel = abs(prev_obj - obj) / max(abs(prev_obj), 1e-12)
        if obj > prev_obj + 1e-12:
            monotone = False
        prev_obj = obj
        rec = dict(iter=it, step=float(np.mean(list(step.values()))), dM=dM,
                   rel_change=rel, **terms.as_dict())
        history.append(rec)
        if opt.verbose and (it % opt.log_every == 0):
            print(_log_line(it, terms, rec))
        stalled = stalled + 1 if rel < opt.tol_objective else 0
        if (rel < opt.tol_objective and dM < opt.tol_membership) or \
                stalled >= opt.patience:
            converged = True
            break

    loss, terms = model.objective(chunk=chunk)
    return FitResult(model=model, M=model.numpy_memberships(), terms=terms,
                     history=history, converged=converged, monotone=monotone,
                     n_iter=it, seed=opt.seed, wall_time=time.time() - t0)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def fit(
    chain: ReferenceChain,
    Z: Sequence[np.ndarray],
    model_cfg: ModelConfig = ModelConfig(),
    opt_cfg: OptimConfig = OptimConfig(),
    U_init: Optional[Sequence[np.ndarray]] = None,
    chunk: Optional[int] = None,
) -> FitResult:
    """Fit memberships, optionally from several initialisations.

    With ``opt_cfg.n_init > 1`` every restart is run to convergence, the best
    objective is returned, and the agreement between restarts is recorded in
    ``FitResult.restarts`` (initialisation stability is a reported quantity,
    spec 1.21, not a nuisance).
    """
    best: Optional[FitResult] = None
    restarts: List[dict] = []
    all_labels: List[List[np.ndarray]] = []

    for r in range(max(1, opt_cfg.n_init)):
        seed = opt_cfg.seed + r
        set_seed(seed)
        if U_init is not None and r == 0:
            U0 = [np.asarray(u) for u in U_init]
        else:
            method = opt_cfg.init if r == 0 else "random"
            U0 = initialize_logits(Z, model_cfg.K, method=method,
                                   scale=opt_cfg.init_logit_scale, seed=seed)
        model = CoarseGrainModel(chain, Z, model_cfg, U_init=U0)
        if opt_cfg.verbose:
            print(f"[fit] restart {r} (seed={seed}, init="
                  f"{'given' if (U_init is not None and r == 0) else (opt_cfg.init if r == 0 else 'random')})")
        cfg_r = OptimConfig(**{**opt_cfg.__dict__, "seed": seed})
        if opt_cfg.method == "full_gradient":
            res = _run_full_gradient(model, cfg_r, chunk)
        elif opt_cfg.method == "block_coordinate":
            res = _run_block_coordinate(model, cfg_r, chunk)
        else:
            raise ValueError(f"unknown optimiser {opt_cfg.method!r}")
        restarts.append(res.summary())
        all_labels.append(res.labels)
        if best is None or res.objective < best.objective:
            best = res

    if len(all_labels) > 1:
        best.restarts = restarts
        agree = _restart_agreement(all_labels)
        for rec, ari in zip(best.restarts, agree):
            rec["mean_ari_to_others"] = ari
    else:
        best.restarts = restarts
    return best


def _restart_agreement(all_labels: List[List[np.ndarray]]) -> List[float]:
    """Mean adjusted Rand index of each restart against all the others."""
    try:
        from sklearn.metrics import adjusted_rand_score as ari
    except Exception:  # pragma: no cover
        return [float("nan")] * len(all_labels)
    R = len(all_labels)
    out = []
    for i in range(R):
        vals = []
        for j in range(R):
            if i == j:
                continue
            vals.append(float(np.mean([ari(a, b) for a, b in
                                       zip(all_labels[i], all_labels[j])])))
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out
