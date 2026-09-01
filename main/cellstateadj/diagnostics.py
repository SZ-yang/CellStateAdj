"""Post-fit diagnostics (spec 1.16-1.18) plus the degeneracy checks.

These are not optional extras -- they carry the interpretation.

* ``V+/V-`` (Eqs. 29-30): within-state fingerprint dispersion.  Deliberately
  NOT driven to zero by the objective.
* ``G+/G-`` (Eqs. 31-32): the same dispersion measured against a cross-fitted
  smooth predictor of the fingerprint from expression geometry.  Because
  P^ref is built from distances in z-space, a state straddling a geometric
  boundary gets a high V for free.  G is what makes V interpretable:

      high V, low  G -> heterogeneity is explained by smooth expression geometry
      high V, high G -> excess non-smooth transition heterogeneity

  The second is NECESSARY BUT NOT SUFFICIENT evidence for fate priming.  It is
  also consistent with coupling noise, representation error, or a bad null.
  Never report it as priming on its own.
* ``N_child / N_parent`` (Eqs. 33-34): population-level branching and merging,
  which is a *different* claim from within-state heterogeneity.  No mode
  counting on A_t(k,:) -- state indices carry no topology.
* ``degeneracy_check``: the empirical demonstration that fingerprints taken
  from Phat are constant within a state for ANY assignment (Degeneracy 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch

from .config import DiagnosticsConfig
from .model import CoarseGrainModel
from .utils import entropy

_EPS = 1e-30


# ---------------------------------------------------------------------------
# geometric null
# ---------------------------------------------------------------------------

def _median_bandwidth(Z: np.ndarray, rng: np.random.Generator, n_probe: int = 2000) -> float:
    n = Z.shape[0]
    idx = rng.choice(n, size=int(min(n_probe, n)), replace=False)
    Zs = Z[idx]
    d2 = ((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(len(idx), k=1)
    if iu[0].size == 0:
        return 1.0
    med = np.sqrt(np.median(d2[iu]))
    return float(med if med > 0 else 1.0)


def _make_folds(n: int, n_folds: int, replicate: Optional[np.ndarray],
                rng: np.random.Generator) -> np.ndarray:
    """Fold assignment that keeps a cell and its replicate together.

    The null must be trained without the held-out cell AND without the sample
    it came from; otherwise a batch-specific quirk leaks into the "smooth
    geometry" prediction and G is biased down.
    """
    if replicate is None:
        return rng.integers(0, n_folds, size=n)
    groups = np.unique(replicate)
    if len(groups) >= 2:
        assign = {gv: i % min(n_folds, len(groups)) for i, gv in
                  enumerate(rng.permutation(groups))}
        return np.array([assign[r] for r in replicate])
    return rng.integers(0, n_folds, size=n)


def nadaraya_watson_crossfit(
    Z: np.ndarray,
    F: np.ndarray,
    folds: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Cross-fitted kernel-regression prediction of the fingerprint from z.

    A weighted average of simplex vectors stays in the simplex, so the
    prediction is a valid distribution and KL(f || fhat) is well defined.
    """
    n, K = F.shape
    out = np.zeros((n, K))
    h2 = 2.0 * bandwidth ** 2
    for f in np.unique(folds):
        te = np.flatnonzero(folds == f)
        tr = np.flatnonzero(folds != f)
        if tr.size == 0:
            out[te] = F.mean(0)
            continue
        d2 = ((Z[te][:, None, :] - Z[tr][None, :, :]) ** 2).sum(-1)
        w = np.exp(-(d2 - d2.min(axis=1, keepdims=True)) / h2)
        denom = w.sum(1, keepdims=True)
        denom[denom <= 0] = 1.0
        pred = (w @ F[tr]) / denom
        s = pred.sum(1, keepdims=True)
        s[s <= 0] = 1.0
        out[te] = pred / s
    return out


def geometric_null(
    Z: np.ndarray,
    F: np.ndarray,
    a: np.ndarray,
    M: np.ndarray,
    g: np.ndarray,
    cfg: DiagnosticsConfig,
    replicate: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """G_tk for one timepoint and one direction (Eqs. 31-32)."""
    rng = np.random.default_rng(cfg.seed) if rng is None else rng
    n = Z.shape[0]
    if n > cfg.null_max_cells:
        sub = np.sort(rng.choice(n, size=cfg.null_max_cells, replace=False))
    else:
        sub = np.arange(n)
    Zs, Fs, Ms = Z[sub], F[sub], M[sub]
    a_s = a[sub]
    rep_s = None if replicate is None else replicate[sub]
    h = cfg.null_bandwidth or _median_bandwidth(Zs, rng)
    h *= cfg.null_bandwidth_scale
    folds = _make_folds(len(sub), cfg.null_n_folds, rep_s, rng)
    Fhat = nadaraya_watson_crossfit(Zs, Fs, folds, h)
    kl = (Fs * (np.log(np.maximum(Fs, _EPS)) - np.log(np.maximum(Fhat, _EPS)))).sum(1)
    num = (a_s[:, None] * Ms * kl[:, None]).sum(0)
    den = (a_s[:, None] * Ms).sum(0)
    return num / np.maximum(den, _EPS)


# ---------------------------------------------------------------------------
# results container
# ---------------------------------------------------------------------------

def _quadrant(v: float, g: float, v_threshold: float, g_threshold: float) -> str:
    """The (V, G) quadrant label of spec 1.17.

    * ``geometric``      -- high V, low G: the heterogeneity is largely
      predictable from smooth expression geometry.
    * ``excess_nonsmooth`` -- high V, high G: not captured by the smooth null.
      NECESSARY BUT NOT SUFFICIENT evidence for fate priming; also consistent
      with coupling noise, representation error, or an inadequate null model.
      Never report it as priming without the supplementary checks.
    * ``homogeneous``    -- low V.
    """
    if not (np.isfinite(v) and np.isfinite(g)):
        return "unevaluated"
    if v < v_threshold:
        return "homogeneous"
    return "excess_nonsmooth" if g >= g_threshold else "geometric"


@dataclass
class Diagnostics:
    g: List[np.ndarray]                       # (T,) state masses
    k_eff: List[float]
    T_mat: List[np.ndarray]                   # (T-1,) induced transitions
    A: List[np.ndarray]
    B: List[np.ndarray]
    V_plus: List[Optional[np.ndarray]]
    V_minus: List[Optional[np.ndarray]]
    G_plus: List[Optional[np.ndarray]] = field(default_factory=list)
    G_minus: List[Optional[np.ndarray]] = field(default_factory=list)
    n_child: List[Optional[np.ndarray]] = field(default_factory=list)
    n_parent: List[Optional[np.ndarray]] = field(default_factory=list)
    mu: List[np.ndarray] = field(default_factory=list)
    active: List[np.ndarray] = field(default_factory=list)
    notes: Dict[str, object] = field(default_factory=dict)

    def event_table(self, v_threshold: float = 0.1, n_threshold: float = 1.2,
                    g_threshold: Optional[float] = None):
        """Classify each state as coherent divergence / outgoing heterogeneity /
        coherent merging / incoming heterogeneity (spec 1.18).

        These are different biological claims and the method separates them;
        that separation is one of the defensible contributions, so the labels
        are reported explicitly rather than collapsed into one "branching"
        score.

        Geometry is reported as a QUADRANT of (V, G), never as ``V - G``.  There
        is no ``V = V_geometry + V_residual`` decomposition: Eqs. 29 and 31
        compare the fingerprints against different reference distributions
        (the KL barycentre phi_k versus the smooth predictor fhat(z)), so their
        difference is not a residual and carries no guaranteed sign.  In
        practice it is routinely NEGATIVE, because a cross-fitted smooth
        predictor is often a worse reference than the within-state mean.
        """
        g_threshold = v_threshold if g_threshold is None else g_threshold
        rows = []
        T = len(self.g)
        for t in range(T):
            for k in range(len(self.g[t])):
                if not (self.active[t][k] if self.active else True):
                    continue
                nch = self.n_child[t][k] if (t < T - 1 and self.n_child) else np.nan
                npa = self.n_parent[t][k] if (t > 0 and self.n_parent) else np.nan
                vp = self.V_plus[t][k] if self.V_plus[t] is not None else np.nan
                vm = self.V_minus[t][k] if self.V_minus[t] is not None else np.nan
                gp = (self.G_plus[t][k] if (self.G_plus and self.G_plus[t] is not None)
                      else np.nan)
                gm = (self.G_minus[t][k] if (self.G_minus and self.G_minus[t] is not None)
                      else np.nan)
                labels = []
                if np.isfinite(nch) and nch > n_threshold and vp < v_threshold:
                    labels.append("coherent_divergence")
                if np.isfinite(vp) and vp >= v_threshold:
                    labels.append("outgoing_heterogeneity")
                if np.isfinite(npa) and npa > n_threshold and vm < v_threshold:
                    labels.append("coherent_merging")
                if np.isfinite(vm) and vm >= v_threshold:
                    labels.append("incoming_heterogeneity")
                rows.append(dict(t=t, state=k, mass=float(self.g[t][k]),
                                 n_child=nch, n_parent=npa,
                                 V_plus=vp, V_minus=vm, G_plus=gp, G_minus=gm,
                                 geometry_plus=_quadrant(vp, gp, v_threshold,
                                                         g_threshold),
                                 geometry_minus=_quadrant(vm, gm, v_threshold,
                                                          g_threshold),
                                 label="+".join(labels) if labels else "stable"))
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except Exception:  # pragma: no cover
            return rows

    def dag_edges(self, min_mass: float = 1e-4):
        """Time-layered DAG edges: (t, k) -> (t+1, l) with transported mass.

        Edges are "transport-implied developmental compatibility", never
        observed lineage.
        """
        edges = []
        for t, Tt in enumerate(self.T_mat):
            for k in range(Tt.shape[0]):
                for l in range(Tt.shape[1]):
                    if Tt[k, l] >= min_mass:
                        edges.append(dict(t=t, source=int(k), target=int(l),
                                          mass=float(Tt[k, l]),
                                          forward=float(self.A[t][k, l]),
                                          reverse=float(self.B[t][l, k])))
        return edges


def compute_diagnostics(
    model: CoarseGrainModel,
    cfg: DiagnosticsConfig = DiagnosticsConfig(),
    replicate: Optional[Sequence[np.ndarray]] = None,
    Z: Optional[Sequence[np.ndarray]] = None,
) -> Diagnostics:
    """Everything in spec 1.16-1.18 for a fitted model."""
    rng = np.random.default_rng(cfg.seed)
    with torch.no_grad():
        M = model.memberships()
        Ts, As, Bs, g = model.induced_transitions(M)
        Fp, Fm = model.fingerprints(M)
        mus = model.expression_prototypes(M, g)

        V_plus: List[Optional[np.ndarray]] = []
        V_minus: List[Optional[np.ndarray]] = []
        phis_p: List[Optional[torch.Tensor]] = []
        phis_m: List[Optional[torch.Tensor]] = []
        for t in range(model.T):
            if Fp[t] is not None:
                phi = model.prototypes(M, Fp[t], g, t)
                phis_p.append(phi)
                V_plus.append(model.within_state_dispersion(Fp[t], phi, M, g, t)
                              .cpu().numpy())
            else:
                phis_p.append(None)
                V_plus.append(None)
            if Fm[t] is not None:
                phi = model.prototypes(M, Fm[t], g, t)
                phis_m.append(phi)
                V_minus.append(model.within_state_dispersion(Fm[t], phi, M, g, t)
                               .cpu().numpy())
            else:
                phis_m.append(None)
                V_minus.append(None)

        g_np = [x.cpu().numpy() for x in g]
        M_np = [x.cpu().numpy() for x in M]
        A_np = [x.cpu().numpy() for x in As]
        B_np = [x.cpu().numpy() for x in Bs]
        T_np = [x.cpu().numpy() for x in Ts]
        a_np = [x.cpu().numpy() for x in model.a]
        Fp_np = [None if f is None else f.cpu().numpy() for f in Fp]
        Fm_np = [None if f is None else f.cpu().numpy() for f in Fm]
        mu_np = [x.cpu().numpy() for x in mus]

    Znp = ([np.asarray(z) for z in Z] if Z is not None
           else [z.cpu().numpy() for z in model.Z])

    n_child = [np.exp(entropy(A, axis=1)) for A in A_np] + [None]
    n_parent = [None] + [np.exp(entropy(B, axis=1)) for B in B_np]

    G_plus: List[Optional[np.ndarray]] = []
    G_minus: List[Optional[np.ndarray]] = []
    if cfg.geometric_null:
        for t in range(model.T):
            rep = None if replicate is None else np.asarray(replicate[t])
            G_plus.append(None if Fp_np[t] is None else geometric_null(
                Znp[t], Fp_np[t], a_np[t], M_np[t], g_np[t], cfg, rep, rng))
            G_minus.append(None if Fm_np[t] is None else geometric_null(
                Znp[t], Fm_np[t], a_np[t], M_np[t], g_np[t], cfg, rep, rng))
    else:
        G_plus = [None] * model.T
        G_minus = [None] * model.T

    return Diagnostics(
        g=g_np,
        k_eff=[float(np.exp(entropy(x, axis=0))) for x in g_np],
        T_mat=T_np, A=A_np, B=B_np,
        V_plus=V_plus, V_minus=V_minus, G_plus=G_plus, G_minus=G_minus,
        n_child=n_child, n_parent=n_parent, mu=mu_np,
        active=[x > model.cfg.g_min for x in g_np],
        notes={"K": model.K, "epsilon": model.chain.epsilon,
               "geometric_null": cfg.geometric_null},
    )


# ---------------------------------------------------------------------------
# degeneracy checks (handoff step 4 -- DO NOT PROCEED IF THIS FAILS)
# ---------------------------------------------------------------------------

def fingerprints_from_reconstruction(model: CoarseGrainModel, t: int) -> np.ndarray:
    """f+ computed from Phat_t instead of P^ref_t -- the WRONG way.

    Analytically this equals M_t(i,:) B_t with B_t independent of i, so all
    cells sharing a hard assignment get the same fingerprint no matter what the
    assignment is.  Provided only so the ablation can be run and reported.
    """
    with torch.no_grad():
        M = model.memberships()
        Ts, As, Bs, g = model.induced_transitions(M)
        W = Ts[t] / (g[t].clamp_min(_EPS)[:, None] * g[t + 1].clamp_min(_EPS)[None, :])
        # Phat(i,:) M_{t+1} = a_i * M_t(i,:) W (Q_{t+1}^T M_{t+1}),  and the
        # row sum divides out the a_i exactly.
        S = (model.a[t + 1][:, None] * M[t + 1]).transpose(0, 1) @ M[t + 1]   # (K, K)
        num = (M[t] @ W) @ S
        den = num.sum(1, keepdim=True).clamp_min(_EPS)
        return (num / den).cpu().numpy()


def degeneracy_check(model: CoarseGrainModel, t: int = 0,
                     tol: float = 1e-6) -> dict:
    """Empirical Degeneracy-1 demonstration + the non-degeneracy conditions.

    Read the outputs in this order; the first two are the sharp statements and
    the third is a weaker summary that is easy to misreport.

    * ``phat_rank_one_residual`` -- max deviation of the Phat fingerprints from
      ``M_t B_t`` for a single i-independent B_t.  Zero to floating point is
      THE statement of Degeneracy 1.
    * ``phat_max_within_hard_state_deviation`` -- spread of Phat fingerprints
      among cells sharing a hard assignment.  Zero for hard memberships.
    * ``phat_within_state_spread`` -- weighted V+ using Phat fingerprints.
      This is NOT zero for soft memberships: f+ = M_t(i,:) B_t still varies
      with i through the membership row itself, so V+ picks up membership
      softness.  It is zero only in the hard limit.  Compare it against
      ``ref_within_state_spread`` as a ratio, and do not report it as "zero".

    ``ref_within_state_spread`` must be clearly larger for the method to be
    measuring anything real.
    """
    with torch.no_grad():
        M = model.memberships()
        Ts, As, Bs, g = model.induced_transitions(M)
        Fp, _ = model.fingerprints(M)
        phi = model.prototypes(M, Fp[t], g, t)
        V_ref = model.within_state_dispersion(Fp[t], phi, M, g, t).cpu().numpy()
        gt = g[t].cpu().numpy()
        ref_spread = float((gt * V_ref).sum() / max(gt.sum(), _EPS))

        F_phat = fingerprints_from_reconstruction(model, t)
        Ft = torch.as_tensor(F_phat, dtype=model.dtype, device=model.device)
        phi_phat = model.prototypes(M, Ft, g, t)
        V_phat = model.within_state_dispersion(Ft, phi_phat, M, g, t).cpu().numpy()
        phat_spread = float((gt * V_phat).sum() / max(gt.sum(), _EPS))

        # hard-assigned cells sharing a state should have identical Phat
        # fingerprints; measure the max deviation within the largest state
        lab = M[t].argmax(1).cpu().numpy()
        big = int(np.bincount(lab, minlength=model.K).argmax())
        idx = np.flatnonzero(lab == big)
        within = (F_phat[idx] - F_phat[idx].mean(0)) if idx.size else np.zeros((1, model.K))
        # rank-one check: f_phat = softmax-normalised M_t B for a fixed B
        Bmat = np.linalg.lstsq(M[t].cpu().numpy(), F_phat, rcond=None)[0]
        resid = float(np.abs(M[t].cpu().numpy() @ Bmat - F_phat).max())

    ok = ref_spread > tol and resid < tol
    ratio = ref_spread / max(phat_spread, 1e-30)
    return {
        "interval": t,
        "ref_within_state_spread": ref_spread,
        "phat_within_state_spread": phat_spread,
        "spread_ratio_ref_over_phat": ratio,
        "phat_max_within_hard_state_deviation": float(np.abs(within).max()),
        "phat_rank_one_residual": resid,
        "phat_factors_through_M": bool(resid < tol),
        "ref_fingerprints_informative": bool(ref_spread > tol),
        "mean_membership_max": float(M[t].max(1).values.mean()),
        "verdict": (
            f"OK: Phat fingerprints factor exactly through M (residual "
            f"{resid:.2e}), so they carry no transition information beyond the "
            f"assignment itself; P^ref fingerprints do vary within states "
            f"(spread ratio {ratio:.1f}x). Note the Phat spread is nonzero only "
            f"because the memberships are soft."
            if ok else "CHECK FAILED -- inspect before proceeding"),
    }


def normalized_transition_ratios(model: CoarseGrainModel) -> dict:
    """R+,t and R-,t of Eqs. 27-28 -- for REPORTING only.

        R+,t = I(I_t; Z_{t+1} | Z_t) / (I(I_t; Z_{t+1}) + delta_I)

    the fraction of transition information *not* captured by the current state.
    The denominator is the same quantity with a single lumped state, i.e. the
    total transition information available at this K.

    These ratios must not be used on their own for state-number selection:
    coarsening the neighbouring state space shrinks both numerator and
    denominator (Degeneracy 3).
    """
    delta_I = 1e-12
    out = {"R_plus": [], "R_minus": []}
    with torch.no_grad():
        M = model.memberships()
        g = model.state_masses(M)
        Fp, Fm = model.fingerprints(M)
        for t in range(model.T):
            for key, F in (("R_plus", Fp[t]), ("R_minus", Fm[t])):
                if F is None:
                    out[key].append(float("nan"))
                    continue
                a = model.a[t]
                phi = model.prototypes(M, F, g, t)
                cond = float(model._fingerprint_loss(F, phi, a, g[t]))
                # total: one lumped state
                bar = (a[:, None] * F).sum(0)
                bar = bar / bar.sum().clamp_min(_EPS)
                total = float((a * (F * (torch.log(F.clamp_min(_EPS))
                                         - torch.log(bar.clamp_min(_EPS)))).sum(1)).sum())
                out[key].append(cond / (total + delta_I))
    return out


def membership_sensitivity(M_a: Sequence[np.ndarray], M_b: Sequence[np.ndarray]) -> dict:
    """How much two fits differ: mean ARI of hard labels and mean L1 of M.

    Used for "removing L_pm measurably changes M_t" (handoff step 4, check ii)
    and for stability reporting.
    """
    try:
        from sklearn.metrics import adjusted_rand_score as ari
    except Exception:  # pragma: no cover
        ari = None
    aris, l1 = [], []
    for a, b in zip(M_a, M_b):
        if ari is not None:
            aris.append(float(ari(a.argmax(1), b.argmax(1))))
        l1.append(float(np.abs(a - b).sum(1).mean()))
    return {"mean_ari": float(np.mean(aris)) if aris else float("nan"),
            "mean_l1_membership_change": float(np.mean(l1)),
            "per_timepoint_ari": aris, "per_timepoint_l1": l1}


def local_permutation_test(
    Z: np.ndarray,
    F: np.ndarray,
    a: np.ndarray,
    M: np.ndarray,
    g: np.ndarray,
    n_neighbors: int = 30,
    n_perm: int = 20,
    seed: int = 0,
) -> dict:
    """Permute fingerprints within expression neighbourhoods and recompute V.

    Supplementary check for spec 1.17: if the observed V sits inside the
    permutation null, the heterogeneity is compatible with smooth local
    geometry plus noise.
    """
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    nbr = np.argsort(d2, axis=1)[:, :n_neighbors]

    def _V(Fx):
        num = (a[:, None] * Fx)
        phi = (M.T @ num) / np.maximum(g[:, None], _EPS)
        kl = ((Fx * (np.log(np.maximum(Fx, _EPS))))[:, None, :]
              - Fx[:, None, :] * np.log(np.maximum(phi, _EPS))[None, :, :]).sum(-1)
        return ((a[:, None] * M * kl).sum(0) / np.maximum(g, _EPS))

    obs = _V(F)
    null = np.zeros((n_perm, len(g)))
    for r in range(n_perm):
        pick = nbr[np.arange(n), rng.integers(0, n_neighbors, size=n)]
        null[r] = _V(F[pick])
    p = (null >= obs[None, :]).mean(0)
    return {"V_observed": obs, "V_null_mean": null.mean(0),
            "V_null_sd": null.std(0), "p_value": p}
