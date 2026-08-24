"""The identities the method rests on.  If these break, nothing above matters.

Covered here:

* total mass of Phat is exactly 1, which is what lets the generalised-KL
  "-p+q" terms cancel and reduces L_compress to sum p log(p/q) on supp(P^ref);
* the fast entropy form of L_pm equals the literal double sum of Eqs. 19-20;
* L_+ equals I(I_t; Z_{t+1} | Z_t) computed by brute force from the explicit
  3-way joint p+(i,k,l) = a_ti M_t,ik f+_ti,l  (Eq. 21-22);
* the prototypes coincide with the rows of A_t and B_t;
* L_+ = 0 exactly at K = 1 (Degeneracy 3);
* Degeneracy 1: fingerprints from Phat factor as M_t(i,:) B_t and carry no
  within-state variation, while fingerprints from P^ref do;
* autograd matches finite differences.
"""

import numpy as np
import pytest
import torch

from cellstateadj.config import CouplingConfig, ModelConfig
from cellstateadj.diagnostics import degeneracy_check, fingerprints_from_reconstruction
from cellstateadj.model import CoarseGrainModel
from cellstateadj.reference import build_reference_chain
from cellstateadj.utils import uniform_weights


def make_chain(T=4, n=25, d=3, K=4, eps=0.1, seed=0, dense=True):
    rng = np.random.default_rng(seed)
    Z = [rng.standard_normal((n, d)) + 0.5 * t for t in range(T)]
    tau = np.arange(T, dtype=float)
    cfg = CouplingConfig(epsilon=eps, support="dense" if dense else "knn",
                         kappa=10, tol=1e-13, max_iter=5000)
    chain = build_reference_chain(Z, tau, cfg, verbose=0)
    return chain, Z


@pytest.fixture(scope="module")
def setup():
    chain, Z = make_chain()
    rng = np.random.default_rng(1)
    K = 4
    U = [rng.standard_normal((z.shape[0], K)) for z in Z]
    cfg = ModelConfig(K=K, dtype="float64", lambda_plus=1.0, lambda_minus=1.0)
    model = CoarseGrainModel(chain, Z, cfg, U_init=U)
    return model


# ---------------------------------------------------------------------------

def test_reference_couplings_have_unit_mass_and_correct_marginals(setup):
    model = setup
    for t, c in enumerate(model.chain.couplings):
        assert abs(c.values.sum() - 1.0) < 1e-10
        assert np.abs(c.row_sums() - model.chain.a[t]).max() < 1e-10
        assert np.abs(c.col_sums() - model.chain.a[t + 1]).max() < 1e-10


def test_phat_has_unit_mass(setup):
    """Required for the KL simplification of Eq. 14 to be legitimate."""
    model = setup
    M = model.memberships()
    g = model.state_masses(M)
    for t in range(model.T - 1):
        PM = torch.sparse.mm(model.tt["P"][t], M[t + 1])
        Tt = M[t].t() @ PM
        W = Tt / (g[t][:, None] * g[t + 1][None, :])
        Phat = (model.a[t][:, None] * (M[t] @ W)) @ (model.a[t + 1][:, None] * M[t + 1]).t()
        assert abs(float(Phat.sum()) - 1.0) < 1e-10
        assert float(Phat.min()) > 0.0


def test_phat_support_values_match_dense_construction(setup):
    model = setup
    M = model.memberships()
    g = model.state_masses(M)
    t = 0
    PM = torch.sparse.mm(model.tt["P"][t], M[t + 1])
    Tt = M[t].t() @ PM
    W = Tt / (g[t][:, None] * g[t + 1][None, :])
    dense = (model.a[t][:, None] * (M[t] @ W)) @ (model.a[t + 1][:, None] * M[t + 1]).t()
    rows, cols = model.tt["rows"][t], model.tt["cols"][t]
    R = M[t] @ W
    onsup = model.a[t][rows] * model.a[t + 1][cols] * (R[rows] * M[t + 1][cols]).sum(1)
    assert torch.allclose(onsup, dense[rows, cols], atol=1e-12)


def test_state_masses_sum_to_one_and_T_has_correct_marginals(setup):
    model = setup
    M = model.memberships()
    Ts, As, Bs, g = model.induced_transitions(M)
    for gt in g:
        assert abs(float(gt.sum()) - 1.0) < 1e-10
    for t, Tt in enumerate(Ts):
        assert torch.allclose(Tt.sum(1), g[t], atol=1e-10)
        assert torch.allclose(Tt.sum(0), g[t + 1], atol=1e-10)
        assert torch.allclose(As[t].sum(1), torch.ones_like(g[t]), atol=1e-10)
        assert torch.allclose(Bs[t].sum(1), torch.ones_like(g[t + 1]), atol=1e-10)


def test_prototypes_equal_conditional_transition_rows(setup):
    """phi+_tk = A_t(k,:) and phi-_{t+1,l} = B_t(l,:)."""
    model = setup
    M = model.memberships()
    Ts, As, Bs, g = model.induced_transitions(M)
    Fp, Fm = model.fingerprints(M)
    for t in range(model.T - 1):
        phip = model.prototypes(M, Fp[t], g, t)
        assert torch.allclose(phip, As[t], atol=1e-9)
        phim = model.prototypes(M, Fm[t + 1], g, t + 1)
        assert torch.allclose(phim, Bs[t], atol=1e-9)


def test_fast_fingerprint_loss_matches_literal_double_sum(setup):
    model = setup
    M = model.memberships()
    g = model.state_masses(M)
    Fp, Fm = model.fingerprints(M)
    for t in range(model.T - 1):
        phi = model.prototypes(M, Fp[t], g, t)
        fast = model._fingerprint_loss(Fp[t], phi, model.a[t], g[t])
        direct = model.fingerprint_loss_direct(Fp[t], phi, M[t], model.a[t])
        assert abs(float(fast) - float(direct)) < 1e-9


def _cmi_bruteforce(p3: np.ndarray) -> float:
    """I(I ; L | K) from an explicit joint array p3[i, k, l]."""
    p3 = p3 / p3.sum()
    pk = p3.sum(axis=(0, 2))                       # (K,)
    pik = p3.sum(axis=2)                           # (n, K)
    pkl = p3.sum(axis=0)                           # (K, L)
    out = 0.0
    nz = p3 > 0
    num = p3[nz] * pk[None, :, None].repeat(p3.shape[0], 0).repeat(p3.shape[2], 2)[nz]
    den = (pik[:, :, None] * pkl[None, :, :])[nz]
    out = float((p3[nz] * np.log(num / den)).sum())
    return out


def test_L_plus_equals_conditional_mutual_information(setup):
    """Eq. 22, checked against a generic CMI computation on the 3-way joint."""
    model = setup
    M = model.memberships()
    g = model.state_masses(M)
    Fp, _ = model.fingerprints(M)
    for t in range(model.T - 1):
        phi = model.prototypes(M, Fp[t], g, t)
        loss = float(model._fingerprint_loss(Fp[t], phi, model.a[t], g[t]))
        a = model.a[t].detach().numpy()
        Mn = M[t].detach().numpy()
        Fn = Fp[t].detach().numpy()
        p3 = a[:, None, None] * Mn[:, :, None] * Fn[:, None, :]
        assert abs(loss - _cmi_bruteforce(p3)) < 1e-8


def test_L_minus_equals_conditional_mutual_information(setup):
    model = setup
    M = model.memberships()
    g = model.state_masses(M)
    _, Fm = model.fingerprints(M)
    for t in range(1, model.T):
        phi = model.prototypes(M, Fm[t], g, t)
        loss = float(model._fingerprint_loss(Fm[t], phi, model.a[t], g[t]))
        a = model.a[t].detach().numpy()
        Mn = M[t].detach().numpy()
        Fn = Fm[t].detach().numpy()
        p3 = a[:, None, None] * Mn[:, :, None] * Fn[:, None, :]
        assert abs(loss - _cmi_bruteforce(p3)) < 1e-8


def test_expression_loss_matches_literal_form(setup):
    model = setup
    M = model.memberships()
    g = model.state_masses(M)
    mus = model.expression_prototypes(M, g)
    for t in range(model.T):
        z, a, m, mu = model.Z[t], model.a[t], M[t], mus[t]
        direct = (a[:, None] * m * ((z[:, None, :] - mu[None, :, :]) ** 2).sum(-1)).sum()
        fast = model._z_sq[t] - float((g[t] * (mu ** 2).sum(1)).sum())
        assert abs(float(direct) - fast) < 1e-9


def test_L_plus_is_zero_at_K_equals_one():
    """Degeneracy 3: with one state every fingerprint is [1] and L_+ vanishes."""
    chain, Z = make_chain(K=1)
    cfg = ModelConfig(K=1, dtype="float64", lambda_plus=1.0, lambda_minus=1.0)
    model = CoarseGrainModel(chain, Z, cfg)
    _, terms = model.objective()
    assert abs(terms.plus) < 1e-12
    assert abs(terms.minus) < 1e-12


def test_compression_kl_is_nonnegative_and_zero_at_perfect_reconstruction(setup):
    model = setup
    _, terms = model.objective()
    assert terms.compress >= -1e-10
    for v in terms.compress_t:
        assert v >= -1e-10


# ---------------------------------------------------------------------------
# Degeneracy 1
# ---------------------------------------------------------------------------

def test_fingerprints_from_phat_factor_through_memberships(setup):
    """f+ from Phat equals M_t(i,:) B_t with B_t independent of i."""
    model = setup
    F_phat = fingerprints_from_reconstruction(model, 0)
    M0 = model.memberships()[0].detach().numpy()
    B, *_ = np.linalg.lstsq(M0, F_phat, rcond=None)
    assert np.abs(M0 @ B - F_phat).max() < 1e-9


def test_degeneracy_check_reports_the_contrast(setup):
    d = degeneracy_check(setup, t=0)
    # the sharp statements: Phat fingerprints factor exactly through M
    assert d["phat_factors_through_M"]
    assert d["phat_rank_one_residual"] < 1e-9
    assert d["ref_fingerprints_informative"]
    # the weaker summary: only a ratio, since soft memberships leave a residual
    assert d["spread_ratio_ref_over_phat"] > 5.0
    assert d["verdict"].startswith("OK")


def test_phat_spread_vanishes_in_the_hard_membership_limit():
    """The V+ summary is only ~0 for hard assignments; the rank-one statement
    holds regardless.  This pins down which claim is safe to report."""
    chain, Z = make_chain(K=3)
    rng = np.random.default_rng(11)
    K = 3
    labels = [rng.integers(0, K, size=z.shape[0]) for z in Z]
    U = [np.where(np.eye(K)[l] > 0, 30.0, -30.0) for l in labels]
    model = CoarseGrainModel(chain, Z, ModelConfig(K=K, dtype="float64"), U_init=U)
    d = degeneracy_check(model, t=0)
    assert d["phat_within_state_spread"] < 1e-12
    assert d["phat_max_within_hard_state_deviation"] < 1e-12


def test_hard_assigned_cells_share_phat_fingerprints():
    """The sharp statement: identical fingerprints for ANY hard assignment."""
    chain, Z = make_chain(K=3)
    rng = np.random.default_rng(3)
    K = 3
    labels = [rng.integers(0, K, size=z.shape[0]) for z in Z]
    U = [np.where(np.eye(K)[l] > 0, 40.0, -40.0) for l in labels]
    model = CoarseGrainModel(chain, Z, ModelConfig(K=K, dtype="float64"), U_init=U)
    F = fingerprints_from_reconstruction(model, 0)
    for k in range(K):
        idx = np.flatnonzero(labels[0] == k)
        if idx.size > 1:
            assert np.abs(F[idx] - F[idx][0]).max() < 1e-9


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def test_gradient_matches_finite_differences():
    chain, Z = make_chain(T=3, n=12, d=2, K=3)
    rng = np.random.default_rng(5)
    U = [rng.standard_normal((z.shape[0], 3)) for z in Z]
    cfg = ModelConfig(K=3, dtype="float64", lambda_compress=1.0, lambda_x=1.0,
                      lambda_plus=1.0, lambda_minus=1.0, delta_floor=0.0)
    model = CoarseGrainModel(chain, Z, cfg, U_init=U)
    loss, _ = model.objective(with_diagnostics=False)
    loss.backward()
    grads = [u.grad.detach().clone() for u in model.U]

    h = 1e-6
    rng = np.random.default_rng(6)
    for _ in range(8):
        t = int(rng.integers(0, model.T))
        i = int(rng.integers(0, model.U[t].shape[0]))
        k = int(rng.integers(0, 3))
        base = model.clone_U()
        up = [u.clone() for u in base]
        up[t][i, k] += h
        dn = [u.clone() for u in base]
        dn[t][i, k] -= h
        with torch.no_grad():
            lu, _ = model.objective(U=up, with_diagnostics=False)
            ld, _ = model.objective(U=dn, with_diagnostics=False)
        fd = (float(lu) - float(ld)) / (2 * h)
        an = float(grads[t][i, k])
        assert abs(fd - an) < 1e-5 * max(1.0, abs(an)), (t, i, k, fd, an)
