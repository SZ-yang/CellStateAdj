"""Birth-death SDE simulator with ground-truth states and transitions.

Handoff step 2.  This is NOT optional: it is the only place ground truth
exists, and the objective is non-convex with self-referential features, so on
real data a wrong answer is indistinguishable from a surprising one.

Model
-----
A population of cells lives in a latent space R^d.  Each cell carries a
discrete ground-truth state s; between jumps its latent position follows an
Ornstein-Uhlenbeck SDE pulled toward that state's (possibly time-varying)
centre::

    dz = -theta (z - c_s(t)) dt + sigma dW

State jumps happen with time-dependent rates Q_{s->s'}(t); cells divide at rate
beta_s(t) (the daughter inherits z plus a little noise) and die at rate
delta_s(t).  Sampling at tau_1..tau_T is destructive: different cells at each
time, exactly as in the real data.

Observation model::

    X_ig ~ NegBinomial(mean = mu_g(z_i), dispersion = theta_g),
    log mu_g(z) = b_g + w_g . z + log(library size) [+ batch offset]

Ground truth returned
---------------------
* the discrete state of every sampled cell;
* ``T_true[t]``: the state-level lineage-flow matrix between tau_t and tau_{t+1}
  computed over the *whole* population, normalised to total mass 1;
* ``ancestor_state[t]``: for each cell sampled at tau_t, the state of its
  ancestor at tau_{t-1};
* ``ancestor_row[t]``: the index of that ancestor within the population
  snapshot at tau_{t-1}, so a ground-truth cell-level coupling can be built on
  the cells that were sampled at both ends.

Required scenarios (all provided in :data:`SCENARIOS`): persistent states,
gradual drift, branching, merging, temporary disappearance and recurrence,
unequal intervals, missing timepoints, expansion/contraction, transcriptionally
similar states with different transition roles, transcriptionally distinct
states with similar transition roles, and replicate batch effects.  The last
two similar/distinct cases are the decisive ones -- they are exactly where
transition-defined states must differ from expression-defined ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .data import TimeSeriesData

Array = np.ndarray


# ---------------------------------------------------------------------------
# scenario specification
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """A fully specified ground-truth process."""

    name: str
    n_states: int
    d: int
    centers: Array                                   # (S, d) at time tau[0]
    tau: Array                                       # observation times
    rate_events: List[Tuple[float, float, int, int, float]] = field(default_factory=list)
    #   (t_start, t_end, from_state, to_state, rate)
    centers_end: Optional[Array] = None              # (S, d); linear drift if set
    birth: Optional[Array] = None                    # (S,)
    death: Optional[Array] = None                    # (S,)
    initial: Optional[Array] = None                  # (S,) initial distribution
    n_init: int = 2000
    n_sample: int = 400
    max_population: int = 20000
    sigma: float = 0.15                              # SDE noise
    theta_ou: float = 2.0                            # pull strength
    dt: float = 0.02
    n_replicates: int = 2
    batch_sigma: float = 0.0                         # per-replicate gene offsets
    seed: int = 0
    notes: str = ""

    def centers_at(self, t: float) -> Array:
        if self.centers_end is None:
            return self.centers
        span = float(self.tau[-1] - self.tau[0])
        w = 0.0 if span <= 0 else float(np.clip((t - self.tau[0]) / span, 0.0, 1.0))
        return (1 - w) * self.centers + w * self.centers_end

    def rates_at(self, t: float) -> Array:
        """Off-diagonal jump-rate matrix at time t."""
        Q = np.zeros((self.n_states, self.n_states))
        for (t0, t1, i, j, r) in self.rate_events:
            if t0 <= t < t1:
                Q[i, j] += r
        np.fill_diagonal(Q, 0.0)
        return Q

    def birth_rates(self) -> Array:
        return np.zeros(self.n_states) if self.birth is None else np.asarray(self.birth, float)

    def death_rates(self) -> Array:
        return np.zeros(self.n_states) if self.death is None else np.asarray(self.death, float)

    def initial_distribution(self) -> Array:
        if self.initial is None:
            p = np.zeros(self.n_states)
            p[0] = 1.0
            return p
        p = np.asarray(self.initial, float)
        return p / p.sum()


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    data: TimeSeriesData
    Z_true: List[Array]
    states: List[Array]
    T_true: List[Array]
    ancestor_state: List[Optional[Array]]
    ancestor_row: List[Optional[Array]]
    sampled_index: List[Array]           # index of sampled cells in the snapshot
    population_size: List[int]
    scenario: Scenario
    gene_params: dict

    def state_masses(self) -> List[Array]:
        out = []
        for s in self.states:
            c = np.bincount(s, minlength=self.scenario.n_states).astype(float)
            out.append(c / max(c.sum(), 1))
        return out

    def true_forward(self, t: int) -> Array:
        """Row-normalised ground-truth A_t (conditional on the source state)."""
        Tt = self.T_true[t]
        rs = Tt.sum(1, keepdims=True)
        rs[rs <= 0] = 1.0
        return Tt / rs


def _sample_nb(mu: Array, theta: Array, rng: np.random.Generator) -> Array:
    """X ~ NB with mean mu and gene-wise dispersion theta (variance mu+mu^2/theta)."""
    p = theta[None, :] / (theta[None, :] + np.maximum(mu, 1e-12))
    return rng.negative_binomial(n=np.broadcast_to(theta[None, :], mu.shape), p=p)


def make_gene_params(d: int, n_genes: int, rng: np.random.Generator,
                     loading_scale: float = 1.0, base_expression: float = 1.0,
                     dispersion: float = 5.0) -> dict:
    W = loading_scale * rng.standard_normal((n_genes, d)) / np.sqrt(d)
    b = np.log(base_expression) + 0.5 * rng.standard_normal(n_genes)
    theta = np.full(n_genes, dispersion) * np.exp(0.3 * rng.standard_normal(n_genes))
    return {"W": W, "b": b, "theta": theta}


def simulate(
    scenario: Scenario,
    n_genes: int = 500,
    library_size: float = 5000.0,
    gene_params: Optional[dict] = None,
    verbose: int = 0,
) -> SimulationResult:
    """Run the birth-death SDE and emit counts at the observation times."""
    sc = scenario
    rng = np.random.default_rng(sc.seed)
    tau = np.asarray(sc.tau, float)
    if np.any(np.diff(tau) <= 0):
        raise ValueError("scenario.tau must be strictly increasing")

    # -- initial population ------------------------------------------------
    n0 = sc.n_init
    state = rng.choice(sc.n_states, size=n0, p=sc.initial_distribution())
    C0 = sc.centers_at(tau[0])
    z = C0[state] + sc.sigma * rng.standard_normal((n0, sc.d))
    replicate = rng.integers(0, sc.n_replicates, size=n0)
    anc_state = state.copy()          # ancestor's state at the last snapshot
    anc_row = np.arange(n0)           # ancestor's row in the last snapshot

    beta, delta = sc.birth_rates(), sc.death_rates()

    snapshots: List[dict] = []
    t = float(tau[0])
    next_obs = 0

    def take_snapshot(tnow: float) -> None:
        snapshots.append({
            "t": tnow,
            "state": state.copy(),
            "z": z.copy(),
            "replicate": replicate.copy(),
            "anc_state": anc_state.copy(),
            "anc_row": anc_row.copy(),
            "N": len(state),
        })

    take_snapshot(t)
    next_obs = 1
    anc_state = state.copy()
    anc_row = np.arange(len(state))

    while next_obs < len(tau):
        step = min(sc.dt, float(tau[next_obs] - t))
        if step <= 0:
            take_snapshot(t)
            next_obs += 1
            anc_state = state.copy()
            anc_row = np.arange(len(state))
            continue

        n = len(state)
        if n == 0:
            raise RuntimeError("population went extinct; lower the death rates")

        # -- OU drift + diffusion -----------------------------------------
        C = sc.centers_at(t)
        z = (z + sc.theta_ou * (C[state] - z) * step
             + sc.sigma * np.sqrt(step) * rng.standard_normal(z.shape))

        # -- state jumps ---------------------------------------------------
        Q = sc.rates_at(t)
        if Q.any():
            total = Q.sum(1)
            jump = rng.random(n) < (1.0 - np.exp(-total[state] * step))
            idx = np.flatnonzero(jump)
            if idx.size:
                # vectorised destination sampling: one inverse-CDF draw per
                # source state rather than a Python loop over jumping cells
                src = state[idx]
                u = rng.random(idx.size)
                for k in np.unique(src):
                    if total[k] <= 0:
                        continue
                    sel = idx[src == k]
                    cdf = np.cumsum(Q[k] / total[k])
                    state[sel] = np.searchsorted(cdf, u[src == k], side="right")
                np.clip(state, 0, sc.n_states - 1, out=state)

        # -- death ---------------------------------------------------------
        if delta.any():
            die = rng.random(n) < (1.0 - np.exp(-delta[state] * step))
            keep = ~die
            state, z, replicate = state[keep], z[keep], replicate[keep]
            anc_state, anc_row = anc_state[keep], anc_row[keep]

        # -- birth -----------------------------------------------------------
        if beta.any() and len(state) > 0:
            n = len(state)
            div = rng.random(n) < (1.0 - np.exp(-beta[state] * step))
            idx = np.flatnonzero(div)
            if idx.size:
                state = np.concatenate([state, state[idx]])
                z = np.concatenate([z, z[idx] + 0.05 * sc.sigma
                                    * rng.standard_normal((idx.size, sc.d))])
                replicate = np.concatenate([replicate, replicate[idx]])
                anc_state = np.concatenate([anc_state, anc_state[idx]])
                anc_row = np.concatenate([anc_row, anc_row[idx]])

        # -- cap the population (uniform thinning keeps composition) --------
        if len(state) > sc.max_population:
            pick = np.sort(rng.choice(len(state), size=sc.max_population, replace=False))
            state, z, replicate = state[pick], z[pick], replicate[pick]
            anc_state, anc_row = anc_state[pick], anc_row[pick]

        t += step
        if t >= tau[next_obs] - 1e-12:
            take_snapshot(float(tau[next_obs]))
            next_obs += 1
            anc_state = state.copy()
            anc_row = np.arange(len(state))
            if verbose:
                print(f"  [sim] t={t:.3f} N={len(state)} "
                      f"states={np.bincount(state, minlength=sc.n_states)}")

    # -- ground-truth state-level flow ------------------------------------
    T_true = []
    for k in range(len(snapshots) - 1):
        nxt = snapshots[k + 1]
        M = np.zeros((sc.n_states, sc.n_states))
        np.add.at(M, (nxt["anc_state"], nxt["state"]), 1.0)
        T_true.append(M / max(M.sum(), 1.0))

    # -- destructive sampling + observation model --------------------------
    if gene_params is None:
        gene_params = make_gene_params(sc.d, n_genes, rng)
    W, b, theta_g = gene_params["W"], gene_params["b"], gene_params["theta"]
    batch_off = (sc.batch_sigma * rng.standard_normal((sc.n_replicates, n_genes))
                 if sc.batch_sigma > 0 else np.zeros((sc.n_replicates, n_genes)))

    X, Z_true, states, reps, obs = [], [], [], [], []
    anc_states_out: List[Optional[Array]] = []
    anc_rows_out: List[Optional[Array]] = []
    sampled_index: List[Array] = []
    pop_sizes: List[int] = []

    for k, snap in enumerate(snapshots):
        N = snap["N"]
        m = int(min(sc.n_sample, N))
        pick = np.sort(rng.choice(N, size=m, replace=False))
        zt = snap["z"][pick]
        st = snap["state"][pick]
        rp = snap["replicate"][pick]

        eta = b[None, :] + zt @ W.T + batch_off[rp]
        mu = np.exp(eta)
        mu = mu / mu.sum(1, keepdims=True) * library_size
        counts = _sample_nb(mu, theta_g, rng)

        X.append(counts)
        Z_true.append(zt)
        states.append(st)
        reps.append(rp)
        sampled_index.append(pick)
        pop_sizes.append(N)
        anc_states_out.append(None if k == 0 else snap["anc_state"][pick])
        anc_rows_out.append(None if k == 0 else snap["anc_row"][pick])
        obs.append({"true_state": st, "replicate": rp})

    data = TimeSeriesData(X=X, tau=tau, replicate=reps, obs=obs)
    return SimulationResult(
        data=data, Z_true=Z_true, states=states, T_true=T_true,
        ancestor_state=anc_states_out, ancestor_row=anc_rows_out,
        sampled_index=sampled_index, population_size=pop_sizes,
        scenario=sc, gene_params=gene_params,
    )


def true_coupling(result: SimulationResult, t: int) -> Optional[np.ndarray]:
    """Ground-truth cell-level coupling between the cells sampled at t and t+1.

    Only cells at t+1 whose ancestor was itself sampled at t contribute, so the
    result is a partial coupling; it is normalised to mass 1 and is meant for
    qualitative comparison against P^ref, not as a balanced plan.
    """
    anc_rows = result.ancestor_row[t + 1]
    if anc_rows is None:
        return None
    src = result.sampled_index[t]
    pos = {int(v): i for i, v in enumerate(src)}
    n, m = len(src), len(result.sampled_index[t + 1])
    P = np.zeros((n, m))
    for j, ar in enumerate(anc_rows):
        i = pos.get(int(ar))
        if i is not None:
            P[i, j] += 1.0
    s = P.sum()
    return P / s if s > 0 else P


# ---------------------------------------------------------------------------
# scenario library
# ---------------------------------------------------------------------------

def _ring_centers(S: int, d: int, radius: float = 3.0, seed: int = 0) -> Array:
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((S, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return radius * C


def scenario_persistent(seed: int = 0, T: int = 8, **kw) -> Scenario:
    """Three stable states, no transitions -- the trivial recovery case."""
    d = 4
    return Scenario(name="persistent", n_states=3, d=d,
                    centers=_ring_centers(3, d, seed=seed),
                    tau=np.arange(T, dtype=float),
                    initial=np.ones(3) / 3, seed=seed,
                    notes="no transitions; states must be recovered as stable",
                    **kw)


def scenario_drift(seed: int = 0, T: int = 8, **kw) -> Scenario:
    """Two states whose centres translate steadily -- expression changes, roles do not."""
    d = 4
    C0 = _ring_centers(2, d, seed=seed)
    return Scenario(name="drift", n_states=2, d=d, centers=C0,
                    centers_end=C0 + 4.0, tau=np.arange(T, dtype=float),
                    initial=np.ones(2) / 2, seed=seed,
                    notes="gradual drift; identity should not fragment along time",
                    **kw)


def scenario_branching(seed: int = 0, T: int = 8, **kw) -> Scenario:
    """A -> {B, C} after mid-series: one state, two futures."""
    d = 4
    C = _ring_centers(3, d, seed=seed)
    half = (T - 1) / 2.0
    return Scenario(name="branching", n_states=3, d=d, centers=C,
                    tau=np.arange(T, dtype=float),
                    rate_events=[(half, 1e9, 0, 1, 0.8), (half, 1e9, 0, 2, 0.8)],
                    initial=np.array([1.0, 0.0, 0.0]), seed=seed,
                    notes="divergence; positive control for N_child > 1",
                    **kw)


def scenario_merging(seed: int = 0, T: int = 8, **kw) -> Scenario:
    """{A, B} -> C: convergence, which a tree cannot represent."""
    d = 4
    C = _ring_centers(3, d, seed=seed)
    half = (T - 1) / 2.0
    return Scenario(name="merging", n_states=3, d=d, centers=C,
                    tau=np.arange(T, dtype=float),
                    rate_events=[(half, 1e9, 0, 2, 0.8), (half, 1e9, 1, 2, 0.8)],
                    initial=np.array([0.5, 0.5, 0.0]), seed=seed,
                    notes="convergence; N_parent > 1 at the destination",
                    **kw)


def scenario_recurrence(seed: int = 0, T: int = 10, **kw) -> Scenario:
    """A state that empties out and is repopulated later."""
    d = 4
    C = _ring_centers(3, d, seed=seed)
    return Scenario(name="recurrence", n_states=3, d=d, centers=C,
                    tau=np.arange(T, dtype=float),
                    rate_events=[(1.0, 3.0, 1, 0, 2.0),      # B empties into A
                                 (6.0, 1e9, 0, 1, 1.0)],     # and refills later
                    initial=np.array([0.4, 0.6, 0.0]), seed=seed,
                    notes="temporary disappearance and recurrence; state indices "
                          "are time-local so this must not be forced to reuse a label",
                    **kw)


def scenario_unequal_intervals(seed: int = 0, **kw) -> Scenario:
    """Non-uniform tau -- the cost divides by dtau, so this must not bias states."""
    d = 4
    C = _ring_centers(3, d, seed=seed)
    tau = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 4.5, 6.0, 9.0])
    return Scenario(name="unequal_intervals", n_states=3, d=d, centers=C, tau=tau,
                    rate_events=[(2.0, 1e9, 0, 1, 0.5), (4.0, 1e9, 1, 2, 0.5)],
                    initial=np.array([1.0, 0.0, 0.0]), seed=seed,
                    notes="unequal dtau", **kw)


def scenario_missing_timepoints(seed: int = 0, **kw) -> Scenario:
    """A long gap in the middle of an otherwise dense series."""
    d = 4
    C = _ring_centers(3, d, seed=seed)
    tau = np.array([0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0])
    return Scenario(name="missing_timepoints", n_states=3, d=d, centers=C, tau=tau,
                    rate_events=[(2.0, 1e9, 0, 1, 0.4), (3.0, 1e9, 1, 2, 0.4)],
                    initial=np.array([1.0, 0.0, 0.0]), seed=seed,
                    notes="transitions occurring entirely inside the gap are "
                          "unrecoverable by construction", **kw)


def scenario_expansion_contraction(seed: int = 0, T: int = 8, **kw) -> Scenario:
    """Unequal birth/death: one state expands while another contracts."""
    d = 4
    C = _ring_centers(3, d, seed=seed)
    return Scenario(name="expansion_contraction", n_states=3, d=d, centers=C,
                    tau=np.arange(T, dtype=float),
                    birth=np.array([0.5, 0.0, 0.1]),
                    death=np.array([0.0, 0.4, 0.1]),
                    rate_events=[(1.0, 1e9, 0, 2, 0.3)],
                    initial=np.array([0.4, 0.4, 0.2]), seed=seed,
                    notes="balanced OT cannot see growth; abundance changes are "
                          "reported, not modelled (handoff s9)",
                    **kw)


def scenario_similar_expression_different_role(
    seed: int = 0, T: int = 8, separation: float = 0.3, **kw
) -> Scenario:
    """DECISIVE CASE 1.

    Two source states sit almost on top of each other in expression
    (``separation`` in units of the SDE noise scale) but have disjoint futures.
    Expression clustering must merge them; transition-defined states should not.
    ``separation -> 0`` is the identifiability limit: with identical profiles no
    method can separate them from snapshots alone.
    """
    d = 4
    base = _ring_centers(4, d, seed=seed)
    rng = np.random.default_rng(seed + 77)
    off = rng.standard_normal(d)
    off /= np.linalg.norm(off)
    C = base.copy()
    C[1] = C[0] + separation * off            # A and B nearly coincide
    half = (T - 1) / 2.0
    return Scenario(name="similar_expression_different_role", n_states=4, d=d,
                    centers=C, tau=np.arange(T, dtype=float),
                    rate_events=[(half, 1e9, 0, 2, 1.0), (half, 1e9, 1, 3, 1.0)],
                    initial=np.array([0.5, 0.5, 0.0, 0.0]), seed=seed,
                    notes=f"A~B in z (sep={separation}) but A->C, B->D",
                    **kw)


def scenario_distinct_expression_same_role(seed: int = 0, T: int = 8, **kw) -> Scenario:
    """DECISIVE CASE 2.

    Two well-separated expression clusters with identical transition roles
    (both feed the same descendant).  Expression clustering splits them;
    a purely transition-defined criterion would merge them -- which is why
    L_expression is in the objective at all.
    """
    d = 4
    C = _ring_centers(3, d, radius=4.0, seed=seed)
    half = (T - 1) / 2.0
    return Scenario(name="distinct_expression_same_role", n_states=3, d=d, centers=C,
                    tau=np.arange(T, dtype=float),
                    rate_events=[(half, 1e9, 0, 2, 1.0), (half, 1e9, 1, 2, 1.0)],
                    initial=np.array([0.5, 0.5, 0.0]), seed=seed,
                    notes="A and B far apart in z, both -> C",
                    **kw)


def scenario_replicate_batch(seed: int = 0, T: int = 8, batch_sigma: float = 0.4,
                             **kw) -> Scenario:
    """Branching plus gene-level offsets between culture replicates."""
    s = scenario_branching(seed=seed, T=T, **kw)
    s.name = "replicate_batch"
    s.batch_sigma = batch_sigma
    s.n_replicates = 2
    s.notes = "branching with replicate batch effects in the observation model"
    return s


SCENARIOS: Dict[str, Callable[..., Scenario]] = {
    "persistent": scenario_persistent,
    "drift": scenario_drift,
    "branching": scenario_branching,
    "merging": scenario_merging,
    "recurrence": scenario_recurrence,
    "unequal_intervals": scenario_unequal_intervals,
    "missing_timepoints": scenario_missing_timepoints,
    "expansion_contraction": scenario_expansion_contraction,
    "similar_expression_different_role": scenario_similar_expression_different_role,
    "distinct_expression_same_role": scenario_distinct_expression_same_role,
    "replicate_batch": scenario_replicate_batch,
}


def make(name: str, **kw) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; available: {sorted(SCENARIOS)}")
    return SCENARIOS[name](**kw)
