"""The simulator must actually produce the structures it advertises."""

import numpy as np
import pytest

from cellstateadj import simulate as sim


def test_all_required_scenarios_exist():
    required = {
        "persistent", "drift", "branching", "merging", "recurrence",
        "unequal_intervals", "missing_timepoints", "expansion_contraction",
        "similar_expression_different_role", "distinct_expression_same_role",
        "replicate_batch",
    }
    assert required <= set(sim.SCENARIOS)


@pytest.mark.parametrize("name", sorted(sim.SCENARIOS))
def test_scenario_runs_and_shapes_are_consistent(name):
    sc = sim.make(name, seed=0)
    sc.n_init, sc.n_sample, sc.max_population, sc.dt = 300, 80, 1500, 0.1
    res = sim.simulate(sc, n_genes=40)
    d = res.data
    assert d.T == len(sc.tau)
    assert all(x.shape[0] == len(s) for x, s in zip(d.X, res.states))
    assert all(x.shape[1] == 40 for x in d.X)
    assert all(x.min() >= 0 for x in d.X)
    assert len(res.T_true) == d.T - 1
    for M in res.T_true:
        assert abs(M.sum() - 1.0) < 1e-9


def test_branching_produces_two_children():
    sc = sim.make("branching", seed=0, T=8)
    sc.n_init, sc.n_sample, sc.dt = 800, 200, 0.05
    res = sim.simulate(sc, n_genes=30)
    A = res.true_forward(len(res.T_true) - 1)
    nchild = np.exp(-(A * np.log(np.maximum(A, 1e-30))).sum(1))
    assert nchild.max() > 1.3


def test_merging_produces_two_parents():
    sc = sim.make("merging", seed=0, T=8)
    sc.n_init, sc.n_sample, sc.dt = 800, 200, 0.05
    res = sim.simulate(sc, n_genes=30)
    T_last = res.T_true[-1]
    col = T_last[:, 2]
    assert (col > 0.02).sum() >= 2


def test_similar_expression_states_really_do_overlap():
    """The decisive scenario is only decisive if the two states are close in z."""
    sc = sim.make("similar_expression_different_role", seed=0, separation=0.3)
    sc.n_init, sc.n_sample, sc.dt = 600, 200, 0.05
    res = sim.simulate(sc, n_genes=30)
    z0 = res.Z_true[0]
    s0 = res.states[0]
    if (s0 == 0).sum() > 5 and (s0 == 1).sum() > 5:
        between = np.linalg.norm(z0[s0 == 0].mean(0) - z0[s0 == 1].mean(0))
        within = z0[s0 == 0].std(0).mean() + z0[s0 == 1].std(0).mean()
        assert between < 3 * within, (between, within)


def test_expansion_contraction_changes_composition():
    sc = sim.make("expansion_contraction", seed=0, T=8)
    sc.n_init, sc.n_sample, sc.dt = 800, 200, 0.05
    res = sim.simulate(sc, n_genes=30)
    masses = res.state_masses()
    assert abs(masses[-1][0] - masses[0][0]) > 0.05


def test_replicate_batch_effect_is_visible_in_counts():
    sc = sim.make("replicate_batch", seed=0, T=5, batch_sigma=1.0)
    sc.n_init, sc.n_sample, sc.dt = 400, 150, 0.1
    res = sim.simulate(sc, n_genes=50)
    X, rep = res.data.X[0].astype(float), res.data.replicate[0]
    lx = np.log1p(X / X.sum(1, keepdims=True) * 1e4)
    d = np.abs(lx[rep == 0].mean(0) - lx[rep == 1].mean(0)).mean()
    assert d > 1e-3


def test_true_coupling_is_a_valid_partial_plan():
    sc = sim.make("branching", seed=0, T=5)
    sc.n_init, sc.n_sample, sc.dt = 500, 400, 0.1
    res = sim.simulate(sc, n_genes=20)
    P = sim.true_coupling(res, 0)
    assert P is not None
    assert P.min() >= 0
    assert abs(P.sum() - 1.0) < 1e-9 or P.sum() == 0
