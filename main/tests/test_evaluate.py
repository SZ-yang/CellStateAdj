"""Matching and recovery metrics."""

import numpy as np

from cellstateadj.evaluate import match_states, state_recovery, transition_recovery


def test_match_states_joint_is_correct():
    """Guards the np.add.at-through-a-transposed-view construction."""
    M = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    true = np.array([0, 0, 1, 1])
    a = np.full(4, 0.25)
    assign, joint = match_states(M, true, n_true=2, a=a)
    expected = np.array([[0.5, 0.0], [0.0, 0.5]])
    assert np.allclose(joint, expected), joint
    assert joint.shape == (2, 2)
    assert abs(joint.sum() - 1.0) < 1e-12
    assert list(assign) == [0, 1]


def test_match_states_handles_permuted_labels():
    M = np.eye(3)[np.array([2, 2, 0, 0, 1, 1])]
    true = np.array([0, 0, 1, 1, 2, 2])
    assign, _ = match_states(M, true, n_true=3)
    # learned state 2 -> true 0, learned 0 -> true 1, learned 1 -> true 2
    assert assign[2] == 0 and assign[0] == 1 and assign[1] == 2


def test_perfect_recovery_scores_one():
    M = [np.eye(3)[np.array([0, 1, 2, 0])] for _ in range(2)]
    true = [np.array([0, 1, 2, 0]) for _ in range(2)]
    r = state_recovery(M, true)
    assert r["mean_ari"] == 1.0


def test_transition_recovery_is_zero_error_on_an_exact_match():
    n, K = 6, 3
    lab0 = np.array([0, 0, 1, 1, 2, 2])
    lab1 = np.array([0, 1, 1, 2, 2, 0])
    M = [np.eye(K)[lab0], np.eye(K)[lab1]]
    # ground-truth flow implied by treating learned == true states
    a0 = np.full(n, 1.0 / n)
    T = np.zeros((K, K))
    for i in range(n):
        T[lab0[i], lab1[i]] += 1.0 / n
    g = [np.array([np.sum(a0[lab0 == k]) for k in range(K)])]
    A = [T / np.maximum(g[0][:, None], 1e-30)]
    out = transition_recovery(A, g, M, [lab0, lab1], [T], n_true=K)
    assert out["mean_l1_error"] < 1e-9, out
