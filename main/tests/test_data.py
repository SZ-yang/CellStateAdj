"""Container and subsampling behaviour."""

import numpy as np
import pytest

from cellstateadj.data import (TimeSeriesData, split_half_by_replicate,
                               subsample_cells, subsample_timepoints)


def toy(T=8, n=30, G=10, seed=0):
    rng = np.random.default_rng(seed)
    X = [rng.poisson(3, size=(n, G)) for _ in range(T)]
    rep = [rng.integers(0, 2, n) for _ in range(T)]
    return TimeSeriesData(X=X, tau=np.arange(T, dtype=float) * 0.5, replicate=rep)


def test_requires_increasing_tau():
    with pytest.raises(ValueError):
        TimeSeriesData(X=[np.zeros((2, 2))] * 2, tau=[1.0, 1.0])


def test_delta_tau_subsampling():
    d = toy(T=9)
    s = subsample_timepoints(d, stride=4)
    assert s.T == 3
    assert np.allclose(s.dtau, 2.0)          # 4 x 0.5
    assert np.allclose(d.dtau, 0.5)


def test_cell_subsampling_keeps_both_replicates():
    d = toy(n=100)
    s = subsample_cells(d, n_per_timepoint=20, seed=0)
    assert all(x.shape[0] == 20 for x in s.X)
    assert all(len(np.unique(r)) == 2 for r in s.replicate)


def test_split_half_by_replicate_is_disjoint_and_complete():
    d = toy(n=40)
    a, b = split_half_by_replicate(d, seed=0)
    for t in range(d.T):
        assert a.X[t].shape[0] + b.X[t].shape[0] == d.X[t].shape[0]
