import numpy as np
import pytest
from scipy import sparse

from hive.swarm import graph_voter, well_mixed_voter


def test_equal_rewards_give_frequency_proportional_votes():
    types = np.array([0] * 30 + [1] * 10)
    rewards = np.ones(40)
    out = well_mixed_voter(types, rewards, np.random.default_rng(0), n_types=2)
    mass = np.bincount(out, minlength=2) / out.size
    assert mass[0] == pytest.approx(0.75, abs=0.12)


def test_well_mixed_never_invents_an_absent_type():
    types = np.array([0, 2] * 50)
    out = well_mixed_voter(types, np.ones(100), np.random.default_rng(0), n_types=3)
    assert 1 not in set(out.tolist())


def test_isolated_node_keeps_its_type():
    adj = sparse.csr_matrix((3, 3))
    types = np.array([1, 1, 1])
    out = graph_voter(types, np.ones(3), adj, np.random.default_rng(0), n_types=3)
    assert np.array_equal(out, types)


def test_graph_voter_only_produces_types_present_in_neighbourhood():
    # path 0 - 1 - 2; node 0 only observes node 1
    adj = sparse.csr_matrix(([1.0, 1.0], ([0, 1], [1, 2])), shape=(3, 3))
    adj = (adj + adj.T).tocsr()
    types = np.array([0, 1, 2])
    out = graph_voter(types, np.ones(3), adj, np.random.default_rng(0), n_types=3)
    assert out[0] == 1
    assert out[2] == 1


def test_graph_voter_weights_neighbours_by_reward():
    # star: node 0 sees nodes 1 (type 0, reward 0.1) and 2 (type 1, reward 0.9)
    adj = sparse.csr_matrix(([1.0, 1.0], ([0, 0], [1, 2])), shape=(3, 3))
    adj = (adj + adj.T).tocsr()
    types = np.array([0, 0, 1])
    rewards = np.array([0.0, 0.1, 0.9])
    rng = np.random.default_rng(0)
    picks = [
        graph_voter(types, rewards, adj, rng, n_types=2)[0] for _ in range(2000)
    ]
    assert np.mean(np.array(picks) == 1) == pytest.approx(0.9, abs=0.03)
