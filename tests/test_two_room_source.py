import numpy as np

from hive.bandit import make_q
from hive.graph_sweep import (
    initial_types_source_room,
    room_mask,
    simulate_two_room_source,
)
from hive.graphs import two_room_graph


def test_source_initializer_keeps_best_out_of_the_target_room():
    N, n = 40, 5
    q = make_q("middle", n)
    best = int(np.argmax(q))
    source = room_mask(N, 0)
    types = initial_types_source_room(np.random.default_rng(0), n, N, best, source)
    assert np.any(types[source] == best)
    assert not np.any(types[~source] == best)
    assert types.size == N
    assert set(np.unique(types)) <= set(range(n))


def test_target_room_starts_without_the_best_option():
    N, n = 40, 5
    q = make_q("middle", n)
    result = simulate_two_room_source(
        two_room_graph(N, bridge=3, seed=0),
        q,
        steps=10,
        seeds=2,
        n_types=n,
    )
    assert result["target_curves"][:, 0].tolist() == [0.0, 0.0]
    assert np.all(result["source_curves"][:, 0] > 0)
    assert result["target_curves"].shape == (2, 11)
    assert result["t50_target"].shape == (2,)


def test_source_room_experiment_reports_thresholds_and_hits():
    N, n = 60, 5
    q = make_q("middle", n)
    result = simulate_two_room_source(
        two_room_graph(N, bridge=6, seed=0),
        q,
        steps=200,
        seeds=3,
        n_types=n,
        source_room=1,
    )
    assert np.all(result["final_target"] >= 0)
    assert np.all(result["final_target"] <= 1)
    assert result["hit50_target"].dtype == bool
    assert result["t50_target"].shape == (3,)
    assert result["curves"].shape == (3, 201)
