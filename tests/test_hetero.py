import numpy as np
import pytest
from scipy import sparse

from hive.graphs import is_connected, two_room_hetero
from hive.hetero import (
    bootstrap_hetero,
    compare_hetero_predictors,
    endpoint_stats,
    layout_name,
    placement_effects,
    run_hetero,
    summarise_hetero,
)


def _endpoints(adj, half):
    cross = adj[:half, half:].tocoo()
    return cross.row, cross.col + half


def test_hetero_graph_is_connected_and_symmetric():
    adj = two_room_hetero(120, bridge=3, seed=0)
    assert is_connected(adj)
    assert (adj != adj.T).nnz == 0
    assert adj.diagonal().sum() == 0


def test_hub_and_leaf_placement_pick_extreme_degrees():
    n, half = 200, 100
    hub = two_room_hetero(n, bridge=3, seed=1, source="hub", target="hub")
    leaf = two_room_hetero(n, bridge=3, seed=1, source="leaf", target="leaf")
    deg_h = np.asarray(hub.sum(axis=1)).ravel()
    deg_l = np.asarray(leaf.sum(axis=1)).ravel()
    src_h, tgt_h = _endpoints(hub, half)
    src_l, tgt_l = _endpoints(leaf, half)
    assert deg_h[tgt_h].mean() > 3 * deg_l[tgt_l].mean()
    assert deg_h[src_h].mean() > 3 * deg_l[src_l].mean()
    # Leaves of a BA(m=3) graph have degree 3 (+1 for the bridge edge).
    assert deg_l[tgt_l].max() <= 4


def test_placement_is_independent_per_side():
    base = two_room_hetero(200, bridge=2, seed=2, source="random", target="random")
    tgt = two_room_hetero(200, bridge=2, seed=2, source="random", target="hub")
    d_base = np.asarray(base.sum(axis=1)).ravel()
    d_tgt = np.asarray(tgt.sum(axis=1)).ravel()
    _, tb = _endpoints(base, 100)
    _, tt = _endpoints(tgt, 100)
    assert d_tgt[tt].mean() > d_base[tb].mean()


def test_regular_control_has_no_hubs():
    adj = two_room_hetero(120, bridge=2, seed=0, kind="regular", m=3)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    assert set(np.unique(deg)) <= {6, 7}


def test_invalid_placement_and_kind_raise():
    with pytest.raises(ValueError):
        two_room_hetero(60, bridge=2, seed=0, target="middle")
    with pytest.raises(ValueError):
        two_room_hetero(60, bridge=2, seed=0, kind="star")


def test_endpoint_stats_on_a_tiny_graph():
    # two triangles 0-1-2 and 3-4-5 joined by edge 0-3
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (0, 3)]
    r = [a for a, b in edges] + [b for a, b in edges]
    c = [b for a, b in edges] + [a for a, b in edges]
    adj = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(6, 6))
    s = endpoint_stats(adj)
    assert s["bridge_edges"] == 1
    # node 0 has degree 3 (two triangle edges + bridge), node 3 likewise
    assert s["inv_end_target"] == pytest.approx(1 / 3)
    assert s["inv_end_source"] == pytest.approx(1 / 3)
    assert s["mean_deg_target"] == pytest.approx((3 + 2 + 2) / 3)


def _small_layouts():
    return [
        {"kind": "ba", "m": 3, "source": "random", "target": "random"},
        {"kind": "ba", "m": 3, "source": "random", "target": "hub"},
        {"kind": "ba", "m": 3, "source": "random", "target": "leaf"},
        {"kind": "regular", "m": 3, "source": "random", "target": "random"},
    ]


def test_run_hetero_summary_and_fits():
    df = run_hetero(80, 120, [1, 3], _small_layouts(), graph_seeds=2, seeds=3, n_types=5)
    assert len(df) == 4 * 2 * 2 * 3
    assert set(df["layout"]) == {layout_name(layout) for layout in _small_layouts()}
    summary = summarise_hetero(df, n_boot=30)
    assert len(summary) == 8
    assert (summary["graphs"] == 2).all()
    hub = summary[summary["target"] == "hub"]["inv_end_target"].mean()
    leaf = summary[summary["target"] == "leaf"]["inv_end_target"].mean()
    assert leaf > hub
    fits = compare_hetero_predictors(summary)
    assert {"single", "target_and_source", "target_and_room", "target_and_tau"} <= set(fits)
    cis = bootstrap_hetero(df, "t_first", "hit_first", ["inv_end_target", "tau"], n_boot=15)
    assert set(cis) <= {"inv_end_target", "tau"}
    effects = placement_effects(summary)
    assert {"source", "target", "ratio_median"} <= set(effects.columns)


def test_hetero_figure_is_written(tmp_path):
    from hive.plots import plot_two_room_hetero

    df = run_hetero(80, 120, [1, 3], _small_layouts(), graph_seeds=2, seeds=3, n_types=5)
    paths = plot_two_room_hetero(summarise_hetero(df, n_boot=20), tmp_path)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
