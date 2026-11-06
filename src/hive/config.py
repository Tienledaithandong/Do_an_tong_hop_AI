"""Shared experiment configuration.

Single source of truth for the knobs used by ``scripts/run_all.py`` and
``scripts/check_heavy_tail.py`` so the quick/full runs and the heavy-tail
reference values cannot drift apart.
"""

from __future__ import annotations

# Graph families swept in Experiment B. Kept in one place so scripts and
# report do not hardcode the same lists in two places.
WS_P: list[float] = [0.0, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
RGG_C: list[float] = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]
BRIDGE_P: list[float] = [1.0, 3.0, 6.0]

# Defaults for scripts/run_all.py. ``full`` runs the complete experiments;
# ``quick`` is a smoke test that finishes in a few seconds.
RUN_ALL: dict[str, dict[str, dict]] = {
    "full": {
        "reproduction": {"N": 1000, "steps": 300, "seeds": 100},
        "graph": {
            "N": 500,
            "steps": 400,
            "seeds": 100,
            "ws_p": WS_P,
            "rgg_c": RGG_C,
        },
    },
    "quick": {
        "reproduction": {"N": 300, "steps": 120, "seeds": 20},
        "graph": {
            "N": 200,
            "steps": 120,
            "seeds": 12,
            "ws_p": [0.01, 0.1, 1.0],
            "rgg_c": [1.4, 2.0],
        },
    },
}

# Calibration line used by scripts/check_heavy_tail.py to judge the
# t90 ~ L law on heavy-tailed networks. Fitted in the report on the 16
# non-intervention configurations (see ``hive.plots.fit_law``).
HEAVY_TAIL: dict = {
    "n": 500,
    "steps": 400,
    "seeds": 100,
    "n_types": 10,
    "scenario": "middle",
    "intercept": 44.85,
    "slope": 3.46,
}

# Source-room intervention: the best option starts in room A only, so the
# bridge is the sole channel for it to reach room B. This is deliberately
# different from the original sweep, where both rooms were already seeded.
TWO_ROOM_SOURCE: dict = {
    "N": 300,
    "steps": 600,
    "seeds": 30,
    "n_types": 10,
    "scenario": "middle",
    "bridges": [1, 3, 6, 12],
}

# Replicated source-room experiment (scripts/check_two_room_replicates.py).
# ``sizes`` maps N to the number of steps; the horizon grows with N because the
# wait at a narrow bridge scales with the room size.
TWO_ROOM_REPLICATES: dict[str, dict] = {
    "full": {
        "sizes": {200: 800, 400: 1600, 600: 2400},
        "bridges": [1, 2, 4, 8, 16],
        "graph_seeds": 4,
        "seeds": 20,
    },
    "quick": {
        "sizes": {60: 200, 100: 300},
        "bridges": [1, 4],
        "graph_seeds": 2,
        "seeds": 4,
    },
}

# Sparse-room version of the replicated source-room experiment. Rooms are
# random regular graphs, so at the same bridge width ``tau`` drops by roughly
# ``degree / (N / 2)`` while the bridge-per-node ratio ``phi`` is unchanged.
TWO_ROOM_SPARSE: dict[str, dict] = {
    "full": {
        "sizes": {200: 800, 400: 1600, 600: 2400},
        "bridges": [1, 2, 4, 8, 16],
        "room_degrees": [8, 24],
        "graph_seeds": 4,
        "seeds": 20,
    },
    "quick": {
        "sizes": {60: 200, 100: 300},
        "bridges": [1, 4],
        "room_degrees": [6],
        "graph_seeds": 2,
        "seeds": 4,
    },
}

# Uneven-degree rooms with the bridge attached to hubs, leaves or random
# nodes (scripts/check_two_room_hetero.py). Barabasi-Albert rooms with
# ``m = 3`` have mean degree ~6; the 6-regular layout is the control with the
# same mean degree and no hubs.
TWO_ROOM_HETERO: dict[str, dict] = {
    "full": {
        "N": 400,
        "steps": 1000,
        "bridges": [1, 2, 4, 8],
        "m": 3,
        "graph_seeds": 5,
        "seeds": 20,
    },
    "quick": {
        "N": 80,
        "steps": 200,
        "bridges": [1, 4],
        "m": 3,
        "graph_seeds": 2,
        "seeds": 4,
    },
}
