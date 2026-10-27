import time

import numpy as np
from scipy import sparse

from hive.bandit import Bandit, make_q
from hive.graphs import degree_stats, is_connected, relaxation_time, rgg_graph, ws_graph
from hive.metrics import best_mass, t90
from hive.swarm import graph_voter

N = 500
q = make_q("middle")
best = int(np.argmax(q))

rc = np.sqrt(np.log(N) / (np.pi * N))
for c in [1.0, 1.2, 1.4]:
    adj = rgg_graph(N, c * rc, seed=0)
    ncomp, _ = sparse.csgraph.connected_components(adj, directed=False)
    print("rgg", c, "components", ncomp, "deg", degree_stats(adj)["mean"])

for p in [0.01, 0.05, 0.1, 1.0]:
    adj = ws_graph(N, 6, p, seed=0)
    lam, tau = relaxation_time(adj)
    t0 = time.time()
    n_seeds = 10
    ts = []
    finals = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        bandit = Bandit(q, rng=rng)
        types = np.repeat(np.arange(10), N // 10)
        curve = [best_mass(types, best)]
        for _ in range(200):
            rewards = bandit.pull(types)
            types = graph_voter(types, rewards, adj, rng, n_types=10)
            curve.append(best_mass(types, best))
        ts.append(t90(np.array(curve))[0])
        finals.append(np.mean(curve[-20:]))
    print(
        f"ws p={p} tau={tau:8.1f} t90 median={np.median(ts):6.1f} "
        f"final={np.mean(finals):.3f} time/seed={1000*(time.time()-t0)/n_seeds:.1f}ms"
    )
