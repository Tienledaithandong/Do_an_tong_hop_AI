import time

import numpy as np

from hive.graphs import degree_stats, relaxation_time, rgg_graph, ws_graph

t0 = time.time()
for p in [0, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]:
    lam, tau = relaxation_time(ws_graph(500, 6, p, seed=0))
    print(f"ws   p={p:<7} lam2={lam:.6f} tau={tau:10.1f}")

rc = np.sqrt(np.log(500) / (np.pi * 500))
print("r_c", round(rc, 4))
for c in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
    adj = rgg_graph(500, c * rc, seed=0)
    lam, tau = relaxation_time(adj)
    stats = degree_stats(adj)
    print(
        f"rgg c={c:<4} deg={stats['mean']:5.1f} cv={stats['cv']:.2f} "
        f"lam2={lam:.4f} tau={tau:10.1f}"
    )
print("elapsed", round(time.time() - t0, 1))
