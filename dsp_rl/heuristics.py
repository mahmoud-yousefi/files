"""Strong baselines for TDSP.

TDSP is exactly a set-cover problem: with universe U = V and set_v = N(v) for every
vertex v, a total dominating set is a subfamily covering U (every vertex must be a
neighbour of some chosen vertex). So the classical set-cover machinery applies:

  - greedy_tds      : the standard ln(n) greedy  (kept in tdsp.py)
  - remove_redundant: drop any chosen vertex whose removal keeps total domination
  - grasp_tds       : multi-start randomized greedy (GRASP) + redundancy removal

These give a much stronger opponent than plain greedy for the chapter-6 comparison.
The ILP optimum (baselines.exact(gd,"tot")) is used as ground truth on small graphs.
"""
import numpy as np


def tds_counts(gd, S):
    """coverage[u] = number of chosen neighbours of u."""
    c = np.zeros(gd.n, dtype=np.int64)
    for v in S:
        c[gd.nbrs[v]] += 1
    return c


def is_total(gd, S):
    return bool((tds_counts(gd, S) > 0).all())


def remove_redundant(gd, S):
    """Greedily drop chosen vertices that are not needed (checks counts, O(n*m))."""
    S = list(S); cnt = tds_counts(gd, S)
    for v in list(S):
        nb = gd.nbrs[v]
        if (cnt[nb] > 1).all():          # every neighbour still covered by someone else
            cnt[nb] -= 1; S.remove(v)
    return S


def _gains(gd, unc):
    g = np.zeros(gd.n, dtype=np.int64)
    for v in range(gd.n):
        g[v] = int(unc[gd.nbrs[v]].sum())
    return g


def randomized_greedy(gd, rng, alpha=0.3):
    """GRASP construction: at each step pick uniformly from the RCL of near-best gains."""
    unc = np.ones(gd.n, dtype=bool); S = []
    while unc.any():
        g = _gains(gd, unc)
        mx, mn = int(g.max()), int(g.min())
        if mx == 0:                       # isolated vertex: no TDS exists
            break
        thr = mx - alpha * (mx - mn)
        cand = np.flatnonzero(g >= max(thr, 1))
        v = int(rng.choice(cand))
        S.append(v); unc[gd.nbrs[v]] = False
    return S


def grasp_tds(gd, iters=40, rng=None, alpha=0.3):
    """Multi-start randomized greedy with redundancy removal; returns the best set."""
    rng = rng or np.random.default_rng(0)
    best = remove_redundant(gd, randomized_greedy(gd, rng, alpha))
    for _ in range(iters - 1):
        S = remove_redundant(gd, randomized_greedy(gd, rng, alpha))
        if len(S) < len(best):
            best = S
    return sorted(best)
