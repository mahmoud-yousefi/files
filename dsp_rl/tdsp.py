"""TDSP (total dominating set) utilities and the two-stage DSP->TDSP construction.

Bound (constructive, proved for any dominating set D of a graph with no isolated vertex).
Attach to every v in D one neighbour f(v); let S = D u f(D). Then S is total-dominating:
  - every v in D has its chosen neighbour f(v) in S;
  - every v not in D is dominated by D, and its dominator p in D lies outside D
    (a minimum D has no p whose whole closed neighbourhood is inside D, else D\\{p}
    still dominates), so p in S is a neighbour of v.
Hence |S| = |D| + |f(D)| <= 2|D|. No minimality of D is needed, only no isolated vertices.
"""
import numpy as np
from baselines import is_total_dominating
from env import DSPEnv


def has_isolated(gd):
    """A graph with an isolated vertex has no total dominating set at all."""
    return bool((gd.deg == 0).any())


def model_dominating_set(agent, gd, env=None):
    """Run the trained policy greedily to a terminal state and return the selected set.
    With Chen's mask (DECISIONS D1) the result is an independent dominating set."""
    env = env or DSPEnv()
    vf, done = env.reset(gd), False
    while not done:
        q = agent.q_values(gd, vf); c = env.candidates()
        vf, _, done = env.step(int(c[np.argmax(q[c])]))
    return set(env.selected)


def cover_D_greedy(gd, D):
    """Stage 2 as greedy set-cover over D: repeatedly pick the vertex dominating the most
    still-uncovered members of D, so one shared neighbour covers two members at once."""
    nbr = [set(gd.nbrs[u].tolist()) for u in range(gd.n)]
    unc, A = set(int(v) for v in D), set()
    while unc:
        best = max(range(gd.n), key=lambda u: len(unc & nbr[u]))
        if not (unc & nbr[best]):
            break                                   # an isolated member: nothing can cover it
        A.add(best); unc -= nbr[best]
    return A


def two_stage_tdsp(gd, D):
    """Return (S, A) with S = D u A a total dominating set. Asserts the construction holds."""
    D = set(int(v) for v in D)
    A = cover_D_greedy(gd, D)
    S = D | A
    assert is_total_dominating(gd, S), "two-stage construction failed to total-dominate"
    assert len(S) <= 2 * len(D), "constructive bound |S| <= 2|D| violated"
    return S, A


def greedy_tds(gd):
    """Native total-domination greedy baseline: repeatedly pick the vertex dominating the
    most still-undominated vertices."""
    nbr = [set(gd.nbrs[u].tolist()) for u in range(gd.n)]
    unc, S = set(range(gd.n)), set()
    while unc:
        best = max(range(gd.n), key=lambda u: len(unc & nbr[u]))
        S.add(best); unc -= nbr[best]
    return S
