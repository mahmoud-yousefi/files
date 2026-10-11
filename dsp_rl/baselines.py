"""Greedy baseline, exact ILP (ground truth) and validity checks."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


def greedy_ds(gd):
    """Classical greedy: repeatedly pick the vertex (any vertex) dominating most undominated vertices."""
    und = np.ones(gd.n, dtype=np.int64)
    sel = []
    while und.any():
        gain = und + np.bincount(gd.row, weights=und[gd.col], minlength=gd.n)
        v = int(np.argmax(gain)); sel.append(v)
        und[v] = 0; und[gd.nbrs[v]] = 0
    return sel


def greedy_ds_masked(gd, tie="first", rng=None):
    """Greedy restricted to undominated vertices (same action set as Chen's masked policy).
    Matches the paper's greedy values on 15/22 grid instances of Table 3 (see DECISIONS D9)."""
    und = np.ones(gd.n, dtype=np.int64)
    sel = []
    while und.any():
        gain = und + np.bincount(gd.row, weights=und[gd.col], minlength=gd.n)
        gain = np.where(und == 1, gain, -1)
        v = int(np.argmax(gain)) if tie == "first" else int(rng.choice(np.flatnonzero(gain == gain.max())))
        sel.append(v); und[v] = 0; und[gd.nbrs[v]] = 0
    return sel


def is_dominating(gd, D):
    dom = np.zeros(gd.n, bool); dom[list(D)] = True
    for v in D: dom[gd.nbrs[v]] = True
    return bool(dom.all())


def is_independent(gd, D):
    s = set(D)
    return all(not (set(gd.nbrs[v].tolist()) & s) for v in D)


def is_total_dominating(gd, D):
    s = set(D)
    return all(bool(set(gd.nbrs[v].tolist()) & s) for v in range(gd.n))


def exact(gd, kind="dom", time_limit=120):
    """kind: 'dom' (gamma), 'ind' (independent domination), 'tot' (total domination gamma_t).
    Returns (value, proven_optimal)."""
    v, _, ok = exact_solution(gd, kind, time_limit)
    return v, ok


def exact_solution(gd, kind="dom", time_limit=120):
    """Like exact() but also returns the optimal vertex set, if the solve finished.
    Returns (integer_value, set_of_selected_vertices_or_None, proven_optimal)."""
    v, sel, ok = solve_ilp(gd, kind, time_limit)
    return v, ({int(i) for i in np.flatnonzero(sel)} if sel is not None else None), ok


def solve_ilp(gd, kind="dom", time_limit=120):
    """Build the ILP, solve with scipy.milp; returns (value, x_bool_array_or_None, proven)."""
    n = gd.n
    ne = len(gd.row) // 2
    A = lil_matrix((n + (ne if kind == "ind" else 0), n))
    lb, ub = [], []
    for v in range(n):
        if kind != "tot": A[v, v] = 1
        for u in gd.nbrs[v]: A[v, u] = 1
        lb.append(1); ub.append(np.inf)
    if kind == "ind":
        r = n
        for a, b in zip(gd.row[:ne], gd.col[:ne]):
            A[r, a] = 1; A[r, b] = 1; lb.append(-np.inf); ub.append(1); r += 1
    res = milp(np.ones(n), constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=np.ones(n),
               bounds=Bounds(0, 1), options={"time_limit": time_limit})
    ok = bool(res.status == 0 and res.x is not None and res.fun is not None)
    x = None if not ok else np.round(res.x).astype(int)
    val = int(round(res.fun)) if ok else None
    return val, x, ok
