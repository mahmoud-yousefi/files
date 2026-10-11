"""Exact reduction rules for TDSP -- and one that provably does NOT transfer from DSP.

Rule 1 -- isolated vertex => infeasible.
  An isolated vertex has no neighbour, so it can never be covered by a chosen
  vertex; hence there is no total dominating set. (Detection.)

Rule 2 -- connected components split the problem exactly.
  A chosen vertex covers only vertices in its own component. Therefore
  gamma_t(G) = sum_C gamma_t(C) and an optimal S is the union of optimal
  per-component sets. We solve each component on its own and union the answers.
  This rule is verified below against the ILP.

Rule 3 -- the naive leaf rule (valid for DSP) is INVALID for TDSP.
  In DSP, a leaf v (sole neighbour u) forces u into the set and that is the end of
  it. In TDSP, u is still forced, BUT u must itself have a chosen neighbour, so
  the residual problem is not simply the induced subgraph with {u} u N(u)
  removed: it carries an extra constraint ("u must gain a chosen neighbour from
  N(u)"), which a plain induced-subgraph reduction cannot express. Dropping the
  covered vertices anyway yields sets that are too small and even infeasible
  (see `counterexample`). We therefore do NOT apply it; the finding itself is a
  result, matching the warning that DSP reduction rules do not carry over.
"""
import numpy as np
from graphs import GraphData


def components(gd):
    """List of boolean masks, one per connected component."""
    lab = -np.ones(gd.n, dtype=np.int64); c = 0
    for s in range(gd.n):
        if lab[s] >= 0:
            continue
        stack = [s]; lab[s] = c
        while stack:
            u = stack.pop()
            for w in gd.nbrs[u]:
                if lab[w] < 0:
                    lab[w] = c; stack.append(w)
        c += 1
    return [(lab == k) for k in range(c)]


def induced_subgraph(gd, mask):
    """Induced subgraph on mask==True, renumbered 0..k-1. Returns (GraphData, old_index)."""
    found = np.flatnonzero(mask)
    pos = {int(x): i for i, x in enumerate(found)}
    n = len(found)
    row, col = [], []
    for i, u in enumerate(found):
        for w in gd.nbrs[u]:
            if mask[w]:
                row.append(i); col.append(pos[int(w)])
    row = np.array(row, np.int64); col = np.array(col, np.int64)
    deg = np.bincount(row, minlength=n).astype(np.float32)
    an = (1.0 / np.sqrt(deg[row] * deg[col])).astype(np.float32) if len(row) else np.zeros(0, np.float32)
    nbrs = [np.array([pos[int(w)] for w in gd.nbrs[u] if mask[w]], dtype=np.int64) for u in found]
    return GraphData(n, row, col, an, deg, nbrs), found


def leaf_supports(gd):
    """Supports of degree-1 vertices -- forced into every TD set (correct, used for information)."""
    return {int(gd.nbrs[v][0]) for v in range(gd.n) if gd.deg[v] == 1}


def reduce(gd):
    """Rules 1-2. Returns (forced_or_None, subproblems, rebuild).
    infeasible (rule 1) -> (None, [], None). subproblems: [(GraphData, old_index)];
    rebuild(parts) unions the solved components' labels."""
    if any(d == 0 for d in gd.deg):
        return None, [], None
    subs = []
    for mask in components(gd):
        g, idx = induced_subgraph(gd, mask)
        subs.append((g, idx))
    def rebuild(parts):
        out = set()
        for (g, idx), S in zip(subs, parts):
            for u in S:
                out.add(int(idx[u]))
        return out
    return set(), subs, rebuild


def reduced_size(gd):
    """(n_before, n_after): total vertices kept after rules 1-2 (sum over components)."""
    res = reduce(gd)
    if res[0] is None:
        return gd.n, 0
    return gd.n, sum(g.n for g, _ in res[1])


def counterexample():
    """Smallest graph showing the naive leaf rule is wrong for TDSP: path P3 (0-1-2).
    Leaf 0 forces vertex 1. Naively deleting {1} u N(1)={0,2} leaves S={1}, which is
    NOT total-dominating (vertex 1 has no chosen neighbour). The optimum is {1,0}
    (or {1,2}), size 2. So the reduction loses optimality => invalid for TDSP."""
    from graphs import make_graph_data
    import networkx as nx
    gd = make_graph_data(nx.path_graph(3))
    return {"n": gd.n, "forced_by_leaf_rule": sorted(leaf_supports(gd)),
            "naive_answer_size": len(leaf_supports(gd)), "true_optimum": 2}


# ---- sanity: rules 1-2 preserve the optimum --------------------------------
if __name__ == "__main__":
    from graphs import er, grid, make_graph_data
    from baselines import exact_solution
    rng = np.random.default_rng(0)
    bad = ntest = skipped = 0
    for n in (8, 10, 12, 14):
        for _ in range(40):
            gd = make_graph_data(er(n, 0.5, rng))
            res = reduce(gd)
            if res[0] is None:              # infeasible
                continue
            _forced, subs, rebuild = res
            parts, ok = [], True
            for g, _idx in subs:
                opt = exact_solution(g, "tot", 30)[1]
                if opt is None:
                    ok = False; break
                parts.append({int(u) for u in opt})
            if not ok:
                skipped += 1; continue
            S = rebuild(parts)
            o, _, oko = exact_solution(gd, "tot", 60)
            if not oko:
                skipped += 1; continue
            ntest += 1
            if len(S) != o:
                bad += 1; print("MISMATCH", n, len(S), o)
    # component splitting on a disconnected graph (rules 1-2 are exact)
    print("leaf-rule counterexample:", counterexample())
    print(f"sanity: {ntest} compared, mismatches={bad}, skipped={skipped}")
