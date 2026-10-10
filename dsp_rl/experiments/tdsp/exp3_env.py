"""Experiment 3 (chapter 5): a redesigned action space for TDSP, and its verification.

The paper's mask cannot yield a total dominating set (exp2). Here we change only
the rules of the environment (tdsp_env.TDSPEnv): any not-yet-chosen vertex may be
picked, picking v satisfies N(v) but not v, and the episode ends when every vertex
has a neighbour in S. The RL brain is untouched.

This script verifies the mechanics and that two simple policies built on the new
rules always terminate in a genuine total dominating set, and how their size
compares with the native greedy baseline and the optimum gamma_t.

Run from the dsp_rl directory:
  python experiments/tdsp/exp3_env.py
"""
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from graphs import make_graph_data, grid, er, ba, hexa, tri
from baselines import exact, is_total_dominating, greedy_ds_masked
from tdsp_env import TDSPEnv
from tdsp import greedy_tds, has_isolated
from evaluate import GRID_SHAPES, HEX_SHAPES

HEADER = ["family", "n", "policy", "|S|", "total_dominating", "steps", "gamma_t", "|S|/gamma_t", "greedy_tds"]


def random_policy(env, gd, rng):
    vf, done, steps = env.reset(gd), False, 0
    while not done:
        vf, _, done = env.step(int(rng.choice(env.candidates()))); steps += 1
    return set(env.selected), steps


def greedy_env_policy(env, gd):
    """Greedy under the new rules: pick the vertex satisfying the most new vertices."""
    vf, done, steps = env.reset(gd), False, 0
    nbr = [set(gd.nbrs[u].tolist()) for u in range(gd.n)]
    while not done:
        best = max(env.candidates(), key=lambda u: len({w for w in nbr[u] if vf[w] == 0}))
        vf, _, done = env.step(int(best)); steps += 1
    return set(env.selected), steps


def mechanics_check(gd, rng, env):
    """Unit-check the rules: picking v satisfies N(v) but not v itself."""
    env.reset(gd)
    v = int(rng.choice(env.candidates()))
    nb = set(gd.nbrs[v].tolist())
    vf0 = env.vf.copy()
    vf, _, _ = env.step(v)
    assert env.ins[v] == 1 and v not in env.candidates()
    assert all(vf[u] == 1 for u in nb), "neighbours of v must be satisfied"
    assert vf[v] == vf0[v], "v must NOT become satisfied by being picked"
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tdsp/exp3")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--ilp_time", type=int, default=60)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(11)
    env = TDSPEnv()

    rows, mech = [], 0
    def run(family, gd, do_opt):
        nonlocal mech
        if has_isolated(gd):
            return
        gt = exact(gd, "tot", a.ilp_time)[0] if do_opt else None
        gtds = len(greedy_tds(gd))
        for name, fn in [("random", lambda: random_policy(env, gd, rng)),
                         ("greedy_env", lambda: greedy_env_policy(env, gd))]:
            S, steps = fn()
            ok = is_total_dominating(gd, S)
            assert ok, f"{name} on n={gd.n} did not total-dominate"
            mech += mechanics_check(gd, rng, env)
            rows.append({"family": family, "n": gd.n, "policy": name, "|S|": len(S),
                         "total_dominating": ok, "steps": steps,
                         "gamma_t": gt if gt is not None else "",
                         "|S|/gamma_t": round(len(S) / gt, 3) if gt else "",
                         "greedy_tds": gtds})

    for fam, shapes, gen in [("GRID", GRID_SHAPES, grid), ("HEX", HEX_SHAPES, hexa)]:
        for n, sh in shapes.items():
            run(fam, make_graph_data(gen(*sh)), n <= 200)
    run("TRI", make_graph_data(tri(6, 6)), True)
    for fam, gen in [("ER", lambda: er(40, 0.4, rng)), ("BA", lambda: ba(40, 4, rng))]:
        for _ in range(a.reps):
            run(fam, make_graph_data(gen()), True)

    with open(f"{a.out}/env.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER); w.writeheader(); w.writerows(rows)
    with open(f"{a.out}/env.md", "w") as f:
        f.write("# Exp3: redesigned action space for TDSP\n\n")
        f.write(f"graphs/rows={len(rows)}  mechanics checks passed={mech}\n")
        f.write("Every policy run ended in a genuine total dominating set.\n\n")
        f.write("| " + " | ".join(HEADER) + " |\n|" + "---|" * len(HEADER) + "\n")
        for r in rows: f.write("| " + " | ".join(str(r[c]) for c in HEADER) + " |\n")
    print(f"rows={len(rows)}  mechanics checks passed={mech}")
    print(open(f"{a.out}/env.md").read()[:600])


if __name__ == "__main__":
    main()
