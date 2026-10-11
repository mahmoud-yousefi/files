"""Experiment 6 (chapter 5): a stronger baseline for TDSP.

Plain greedy is a weak opponent; TDSP is a set-cover problem, so we add the standard
set-cover baselines -- redundancy removal and a GRASP multi-start randomized greedy --
and compare the trained RL model against greedy, GRASP and the ILP optimum.

Run from the dsp_rl directory:
  python experiments/tdsp/exp6_baselines.py \
     --model size=runs/tdsp_er20_s0_size_v2/model.pt --out results/tdsp/exp6
"""
import argparse, csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from graphs import make_graph_data, grid, er, ba, hexa, tri
from baselines import exact
from tdsp_env import TDSPEnv
from tdsp import greedy_tds, has_isolated
from heuristics import grasp_tds, is_total
from agent import agent_for_model

HEADER = ["family", "n", "source", "|S|", "gamma_t", "|S|/gamma_t", "ratio_vs_greedy", "time_s"]


def solve_rl(agent, env, gd):
    t = time.perf_counter(); s = env.reset(gd); done = False
    while not done:
        c = env.candidates(); q = agent.q_values(gd, s)
        s, _, done = env.step(int(c[np.argmax(q[c])]))
    dt = time.perf_counter() - t
    assert is_total(gd, env.selected)
    return set(env.selected), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="name=path; repeatable")
    ap.add_argument("--out", default="results/tdsp/exp6")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--grasp_iters", type=int, default=40)
    ap.add_argument("--ilp_time", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    agents = {}
    for spec in a.model:
        name, path = spec.split("=", 1)
        agents[name] = agent_for_model(path, device=a.device)
        print(f"loaded {name}: {path}")
    env = TDSPEnv(); rng = np.random.default_rng(2024)
    rows = []

    def one(family, gd):
        if has_isolated(gd):
            return
        gt = exact(gd, "tot", a.ilp_time)[0] if gd.n <= 120 else None
        res = {}
        t = time.perf_counter(); res["greedy"] = greedy_tds(gd); tg = time.perf_counter() - t
        t = time.perf_counter(); res["grasp"] = grasp_tds(gd, a.grasp_iters, np.random.default_rng(gd.n)); tp = time.perf_counter() - t
        for name, ag in agents.items():
            S, dt = solve_rl(ag, env, gd); res[name] = S
            _add(rows, family, gd, name, S, gt, res["greedy"], dt)
        _add(rows, family, gd, "greedy", res["greedy"], gt, res["greedy"], tg)
        _add(rows, family, gd, "grasp", res["grasp"], gt, res["greedy"], tp)

    def _add(rows, family, gd, src, S, gt, greedy, dt):
        rows.append({"family": family, "n": gd.n, "source": src, "|S|": len(S),
                     "gamma_t": gt if gt else "", "|S|/gamma_t": round(len(S)/gt, 3) if gt else "",
                     "ratio_vs_greedy": round(len(S)/len(greedy), 3), "time_s": round(dt, 5)})

    for sh in [(4, 5), (6, 6), (7, 9), (10, 10)]:
        one("GRID", make_graph_data(grid(*sh)))
    for sh in [(4, 5), (6, 6), (8, 10)]:
        one("HEX", make_graph_data(hexa(*sh)))
    one("TRI", make_graph_data(tri(6, 6)))
    for n in [20, 40, 80, 100, 200, 300]:
        for _ in range(a.reps):
            one("ER", make_graph_data(er(n, 0.4, rng)))
            one("BA", make_graph_data(ba(n, 4, rng)))

    keys = ["family", "n", "source"]; group = {}
    for r in rows:
        group.setdefault(tuple(r[k] for k in keys), []).append(r)
    agg = []
    for key, rs in group.items():
        out = dict(zip(keys, key))
        for c in ["|S|", "gamma_t", "|S|/gamma_t", "ratio_vs_greedy", "time_s"]:
            vals = [r[c] for r in rs if r[c] != ""]
            out[c] = round(float(np.mean(vals)), 3) if vals else ""
        agg.append(out)
    agg.sort(key=lambda r: (r["family"], r["n"], r["source"]))
    with open(f"{a.out}/baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER); w.writeheader(); w.writerows(agg)
    with open(f"{a.out}/baselines.md", "w") as f:
        f.write("# Exp6: TDSP baselines (greedy / GRASP) vs model vs optimum\n\n")
        f.write("| " + " | ".join(HEADER) + " |\n|" + "---|" * len(HEADER) + "\n")
        for r in agg: f.write("| " + " | ".join(str(r[c]) for c in HEADER) + " |\n")
    print(open(f"{a.out}/baselines.md").read())


if __name__ == "__main__":
    main()
