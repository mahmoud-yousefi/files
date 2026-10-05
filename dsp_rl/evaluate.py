"""Evaluation mirroring paper Sec. 4.3.2. Writes CSV + a markdown table into --out.
  python evaluate.py grid_opt --model runs/er20_s0/model.pt --out results/table3
  python evaluate.py er_ba    --model ... --graphs 100
  python evaluate.py lattice  --model ...
"""
import argparse, os, time, csv
import numpy as np, torch
from graphs import *
from env import DSPEnv
from agent import DDQNAgent
from baselines import greedy_ds, greedy_ds_masked, exact, is_dominating

TABLE3 = [(5,4),(5,5),(5,6),(5,7),(5,8),(5,9),(5,10),(6,6),(6,7),(6,8),(6,9),(6,10),(7,7),(7,8),(7,9),(7,10),(8,8),(8,9),(8,10),(9,9),(9,10),(10,10)]
GRID_SHAPES = {20:(4,5),36:(6,6),63:(7,9),81:(9,9),100:(10,10),200:(10,20),300:(15,20),400:(20,20),500:(20,25)}   # shapes inferred (D10)
HEX_SHAPES = {22:(1,5),38:(3,4),58:(4,5),82:(5,6),96:(6,6),196:(8,10),306:(10,13),418:(13,14),510:(15,15)}       # networkx sizes that match Table 2 (D10)


def solve(agent, env, gd):
    t = time.perf_counter(); vf = env.reset(gd); done = False
    while not done:
        q = agent.q_values(gd, vf); c = env.candidates(); vf, _, done = env.step(int(c[np.argmax(q[c])]))
    dt = time.perf_counter() - t
    assert is_dominating(gd, env.selected)
    return len(env.selected), dt


def timed(fn, gd):
    t = time.perf_counter(); s = fn(gd); return len(s), time.perf_counter() - t


def md(path, header, rows):
    with open(path, "w") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows: f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    with open(path.replace(".md", ".csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["grid_opt", "er_ba", "lattice"])
    ap.add_argument("--model", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--graphs", type=int, default=100); ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--nf", type=int, default=32); ap.add_argument("--agg", default="sum"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--ilp_time", type=int, default=120)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    agent = DDQNAgent(a.nf, 3, a.agg, device=a.device); agent.load(a.model); env = DSPEnv(); rng = np.random.default_rng(a.seed)
    rows = []
    if a.mode == "grid_opt":      # paper Table 3
        hdr = ["m", "n", "model", "greedy_any", "greedy_masked", "opt_gamma", "opt_proven", "delta_model", "delta_greedy_masked"]
        for m, n in TABLE3:
            gd = make_graph_data(grid(m, n)); k, _ = solve(agent, env, gd); g1 = len(greedy_ds(gd)); g2 = len(greedy_ds_masked(gd))
            o, ok = exact(gd, "dom", a.ilp_time); rows.append([m, n, k, g1, g2, o, ok, k - o, g2 - o])
        rows.append(["mean", "", np.mean([r[2] for r in rows]), np.mean([r[3] for r in rows]), np.mean([r[4] for r in rows]), "", "", np.mean([r[7] for r in rows]), np.mean([r[8] for r in rows])])
    elif a.mode == "er_ba":       # paper Table 1 (metric = cardinality |D|, see D11)
        hdr = ["family", "n", "model_size", "greedy_any_size", "greedy_masked_size", "model_time_s", "greedy_time_s"]
        for fam in ["BA", "ER"]:
            for n in [20, 40, 80, 100, 200, 300, 400, 500]:
                ms, gs, gm, mt, gt = [], [], [], [], []
                for _ in range(a.graphs):
                    gd = make_graph_data(ba(n, 18, rng) if fam == "BA" else er(n, 0.8, rng))
                    k, t = solve(agent, env, gd); ms.append(k); mt.append(t)
                    k, t = timed(greedy_ds, gd); gs.append(k); gt.append(t); gm.append(len(greedy_ds_masked(gd)))
                rows.append([fam, n, round(np.mean(ms), 2), round(np.mean(gs), 2), round(np.mean(gm), 2), round(np.mean(mt), 5), round(np.mean(gt), 5)])
    else:                         # paper Table 2, grid + hex (triangular: shapes not recoverable, D10)
        hdr = ["family", "n", "shape", "model", "greedy_any", "greedy_masked"]
        for fam, shapes, gen in [("GRID", GRID_SHAPES, grid), ("HEX", HEX_SHAPES, hexa)]:
            for n, sh in shapes.items():
                gd = make_graph_data(gen(*sh)); k, _ = solve(agent, env, gd)
                rows.append([fam, gd.n, sh, k, len(greedy_ds(gd)), len(greedy_ds_masked(gd))])
    md(f"{a.out}/{a.mode}.md", hdr, rows); print(open(f"{a.out}/{a.mode}.md").read())


if __name__ == "__main__":
    main()
