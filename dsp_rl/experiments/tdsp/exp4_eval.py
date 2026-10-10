"""Experiment 4 (chapter 5): evaluate RL trained directly on TDSP.

Compares the trained models (one per reward) against the native TDSP greedy and,
where ILP is feasible, the optimum gamma_t. Models are trained on ER n=20; the
sweep also tests family/size generalisation.

Run from the dsp_rl directory:
  python experiments/tdsp/exp4_eval.py \
     --model size=runs/tdsp_er20_s0_size/model.pt --model paper=runs/tdsp_er20_s0_paper/model.pt \
     --out results/tdsp/exp4
"""
import argparse, csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from graphs import make_graph_data, grid, er, ba, hexa, tri
from baselines import exact, is_total_dominating
from tdsp_env import TDSPEnv
from tdsp import greedy_tds, has_isolated
from agent import DDQNAgent

HEADER = ["family", "n", "source", "|S|", "gamma_t", "|S|/gamma_t", "greedy_tds", "ratio_vs_greedy", "time_s"]


def solve(agent, env, gd):
    t = time.perf_counter(); s = env.reset(gd); done = False
    while not done:
        c = env.candidates(); q = agent.q_values(gd, s)
        s, _, done = env.step(int(c[np.argmax(q[c])]))
    dt = time.perf_counter() - t
    assert is_total_dominating(gd, env.selected), "model output is not total-dominating"
    return set(env.selected), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=path; repeatable")
    ap.add_argument("--out", default="results/tdsp/exp4")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--nf", type=int, default=32); ap.add_argument("--agg", default="sum")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--ilp_time", type=int, default=60)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    env = TDSPEnv(); agents = {}
    for spec in a.model:
        name, path = spec.split("=", 1)
        ag = DDQNAgent(a.nf, 3, a.agg, in_ch=env.n_channels, device=a.device); ag.load(path)
        agents[name] = ag
        print(f"loaded {name}: {path}")

    rng = np.random.default_rng(2024)
    rows = []
    def one(family, gd):
        if has_isolated(gd):
            return
        gt = exact(gd, "tot", a.ilp_time)[0] if gd.n <= 120 else None
        gtds = len(greedy_tds(gd))
        for name, ag in agents.items():
            S, dt = solve(ag, env, gd)
            rows.append({"family": family, "n": gd.n, "source": name, "|S|": len(S),
                         "gamma_t": gt if gt else "", "|S|/gamma_t": round(len(S)/gt, 3) if gt else "",
                         "greedy_tds": gtds, "ratio_vs_greedy": round(len(S)/gtds, 3), "time_s": round(dt, 5)})
        rows.append({"family": family, "n": gd.n, "source": "greedy_tds", "|S|": gtds,
                     "gamma_t": gt if gt else "", "|S|/gamma_t": round(gtds/gt, 3) if gt else "",
                     "greedy_tds": gtds, "ratio_vs_greedy": 1.0, "time_s": ""})

    for sh in [(4, 5), (6, 6), (7, 9), (10, 10)]:
        one("GRID", make_graph_data(grid(*sh)))
    for sh in [(4, 5), (6, 6), (8, 10)]:
        one("HEX", make_graph_data(hexa(*sh)))
    one("TRI", make_graph_data(tri(6, 6)))
    for n in [20, 40, 80, 100, 200, 300]:
        for _ in range(a.reps):
            one("ER", make_graph_data(er(n, 0.4, rng)))
            one("BA", make_graph_data(ba(n, 4, rng)))

    # aggregate reps for random families
    keys = ["family", "n", "source"]; group = {}
    for r in rows:
        group.setdefault(tuple(r[k] for k in keys), []).append(r)
    agg = []
    for key, rs in group.items():
        out = dict(zip(keys, key))
        for c in ["|S|", "gamma_t", "|S|/gamma_t", "greedy_tds", "ratio_vs_greedy", "time_s"]:
            vals = [r[c] for r in rs if r[c] != ""]
            out[c] = round(float(np.mean(vals)), 3) if vals else ""
        agg.append(out)
    agg.sort(key=lambda r: (r["family"], r["n"], r["source"]))

    with open(f"{a.out}/eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER); w.writeheader(); w.writerows(agg)
    with open(f"{a.out}/eval.md", "w") as f:
        f.write("# Exp4: RL trained directly on TDSP\n\n")
        f.write("models: " + ", ".join(f"{k}={v}" for k, v in [(s.split('=',1)[0], s.split('=',1)[1]) for s in a.model]) + "\n\n")
        f.write("| " + " | ".join(HEADER) + " |\n|" + "---|" * len(HEADER) + "\n")
        for r in agg: f.write("| " + " | ".join(str(r[c]) for c in HEADER) + " |\n")
    print(open(f"{a.out}/eval.md").read())


if __name__ == "__main__":
    main()
