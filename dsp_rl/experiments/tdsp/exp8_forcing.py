"""Experiment 8 (chapter 5): does exact forced-move knowledge help RL on TDSP?

Exp7 showed the classic reduction rules never fire on our dense benchmarks. Here we
change the *test distribution* to sparse, leaf-rich families (uniform random labeled
trees, and BA m=1 recursive trees) where the forced set F = leaf supports is never
empty, and measure what forcing actually buys.

Claim separation (keep these apart in the thesis):
  phase 0 (local, no training)  -- the exp5 dense-trained model, zero-shot on sparse
                                    graphs, with and without the forced seed. Any gap
                                    mixes distribution shift with the effect of forcing.
  phase 1 (Colab, two trainings) -- arm A: trained on trees, plain env;
                                    arm B: trained on trees, every episode seeded with F.
                                    Same budget/protocol as exp5, so learning curves are
                                    comparable and forcing is the only difference.

The forcing itself is *sound*: every leaf support lies in every total dominating set
(a leaf's only neighbour must be chosen to satisfy it), so seeding S0 = F removes no
optimal solution. This is NOT the invalid exp7 "delete-and-recurse" reduction: the
residual constraint (each u in F still needs a chosen neighbour) is carried by the
environment's own mechanics (vf stays 0 at u until RL picks one of its neighbours).

Run from the dsp_rl directory:
  python experiments/tdsp/exp8_forcing.py phase0 \
     --model size_v2=runs/tdsp_er20_s0_size_v2/model.pt --out results/tdsp/exp8
  python experiments/tdsp/exp8_forcing.py eval \
     --model armA=.../armA_sparse_base/model.pt --model armB=.../armB_sparse_forced/model.pt \
     --model size_v2=runs/tdsp_er20_s0_size_v2/model.pt --out results/tdsp/exp8
  python experiments/tdsp/exp8_forcing.py convergence \
     --log armA=.../armA_sparse_base/log.csv --log armB=.../armB_sparse_forced/log.csv \
     --out results/tdsp/exp8
"""
import argparse, csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from graphs import make_graph_data, grid, er, ba, hexa, tri, tree, ba1
from baselines import exact
from tdsp_env import TDSPEnv
from tdsp import greedy_tds, has_isolated
from reductions import leaf_supports
from heuristics import grasp_tds, is_total
from agent import agent_for_model

HEADER = ["family", "n", "source", "|S|", "mean|F|", "gamma_t", "|S|/gamma_t",
          "ratio_vs_greedy", "rl_steps", "time_s"]
SPARSE = {"TREE": tree, "BA1": ba1}


def sparse_instances(reps, rng, sizes=(20, 40, 100, 200)):
    return [(fam, n, make_graph_data(gen(n, rng))) for fam, gen in SPARSE.items()
            for n in sizes for _ in range(reps)]


def dense_instances(reps, rng):
    out = []
    for sh in [(4, 5), (6, 6), (7, 9), (10, 10)]:
        out.append(("GRID", 0, make_graph_data(grid(*sh))))
    for sh in [(4, 5), (6, 6), (8, 10)]:
        out.append(("HEX", 0, make_graph_data(hexa(*sh))))
    out.append(("TRI", 0, make_graph_data(tri(6, 6))))
    for n in [20, 40, 100, 300]:
        for _ in range(reps):
            out.append(("ER", n, make_graph_data(er(n, 0.4, rng))))
            out.append(("BA", n, make_graph_data(ba(n, 4, rng))))
    return [(f, gd.n, gd) for f, _, gd in out]


def solve_rl(agent, env, gd, forced=()):
    """Greedy roll-out of the policy; with forced seed F, `rl_steps` counts only the
    decisions the agent actually made (|S| still includes F, as reported)."""
    t = time.perf_counter()
    s = env.reset(gd, forced=forced) if forced else env.reset(gd)
    done = getattr(env, "done", False)
    while not done:
        c = env.candidates(); q = agent.q_values(gd, s)
        s, _, done = env.step(int(c[np.argmax(q[c])]))
    dt = time.perf_counter() - t
    assert is_total(gd, env.selected), "policy output is not total-dominating"
    return set(env.selected), len(env.selected) - len(forced), dt


def _rows_add(rows, family, gd, src, S, F, gt, greedy, steps, dt):
    rows.append({"family": family, "n": gd.n, "source": src, "|S|": len(S),
                 "mean|F|": F, "gamma_t": gt if gt else "",
                 "|S|/gamma_t": round(len(S) / gt, 3) if gt else "",
                 "ratio_vs_greedy": round(len(S) / len(greedy), 3),
                 "rl_steps": steps, "time_s": round(dt, 5)})


def aggregate(rows, keys=("family", "n", "source")):
    group = {}
    for r in rows:
        group.setdefault(tuple(r[k] for k in keys), []).append(r)
    agg = []
    for key, rs in group.items():
        out = dict(zip(keys, key))
        for c in ["|S|", "mean|F|", "gamma_t", "|S|/gamma_t", "ratio_vs_greedy", "rl_steps", "time_s"]:
            vals = [r[c] for r in rs if r[c] != ""]
            out[c] = round(float(np.mean(vals)), 3) if vals else ""
        agg.append(out)
    return sorted(agg, key=lambda r: (r["family"], r["n"], r["source"]))


def write_table(path_base, title, agg):
    with open(path_base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER); w.writeheader(); w.writerows(agg)
    with open(path_base + ".md", "w") as f:
        f.write(f"# {title}\n\n| " + " | ".join(HEADER) + " |\n|" + "---|" * len(HEADER) + "\n")
        for r in agg:
            f.write("| " + " | ".join(str(r[c]) for c in HEADER) + " |\n")
    print(open(path_base + ".md").read())


def phase0(a):
    rng = np.random.default_rng(2024)
    # ---- part 1: how much forcing material do the sparse families offer? --------
    stats = []
    for fam, gen in SPARSE.items():
        for n in (20, 40, 100, 200):
            leaves, supports, with_f = [], [], 0
            for _ in range(a.reps):
                gd = make_graph_data(gen(n, rng))
                lv = sum(1 for v in range(gd.n) if gd.deg[v] == 1)
                F = leaf_supports(gd)
                leaves.append(lv); supports.append(len(F)); with_f += bool(F)
            stats.append({"family": fam, "n": n, "instances": a.reps,
                          "pct_with_F": round(100 * with_f / a.reps, 1),
                          "mean_leaves": round(float(np.mean(leaves)), 2),
                          "mean|F|": round(float(np.mean(supports)), 2),
                          "mean|F|/n": round(float(np.mean(supports)) / n, 3)})
    with open(f"{a.out}/phase0_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)
    with open(f"{a.out}/phase0_stats.md", "w") as f:
        cols = list(stats[0])
        f.write("# Exp8 phase 0 (a): forcing material in the sparse families\n\n"
                "Every instance of TREE/BA1 has leaves, so the forced set F is never empty "
                "(contrast exp7: F empty on all dense families).\n\n")
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in stats:
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    print(open(f"{a.out}/phase0_stats.md").read())

    # ---- part 2: zero-shot, the exp5 dense model on sparse graphs +- forcing ----
    agents = {name: agent_for_model(path, device=a.device) for name, path in _models(a)}
    env = TDSPEnv()
    rows = []
    for fam, n, gd in sparse_instances(a.reps, rng, sizes=(20, 40)):
        if has_isolated(gd):
            continue
        F = leaf_supports(gd)
        gt = exact(gd, "tot", a.ilp_time)[0]
        greedy = greedy_tds(gd)
        t = time.perf_counter(); grasp = grasp_tds(gd, a.grasp_iters, np.random.default_rng(gd.n)); tg = time.perf_counter() - t
        _rows_add(rows, fam, gd, "greedy", greedy, len(F), gt, greedy, len(greedy), 0.0)
        _rows_add(rows, fam, gd, "grasp", grasp, len(F), gt, greedy, len(grasp), tg)
        for name, ag in agents.items():
            S, steps, dt = solve_rl(ag, env, gd)
            _rows_add(rows, fam, gd, name, S, len(F), gt, greedy, steps, dt)
            S, steps, dt = solve_rl(ag, env, gd, forced=F)
            _rows_add(rows, fam, gd, name + "+F", S, len(F), gt, greedy, steps, dt)
    write_table(f"{a.out}/phase0_zero_shot",
                "Exp8 phase 0 (b): zero-shot -- exp5 dense-trained model on sparse graphs",
                aggregate(rows))


def eval_models(a):
    """Phase 1 (Colab): the unified final exp8 table -- trained arms +- F on sparse,
    plain on dense/lattices (cross-family check that sparse training cost nothing there)."""
    agents = {name: agent_for_model(path, device=a.device) for name, path in _models(a)}
    env = TDSPEnv(); rng = np.random.default_rng(2024)
    rows = []
    for fam, n, gd in sparse_instances(a.reps, rng):
        if has_isolated(gd):
            continue
        F = leaf_supports(gd)
        gt = exact(gd, "tot", a.ilp_time)[0] if gd.n <= 120 else None
        greedy = greedy_tds(gd)
        t = time.perf_counter(); grasp = grasp_tds(gd, a.grasp_iters, np.random.default_rng(gd.n)); tg = time.perf_counter() - t
        _rows_add(rows, fam, gd, "greedy", greedy, len(F), gt, greedy, len(greedy), 0.0)
        _rows_add(rows, fam, gd, "grasp", grasp, len(F), gt, greedy, len(grasp), tg)
        for name, ag in agents.items():
            S, steps, dt = solve_rl(ag, env, gd)
            _rows_add(rows, fam, gd, name, S, len(F), gt, greedy, steps, dt)
            S, steps, dt = solve_rl(ag, env, gd, forced=F)
            _rows_add(rows, fam, gd, name + "+F", S, len(F), gt, greedy, steps, dt)
    for fam, n, gd in dense_instances(a.reps, rng):          # cross-family, no F rows
        if has_isolated(gd):
            continue
        gt = exact(gd, "tot", a.ilp_time)[0] if gd.n <= 120 else None
        greedy = greedy_tds(gd)
        t = time.perf_counter(); grasp = grasp_tds(gd, a.grasp_iters, np.random.default_rng(gd.n)); tg = time.perf_counter() - t
        _rows_add(rows, fam, gd, "greedy", greedy, 0, gt, greedy, len(greedy), 0.0)
        _rows_add(rows, fam, gd, "grasp", grasp, 0, gt, greedy, len(grasp), tg)
        for name, ag in agents.items():
            S, steps, dt = solve_rl(ag, env, gd)
            _rows_add(rows, fam, gd, name, S, 0, gt, greedy, steps, dt)
    write_table(f"{a.out}/phase1_eval", "Exp8 phase 1: final eval -- arms A/B vs baselines", aggregate(rows))


def convergence(a):
    """Phase 1 (Colab): learning-curve comparison from the two arms' log.csv files."""
    curves, summary = [], []
    for name, path in _logs(a):
        pts = [(int(r["episode"]), float(r["val_mean_size"]))
               for r in csv.DictReader(open(path)) if r["val_mean_size"]]
        final = min(s for _, s in pts)
        conv = next((ep for ep, s in pts if s <= final + 0.05), None)  # first touch of near-final
        summary.append({"arm": name, "episodes": pts[-1][0], "final_val_mean_size": final,
                        "first_near_final_episode": conv})
        for ep, s in pts:
            curves.append({"arm": name, "episode": ep, "val_mean_size": s})
    with open(f"{a.out}/phase1_convergence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "episode", "val_mean_size"]); w.writeheader(); w.writerows(curves)
    cols = list(summary[0])
    with open(f"{a.out}/phase1_convergence_summary.md", "w") as f:
        f.write("# Exp8 phase 1: convergence of the two arms\n\n"
                "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in summary:
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    print(open(f"{a.out}/phase1_convergence_summary.md").read())
    try:                                                       # plot for the thesis (optional)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, _ in _logs(a):
            xs = [c["episode"] for c in curves if c["arm"] == name]
            ys = [c["val_mean_size"] for c in curves if c["arm"] == name]
            ax.plot(xs, ys, label=name, lw=1.2)
        ax.set_xlabel("episode"); ax.set_ylabel("validation mean |S|"); ax.legend()
        ax.set_title("exp8: convergence with vs without forced seeding")
        fig.tight_layout(); fig.savefig(f"{a.out}/phase1_convergence.png", dpi=150)
        print("plot:", f"{a.out}/phase1_convergence.png")
    except ImportError:
        print("matplotlib not available -- skipped the .png (curves are in the .csv)")


def _models(a):
    return [spec.split("=", 1) for spec in a.model]


def _logs(a):
    return [spec.split("=", 1) for spec in a.log]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["phase0", "eval", "convergence"])
    ap.add_argument("--model", action="append", default=[], help="name=path; repeatable")
    ap.add_argument("--log", action="append", default=[], help="arm=path/to/log.csv; repeatable")
    ap.add_argument("--out", default="results/tdsp/exp8")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--grasp_iters", type=int, default=40)
    ap.add_argument("--ilp_time", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    {"phase0": phase0, "eval": eval_models, "convergence": convergence}[a.mode](a)


if __name__ == "__main__":
    main()
