"""Worked example for chapter 5: the whole pipeline on graphs small enough to draw.

Two things the chapter needs and that no aggregate CSV can provide:

  1. `fig5_11_p4`: why Chen's action mask cannot solve TDSP, on P4 (4 vertices).
     The mask forces S to be independent; on P4 the best independent dominating
     set has size 2 ({0,2}) and vertex 0 has no neighbour in S, so it is *not* a
     total dominating set. The optimum TDS has size 2 as well ({1,2}), i.e. the
     failure is not about size but about feasibility.

  2. `fig5_12_trace`: the proposed pipeline, step by step, on a 12-vertex tree:
     leaf forcing seeds F, then the trained policy (armA + F) decides the rest.
     For every RL decision the table records the legal set, the top-k gain set,
     the Q values the network assigns and the reward actually collected, so the
     chapter can walk through state -> action -> reward -> next state explicitly.

Outputs (under results/tdsp/ch5):
  example_trace.md, example_steps.csv, example_p4.csv, figs/fig5_11_p4.pdf,
  figs/fig5_12_trace.pdf

Run from the dsp_rl directory.
"""
import argparse, csv, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from graphs import make_graph_data, tree                      # noqa: E402
from tdsp import greedy_tds                                   # noqa: E402
from baselines import greedy_ds_masked, is_total_dominating, exact_solution  # noqa: E402
from tdsp_env import TDSPEnv                                  # noqa: E402
from reductions import leaf_supports                          # noqa: E402
from agent import agent_for_model                             # noqa: E402
from baselines import exact                                   # noqa: E402

MODEL = "runs/armA_sparse_base/model.pt"
TREE_N, TREE_SEED, TOPK = 12, 5, 3


def gains(env):
    """gain(v) = how many *unsatisfied* neighbours v would satisfy (exp10 definition)."""
    g = np.zeros(env.gd.n, dtype=np.int64)
    for v in range(env.gd.n):
        g[v] = int((env.vf[env.gd.nbrs[v]] == 0).sum())
    return g


def rollout_with_trace(agent, gd, forced, env, k=None):
    """One episode, recording every RL decision (state, action, reward, Q)."""
    state = env.reset(gd, forced=forced)
    steps, S, done = [], list(forced), bool(env.done)
    while not done:
        c = env.candidates()
        if len(c) == 0:
            break
        g = gains(env)
        legal = list(c) if k is None or len(c) <= k else \
            [int(v) for v in c[np.lexsort((c, -g[c]))][:k]]        # top-k by gain, ties by index
        q = agent.q_values(gd, state)[legal] if agent is not None else np.zeros(len(legal))
        a = int(legal[int(np.argmax(q))]) if agent is not None else int(legal[0])
        before = env.vf.copy()
        state, r, done = env.step(a)
        newly = [int(u) for u in np.where(env.vf > before)[0]]
        steps.append({"action": a, "legal": legal, "gain": {int(v): int(g[v]) for v in c},
                      "q": {int(v): round(float(x), 4) for v, x in zip(legal, q)},
                      "satisfied_new": newly, "reward": r, "S_size": len(S) + 1})
        S.append(a)
    return S, steps


def draw(ax, gd, pos, S, forced, newly=None, title=""):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(gd.n))
    G.add_edges_from(zip(*[gd.row[:len(gd.row) // 2], gd.col[:len(gd.col) // 2]]))
    colours, sizes = [], []
    for v in range(gd.n):
        if v in S:
            colours.append("#D1495B")
        elif v in ((newly or []) or []):
            colours.append("#EDAE49")
        elif v in forced:
            colours.append("#2E5EAA")
        else:
            colours.append("#DDDDDD")
        sizes.append(420 if v in S else 320)
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, edge_color="#888888")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colours, node_size=sizes,
                           edgecolors="#333333", linewidths=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="#111111")
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tdsp/ch5")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    os.makedirs(os.path.join(a.out, "figs"), exist_ok=True)

    # ---------------------------------------------------------------- 1. P4
    p4 = nx.path_graph(4)
    gd4 = make_graph_data(p4)
    S4 = list(greedy_ds_masked(gd4))     # Chen's mask, greedy inside it (exp2 definition)
    opt4 = exact_solution(gd4, "tot")[1]
    greedy4 = greedy_tds(gd4)
    with open(os.path.join(a.out, "example_p4.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "S", "|S|", "total_dominating"])
        w.writeheader()
        for name, S in (("chen_mask", S4), ("greedy_tds", greedy4), ("optimum", opt4)):
            w.writerow({"source": name, "S": " ".join(map(str, sorted(S))), "|S|": len(S),
                        "total_dominating": is_total_dominating(gd4, set(S))})

    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.2))
    pos4 = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0)}
    draw(axs[0], gd4, pos4, set(), (), title="(a) P4, no decision yet")
    draw(axs[1], gd4, pos4, set(S4), (), title=f"(b) Chen mask: S={sorted(S4)} — not total")
    draw(axs[2], gd4, pos4, set(opt4), (), title=f"(c) optimal TDS: S={sorted(opt4)}")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "figs", "fig5_11_p4.pdf"))
    fig.savefig(os.path.join(a.out, "figs", "fig5_11_p4.png"))
    plt.close(fig)

    # ---------------------------------------------------------------- 2. trace
    rng = np.random.default_rng(TREE_SEED)
    G = tree(TREE_N, rng)
    gd = make_graph_data(G)
    F = leaf_supports(gd)
    leaves = sorted(v for v in range(gd.n) if gd.deg[v] == 1)
    env = TDSPEnv()
    ag = agent_for_model(a.model)
    # also show the mask-only control (k=3) to make the Q/mask split concrete
    S_free, steps = rollout_with_trace(ag, gd, F, env)
    S_k3, steps_k3 = rollout_with_trace(ag, gd, F, env, k=TOPK)
    gt = greedy_tds(gd)
    opt = exact_solution(gd, "tot")[1]
    env.reset(gd, forced=F)
    S_f, _ = rollout_with_trace(None, gd, F, TDSPEnv(), k=1)     # = greedy from F

    pos = nx.kamada_kawai_layout(G)
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.4))
    draw(axs[0], gd, pos, set(F), F, title=f"(a) forced leaf supports F={sorted(F)}")
    draw(axs[1], gd, pos, set(F) | {steps[0]["action"]}, F,
         title=f"(b) after 1 RL decision (v={steps[0]['action']}, r={steps[0]['reward']})")
    draw(axs[2], gd, pos, set(S_free), F,
         title=f"(c) final S (|S|={len(S_free)}, optimum={len(opt)})")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "figs", "fig5_12_trace.pdf"))
    fig.savefig(os.path.join(a.out, "figs", "fig5_12_trace.png"))
    plt.close(fig)

    # ---------------------------------------------------------------- 3. tables
    with open(os.path.join(a.out, "example_steps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "S_before", "forced", "legal",
                                          "topgain", "q_values", "action", "newly_satisfied",
                                          "reward", "|S|_after"])
        w.writeheader()
        S = list(F)
        for i, st in enumerate(steps, 1):
            tg = sorted(st["gain"].items(), key=lambda kv: -kv[1])[:TOPK]
            w.writerow({"step": i, "S_before": " ".join(map(str, S)), "forced": " ".join(map(str, F)),
                        "legal": " ".join(map(str, st["legal"])),
                        "topgain": " ".join(f"{v}:{g}" for v, g in tg),
                        "q_values": " ".join(f"{v}:{q}" for v, q in sorted(st["q"].items())),
                        "action": st["action"],
                        "newly_satisfied": " ".join(map(str, st["satisfied_new"])),
                        "reward": st["reward"], "|S|_after": st["S_size"]})
            S.append(st["action"])

    lines = ["# Worked example (chapter 5, section 5-3 and 5-4)", "",
             f"Graph: uniform random labelled tree, n={TREE_N}, seed={TREE_SEED}.",
             f"Leaves: {leaves}. Forced set F = support of the leaves = {sorted(F)} "
             f"(|F|={len(F)});", "every support is in *every* total dominating set, so forcing it is valid.", "",
             "## Sizes", "",
             "| method | S | \\|S\\| | total dominating |", "|---|---|---|---|"]
    for name, S in (("greedy_tds (no forcing)", gt), ("F only (seed)", set(F)),
                    ("F + policy, free action set", set(S_free)),
                    ("F + policy, top-3 gain mask", set(S_k3)),
                    ("F + greedy inside mask (k=1)", set(S_f)), ("ILP optimum", opt)):
        lines.append(f"| {name} | {{{', '.join(map(str, sorted(S)))}}} | {len(S)} | "
                     f"{is_total_dominating(gd, set(S))} |")
    lines += ["", "## Step-by-step (free action set, armA + F)", "",
              "| step | S before | legal candidates | top-3 by gain | Q of legal | action | newly satisfied | r | \\|S\\| |",
              "|---|---|---|---|---|---|---|---|---|"]
    S = list(F)
    for i, st in enumerate(steps, 1):
        tg = sorted(st["gain"].items(), key=lambda kv: -kv[1])[:TOPK]
        qs = " ".join(f"{v}:{q:+.3f}" for v, q in sorted(st["q"].items()))
        lines.append(f"| {i} | {{{', '.join(map(str, S))}}} | {{{', '.join(map(str, st['legal']))}}} | "
                     f"{', '.join(f'{v}({g})' for v, g in tg)} | {qs} | **{st['action']}** | "
                     f"{{{', '.join(map(str, st['satisfied_new']))}}} | {st['reward']} | {st['S_size']} |")
        S.append(st["action"])
    lines += ["", "## The reward argument on this instance", "",
              f"Under the paper's gain reward every vertex is satisfied exactly once, so the return is "
              f"always Σgain = n = {gd.n} regardless of how many vertices are chosen — the policy gets no "
              f"signal for size. Under the size reward each of the {len(steps)} RL decisions pays r = −1, "
              f"so the return is exactly −|S_RL| = −{len(steps)}, i.e. the objective itself.", ""]
    with open(os.path.join(a.out, "example_trace.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:26]))
    print(f"... full trace in {a.out}/example_trace.md")


if __name__ == "__main__":
    main()
