"""Experiment 2 (chapter 5): is Chen's action mask even usable for TDSP?

Chen's mask only allows selecting vertices with vf==0 (undominated). Any such run
ends with an independent set (DECISIONS D1). An independent set is never
total-dominating, because no member of it has a neighbour inside the set. So the
paper's action space cannot produce a TDSP solution at all -- for any graph.
This experiment verifies that claim empirically for a random-masked policy, the
masked greedy policy, and the trained model.

Run from the dsp_rl directory:
  python experiments/tdsp/exp2_action_space.py --model runs/er20_s0/model.pt
"""
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from graphs import make_graph_data, grid, er, ba, hexa, tri
from baselines import greedy_ds_masked, is_independent, is_total_dominating
from env import DSPEnv
from tdsp import independence_witness, masked_policy_set, has_isolated
from evaluate import GRID_SHAPES, HEX_SHAPES

HEADER = ["family", "n", "source", "|D|", "independent", "total_dominating", "witness"]


def random_masked_set(gd, rng):
    """Replay the mask with uniformly random actions (the epsilon=1 policy)."""
    env, vf, done = DSPEnv(), None, False
    vf = env.reset(gd)
    while not done:
        vf, _, done = env.step(int(rng.choice(env.candidates())))
    return set(env.selected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/er20_s0/model.pt")
    ap.add_argument("--out", default="results/tdsp/exp2")
    ap.add_argument("--reps", type=int, default=20, help="random graphs per ER/BA size")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nf", type=int, default=32); ap.add_argument("--agg", default="sum")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    agent = None
    if a.model and os.path.exists(a.model):
        from agent import DDQNAgent
        agent = DDQNAgent(a.nf, 3, a.agg, device=a.device); agent.load(a.model)
        print(f"loaded model: {a.model}")

    rows = []
    def check(family, gd, source, D):
        wit = independence_witness(gd, D)
        rows.append({"family": family, "n": gd.n, "source": source, "|D|": len(D),
                     "independent": bool(is_independent(gd, D)),
                     "total_dominating": bool(is_total_dominating(gd, D)),
                     "witness": ("" if wit is None else wit)})

    rng = np.random.default_rng(7)
    # deterministic lattices (one instance each) + random ER/BA
    inst = [("GRID", make_graph_data(grid(*sh))) for sh in GRID_SHAPES.values()]
    inst += [("HEX", make_graph_data(hexa(*sh))) for sh in HEX_SHAPES.values()]
    inst += [("TRI", make_graph_data(tri(6, 6)))]
    for fam, gd in inst:
        check(fam, gd, "random_masked", random_masked_set(gd, rng))
        check(fam, gd, "greedy_masked", set(greedy_ds_masked(gd)))
        if agent is not None:
            check(fam, gd, "model_masked", masked_policy_set(agent, gd))
    for fam, gen in [("ER", lambda: make_graph_data(er(40, 0.4, rng))),
                     ("BA", lambda: make_graph_data(ba(40, 4, rng)))]:
        for _ in range(a.reps):
            gd = gen()
            if has_isolated(gd):
                continue
            check(fam, gd, "random_masked", random_masked_set(gd, rng))
            check(fam, gd, "greedy_masked", set(greedy_ds_masked(gd)))
            if agent is not None:
                check(fam, gd, "model_masked", masked_policy_set(agent, gd))

    n = len(rows)
    ind = sum(r["independent"] for r in rows)
    tot = sum(r["total_dominating"] for r in rows)
    with open(f"{a.out}/action_space.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER); w.writeheader(); w.writerows(rows)
    with open(f"{a.out}/action_space.md", "w") as f:
        f.write("# Exp2: can Chen's action mask solve TDSP?\n\n")
        f.write(f"model={a.model}  graphs={n}\n\n")
        f.write(f"**Independent outputs: {ind}/{n}.  Total-dominating outputs: {tot}/{n}.**\n\n")
        f.write("Every masked run is independent by construction and therefore never total-dominating "
                "(the `witness` column names a member of D with no neighbour in D).\n\n")
        f.write("| " + " | ".join(HEADER) + " |\n|" + "---|" * len(HEADER) + "\n")
        for r in rows: f.write("| " + " | ".join(str(r[c]) for c in HEADER) + " |\n")
    print(f"independent={ind}/{n}  total_dominating={tot}/{n}")
    print(open(f"{a.out}/action_space.md").read()[:700])


if __name__ == "__main__":
    main()
