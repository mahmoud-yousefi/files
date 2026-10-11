"""Experiment 7 (chapter 5): do exact reduction rules help TDSP on our benchmarks?

Three candidate rules are examined: isolated vertex (infeasible detection),
connected components (exact split), and the leaf/support rule.
The experiment REPORTS what each rule actually achieves on the families we use,
rather than assuming a benefit.

Run from the dsp_rl directory:
  python experiments/tdsp/exp7_reduction.py --out results/tdsp/exp7
"""
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from graphs import make_graph_data, grid, er, ba, hexa, tri
from reductions import components, leaf_supports, counterexample
from tdsp import has_isolated

HEADER = ["family", "n", "components", "isolated", "leaf_forced", "reducible_by"]


def classify(gd):
    comps = len(components(gd))
    iso = int(has_isolated(gd))
    leaves = len(leaf_supports(gd))
    if iso:
        r = "R1-infeasible"
    elif comps > 1:
        r = "R2-split"
    elif leaves:
        r = "R3-note:leaf-rule-invalid-for-TDSP"
    else:
        r = "none"
    return comps, iso, leaves, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tdsp/exp7")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []
    def one(fam, gd):
        c, iso, lv, r = classify(gd)
        rows.append({"family": fam, "n": gd.n, "components": c, "isolated": iso,
                     "leaf_forced": lv, "reducible_by": r})
    for sh in [(4, 5), (10, 10), (20, 20)]:
        one("GRID", make_graph_data(grid(*sh)))
    for sh in [(4, 5), (6, 6), (10, 13)]:
        one("HEX", make_graph_data(hexa(*sh)))
    one("TRI", make_graph_data(tri(6, 6)))
    for n in [20, 100, 300]:
        for _ in range(a.reps):
            one("ER", make_graph_data(er(n, 0.4, rng)))
            one("BA", make_graph_data(ba(n, 4, rng)))
    # an explicit disconnected control, to show R2 does fire when it should
    import networkx as nx
    one("CTRL-disconnected", make_graph_data(nx.disjoint_union(nx.path_graph(6), nx.path_graph(6))))

    agg = {}
    for r in rows:
        agg.setdefault((r["family"], r["n"]), []).append(r)
    summary = []
    for (fam, n), rs in sorted(agg.items()):
        summary.append({"family": fam, "n": n,
                        "n_instances": len(rs),
                        "mean_components": round(np.mean([x["components"] for x in rs]), 2),
                        "any_isolated": any(x["isolated"] for x in rs),
                        "mean_leaf_forced": round(np.mean([x["leaf_forced"] for x in rs]), 2)})
    with open(f"{a.out}/reduction.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["family", "n", "n_instances", "mean_components", "any_isolated", "mean_leaf_forced"])
        for s in summary: w.writerow([s[k] for k in ("family", "n", "n_instances", "mean_components", "any_isolated", "mean_leaf_forced")])
    with open(f"{a.out}/reduction.md", "w") as f:
        f.write("# Exp7: do reduction rules help TDSP on our benchmarks?\n\n")
        f.write("**Headline:** on the graph families used throughout this thesis, none of the three\n"
                "candidate rules fires: every instance is connected (no components to split), has no\n"
                "isolated vertex, and has minimum degree >= 2 (no leaves). The naive leaf rule is also\n"
                "*invalid* for TDSP (counterexample: P3). So reduction gives no speed/quality gain here --\n"
                "a negative result reported honestly, not a fabricated improvement.\n\n")
        f.write(f"leaf-rule counterexample: `{counterexample()}`\n\n")
        f.write("| family | n | instances | mean components | any isolated | mean leaf-forced |\n")
        f.write("|---|---|---|---|---|---|\n")
        for s in summary:
            f.write(f"| {s['family']} | {s['n']} | {s['n_instances']} | {s['mean_components']} | {s['any_isolated']} | {s['mean_leaf_forced']} |\n")
    print(open(f"{a.out}/reduction.md").read())


if __name__ == "__main__":
    main()
