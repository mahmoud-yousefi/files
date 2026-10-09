"""Experiment 1 (chapter 5): two-stage DSP -> TDSP construction and the |S| <= 2|D| bound.

For every graph we take a dominating set D from several sources (optimal ILP, the trained
paper-style model, greedy), run stage 2 (tdsp.cover_D_greedy), assert the result is
total-dominating, and tabulate |D|, |A|, |S|=|D|+|A| against 2|D| and the optimum gamma_t.

Run from the dsp_rl directory:
  python experiments/tdsp/exp1_construction.py --model runs/er20_s0/model.pt
"""
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from graphs import make_graph_data, grid, er, ba, hexa
from baselines import greedy_ds, greedy_ds_masked, exact, exact_solution, is_total_dominating
from tdsp import cover_D_greedy, two_stage_tdsp, greedy_tds, model_dominating_set, has_isolated
from evaluate import GRID_SHAPES, HEX_SHAPES

SOURCES = ["opt", "model", "greedy_any", "greedy_masked"]
HEADER = ["family", "n", "shape", "source", "|D|", "|A|", "|S|", "2|D|", "slack", "|S|/|D|",
          "gamma_t", "greedyTDS", "|S|/gamma_t", "reps"]


def instances(nmax_er):
    """Deterministic grids/lattices (one instance each) + random ER/BA at several sizes."""
    for n, sh in GRID_SHAPES.items():
        yield "GRID", f"grid{n}", make_graph_data(grid(*sh)), sh, True
    for n, sh in HEX_SHAPES.items():
        yield "HEX", f"hex{n}", make_graph_data(hexa(*sh)), sh, True
    rng = np.random.default_rng(123)
    for n in nmax_er:
        yield "ER", f"ER{n}", lambda: make_graph_data(er(n, 0.4, rng)), "", n <= 100
        yield "BA", f"BA{n}", lambda: make_graph_data(ba(n, 4, rng)), "", n <= 100


def one_row(family, n, shape, src, gd, D, gamma_t, gt_proven, gtds):
    S, A = two_stage_tdsp(gd, D)                      # asserts total domination + bound
    return {
        "family": family, "n": gd.n, "shape": shape, "source": src,
        "|D|": len(D), "|A|": len(A), "|S|": len(S), "2|D|": 2 * len(D),
        "slack": 2 * len(D) - len(S), "|S|/|D|": len(S) / len(D),
        "gamma_t": gamma_t if gt_proven else "",
        "greedyTDS": len(gtds),
        "|S|/gamma_t": (len(S) / gamma_t) if gt_proven else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/er20_s0/model.pt")
    ap.add_argument("--out", default="results/tdsp/exp1")
    ap.add_argument("--reps", type=int, default=5, help="random graphs per ER/BA size")
    ap.add_argument("--nmax", default="20,40,100,200", help="ER/BA sizes to sweep")
    ap.add_argument("--ilp_time", type=int, default=60)
    ap.add_argument("--nf", type=int, default=32); ap.add_argument("--agg", default="sum")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nmax = [int(x) for x in a.nmax.split(",")]

    agent = None
    if a.model and os.path.exists(a.model):
        from agent import DDQNAgent
        agent = DDQNAgent(a.nf, 3, a.agg, device=a.device); agent.load(a.model)
        print(f"loaded model: {a.model}")
    else:
        print("no model loaded; 'model' source skipped")

    rows, skipped_iso = [], 0
    for family, name, gen, shape, do_opt in instances(nmax):
        reps = a.reps if family in ("ER", "BA") else 1
        for _ in range(reps):
            gd = gen() if callable(gen) else gen
            if has_isolated(gd):
                skipped_iso += 1; continue
            gt, gt_ok = (exact(gd, "tot", a.ilp_time) if do_opt else (None, False))
            gtds = greedy_tds(gd)
            Ds = {"greedy_any": greedy_ds(gd), "greedy_masked": greedy_ds_masked(gd)}
            if do_opt:
                _, Dset, ok = exact_solution(gd, "dom", a.ilp_time)
                if ok: Ds["opt"] = Dset
            if agent is not None:
                Ds["model"] = model_dominating_set(agent, gd)
            for src in SOURCES:
                if src in Ds:
                    rows.append(one_row(family, name, shape, src, gd, Ds[src], gt, gt_ok, gtds))
        print(f"  {family} {name}: done", flush=True)

    # aggregate repetitions (ER/BA) by mean over numeric columns
    keys = ["family", "n", "shape", "source"]
    group = {}
    for r in rows:
        group.setdefault(tuple(r[k] for k in keys), []).append(r)
    agg = []
    for key, rs in group.items():
        out = dict(zip(keys, key)); out["reps"] = len(rs)
        for c in HEADER[4:-1]:
            vals = [r[c] for r in rs if r[c] != ""]
            out[c] = round(float(np.mean(vals)), 3) if vals else ""
        agg.append(out)
    agg.sort(key=lambda r: (r["family"], r["n"], r["source"]))

    with open(f"{a.out}/construction.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER); w.writeheader(); w.writerows(agg)
    with open(f"{a.out}/construction.md", "w") as f:
        f.write("# TDSP two-stage construction: |S| <= 2|D|\n")
        f.write(f"model={a.model}  reps(ER/BA)={a.reps}  skipped isolated-vertex graphs={skipped_iso}\n\n")
        f.write("| " + " | ".join(HEADER) + " |\n|" + "---|" * len(HEADER) + "\n")
        for r in agg: f.write("| " + " | ".join(str(r[c]) for c in HEADER) + " |\n")
    print(open(f"{a.out}/construction.md").read())


if __name__ == "__main__":
    main()
