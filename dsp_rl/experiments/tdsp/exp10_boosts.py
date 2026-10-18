"""Experiment 10 (chapter 5): two zero-training boosts on top of the trained policy.

Exp6 was honest -- the RL policy loses to GRASP -- and exp9 (D3QN) showed that more
training capacity is not where the remaining gains are: the dueling head changed the
final |S| not at all (9.5/9.7/9.8 vs 9.5/9.7/9.8). So exp10 asks the other question:

    what can we get for free from the checkpoints we already have?

Branch A -- output refinement (`remove_redundant`, exp6's set-cover heuristic)
    TDSP is exactly set cover, so the classical post-processing applies to ANY total
    dominating set: drop a chosen vertex whose every neighbour is still covered by
    another chosen vertex. Soundness is not an assumption -- the check *is* "still
    total-dominating" -- so size can only go down and the result stays feasible.
    Cost O(|S| * m); strictly dominating-or-equal, never worse.

Branch B -- top-k action masking (gain sweep)
    At each step keep only the k candidates with the largest heuristic gain (number of
    still-unsatisfied neighbours) and let the policy pick among them by Q. The two ends
    of the sweep are known points: k = 1 is exactly `greedy_tds`, k = n is the free
    policy. The curve in between isolates *where* the policy beats greedy: if it peaks
    at small k, light greedy knowledge in the action space helps; if it degrades
    monotonically, the policy's value lives precisely in the steps where it ignores the
    highest-gain vertex.

Both branches compose (mask first, then refine), so the chapter table can carry one row
per family for "model + mask@k + rr".

Verification built in (asserted, not printed):
  * every emitted set is total-dominating                  -> feasibility 100%
  * remove_redundant never increases the size              -> boost cannot backfire
  * k = 1 without forcing reproduces `greedy_tds` exactly   -> the mask is implemented right

Run from the dsp_rl directory:
  python experiments/tdsp/exp10_boosts.py --out results/tdsp/exp10
"""
import argparse, csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from graphs import make_graph_data, grid, er, ba, hexa, tri, tree, ba1
from baselines import exact
from tdsp_env import TDSPEnv
from tdsp import greedy_tds, has_isolated
from heuristics import grasp_tds, is_total, remove_redundant
from reductions import leaf_supports
from agent import agent_for_model

DEFAULT_MODELS = [
    ("armA", "runs/armA_sparse_base/model.pt"),          # exp8 arm A (trained without forcing)
    ("armB", "runs/armB_sparse_forced/model.pt"),        # exp8 arm B (trained with forcing)
    ("size_v2", "runs/tdsp_er20_s0_size_v2/model.pt"),   # exp5 dense-trained, reward = -1
    ("paper_v2", "runs/tdsp_er20_s0_paper_v2/model.pt"), # exp5 dense-trained, Eq.(5) reward
]
SPARSE = {"TREE": tree, "BA1": ba1}
RAW_HEADER = ["inst", "domain", "family", "n", "model", "k", "rr", "source", "|S|", "gamma_t",
              "|S|/gamma_t", "ratio_vs_greedy", "rl_steps", "rr_dropped", "time_s"]

# `inst` is the 1-based index of the graph in this run. It is what makes the raw table
# pairable: every source on the same graph shares the value, so the chapter can quote
# paired differences and bootstrap CIs instead of comparing two independent means.
INST = [0]


# --------------------------------------------------------------------------- instances
def dense_instances(reps, rng):
    out = []
    for sh in [(4, 5), (6, 6), (7, 9), (10, 10)]:
        out.append(("GRID", make_graph_data(grid(*sh))))
    for sh in [(4, 5), (6, 6), (8, 10)]:
        out.append(("HEX", make_graph_data(hexa(*sh))))
    out.append(("TRI", make_graph_data(tri(6, 6))))
    for n in [20, 40, 100, 300]:
        for _ in range(reps):
            out.append(("ER", make_graph_data(er(n, 0.4, rng))))
            out.append(("BA", make_graph_data(ba(n, 4, rng))))
    return [("dense", f, gd.n, gd) for f, gd in out]


def sparse_instances(reps, rng):
    return [("sparse", fam, n, make_graph_data(gen(n, rng)))
            for fam, gen in SPARSE.items() for n in (20, 40, 100, 200) for _ in range(reps)]


# --------------------------------------------------------------------------- rollout
def gains(env):
    """Heuristic gain of every vertex: how many still-unsatisfied vertices it would satisfy."""
    g = np.zeros(env.gd.n, dtype=np.int64)
    for v in range(env.gd.n):
        g[v] = int((env.vf[env.gd.nbrs[v]] == 0).sum())
    return g


def rollout(agent, env, gd, forced=(), k=None, rng=None):
    """Greedy Q roll-out to a terminal set. With `k`, only the k highest-gain candidates
    are legal (ties by index) and Q breaks the tie among them; `agent=None` + `k=1` is
    therefore plain greedy_dts and needs no network call. `rng` replaces Q by a uniform
    pick among the same k -- the control that isolates the network's contribution."""
    t0 = time.perf_counter()
    s = env.reset(gd, forced=forced); done = getattr(env, "done", False)
    while not done:
        c = env.candidates()
        if k is not None and len(c) > k:
            g = gains(env)
            order = c[np.lexsort((c, -g[c]))][:k]        # primary: -gain, secondary: index
        else:
            order = c
        if len(order) == 1:
            v = int(order[0])
        elif rng is not None:
            v = int(rng.choice(order))
        else:
            q = agent.q_values(gd, s); v = int(order[np.argmax(q[order])])
        s, _, done = env.step(v)
    dt = time.perf_counter() - t0
    assert is_total(gd, env.selected), "rollout produced a non-total-dominating set"
    return set(env.selected), dt, len(env.selected) - len(forced)


# --------------------------------------------------------------------------- bookkeeping
def raw_row(domain, family, gd, model, k, rr, source, S, gt, greedy, steps, rr_dropped, dt):
    return {"inst": INST[0], "domain": domain, "family": family, "n": gd.n, "model": model,
            "k": "inf" if k is None else str(k), "rr": int(bool(rr)), "source": source,
            "|S|": len(S), "gamma_t": gt if gt else "",
            "|S|/gamma_t": round(len(S) / gt, 3) if gt else "",
            "ratio_vs_greedy": round(len(S) / len(greedy), 4),
            "rl_steps": steps, "rr_dropped": rr_dropped, "time_s": round(dt, 6)}


def mean_of(rows, key):
    vals = [r[key] for r in rows if r[key] != ""]
    return round(float(np.mean(vals)), 3) if vals else ""


def group(rows, keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def aggregate_raw(rows):
    keys = ("domain", "family", "n", "source")
    agg = []
    for key, rs in group(rows, keys).items():
        row = dict(zip(keys, key))
        for c in ["|S|", "gamma_t", "|S|/gamma_t", "ratio_vs_greedy", "rl_steps", "rr_dropped", "time_s"]:
            row[c] = mean_of(rs, c)
        agg.append(row)
    return sorted(agg, key=lambda r: (r["domain"], r["family"], r["n"], r["source"]))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def write_md_table(f, header, rows):
    f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
    for r in rows:
        f.write("| " + " | ".join(str(r[c]) for c in header) + " |\n")


# --------------------------------------------------------------------------- summary
def source_table(rows):
    keys = ("domain", "source")
    out = []
    for key, rs in sorted(group(rows, keys).items()):
        out.append({
            "domain": key[0], "source": key[1], "instances": len(rs),
            "|S|/greedy": mean_of(rs, "ratio_vs_greedy"),
            "|S|/gamma_t": mean_of(rs, "|S|/gamma_t"),
            "pct_improved": round(100 * sum(1 for r in rs if r["rr_dropped"] > 0) / len(rs), 1),
            "mean_rr_dropped": mean_of(rs, "rr_dropped"),
            "time_s": mean_of(rs, "time_s"),
        })
    return out


def sweep_table(rows, models, ks):
    """model x domain grid: mean |S|/greedy for the free policy and each k, raw and +rr."""
    grid = {}
    for r in rows:
        if r["model"] == "":
            continue
        cell = (r["model"], r["domain"], int(r["rr"]), r["k"])
        grid.setdefault(cell, []).append(r["ratio_vs_greedy"])
    cols = ["free"] + [f"k={k}" for k in ks]
    out = {int(rr): [] for rr in (0, 1)}
    for model in models:
        for domain in ("dense", "sparse"):
            for rr in (0, 1):
                row = {"model": model, "domain": domain}
                for c, kname in zip(cols, ["inf"] + [str(k) for k in ks]):
                    v = grid.get((model, domain, rr, kname))
                    row[c] = round(float(np.mean(v)), 3) if v else ""
                out[rr].append(row)
    return cols, out


def headline(rows):
    """Best config per checkpoint, next to the three baselines that decide the story:
    greedy (mask endpoint), greedy+rr (post-processing on a heuristic) and GRASP."""
    out = []
    for domain in ("dense", "sparse"):
        dom = [r for r in rows if r["domain"] == domain]
        for base in ("greedy", "greedy+rr", "grasp"):
            rs = [r for r in dom if r["source"] == base]
            out.append({"domain": domain, "model": "--", "config": base,
                        "|S|/greedy": mean_of(rs, "ratio_vs_greedy"),
                        "|S|/gamma_t": mean_of(rs, "|S|/gamma_t"), "time_s": mean_of(rs, "time_s")})
        for model in sorted({r["model"] for r in dom if r["model"]}):
            cells = {}
            for r in dom:
                if r["model"] == model:
                    cells.setdefault((r["rr"], r["k"]), []).append(r)
            best = min(cells.items(), key=lambda kv: np.mean([x["ratio_vs_greedy"] for x in kv[1]]))
            (rr, k), rs = best
            cfg = ("@k=" + k if k != "inf" else "free") + (" + rr" if rr else "")
            out.append({"domain": domain, "model": model, "config": cfg,
                        "|S|/greedy": mean_of(rs, "ratio_vs_greedy"),
                        "|S|/gamma_t": mean_of(rs, "|S|/gamma_t"), "time_s": mean_of(rs, "time_s")})
        for base in [k for k in sorted({r["source"] for r in dom}) if k.startswith("random@") and k.endswith("+rr")]:
            rs = [r for r in dom if r["source"] == base]
            out.append({"domain": domain, "model": "--", "config": base,
                        "|S|/greedy": mean_of(rs, "ratio_vs_greedy"),
                        "|S|/gamma_t": mean_of(rs, "|S|/gamma_t"), "time_s": mean_of(rs, "time_s")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="name=path; repeatable")
    ap.add_argument("--out", default="results/tdsp/exp10")
    ap.add_argument("--reps", type=int, default=10, help="instances per size for ER/BA/TREE/BA1")
    ap.add_argument("--ks", default="3,5,10", help="comma list of top-k mask widths")
    ap.add_argument("--grasp_iters", type=int, default=40)
    ap.add_argument("--ilp_time", type=int, default=30)
    ap.add_argument("--ilp_max", type=int, default=120, help="solve the ILP only up to this n")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ks = [int(x) for x in a.ks.split(",")]

    specs = a.model or [f"{n}={p}" for n, p in DEFAULT_MODELS]
    agents = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint for {name} not found: {path}")
        agents[name] = agent_for_model(path, device=a.device)
        print(f"loaded {name}: {path}", flush=True)

    env = TDSPEnv(); rng = np.random.default_rng(2024)
    rows, checks = [], {"instances": 0, "sets_checked": 0, "k1_greedy_mismatch": 0,
                        "rr_increase": 0, "infeasible": 0}

    def refine(gd, S):
        t = time.perf_counter(); R = set(remove_redundant(gd, S)); tr = time.perf_counter() - t
        assert len(R) <= len(S), "remove_redundant increased the size"
        assert is_total(gd, R), "remove_redundant broke total domination"
        return R, tr

    def emit(domain, family, gd, model, k, rr, source, S, gt, greedy, steps, dropped, dt):
        rows.append(raw_row(domain, family, gd, model, k, rr, source, S, gt, greedy, steps, dropped, dt))
        checks["sets_checked"] += 1
        if not is_total(gd, S):
            checks["infeasible"] += 1
            raise AssertionError(f"infeasible set from {source} on {family} n={gd.n}")
        if rr and dropped < 0:
            checks["rr_increase"] += 1
            raise AssertionError(f"remove_redundant increased the size on {family} n={gd.n}")

    instances = dense_instances(a.reps, rng) + sparse_instances(a.reps, rng)
    for i, (domain, family, n, gd) in enumerate(instances, 1):
        if has_isolated(gd):
            continue
        checks["instances"] += 1
        INST[0] = i
        F = leaf_supports(gd); tag = "+F" if F else ""
        gt = exact(gd, "tot", a.ilp_time)[0] if gd.n <= a.ilp_max else None
        greedy = greedy_tds(gd)

        # ---- the mask implementation itself: k=1 without forcing == greedy_tds --------
        S1, _, _ = rollout(None, env, gd, (), k=1)
        if S1 != greedy:
            checks["k1_greedy_mismatch"] += 1
            raise AssertionError(f"k=1 mask != greedy_tds on {family} n={gd.n}")

        emit(domain, family, gd, "", None, 0, "greedy", greedy, gt, greedy, len(greedy), 0, 0.0)
        Rg, trg = refine(gd, greedy)
        emit(domain, family, gd, "", None, 1, "greedy+rr", Rg, gt, greedy, len(greedy),
             len(greedy) - len(Rg), trg)
        t = time.perf_counter(); G = grasp_tds(gd, a.grasp_iters, np.random.default_rng(gd.n))
        tg = time.perf_counter() - t
        emit(domain, family, gd, "", None, 0, "grasp", set(G), gt, greedy, len(G), 0, tg)

        for name, ag in agents.items():
            base = name + tag
            S0, dt0, steps0 = rollout(ag, env, gd, F)
            R0, tr0 = refine(gd, S0)
            emit(domain, family, gd, name, None, 0, base, S0, gt, greedy, steps0, 0, dt0)
            emit(domain, family, gd, name, None, 1, base + "+rr", R0, gt, greedy, steps0,
                 len(S0) - len(R0), dt0 + tr0)
            for k in ks:
                Sk, dtk, stepsk = rollout(ag, env, gd, F, k=k)
                Rk, trk = refine(gd, Sk)
                emit(domain, family, gd, name, k, 0, f"{base}@{k}", Sk, gt, greedy, stepsk, 0, dtk)
                emit(domain, family, gd, name, k, 1, f"{base}@{k}+rr", Rk, gt, greedy, stepsk,
                     len(Sk) - len(Rk), dtk + trk)
        # ---- control: same legal set (top-k by gain), but a coin flip instead of Q ------
        # Emitted once per instance, not once per checkpoint: the roll-out never touches the
        # network (rng only), so inside the checkpoint loop it produced four identical rows
        # per graph and inflated the instance counts of `random@k` in every table.
        for k in ks:
            Cr, dtc, stepsc = rollout(None, env, gd, F, k=k, rng=np.random.default_rng(7000 + gd.n))
            Rc, trc = refine(gd, Cr)
            emit(domain, family, gd, "", k, 0, f"random@{k}", Cr, gt, greedy, stepsc, 0, dtc)
            emit(domain, family, gd, "", k, 1, f"random@{k}+rr", Rc, gt, greedy, stepsc,
                 len(Cr) - len(Rc), dtc + trc)
        if i % 20 == 0:
            print(f"  {i}/{len(instances)} instances", flush=True)

    # ------------------------------------------------------------------ outputs
    write_csv(f"{a.out}/exp10_raw.csv", RAW_HEADER, rows)
    write_csv(f"{a.out}/exp10_boosts.csv",
              ["domain", "family", "n", "source", "|S|", "gamma_t", "|S|/gamma_t",
               "ratio_vs_greedy", "rl_steps", "rr_dropped", "time_s"],
              aggregate_raw(rows))

    src = source_table(rows)
    write_csv(f"{a.out}/exp10_summary.csv", list(src[0]), src)
    cols, sweep = sweep_table(rows, list(agents), ks)
    write_csv(f"{a.out}/exp10_sweep.csv", ["model", "domain"] + cols + ["rr"],
              [dict(r, rr=0) for r in sweep[0]] + [dict(r, rr=1) for r in sweep[1]])

    with open(f"{a.out}/exp10_summary.md", "w") as f:
        head = headline(rows)
        f.write("# Exp10 -- zero-training boosts (output refinement + top-k gain mask)\n\n")
        f.write("All four checkpoints are evaluated with the forced leaf seed F where it is "
                "non-empty (exp8: armB is only valid in that mode). ILP optimum only for "
                f"n <= {a.ilp_max}.\n\n## Checks\n\n")
        f.write(f"- instances evaluated: **{checks['instances']}**\n"
                f"- sets emitted and re-verified as total-dominating: **{checks['sets_checked']}**, "
                f"infeasible: **{checks['infeasible']}**\n"
                f"- `k=1` mask vs `greedy_tds` mismatches: **{checks['k1_greedy_mismatch']}**\n"
                f"- cases where `remove_redundant` increased the size: **{checks['rr_increase']}**\n\n")
        f.write("## Headline: the best configuration each checkpoint can reach\n\n")
        write_md_table(f, ["domain", "model", "config", "|S|/greedy", "|S|/gamma_t", "time_s"], head)
        f.write("\nWhat the table has to be read for: `greedy+rr` is the post-processing boost "
                "applied to the trivial baseline, so a checkpoint only earns its place by "
                "beating that row; `random@k+rr` is a coin flip inside the same top-k gain "
                "set, so any remaining gap to it is the network's contribution.\n\n")
        f.write("## Per source (mean over instances; ratios are per-instance |S|/|greedy|)\n\n")
        write_md_table(f, ["domain", "source", "instances", "|S|/greedy", "|S|/gamma_t",
                           "pct_improved", "mean_rr_dropped", "time_s"], src)
        f.write("\n## Top-k sweep, raw output (mean |S|/greedy)\n\n")
        write_md_table(f, ["model", "domain"] + cols, sweep[0])
        f.write("\n## Top-k sweep, after remove_redundant (mean |S|/greedy)\n\n")
        write_md_table(f, ["model", "domain"] + cols, sweep[1])
    print(open(f"{a.out}/exp10_summary.md").read())


if __name__ == "__main__":
    main()
