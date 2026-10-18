"""Chapter 5 analysis layer: paired statistics + every figure of the chapter.

Nothing here trains or re-evaluates anything; it reads the committed experiment
outputs and produces (a) the paired significance table the chapter quotes and
(b) the figures. Every number in the chapter must be reproducible by:

    python experiments/tdsp/ch5_report.py --stats
    python experiments/tdsp/ch5_report.py --figs

Paired testing is possible only because exp10_raw.csv carries `inst` (the graph
index) — before that column existed the comparisons were between independent means.

Run from the dsp_rl directory.
"""
import argparse, csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results", "tdsp")
OUT_STATS = os.path.join(RES, "ch5")
OUT_FIG = os.path.join(RES, "ch5", "figs")

# ---------------------------------------------------------------- data plumbing
def load(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def by_inst(rows, source, domain):
    """{inst: ratio_vs_greedy} for one source on one domain."""
    out = {}
    for r in rows:
        if r["source"] == source and r["domain"] == domain:
            v = fnum(r["ratio_vs_greedy"])
            if v is not None:
                out[int(r["inst"])] = v
    return out


def boot_ci(x, reps=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    m = rng.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired(a, b):
    """Paired comparison on the intersection of instance ids."""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys], dtype=float)
    lo, hi = boot_ci(d)
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except Exception:                       # scipy missing / degenerate
        p = float("nan")
    return {"n": len(keys), "mean_a": float(np.mean([a[k] for k in keys])),
            "mean_b": float(np.mean([b[k] for k in keys])),
            "diff": float(d.mean()), "lo": lo, "hi": hi, "p": p,
            "wins": int((d < 0).sum()), "ties": int((d == 0).sum()), "losses": int((d > 0).sum())}


# ---------------------------------------------------------------- stats report
COMPARISONS = [
    ("dense", "armA@3+rr", "greedy+rr", "output refinement on the trivial baseline"),
    ("dense", "armA@3+rr", "grasp", "model+boosts vs the strong heuristic"),
    ("dense", "size_v2@3+rr", "grasp", "best dense checkpoint vs GRASP"),
    ("dense", "armA@3+rr", "random@3+rr", "is the Q head worth anything inside the same mask?"),
    ("dense", "armA", "armA+rr", "effect of remove_redundant alone"),
    ("dense", "armA", "armA@3", "effect of the top-3 mask alone"),
    ("dense", "armA@3+rr", "armA", "both boosts together"),
    ("dense", "greedy+rr", "greedy", "remove_redundant on greedy (baseline sanity)"),
    ("dense", "armB@3+rr", "greedy+rr", "forced arm, dense graphs"),
    ("sparse", "armA+F", "greedy+rr", "sparse: model vs post-processed greedy"),
    ("sparse", "armA+F", "grasp", "sparse: model vs GRASP"),
]


def stats_table():
    raw = load("results/tdsp/exp10/exp10_raw.csv")
    cache = {}
    for dom, a, b, _ in COMPARISONS:
        for s in (a, b):
            cache[(dom, s)] = by_inst(raw, s, dom)
    lines, rows = [], []
    for dom, a, b, why in COMPARISONS:
        r = paired(cache[(dom, a)], cache[(dom, b)])
        r.update(domain=dom, a=a, b=b, why=why)
        rows.append(r)
        lines.append(
            f"| {dom} | `{a}` vs `{b}` | {r['n']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | "
            f"{r['diff']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['p']:.3g} | "
            f"{r['wins']}/{r['ties']}/{r['losses']} |")
    head = ["# Chapter 5 — paired tests on exp10",
            "",
            "Metric: `|S| / |greedy_tds|` per graph (lower is better). Differences are `a − b`,",
            "so a negative diff means `a` is smaller. The interval is a 10,000-resample paired",
            "bootstrap of the mean difference; the p-value is a two-sided Wilcoxon signed-rank",
            "test on the same pairs (instances are matched by `inst`).",
            "",
            "| domain | comparison | n | mean a | mean b | mean diff | 95% CI (diff) | p | a<b / = / a>b |",
            "|---|---|---|---|---|---|---|---|---|"]
    os.makedirs(OUT_STATS, exist_ok=True)
    md = os.path.join(OUT_STATS, "paired_tests.md")
    with open(md, "w") as f:
        f.write("\n".join(head + lines) + "\n")
    cols = ["domain", "a", "b", "why", "n", "mean_a", "mean_b", "diff", "lo", "hi", "p",
            "wins", "ties", "losses"]
    with open(os.path.join(OUT_STATS, "paired_tests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 6) if isinstance(r[k], float) else r[k]) for k in cols})
    print("\n".join(head + lines))


# ---------------------------------------------------------------- figures
def figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 9, "figure.dpi": 160, "savefig.bbox": "tight"})
    os.makedirs(OUT_FIG, exist_ok=True)
    made = []

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT_FIG, f"{name}.{ext}"))
        plt.close(fig)
        made.append(name)

    PALETTE = ["#2E5EAA", "#D1495B", "#00798C", "#EDAE49", "#7768AE", "#8B8B8B", "#222222"]

    # --- 5-1 action space: Chen's mask can never produce a total dominating set -----
    rs = load("results/tdsp/exp2/action_space.csv")
    fams = sorted({r["family"] for r in rs})
    ind = [sum(1 for r in rs if r["family"] == f and r["independent"] == "True") for f in fams]
    tot = [sum(1 for r in rs if r["family"] == f and r["total_dominating"] == "True") for f in fams]
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    x = np.arange(len(fams))
    ax.bar(x - 0.18, ind, 0.36, label="independent (what the mask guarantees)", color=PALETTE[0])
    ax.bar(x + 0.18, tot, 0.36, label="total dominating (what TDSP needs)", color=PALETTE[1])
    ax.set_xticks(x, fams)
    ax.set_ylabel("runs")
    ax.set_title(f"Chen's action mask on TDSP: {sum(tot)}/{len(rs)} total dominating sets")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig5_1_action_space")

    # --- 5-2 two-stage construction: the 2|D| bound holds, quality does not -------
    c = load("results/tdsp/exp1/construction.csv")
    srcs = ["opt", "greedy_any", "model"]
    fams = ["GRID", "HEX", "TRI", "ER", "BA"]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    width = 0.26
    for j, s in enumerate(srcs):
        vals, pos = [], []
        for i, f in enumerate(fams):
            v = [fnum(r["|S|/gamma_t"]) for r in c if r["family"] == f and r["source"] == s]
            v = [x for x in v if x is not None]
            if v:
                vals.append(v)
                pos.append(i + (j - 1) * width)
        if vals:
            ax.boxplot(vals, positions=pos, widths=0.22, showfliers=False,
                       medianprops={"color": PALETTE[j]}, boxprops={"color": PALETTE[j]},
                       whiskerprops={"color": PALETTE[j]}, capprops={"color": PALETTE[j]})
    ax.axhline(1.0, color="0.3", lw=0.8, ls=":")
    ax.set_xticks(range(len(fams)), fams)
    ax.set_ylabel(r"$|S| / \gamma_t$")
    ax.set_title("Two-stage construction with D from ILP / greedy / the DSP model")
    for j, s in enumerate(srcs):
        ax.plot([], [], color=PALETTE[j], label=s)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig5_2_construction")

    # --- 5-3 the architecture bug (T7), before vs after ---------------------------
    e4, e5 = load("results/tdsp/exp4/eval.csv"), load("results/tdsp/exp5/eval.csv")
    def mean_ratio(rows, sources, fams_, nmin):
        v = [fnum(r["ratio_vs_greedy"]) for r in rows
             if r["source"] in sources and r["family"] in fams_
             and (fnum(r["n"]) or 0) >= nmin and fnum(r["ratio_vs_greedy"]) is not None]
        return float(np.mean(v)) if v else np.nan
    groups = [("size, ER+BA (n≥80)", "size", "size_v2", ["ER", "BA"], 80),
              ("paper, ER+BA (n≥80)", "paper", "paper_v2", ["ER", "BA"], 80),
              ("HEX", "size", "size_v2", ["HEX"], 0)]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    x = np.arange(len(groups))
    before = [mean_ratio(e4, [g[1]], g[3], g[4]) for g in groups]
    after = [mean_ratio(e5, [g[2]], g[3], g[4]) for g in groups]
    ax.bar(x - 0.18, before, 0.36, label="before: readout $\\sum h_u$", color=PALETTE[1])
    ax.bar(x + 0.18, after, 0.36, label="after: mean aggregation + mean readout", color=PALETTE[2])
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - 0.18, b + 0.05, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.18, a + 0.05, f"{a:.2f}", ha="center", fontsize=8)
    ax.axhline(1.0, color="0.3", lw=0.8, ls=":")
    ax.set_xticks(x, [g[0] for g in groups])
    ax.set_ylabel("|S| / greedy")
    ax.set_title("The scale bug of exp4 and its repair (exp5)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig5_3_architecture")

    # --- 5-4 quality by family: baselines vs model+boosts --------------------------
    raw = load("results/tdsp/exp10/exp10_raw.csv")
    def cell(source, fam, domain):
        v = [fnum(r["ratio_vs_greedy"]) for r in raw
             if r["source"] == source and r["family"] == fam and r["domain"] == domain]
        v = [x for x in v if x is not None]
        return (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v))) if v else np.nan) if v else (np.nan, np.nan)
    dense = ["GRID", "HEX", "TRI", "ER", "BA"]
    sparse = ["TREE", "BA1"]
    series = [("greedy+rr", PALETTE[5]), ("GRASP", PALETTE[6]),
              ("armA@3+rr", PALETTE[0]), ("size_v2@3+rr", PALETTE[2]), ("random@3+rr", PALETTE[3])]
    src_map = {"GRASP": "grasp"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [5, 2]})
    for ax, fams_, dom in ((axes[0], dense, "dense"), (axes[1], sparse, "sparse")):
        x = np.arange(len(fams_))
        for j, (name, col) in enumerate(series):
            src = src_map.get(name, name)
            vals = [cell(src, f, dom)[0] for f in fams_]
            errs = [cell(src, f, dom)[1] for f in fams_]
            ax.bar(x + (j - 2) * 0.16, vals, 0.15, yerr=errs, label=name, color=col,
                   error_kw={"lw": 0.7, "capsize": 1.5})
        ax.axhline(1.0, color="0.3", lw=0.8, ls=":")
        ax.set_xticks(x, fams_)
        ax.set_ylabel("|S| / greedy")
        ax.set_title(dom)
        ax.set_ylim(0.7, 1.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, fontsize=8, loc="lower center")
    fig.suptitle("Quality relative to greedy (mean ± s.e. over graphs)", y=1.02)
    save(fig, "fig5_4_quality_by_family")

    # --- 5-5 top-k sweep ----------------------------------------------------------
    sw = load("results/tdsp/exp10/exp10_sweep.csv")
    models = ["armA", "armB", "size_v2", "paper_v2"]
    dense_fams, sparse_fams = ["GRID", "HEX", "TRI", "ER", "BA"], ["TREE", "BA1"]
    base_k1 = {}
    for dom, fams_ in (("dense", dense_fams), ("sparse", sparse_fams)):
        for rr in (0, 1):
            vals = [cell("greedy" + ("+rr" if rr else ""), f, dom)[0] for f in fams_]
            base_k1[(dom, rr)] = float(np.mean(vals))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    ks = [1, 3, 5, 10]
    for ax, dom in (axes[0], "dense"), (axes[1], "sparse"):
        for j, m in enumerate(models):
            for rr, ls in ((0, "-"), (1, "--")):
                row = [r for r in sw if r["model"] == m and r["domain"] == dom and r["rr"] == str(rr)]
                if not row:
                    continue
                row = row[0]
                # k = 1 is not in the sweep table because it is exactly the greedy baseline
                # (`greedy` for rr=0, `greedy+rr` for rr=1), averaged over that domain.
                ys = [base_k1[(dom, rr)]] + [fnum(row[f"k={k}"]) for k in ks[1:]]
                ax.plot(ks, ys, ls, marker="o", ms=3, color=PALETTE[j], label=m if rr == 0 else None)
        ax.axhline(1.0, color="0.3", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlabel("k (legal set = top-k by gain)")
        ax.set_title(dom + (" (solid: raw, dashed: + rr)" if dom == "dense" else ""))
        ax.set_ylim(0.95, 1.25)
    axes[0].set_ylabel("|S| / greedy")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "fig5_5_topk_sweep")

    # --- 5-6 remove_redundant: dumbbell -------------------------------------------
    summary = load("results/tdsp/exp10/exp10_summary.csv")
    def sval(src, dom):
        v = [fnum(r["|S|/greedy"]) for r in summary if r["source"] == src and r["domain"] == dom]
        return v[0] if v else np.nan
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ys, labels = [], []
    for i, (m, src) in enumerate([("armA", "armA"), ("armB", "armB"),
                                  ("size_v2", "size_v2"), ("paper_v2", "paper_v2")]):
        for dom, off in (("dense", 0.0), ("sparse", -0.18)):
            a, b = sval(src, dom), sval(src + "+rr", dom)
            y = i - off
            ax.plot([a, b], [y, y], color="0.75", lw=2, zorder=1)
            ax.scatter([a], [y], color=PALETTE[1], s=26, zorder=2)
            ax.scatter([b], [y], color=PALETTE[2], s=26, zorder=2)
            ax.text(max(a, b) + 0.006, y, f"{a:.3f} → {b:.3f}", va="center", fontsize=7)
            ys.append(y)
            labels.append(f"{m} ({dom[:2]})")
    ax.set_yticks(ys, labels)
    ax.axvline(1.0, color="0.3", lw=0.8, ls=":")
    ax.set_xlabel("|S| / greedy")
    ax.set_title("remove_redundant: raw (red) → refined (blue)")
    save(fig, "fig5_6_refinement")

    # --- 5-7 convergence ----------------------------------------------------------
    def log(run):
        p = os.path.join(ROOT, "runs", run, "log.csv")
        ep, val = [], []
        with open(p) as f:
            for r in csv.DictReader(f):
                v = fnum(r["val_mean_size"])
                if v is not None:
                    ep.append(int(r["episode"]) / 1000.0)
                    val.append(v)
        return np.array(ep), np.array(val)

    def smooth(v, w=5):
        if len(v) < w:
            return v
        k = np.ones(w) / w
        return np.convolve(v, k, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    for run, lab, col, ls in (("armA_sparse_base", "armA, DDQN", PALETTE[0], "-"),
                              ("armA_sparse_base_d3qn", "armA, D3QN", PALETTE[0], "--"),
                              ("armB_sparse_forced", "armB, DDQN", PALETTE[2], "-"),
                              ("armB_sparse_forced_d3qn", "armB, D3QN", PALETTE[2], "--")):
        ep, val = log(run)
        sm = smooth(val)
        ax.plot(ep[len(ep) - len(sm):], sm, ls, color=col, lw=1.2, label=lab)
    ax.set_xlabel("episodes (×1000)")
    ax.set_ylabel("validation |S|")
    ax.set_title("Q head: DDQN vs D3QN (seed 0)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for base, lab, col in (("armA_sparse_base", "armA (no forcing)", PALETTE[1]),
                           ("armB_sparse_forced", "armB (forced)", PALETTE[2])):
        curves = [log(run) for run in (base, base + "_s1", base + "_s2")]
        ep = curves[0][0]
        sm = np.vstack([smooth(v) for _, v in curves])
        m, lo, hi = sm.mean(0), sm.min(0), sm.max(0)
        ax.plot(ep[len(ep) - len(m):], m, color=col, lw=1.4, label=f"{lab}: mean of 3 seeds")
        ax.fill_between(ep[len(ep) - len(m):], lo, hi, color=col, alpha=0.15, lw=0)
    ax.set_xlabel("episodes (×1000)")
    ax.set_title("Leaf forcing: paired 3-seed convergence")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig5_7_convergence")

    # --- 5-8 RL decisions vs n ----------------------------------------------------
    pe = load("results/tdsp/exp8/phase1_eval.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
    for ax, fam in ((axes[0], "TREE"), (axes[1], "BA1")):
        for j, src in enumerate(["armA", "armA+F", "armB+F"]):
            rows = [r for r in pe if r["family"] == fam and r["source"] == src]
            if not rows:
                continue
            ns = [fnum(r["n"]) for r in rows]
            st = [fnum(r["rl_steps"]) for r in rows]
            ax.plot(ns, st, marker="o", ms=3, color=PALETTE[j], label=src)
        ax.set_title(fam)
        ax.set_xlabel("n")
        ax.set_ylabel("RL decisions per graph")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Leaf forcing removes decisions, not quality (exp8 phase 1)", y=1.02)
    save(fig, "fig5_8_rl_steps")

    # --- 5-9 quality vs time (Pareto) ---------------------------------------------
    keep = ["greedy", "greedy+rr", "grasp", "armA", "armA+rr", "armA@3+rr",
            "armB@3+rr", "size_v2@3+rr", "paper_v2@3+rr", "random@3+rr"]
    pts = []
    for s in keep:
        for dom in ("dense", "sparse"):
            ratio = [fnum(r["|S|/greedy"]) for r in summary if r["source"] == s and r["domain"] == dom]
            tt = [fnum(r["time_s"]) for r in summary if r["source"] == s and r["domain"] == dom]
            if ratio and tt and ratio[0] is not None and tt[0] is not None:
                pts.append((max(tt[0], 1e-4), ratio[0], f"{s} ({dom[0]})"))
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=22, color=PALETTE[0])
    for x, y, lab in pts:
        ax.annotate(lab, (x, y), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("seconds per graph (log)")
    ax.set_ylabel("|S| / greedy")
    ax.axhline(1.0, color="0.3", lw=0.8, ls=":")
    ax.set_title("Quality against inference time")
    save(fig, "fig5_9_time_quality")

    # --- 5-10 the hexagon detail --------------------------------------------------
    hexes = sorted({r["n"] for r in raw if r["family"] == "HEX"}, key=lambda v: int(v))
    srcs = [("greedy", PALETTE[5]), ("greedy+rr", PALETTE[2]), ("grasp", PALETTE[6]),
            ("armA@3+rr", PALETTE[0]), ("size_v2@3+rr", PALETTE[1]), ("random@3+rr", PALETTE[3])]
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    x = np.arange(len(hexes))
    for j, (s, col) in enumerate(srcs):
        vals = []
        for n in hexes:
            v = [fnum(r["ratio_vs_greedy"]) for r in raw
                 if r["family"] == "HEX" and r["n"] == n and r["source"] == s]
            vals.append(float(np.mean(v)) if v else np.nan)
        ax.bar(x + (j - 2.5) * 0.14, vals, 0.13, label=s, color=col)
    ax.axhline(1.0, color="0.3", lw=0.8, ls=":")
    ax.set_xticks(x, [f"HEX {n}" for n in hexes])
    ax.set_ylabel("|S| / greedy")
    ax.set_title("Hexagonal lattice: the one family where the policy wins")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    save(fig, "fig5_10_hex")

    print(f"wrote {len(made)} figures to {OUT_FIG}")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--figs", action="store_true")
    a = ap.parse_args()
    if not (a.stats or a.figs):
        a.stats = a.figs = True
    if a.stats:
        stats_table()
    if a.figs:
        figs()


if __name__ == "__main__":
    main()
