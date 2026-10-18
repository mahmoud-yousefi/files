# Chapter 5 — paired tests on exp10

Metric: `|S| / |greedy_tds|` per graph (lower is better). Differences are `a − b`,
so a negative diff means `a` is smaller. The interval is a 10,000-resample paired
bootstrap of the mean difference; the p-value is a two-sided Wilcoxon signed-rank
test on the same pairs (instances are matched by `inst`).

| domain | comparison | n | mean a | mean b | mean diff | 95% CI (diff) | p | a<b / = / a>b |
|---|---|---|---|---|---|---|---|---|
| dense | `armA@3+rr` vs `greedy+rr` | 85 | 0.990 | 0.995 | -0.0049 | [-0.0191, +0.0101] | 0.485 | 10/69/6 |
| dense | `armA@3+rr` vs `grasp` | 85 | 0.990 | 0.987 | +0.0026 | [-0.0171, +0.0228] | 0.851 | 12/52/21 |
| dense | `size_v2@3+rr` vs `grasp` | 85 | 0.996 | 0.987 | +0.0092 | [-0.0117, +0.0307] | 0.592 | 12/49/24 |
| dense | `armA@3+rr` vs `random@3+rr` | 85 | 0.990 | 1.098 | -0.1085 | [-0.1504, -0.0713] | 1.52e-07 | 38/42/5 |
| dense | `armA` vs `armA+rr` | 85 | 1.205 | 1.059 | +0.1452 | [+0.1095, +0.1833] | 2.35e-09 | 0/38/47 |
| dense | `armA` vs `armA@3` | 85 | 1.205 | 1.009 | +0.1955 | [+0.1546, +0.2393] | 1.98e-11 | 1/25/59 |
| dense | `armA@3+rr` vs `armA` | 85 | 0.990 | 1.205 | -0.2148 | [-0.2573, -0.1746] | 2.84e-12 | 64/20/1 |
| dense | `greedy+rr` vs `greedy` | 88 | 0.995 | 1.000 | -0.0051 | [-0.0098, -0.0016] | 0.0117 | 8/80/0 |
| dense | `armB@3+rr` vs `greedy+rr` | 85 | 1.017 | 0.995 | +0.0218 | [+0.0039, +0.0410] | 0.0351 | 9/57/19 |
| sparse | `armA+F` vs `greedy+rr` | 80 | 0.986 | 0.992 | -0.0058 | [-0.0091, -0.0031] | 0.000426 | 16/64/0 |
| sparse | `armA+F` vs `grasp` | 80 | 0.986 | 0.986 | +0.0001 | [+0.0000, +0.0004] | 0.317 | 0/79/1 |
