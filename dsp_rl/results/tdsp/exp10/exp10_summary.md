# Exp10 -- zero-training boosts (output refinement + top-k gain mask)

All four checkpoints are evaluated with the forced leaf seed F where it is non-empty (exp8: armB is only valid in that mode). ILP optimum only for n <= 120.

## Checks

- instances evaluated: **168**
- sets emitted and re-verified as total-dominating: **9912**, infeasible: **0**
- `k=1` mask vs `greedy_tds` mismatches: **0**
- cases where `remove_redundant` increased the size: **0**

## Headline: the best configuration each checkpoint can reach

| domain | model | config | |S|/greedy | |S|/gamma_t | time_s |
|---|---|---|---|---|---|
| dense | -- | greedy | 1.0 | 1.065 | 0.0 |
| dense | -- | greedy+rr | 0.995 | 1.058 | 0.0 |
| dense | -- | grasp | 0.987 | 1.026 | 0.068 |
| dense | armA | @k=3 + rr | 0.99 | 1.05 | 0.016 |
| dense | armB | @k=5 + rr | 1.015 | 1.081 | 0.016 |
| dense | paper_v2 | @k=3 + rr | 1.048 | 1.119 | 0.016 |
| dense | size_v2 | @k=3 + rr | 0.996 | 1.058 | 0.015 |
| dense | -- | random@10+rr | 1.233 | 1.368 | 0.003 |
| dense | -- | random@3+rr | 1.099 | 1.197 | 0.003 |
| dense | -- | random@5+rr | 1.133 | 1.24 | 0.003 |
| sparse | -- | greedy | 1.0 | 1.014 | 0.0 |
| sparse | -- | greedy+rr | 0.992 | 1.007 | 0.0 |
| sparse | -- | grasp | 0.986 | 1.0 | 0.163 |
| sparse | armA | free | 0.986 | 1.0 | 0.006 |
| sparse | armB | free | 0.986 | 1.0 | 0.005 |
| sparse | paper_v2 | @k=5 + rr | 0.991 | 1.006 | 0.007 |
| sparse | size_v2 | free + rr | 0.988 | 1.002 | 0.005 |
| sparse | -- | random@10+rr | 0.997 | 1.013 | 0.002 |
| sparse | -- | random@3+rr | 0.999 | 1.017 | 0.002 |
| sparse | -- | random@5+rr | 0.999 | 1.016 | 0.002 |

What the table has to be read for: `greedy+rr` is the post-processing boost applied to the trivial baseline, so a checkpoint only earns its place by beating that row; `random@k+rr` is a coin flip inside the same top-k gain set, so any remaining gap to it is the network's contribution.

## Per source (mean over instances; ratios are per-instance |S|/|greedy|)

| domain | source | instances | |S|/greedy | |S|/gamma_t | pct_improved | mean_rr_dropped | time_s |
|---|---|---|---|---|---|---|---|
| dense | armA | 85 | 1.205 | 1.269 | 0.0 | 0.0 | 0.017 |
| dense | armA+F | 3 | 1.181 | 1.0 | 0.0 | 0.0 | 0.023 |
| dense | armA+F+rr | 3 | 1.019 | 1.0 | 33.3 | 5.667 | 0.023 |
| dense | armA+F@10 | 3 | 1.01 | 1.0 | 0.0 | 0.0 | 0.02 |
| dense | armA+F@10+rr | 3 | 1.0 | 1.0 | 33.3 | 0.333 | 0.02 |
| dense | armA+F@3 | 3 | 1.0 | 1.0 | 0.0 | 0.0 | 0.019 |
| dense | armA+F@3+rr | 3 | 0.99 | 1.0 | 33.3 | 0.333 | 0.019 |
| dense | armA+F@5 | 3 | 1.0 | 1.0 | 0.0 | 0.0 | 0.019 |
| dense | armA+F@5+rr | 3 | 1.0 | 1.0 | 0.0 | 0.0 | 0.019 |
| dense | armA+rr | 85 | 1.059 | 1.119 | 55.3 | 1.553 | 0.017 |
| dense | armA@10 | 85 | 1.054 | 1.127 | 0.0 | 0.0 | 0.015 |
| dense | armA@10+rr | 85 | 1.029 | 1.096 | 17.6 | 0.224 | 0.015 |
| dense | armA@3 | 85 | 1.009 | 1.078 | 0.0 | 0.0 | 0.016 |
| dense | armA@3+rr | 85 | 0.99 | 1.051 | 11.8 | 0.129 | 0.016 |
| dense | armA@5 | 85 | 1.027 | 1.096 | 0.0 | 0.0 | 0.016 |
| dense | armA@5+rr | 85 | 1.012 | 1.076 | 10.6 | 0.118 | 0.016 |
| dense | armB | 85 | 1.071 | 1.116 | 0.0 | 0.0 | 0.014 |
| dense | armB+F | 3 | 1.143 | 1.0 | 0.0 | 0.0 | 0.021 |
| dense | armB+F+rr | 3 | 1.114 | 1.0 | 33.3 | 1.0 | 0.021 |
| dense | armB+F@10 | 3 | 1.038 | 1.0 | 0.0 | 0.0 | 0.021 |
| dense | armB+F@10+rr | 3 | 1.038 | 1.0 | 0.0 | 0.0 | 0.021 |
| dense | armB+F@3 | 3 | 1.0 | 1.0 | 0.0 | 0.0 | 0.019 |
| dense | armB+F@3+rr | 3 | 1.0 | 1.0 | 0.0 | 0.0 | 0.019 |
| dense | armB+F@5 | 3 | 1.0 | 1.0 | 0.0 | 0.0 | 0.019 |
| dense | armB+F@5+rr | 3 | 1.0 | 1.0 | 0.0 | 0.0 | 0.019 |
| dense | armB+rr | 85 | 1.066 | 1.112 | 9.4 | 0.118 | 0.015 |
| dense | armB@10 | 85 | 1.034 | 1.103 | 0.0 | 0.0 | 0.015 |
| dense | armB@10+rr | 85 | 1.032 | 1.101 | 3.5 | 0.035 | 0.015 |
| dense | armB@3 | 85 | 1.022 | 1.09 | 0.0 | 0.0 | 0.015 |
| dense | armB@3+rr | 85 | 1.017 | 1.085 | 11.8 | 0.153 | 0.015 |
| dense | armB@5 | 85 | 1.02 | 1.089 | 0.0 | 0.0 | 0.015 |
| dense | armB@5+rr | 85 | 1.016 | 1.084 | 3.5 | 0.035 | 0.015 |
| dense | grasp | 88 | 0.987 | 1.026 | 0.0 | 0.0 | 0.068 |
| dense | greedy | 88 | 1.0 | 1.065 | 0.0 | 0.0 | 0.0 |
| dense | greedy+rr | 88 | 0.995 | 1.058 | 9.1 | 0.148 | 0.0 |
| dense | paper_v2 | 85 | 1.31 | 1.209 | 0.0 | 0.0 | 0.022 |
| dense | paper_v2+F | 3 | 2.333 | 1.1 | 0.0 | 0.0 | 0.07 |
| dense | paper_v2+F+rr | 3 | 1.267 | 1.1 | 33.3 | 37.333 | 0.07 |
| dense | paper_v2+F@10 | 3 | 1.067 | 1.0 | 0.0 | 0.0 | 0.023 |
| dense | paper_v2+F@10+rr | 3 | 1.067 | 1.0 | 0.0 | 0.0 | 0.023 |
| dense | paper_v2+F@3 | 3 | 1.121 | 1.167 | 0.0 | 0.0 | 0.02 |
| dense | paper_v2+F@3+rr | 3 | 1.121 | 1.167 | 0.0 | 0.0 | 0.02 |
| dense | paper_v2+F@5 | 3 | 1.048 | 1.0 | 0.0 | 0.0 | 0.022 |
| dense | paper_v2+F@5+rr | 3 | 1.048 | 1.0 | 0.0 | 0.0 | 0.022 |
| dense | paper_v2+rr | 85 | 1.183 | 1.204 | 8.2 | 4.612 | 0.022 |
| dense | paper_v2@10 | 85 | 1.096 | 1.161 | 0.0 | 0.0 | 0.017 |
| dense | paper_v2@10+rr | 85 | 1.094 | 1.157 | 3.5 | 0.035 | 0.017 |
| dense | paper_v2@3 | 85 | 1.05 | 1.123 | 0.0 | 0.0 | 0.016 |
| dense | paper_v2@3+rr | 85 | 1.045 | 1.117 | 9.4 | 0.094 | 0.016 |
| dense | paper_v2@5 | 85 | 1.066 | 1.131 | 0.0 | 0.0 | 0.016 |
| dense | paper_v2@5+rr | 85 | 1.064 | 1.129 | 4.7 | 0.047 | 0.016 |
| dense | random@10 | 352 | 1.772 | 2.094 | 0.0 | 0.0 | 0.003 |
| dense | random@10+rr | 352 | 1.233 | 1.368 | 65.9 | 2.909 | 0.003 |
| dense | random@3 | 352 | 1.177 | 1.3 | 0.0 | 0.0 | 0.003 |
| dense | random@3+rr | 352 | 1.099 | 1.197 | 29.5 | 0.432 | 0.003 |
| dense | random@5 | 352 | 1.226 | 1.361 | 0.0 | 0.0 | 0.003 |
| dense | random@5+rr | 352 | 1.133 | 1.24 | 42.0 | 0.716 | 0.003 |
| dense | size_v2 | 85 | 1.021 | 1.081 | 0.0 | 0.0 | 0.013 |
| dense | size_v2+F | 3 | 1.0 | 1.0 | 0.0 | 0.0 | 0.015 |
| dense | size_v2+F+rr | 3 | 0.99 | 1.0 | 33.3 | 0.333 | 0.015 |
| dense | size_v2+F@10 | 3 | 1.0 | 1.0 | 0.0 | 0.0 | 0.019 |
| dense | size_v2+F@10+rr | 3 | 0.981 | 1.0 | 33.3 | 0.667 | 0.019 |
| dense | size_v2+F@3 | 3 | 1.01 | 1.0 | 0.0 | 0.0 | 0.02 |
| dense | size_v2+F@3+rr | 3 | 0.99 | 1.0 | 33.3 | 0.667 | 0.02 |
| dense | size_v2+F@5 | 3 | 1.01 | 1.0 | 0.0 | 0.0 | 0.019 |
| dense | size_v2+F@5+rr | 3 | 0.99 | 1.0 | 33.3 | 0.667 | 0.02 |
| dense | size_v2+rr | 85 | 1.012 | 1.074 | 12.9 | 0.176 | 0.013 |
| dense | size_v2@10 | 85 | 1.016 | 1.078 | 0.0 | 0.0 | 0.015 |
| dense | size_v2@10+rr | 85 | 1.009 | 1.071 | 8.2 | 0.106 | 0.015 |
| dense | size_v2@3 | 85 | 1.005 | 1.072 | 0.0 | 0.0 | 0.015 |
| dense | size_v2@3+rr | 85 | 0.996 | 1.06 | 7.1 | 0.082 | 0.015 |
| dense | size_v2@5 | 85 | 1.004 | 1.067 | 0.0 | 0.0 | 0.015 |
| dense | size_v2@5+rr | 85 | 0.998 | 1.059 | 7.1 | 0.082 | 0.015 |
| sparse | armA+F | 80 | 0.986 | 1.0 | 0.0 | 0.0 | 0.006 |
| sparse | armA+F+rr | 80 | 0.986 | 1.0 | 0.0 | 0.0 | 0.006 |
| sparse | armA+F@10 | 80 | 0.986 | 1.0 | 0.0 | 0.0 | 0.007 |
| sparse | armA+F@10+rr | 80 | 0.986 | 1.0 | 0.0 | 0.0 | 0.007 |
| sparse | armA+F@3 | 80 | 0.988 | 1.002 | 0.0 | 0.0 | 0.007 |
| sparse | armA+F@3+rr | 80 | 0.988 | 1.002 | 0.0 | 0.0 | 0.007 |
| sparse | armA+F@5 | 80 | 0.987 | 1.001 | 0.0 | 0.0 | 0.007 |
| sparse | armA+F@5+rr | 80 | 0.987 | 1.001 | 0.0 | 0.0 | 0.007 |
| sparse | armB+F | 80 | 0.986 | 1.0 | 0.0 | 0.0 | 0.005 |
| sparse | armB+F+rr | 80 | 0.986 | 1.0 | 0.0 | 0.0 | 0.005 |
| sparse | armB+F@10 | 80 | 0.986 | 1.0 | 0.0 | 0.0 | 0.007 |
| sparse | armB+F@10+rr | 80 | 0.986 | 1.0 | 0.0 | 0.0 | 0.007 |
| sparse | armB+F@3 | 80 | 0.986 | 1.0 | 0.0 | 0.0 | 0.006 |
| sparse | armB+F@3+rr | 80 | 0.986 | 1.0 | 0.0 | 0.0 | 0.007 |
| sparse | armB+F@5 | 80 | 0.986 | 1.0 | 0.0 | 0.0 | 0.006 |
| sparse | armB+F@5+rr | 80 | 0.986 | 1.0 | 0.0 | 0.0 | 0.006 |
| sparse | grasp | 80 | 0.986 | 1.0 | 0.0 | 0.0 | 0.163 |
| sparse | greedy | 80 | 1.0 | 1.014 | 0.0 | 0.0 | 0.0 |
| sparse | greedy+rr | 80 | 0.992 | 1.007 | 18.8 | 0.375 | 0.0 |
| sparse | paper_v2+F | 80 | 1.033 | 1.048 | 0.0 | 0.0 | 0.006 |
| sparse | paper_v2+F+rr | 80 | 0.999 | 1.016 | 40.0 | 1.587 | 0.006 |
| sparse | paper_v2+F@10 | 80 | 1.004 | 1.022 | 0.0 | 0.0 | 0.007 |
| sparse | paper_v2+F@10+rr | 80 | 0.995 | 1.011 | 17.5 | 0.237 | 0.007 |
| sparse | paper_v2+F@3 | 80 | 0.996 | 1.012 | 0.0 | 0.0 | 0.007 |
| sparse | paper_v2+F@3+rr | 80 | 0.992 | 1.007 | 7.5 | 0.1 | 0.007 |
| sparse | paper_v2+F@5 | 80 | 0.997 | 1.013 | 0.0 | 0.0 | 0.007 |
| sparse | paper_v2+F@5+rr | 80 | 0.991 | 1.006 | 15.0 | 0.188 | 0.007 |
| sparse | random@10 | 320 | 1.21 | 1.257 | 0.0 | 0.0 | 0.002 |
| sparse | random@10+rr | 320 | 0.997 | 1.013 | 83.8 | 5.475 | 0.002 |
| sparse | random@3 | 320 | 1.053 | 1.083 | 0.0 | 0.0 | 0.002 |
| sparse | random@3+rr | 320 | 0.999 | 1.017 | 41.2 | 1.1 | 0.002 |
| sparse | random@5 | 320 | 1.114 | 1.162 | 0.0 | 0.0 | 0.002 |
| sparse | random@5+rr | 320 | 0.999 | 1.016 | 68.8 | 2.337 | 0.002 |
| sparse | size_v2+F | 80 | 0.988 | 1.003 | 0.0 | 0.0 | 0.005 |
| sparse | size_v2+F+rr | 80 | 0.988 | 1.002 | 3.8 | 0.037 | 0.005 |
| sparse | size_v2+F@10 | 80 | 0.989 | 1.003 | 0.0 | 0.0 | 0.007 |
| sparse | size_v2+F@10+rr | 80 | 0.988 | 1.003 | 6.2 | 0.062 | 0.007 |
| sparse | size_v2+F@3 | 80 | 0.989 | 1.003 | 0.0 | 0.0 | 0.007 |
| sparse | size_v2+F@3+rr | 80 | 0.989 | 1.003 | 1.2 | 0.013 | 0.007 |
| sparse | size_v2+F@5 | 80 | 0.989 | 1.003 | 0.0 | 0.0 | 0.007 |
| sparse | size_v2+F@5+rr | 80 | 0.988 | 1.003 | 5.0 | 0.05 | 0.007 |

## Top-k sweep, raw output (mean |S|/greedy)

| model | domain | free | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|
| armA | dense | 1.204 | 1.009 | 1.026 | 1.052 |
| armA | sparse | 0.986 | 0.988 | 0.987 | 0.986 |
| armB | dense | 1.073 | 1.021 | 1.019 | 1.034 |
| armB | sparse | 0.986 | 0.986 | 0.986 | 0.986 |
| size_v2 | dense | 1.02 | 1.005 | 1.004 | 1.015 |
| size_v2 | sparse | 0.988 | 0.989 | 0.989 | 0.989 |
| paper_v2 | dense | 1.345 | 1.053 | 1.065 | 1.095 |
| paper_v2 | sparse | 1.033 | 0.996 | 0.997 | 1.004 |

## Top-k sweep, after remove_redundant (mean |S|/greedy)

| model | domain | free | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|
| armA | dense | 1.058 | 0.99 | 1.012 | 1.028 |
| armA | sparse | 0.986 | 0.988 | 0.987 | 0.986 |
| armB | dense | 1.068 | 1.016 | 1.015 | 1.032 |
| armB | sparse | 0.986 | 0.986 | 0.986 | 0.986 |
| size_v2 | dense | 1.011 | 0.996 | 0.998 | 1.008 |
| size_v2 | sparse | 0.988 | 0.989 | 0.988 | 0.988 |
| paper_v2 | dense | 1.186 | 1.048 | 1.064 | 1.093 |
| paper_v2 | sparse | 0.999 | 0.992 | 0.991 | 0.995 |
