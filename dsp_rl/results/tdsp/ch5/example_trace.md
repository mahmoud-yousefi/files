# Worked example (chapter 5, section 5-3 and 5-4)

Graph: uniform random labelled tree, n=12, seed=5.
Leaves: [1, 2, 3, 4, 11]. Forced set F = support of the leaves = [0, 6, 8, 10] (|F|=4);
every support is in *every* total dominating set, so forcing it is valid.

## Sizes

| method | S | \|S\| | total dominating |
|---|---|---|---|
| greedy_tds (no forcing) | {0, 2, 3, 6, 8, 10} | 6 | True |
| F only (seed) | {0, 6, 8, 10} | 4 | False |
| F + policy, free action set | {0, 2, 3, 6, 8, 10} | 6 | True |
| F + policy, top-3 gain mask | {0, 2, 3, 6, 8, 10} | 6 | True |
| F + greedy inside mask (k=1) | {0, 2, 3, 6, 8, 10} | 6 | True |
| ILP optimum | {0, 5, 6, 7, 8, 10} | 6 | True |

## Step-by-step (free action set, armA + F)

| step | S before | legal candidates | top-3 by gain | Q of legal | action | newly satisfied | r | \|S\| |
|---|---|---|---|---|---|---|---|---|
| 1 | {0, 8, 10, 6} | {1, 2, 3, 4, 5, 7, 9, 11} | 2(1), 3(1), 4(1) | 1:-4.784 2:-3.732 3:-3.767 4:-3.767 5:-3.960 7:-3.998 9:-4.913 11:-4.827 | **2** | {0} | -1.0 | 5 |
| 2 | {0, 8, 10, 6, 2} | {1, 3, 4, 5, 7, 9, 11} | 3(1), 4(1), 5(1) | 1:-2.921 3:-1.884 4:-1.884 5:-2.077 7:-2.980 9:-3.034 11:-2.944 | **3** | {8} | -1.0 | 6 |

## The reward argument on this instance

Under the paper's gain reward every vertex is satisfied exactly once, so the return is always Σgain = n = 12 regardless of how many vertices are chosen — the policy gets no signal for size. Under the size reward each of the 2 RL decisions pays r = −1, so the return is exactly −|S_RL| = −2, i.e. the objective itself.

