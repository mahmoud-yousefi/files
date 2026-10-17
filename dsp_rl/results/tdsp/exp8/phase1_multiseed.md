# Exp8: multi-seed replication (3 seeds x 2 arms, TREE n=20, 200k episodes)

Same protocol as phase 1; seeds 1,2 added to the original seed 0.

| arm | seeds | final val|S| (mean+-std) | first near-final episode (mean+-std) |
|---|---|---|---|
| armA (plain) | 0,1,2 | 9.67 +- 0.15 | 160,000 +- 18,028 |
| armB (forced) | 0,1,2 | 9.67 +- 0.15 | 43,333 +- 10,408 |

Per-run detail:

| arm | seed | final val|S| | first near-final ep |
|---|---|---|---|
| armA (plain) | 0 | 9.5 | 140000 |
| armA (plain) | 1 | 9.8 | 175000 |
| armA (plain) | 2 | 9.7 | 165000 |
| armB (forced) | 0 | 9.5 | 55000 |
| armB (forced) | 1 | 9.8 | 40000 |
| armB (forced) | 2 | 9.7 | 35000 |
