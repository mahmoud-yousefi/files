# Exp7: do reduction rules help TDSP on our benchmarks?

**Headline:** on the graph families used throughout this thesis, none of the three
candidate rules fires: every instance is connected (no components to split), has no
isolated vertex, and has minimum degree >= 2 (no leaves). The naive leaf rule is also
*invalid* for TDSP (counterexample: P3). So reduction gives no speed/quality gain here --
a negative result reported honestly, not a fabricated improvement.

leaf-rule counterexample: `{'n': 3, 'forced_by_leaf_rule': [1], 'naive_answer_size': 1, 'true_optimum': 2}`

| family | n | instances | mean components | any isolated | mean leaf-forced |
|---|---|---|---|---|---|
| BA | 20 | 20 | 1.0 | False | 0.35 |
| BA | 100 | 20 | 1.0 | False | 0.05 |
| BA | 300 | 20 | 1.0 | False | 0.0 |
| CTRL-disconnected | 12 | 1 | 2.0 | False | 4.0 |
| ER | 20 | 20 | 1.0 | False | 0.0 |
| ER | 100 | 20 | 1.0 | False | 0.0 |
| ER | 300 | 20 | 1.0 | False | 0.0 |
| GRID | 20 | 1 | 1.0 | False | 0.0 |
| GRID | 100 | 1 | 1.0 | False | 0.0 |
| GRID | 400 | 1 | 1.0 | False | 0.0 |
| HEX | 58 | 1 | 1.0 | False | 0.0 |
| HEX | 96 | 1 | 1.0 | False | 0.0 |
| HEX | 306 | 1 | 1.0 | False | 0.0 |
| TRI | 28 | 1 | 1.0 | False | 0.0 |
