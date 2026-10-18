# Exp9 — D3QN screening, rebuilt from the run logs

Only the Q head differs (`--dueling`); protocol, seed and reward are identical,
so every row is a paired comparison.

## Per pair (episodes to reach a final-quality plateau)

| pair | plain final | D3QN final | plain ≤9.8 | D3QN ≤9.8 | speed-up | plain ≤10.0 | D3QN ≤10.0 | speed-up |
|---|---|---|---|---|---|---|---|---|
| armA seed0 | 9.5 | 9.5 | 105,000 | 60,000 | 1.75× | 95,000 | 60,000 | 1.58× |
| armA seed1 | 9.8 | 9.8 | 180,000 | 125,000 | 1.44× | 140,000 | 105,000 | 1.33× |
| armA seed2 | 9.72 | 9.7 | 165,000 | 85,000 | 1.94× | 130,000 | 70,000 | 1.86× |
| armB seed0 | 9.5 | 9.5 | 45,000 | 25,000 | 1.8× | 40,000 | 20,000 | 2.0× |

## Means

| group | plain ≤9.8 | D3QN ≤9.8 | plain ≤10.0 | D3QN ≤10.0 |
|---|---|---|---|---|
| armA (3 seeds) | 150,000 | 90,000 | 121,667 | 78,333 |
| armB (seed 0) | 45,000 | 25,000 | 40,000 | 20,000 |

## Reading

- Final quality is the same in both heads (see the two `final` columns, they differ by at
  most 0.02): D3QN buys **no** size improvement, only a shorter climb.
- The speed-up is a *threshold* statistic: it is defined by the plateau level, so a curve
  that dips below the threshold once early is credited immediately. Quote it as
  "reaches the plateau after ~1.7× fewer episodes", never as a wall-clock speed-up.
- `sustained@τ` is the pessimistic twin and it is **not** usable in `armA seed1`, where the
  final value is exactly 9.8 = τ: the curve keeps touching the threshold and the metric
  reports 190k vs 200k for two runs that plateau together. It agrees with `hit@τ` in the
  other three pairs, so the direction of the effect is robust; the exact multiplier is not.
