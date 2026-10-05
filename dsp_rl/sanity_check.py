"""No-torch sanity checks: run this first. Verifies env, greedy, ILP, and the 'masked policy => independent set' inference."""
import numpy as np, networkx as nx
from graphs import *; from env import DSPEnv; from baselines import *
rng = np.random.default_rng(0); env = DSPEnv()
viol = 0
for trial in range(200):
    gd = make_graph_data(er(30, 0.2, rng)); env.reset(gd); done = False; ret = 0
    while not done:
        v = int(rng.choice(env.candidates())); _, r, done = env.step(v); ret += r
    D = env.selected
    assert is_dominating(gd, D)
    viol += (not is_independent(gd, D))
print("random masked policy: non-independent outputs =", viol, "/ 200   (expected 0)")
gd = make_graph_data(grid(7, 7)); g = greedy_ds(gd)
print("grid 7x7: greedy", len(g), "| ILP gamma", exact(gd, "dom"), "| ILP i(G)", exact(gd, "ind"), "| ILP gamma_t", exact(gd, "tot"))
assert is_dominating(gd, g)
for name, G in [("tri(4,6)", tri(4, 6)), ("hex(6,6)", hexa(6, 6))]:
    print(name, "n =", make_graph_data(G).n)
print("OK")
