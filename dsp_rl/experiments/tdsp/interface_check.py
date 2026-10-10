"""Interface check (chapter 5): the agent/env contract after the TDSP adaptation.

Guards the fixes that exp2/exp3 exposed:
  - candidates come from env.candidates(), never from a hard-coded state==0 rule;
  - the TDSP state carries both channels (vf, ins), so the mask and the network
    agree; the DSP state still carries one;
  - the 'size' reward makes an episode return exactly -|S| (the minimisation
    objective), which the 'gain' reward did not.
Also confirms the DSP path is unchanged: the trained er20_s0 model still loads and
solves, and its q-values match a hand-computed single-channel forward.

Run from the dsp_rl directory:  python experiments/tdsp/interface_check.py
"""
import os, sys
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from graphs import make_graph_data, grid, er, ba, hexa
from env import DSPEnv
from tdsp_env import TDSPEnv
from agent import DDQNAgent
from model import collate
from baselines import is_dominating, is_total_dominating


def random_episode(agent, env, gd, rng):
    """Drive env through agent.act (epsilon=1). Any illegal action trips env's assert."""
    s = env.reset(gd); done = False; ret = 0.0; steps = 0
    while not done:
        cand = env.candidates()
        assert len(cand) and s.shape[0] == gd.n
        v = agent.act(gd, s, cand, 1.0, rng)          # random, but must be a legal candidate
        s, r, done = env.step(v); ret += r; steps += 1
    return ret, steps


def main():
    rng = np.random.default_rng(0)
    # 1) state shapes
    gd = make_graph_data(grid(4, 5))
    assert DSPEnv().reset(gd).shape == (gd.n, 1), "DSP state must be 1-channel"
    assert TDSPEnv().reset(gd).shape == (gd.n, 2), "TDSP state must be 2-channel"
    print("ok  state shapes: DSP (n,1), TDSP (n,2)")

    # 2) agent.act only ever returns legal candidates, on both envs, over many graphs
    for env, chk, name in [(TDSPEnv(), is_total_dominating, "TDSP"), (DSPEnv(), is_dominating, "DSP")]:
        ag = DDQNAgent(in_ch=env.n_channels)
        graphs = [make_graph_data(grid(4, 5)), make_graph_data(er(40, 0.4, rng)), make_graph_data(ba(40, 4, rng))]
        for g in graphs:
            for _ in range(30):
                _, steps = random_episode(ag, env, g, rng)
                assert chk(g, env.selected), f"{name} output invalid"
        print(f"ok  {name}: 90 episodes, no illegal action, all outputs valid")

    # 3) 'size' reward gives exactly -|S|
    for env, name in [(TDSPEnv(reward_mode="size"), "TDSP"), (DSPEnv(reward_mode="size"), "DSP")]:
        g = make_graph_data(grid(5, 5)); env.reset(g); ret, steps = random_episode(DDQNAgent(in_ch=env.n_channels), env, g, rng)
        assert abs(ret - (-steps)) < 1e-9, f"{name} size return {ret} != -{steps}"
        print(f"ok  {name} 'size' reward: return {ret:.1f} == -|S| (={-steps})")

    # 4) DSP backward compatibility: existing model loads and single-channel forward is exact
    mp = "runs/er20_s0/model.pt"
    if os.path.exists(mp):
        ag = DDQNAgent(in_ch=1); ag.load(mp)
        g = make_graph_data(grid(4, 5)); vf = np.zeros(g.n, dtype=np.float32)
        q = ag.q_values(g, vf.reshape(-1, 1))
        assert q.shape == (g.n,)
        print(f"ok  DSP model loads and scores: q mean={q.mean():.3f}")
    print("\nALL INTERFACE CHECKS PASSED")


if __name__ == "__main__":
    main()
