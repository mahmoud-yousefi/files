"""TDSP environment: total dominating set as a sequential decision process.

Same interface as env.DSPEnv (reset / candidates / step), so the existing agent
and training loop can drive it *once the agent reads candidates from the env and
the state carries the selection channel*. What differs is the *rules*:

                          DSP (env.py)              TDSP (here)
  action mask     only undominated vertices   any vertex not already chosen
  picking v       marks v and N(v)            marks N(v) only, NOT v itself
  terminal        every vertex dominated      every vertex has a neighbour in S
  state           coverage vf (1 channel)     [coverage vf, selection ins] (2 channels)

The second row is the whole point: a chosen vertex v is satisfied only once one
of its neighbours also joins S -- exactly the total-domination condition. This
is the redesigned action space that DECISIONS T4 requires, since the paper's
vf==0 mask can never produce a total dominating set.

Two channels are needed because vf and ins are different: a vertex can be
selected (ins==1) while still unsatisfied (vf==0), and vice versa. Feeding only
vf aliases those distinct states.
"""
import numpy as np


class TDSPEnv:
    n_channels = 2                                    # state = [vf, ins]

    def __init__(self, alpha=1.0, reward_mode="size"):
        assert reward_mode in ("paper", "gain", "size")
        self.alpha, self.mode = alpha, reward_mode

    def reset(self, gd):
        self.gd = gd
        self.vf = np.zeros(gd.n, dtype=np.uint8)     # 1 = has a neighbour in S (satisfied)
        self.ins = np.zeros(gd.n, dtype=np.uint8)    # 1 = chosen, i.e. a member of S
        self.selected = []
        return self._state()

    def _state(self):
        return np.stack([self.vf, self.ins], axis=1).astype(np.float32)

    def candidates(self):
        return np.flatnonzero(self.ins == 0)          # every vertex not already in S

    def step(self, v):
        assert self.ins[v] == 0, "vertex already chosen"
        nb = self.gd.nbrs[v]
        if self.mode == "size":                       # return = -|S| (for size minimisation)
            r = -1.0
        elif self.mode == "paper":                    # Eq.(5) analogue: +1 new, -1 already
            r = self.alpha * float((1 - 2 * self.vf[nb].astype(np.int64)).sum())
        else:
            r = self.alpha * float((self.vf[nb] == 0).sum())   # gain only
        self.vf[nb] = 1                               # v's neighbours are satisfied, v is not
        self.ins[v] = 1
        self.selected.append(int(v))
        done = bool(self.vf.all())                    # every vertex has a neighbour in S
        return self._state(), r, done
