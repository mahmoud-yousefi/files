"""DSP environment following Chen et al. (2024), Sec. 2.1 and 3.1.
State  : vf in {0,1}^n (1 = dominated). Action: a vertex with vf==0 (DECISIONS D1).
Reward : paper Eq.(5) with w=1 ('paper'), or number of newly dominated vertices ('gain')."""
import numpy as np


class DSPEnv:
    n_channels = 1                                    # state = coverage vf only

    def __init__(self, alpha=1.0, reward_mode="paper"):
        assert reward_mode in ("paper", "gain", "size")
        self.alpha, self.mode = alpha, reward_mode

    def reset(self, gd):
        self.gd = gd
        self.vf = np.zeros(gd.n, dtype=np.uint8)
        self.selected = []
        return self._state()

    def _state(self):
        return self.vf.reshape(-1, 1).astype(np.float32)

    def candidates(self):
        return np.flatnonzero(self.vf == 0)           # S_bar = V - S  (paper Eq.7)

    def step(self, v):
        assert self.vf[v] == 0, "paper's mask forbids selecting a dominated vertex"
        nb = self.gd.nbrs[v]
        if self.mode == "size":                        # return = -|D| (for size minimisation)
            r = -1.0
        elif self.mode == "paper":                     # Eq.(5): alpha * vhat^T (A(i) . w), A(i) has zero diagonal
            r = self.alpha * float((1 - 2 * self.vf[nb].astype(np.int64)).sum())
        else:
            r = self.alpha * float(1 + (self.vf[nb] == 0).sum())
        self.vf[v] = 1
        self.vf[nb] = 1
        self.selected.append(int(v))
        done = bool(self.vf.all())                     # terminal condition: DECISIONS D2
        return self._state(), r, done
