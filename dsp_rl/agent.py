"""DDQN agent (Chen et al. 2024, Sec. 3.2 and Algorithm 1)."""
import copy
import numpy as np
import torch
from model import MPNNQ, collate


class DDQNAgent:
    def __init__(self, nf=32, layers=3, agg="sum", lr=1e-4, gamma=1.0, tau=0.1, device="cpu"):
        self.dev = torch.device(device)
        self.online = MPNNQ(nf, layers, agg).to(self.dev)
        self.target = copy.deepcopy(self.online)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.gamma, self.tau = gamma, tau

    @torch.no_grad()
    def q_values(self, gd, vf):
        return self.online(collate([(gd, vf)], self.dev)).cpu().numpy()

    def act(self, gd, vf, eps, rng):
        cand = np.flatnonzero(vf == 0)
        if rng.random() < eps:
            return int(rng.choice(cand))
        q = self.q_values(gd, vf)
        return int(cand[np.argmax(q[cand])])                                   # Eq.(7)

    def update(self, batch):
        """batch: list of (gd, vf, vf_next, v, r, done)."""
        b0 = collate([(t[0], t[1]) for t in batch], self.dev)
        b1 = collate([(t[0], t[2]) for t in batch], self.dev)
        idx = torch.tensor([b0.offs[i] + batch[i][3] for i in range(len(batch))], device=self.dev)
        q_hat = self.online(b0)[idx]
        with torch.no_grad():
            qo = self.online(b1).cpu().numpy()                                 # action selection: online net
            qt = self.target(b1).cpu().numpy()                                 # action evaluation: target net
            y = np.zeros(len(batch), dtype=np.float32)
            for i, (gd, _, vf2, v, r, done) in enumerate(batch):
                y[i] = r
                if not done:                                                   # DECISIONS D3
                    sl = slice(b1.offs[i], b1.offs[i] + gd.n)
                    cand = np.flatnonzero(vf2 == 0)
                    a = cand[np.argmax(qo[sl][cand])]
                    y[i] += self.gamma * qt[sl][a]
        loss = 0.5 * torch.mean((q_hat - torch.from_numpy(y).to(self.dev)) ** 2)   # Eq.(8)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss)

    @torch.no_grad()
    def soft_update(self):
        for p, pt in zip(self.online.parameters(), self.target.parameters()):
            pt.mul_(1 - self.tau).add_(self.tau * p)                           # Theta_bar <- tau*Theta + (1-tau)*Theta_bar

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        self.online.load_state_dict(torch.load(path, map_location=self.dev)); self.target.load_state_dict(self.online.state_dict())
