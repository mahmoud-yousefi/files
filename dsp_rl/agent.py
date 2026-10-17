"""DDQN agent (Chen et al. 2024, Sec. 3.2 and Algorithm 1)."""
import copy
import numpy as np
import torch
from model import MPNNQ, collate


class DDQNAgent:
    def __init__(self, nf=32, layers=3, agg="sum", in_ch=1, mean_readout=False, dueling=False, lr=1e-4, gamma=1.0, tau=0.1, device="cpu"):
        self.dev = torch.device(device)
        self.online = MPNNQ(nf, layers, agg, in_ch, mean_readout, dueling=dueling).to(self.dev)
        self.target = copy.deepcopy(self.online)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.gamma, self.tau = gamma, tau

    @torch.no_grad()
    def q_values(self, gd, state):
        return self.online(collate([(gd, state)], self.dev)).cpu().numpy()

    def act(self, gd, state, cand, eps, rng):
        """cand: the legal actions (from env.candidates()), not derived from the state."""
        if rng.random() < eps:
            return int(rng.choice(cand))
        q = self.q_values(gd, state)
        return int(cand[np.argmax(q[cand])])                                   # Eq.(7)

    def update(self, batch):
        """batch: list of (gd, state, state_next, v, r, done, cand_next)."""
        b0 = collate([(t[0], t[1]) for t in batch], self.dev)
        b1 = collate([(t[0], t[2]) for t in batch], self.dev)
        idx = torch.tensor([b0.offs[i] + batch[i][3] for i in range(len(batch))], device=self.dev)
        q_hat = self.online(b0)[idx]
        with torch.no_grad():
            qo = self.online(b1).cpu().numpy()                                 # action selection: online net
            qt = self.target(b1).cpu().numpy()                                 # action evaluation: target net
            y = np.zeros(len(batch), dtype=np.float32)
            for i, (gd, _, _, v, r, done, cand) in enumerate(batch):
                y[i] = r
                if not done and len(cand):                                     # DECISIONS D3
                    sl = slice(b1.offs[i], b1.offs[i] + gd.n)
                    a = cand[np.argmax(qo[sl][cand])]
                    y[i] += self.gamma * qt[sl][a]
        loss = 0.5 * torch.mean((q_hat - torch.from_numpy(y).to(self.dev)) ** 2)   # Eq.(8)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

    @torch.no_grad()
    def soft_update(self):
        for p, pt in zip(self.online.parameters(), self.target.parameters()):
            pt.mul_(1 - self.tau).add_(self.tau * p)                           # Theta_bar <- tau*Theta + (1-tau)*Theta_bar

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        self.online.load_state_dict(torch.load(path, map_location=self.dev)); self.target.load_state_dict(self.online.state_dict())
        self.load_path = path


def agent_for_model(model_path, device="cpu", **over):
    """Rebuild an agent matching a saved model, reading its run's config.json so the
    architecture (nf/agg/in_ch/mean_readout) always matches the checkpoint."""
    import json, os
    cfg = {}
    cj = os.path.join(os.path.dirname(model_path), "config.json")
    if os.path.exists(cj):
        cfg = json.load(open(cj))
    env_name = cfg.get("env", "dsp")
    p = dict(nf=cfg.get("nf", 32), agg=cfg.get("agg", "sum"),
             in_ch=cfg.get("in_ch", 2 if env_name == "tdsp" else 1),
             mean_readout=cfg.get("mean_readout", False),
             dueling=cfg.get("dueling", False))
    p.update(over)
    ag = DDQNAgent(layers=3, device=device, **p)
    ag.load(model_path)
    return ag
