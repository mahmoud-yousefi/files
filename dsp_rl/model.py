"""MPNN + Q readout after Chen et al. (2024), Eqs.(1)-(4),(6). Works on batches of graphs (block-diagonal)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Batch:
    pass


def collate(items, device):
    """items: list of (GraphData, vf np.uint8). Returns Batch with offsets/sizes for indexing."""
    off = 0; R, C, AN, G, V, DG, offs, sizes = [], [], [], [], [], [], [], []
    for g, (gd, vf) in enumerate(items):
        R.append(gd.row + off); C.append(gd.col + off); AN.append(gd.an)
        G.append(np.full(gd.n, g, dtype=np.int64)); V.append(vf.astype(np.float32)); DG.append(gd.deg)
        offs.append(off); sizes.append(gd.n); off += gd.n
    b = Batch()
    t = lambda x, dt: torch.from_numpy(np.concatenate(x).astype(dt)).to(device)
    b.row, b.col = t(R, np.int64), t(C, np.int64)
    b.an, b.gid, b.vf, b.deg = t(AN, np.float32), t(G, np.int64), t(V, np.float32), t(DG, np.float32)
    b.ng, b.N = len(items), off
    b.offs, b.sizes = np.array(offs), np.array(sizes)
    return b


class MPNNQ(nn.Module):
    def __init__(self, nf=32, layers=3, agg="sum"):
        super().__init__()
        self.nf, self.agg = nf, agg
        self.t4 = nn.Linear(2, nf)            # Eq.(4)  (n-independent version, DECISIONS D5)
        self.t3 = nn.Linear(nf + 1, nf)       # Eq.(3)  per-edge embedding (DECISIONS D5)
        self.t1 = nn.ModuleList([nn.Linear(3 * nf, nf) for _ in range(layers)])   # Eq.(2)
        self.t2 = nn.ModuleList([nn.Linear(nf, nf) for _ in range(layers)])
        self.t6, self.t7 = nn.Linear(nf, nf), nn.Linear(nf, nf)                   # Eq.(6)
        self.t5 = nn.Linear(2 * nf, 1)

    def forward(self, b):
        N, nf, dev = b.N, self.nf, b.vf.device
        z = lambda *s: torch.zeros(*s, device=dev)
        nvf = z(N).index_add_(0, b.row, b.an * b.vf[b.col])                 # (A_norm vf)_v
        f = F.relu(self.t4(torch.stack([b.vf, nvf], 1)))                    # Eq.(4)
        e = F.relu(self.t3(torch.cat([f[b.col], b.an.unsqueeze(1)], 1)))    # Eq.(3), per edge (v<-w)
        esum = z(N, nf).index_add_(0, b.row, e)
        h = f
        dcl = b.deg.clamp(min=1).unsqueeze(1)
        for l1, l2 in zip(self.t1, self.t2):
            hsum = z(N, nf).index_add_(0, b.row, h[b.col])
            m = torch.cat([b.deg.unsqueeze(1) * h, hsum, esum], 1)          # Eq.(1) with M_t = concat, sum over N(v): n x 3nf
            if self.agg == "mean":
                m = m / dcl
            h = F.relu(l2(F.relu(l1(m))))                                   # Eq.(2)
        s = z(b.ng, nf).index_add_(0, b.gid, h)                             # sum_u m_u
        return self.t5(F.relu(torch.cat([self.t6(s)[b.gid], self.t7(h)], 1))).squeeze(1)   # Eq.(6)
