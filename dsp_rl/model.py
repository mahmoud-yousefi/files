"""MPNN + Q readout after Chen et al. (2024), Eqs.(1)-(4),(6). Works on batches of graphs (block-diagonal)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Batch:
    pass


def collate(items, device):
    """items: list of (GraphData, state) where state is (n,) or (n,k) float.
    Returns Batch with offsets/sizes for indexing."""
    off = 0; R, C, AN, G, ST, DG, offs, sizes = [], [], [], [], [], [], [], []
    for g, (gd, st) in enumerate(items):
        st = np.asarray(st, dtype=np.float32)
        if st.ndim == 1: st = st.reshape(-1, 1)          # 1-channel state -> (n,1)
        R.append(gd.row + off); C.append(gd.col + off); AN.append(gd.an)
        G.append(np.full(gd.n, g, dtype=np.int64)); ST.append(st); DG.append(gd.deg)
        offs.append(off); sizes.append(gd.n); off += gd.n
    b = Batch()
    t = lambda x, dt: torch.from_numpy(np.concatenate(x).astype(dt)).to(device)
    b.row, b.col = t(R, np.int64), t(C, np.int64)
    b.an, b.gid, b.state, b.deg = t(AN, np.float32), t(G, np.int64), t(ST, np.float32), t(DG, np.float32)
    b.ng, b.N = len(items), off
    b.offs, b.sizes = np.array(offs), np.array(sizes)
    b.sizes_t = torch.from_numpy(b.sizes.astype(np.float32)).to(device)
    return b


class MPNNQ(nn.Module):
    def __init__(self, nf=32, layers=3, agg="sum", in_ch=1, mean_readout=False, dueling=False):
        super().__init__()
        self.nf, self.agg, self.in_ch = nf, agg, in_ch
        # mean_readout: divide the graph-level sum by n, so Q does not scale with
        # the graph size. Needed for training/testing across several n (especially
        # with 'size' reward); default off keeps the DSP reproduction bit-identical.
        self.mean_readout = mean_readout
        # dueling (D3QN): Q(s,v) = V(s) + A(s,v) - mean_v A(s,v). The advantage head
        # is the paper's Eq.(6) head; the value head reads only the pooled state.
        # Default off keeps every existing checkpoint bit-identical.
        self.dueling = dueling
        self.t4 = nn.Linear(2 * in_ch, nf)    # Eq.(4); in_ch=1 reproduces the DSP model (D5)
        self.t3 = nn.Linear(nf + 1, nf)       # Eq.(3)  per-edge embedding (DECISIONS D5)
        self.t1 = nn.ModuleList([nn.Linear(3 * nf, nf) for _ in range(layers)])   # Eq.(2)
        self.t2 = nn.ModuleList([nn.Linear(nf, nf) for _ in range(layers)])
        self.t6, self.t7 = nn.Linear(nf, nf), nn.Linear(nf, nf)                   # Eq.(6)
        self.t5 = nn.Linear(2 * nf, 1)
        if dueling:
            self.v_head = nn.Sequential(nn.Linear(nf, nf), nn.ReLU(), nn.Linear(nf, 1))

    def forward(self, b):
        N, nf, dev = b.N, self.nf, b.state.device
        z = lambda *s: torch.zeros(*s, device=dev)
        feats = []
        for j in range(self.in_ch):                                         # per-channel: [c_j, (A_norm c_j)_v]
            cj = b.state[:, j]
            feats += [cj, z(N).index_add_(0, b.row, b.an * cj[b.col])]
        f = F.relu(self.t4(torch.stack(feats, 1)))                          # Eq.(4)
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
        if self.mean_readout:
            s = s / b.sizes_t.unsqueeze(1)                                  # n-independent readout
        a = self.t5(F.relu(torch.cat([self.t6(s)[b.gid], self.t7(h)], 1))).squeeze(1)   # Eq.(6) advantage A(s,v)
        if not self.dueling:
            return a
        v = self.v_head(F.relu(self.t6(s))).squeeze(1)                      # value V(s), one per graph
        amean = z(b.ng).index_add_(0, b.gid, a) / b.sizes_t                 # per-graph mean of A
        return v[b.gid] + a - amean[b.gid]                                  # Q = V + (A - mean A)
