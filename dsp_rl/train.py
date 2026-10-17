"""Train on ER G(n,p) graphs (paper Sec. 4.2), or on the exp8 sparse families.
Examples:
   python train.py --n 20 --episodes 200000 --seed 0 --out runs/er20_s0
   python train.py --env tdsp --graph tree --n 20 --episodes 200000 --seed 0 --out runs/x   # exp8 arm A
   python train.py --env tdsp --graph tree --force_leaf --n 20 ... --out runs/y             # exp8 arm B
Writes: config.json, log.csv (loss, validation |D| and return), model.pt."""
import argparse, collections, csv, json, os, time
import numpy as np, torch
from graphs import sample, make_graph_data
from env import DSPEnv
from tdsp_env import TDSPEnv
from agent import DDQNAgent
from tdsp import has_isolated
from reductions import leaf_supports


def sample_graph(a, rng):
    """A fresh training graph; size drawn from --sizes if given (multi-size training).
    For TDSP, skip graphs with an isolated vertex (no total dominating set exists)."""
    while True:
        n = int(rng.choice(a.sizes))
        gd = make_graph_data(sample(a, n, rng))
        if a.env != "tdsp" or not has_isolated(gd):
            return gd


def forced_set(a, gd):
    """exp8 arm B: seed every episode with the leaf supports F (forced into all TDS)."""
    return leaf_supports(gd) if (a.env == "tdsp" and a.force_leaf) else ()


def run_policy(agent, env, gd, a=None):
    forced = forced_set(a, gd) if a else ()
    s = env.reset(gd, forced=forced) if forced else env.reset(gd)
    done = getattr(env, "done", False); ret = 0.0
    while not done:
        q = agent.q_values(gd, s); cand = env.candidates()
        s, r, done = env.step(int(cand[np.argmax(q[cand])])); ret += r
    return len(env.selected), ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="dsp", choices=["dsp", "tdsp"])
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--p", type=float, default=0.4)
    ap.add_argument("--graph", default="er", choices=["er", "tree", "ba1"],
                    help="training family (exp8: tree/ba1 are sparse, leaf-rich)")
    ap.add_argument("--force_leaf", action="store_true",
                    help="exp8 arm B: seed each episode with the leaf supports F (valid for TDSP)")
    ap.add_argument("--sizes", default=None, help="comma list of training sizes; overrides --n (multi-size, paper-style)")
    ap.add_argument("--episodes", type=int, default=200000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/test")
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.1)                    # not given in paper (D7)
    ap.add_argument("--nf", type=int, default=32)                        # not given in paper (D7)
    ap.add_argument("--buffer", type=int, default=500)                   # paper: 500
    ap.add_argument("--frozen_buffer", action="store_true")              # literal Alg.1 lines 6-8 (D4)
    ap.add_argument("--batch", type=int, default=32)                     # D6
    ap.add_argument("--train_start", type=int, default=100)              # 'n' in Alg.1 (D7)
    ap.add_argument("--target_every", type=int, default=1000)            # paper: 1000
    ap.add_argument("--eps_start", type=float, default=1.0); ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_frac", type=float, default=0.5)         # D8
    ap.add_argument("--reward", default=None, choices=["paper", "gain", "size"]); ap.add_argument("--agg", default=None)
    ap.add_argument("--mean_readout", action="store_true", default=None, help="n-independent graph readout (default on for tdsp)")
    ap.add_argument("--dueling", action="store_true", help="D3QN: split Q into V(s) + A(s,v) - mean A (screening experiment)")
    ap.add_argument("--eval_every", type=int, default=1000); ap.add_argument("--val_graphs", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", action="store_true", help="continue from out/ckpt.pt if present (for Colab restarts)")
    ap.add_argument("--ckpt_every", type=int, default=5000, help="save a resumable checkpoint every N episodes")
    a = ap.parse_args()
    a.sizes = [int(x) for x in a.sizes.split(",")] if a.sizes else [a.n]
    a.reward = a.reward or ("size" if a.env == "tdsp" else "paper")
    if a.agg is None: a.agg = "mean" if a.env == "tdsp" else "sum"
    if a.mean_readout is None: a.mean_readout = (a.env == "tdsp")
    os.makedirs(a.out, exist_ok=True)
    n_channels = 2 if a.env == "tdsp" else 1
    json.dump({**vars(a), "in_ch": n_channels, "torch": torch.__version__}, open(f"{a.out}/config.json", "w"), indent=1)
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    env = TDSPEnv(reward_mode=a.reward) if a.env == "tdsp" else DSPEnv(reward_mode=a.reward)
    agent = DDQNAgent(a.nf, 3, a.agg, in_ch=env.n_channels, mean_readout=a.mean_readout,
                      dueling=a.dueling, lr=a.lr, gamma=a.gamma, tau=a.tau, device=a.device)
    vrng = np.random.default_rng(10_000 + a.seed)
    val = [sample_graph(a, vrng) for _ in range(a.val_graphs)]                  # fixed validation set
    buf = collections.deque(maxlen=None if a.frozen_buffer else a.buffer)
    ckpt = f"{a.out}/ckpt.pt"
    start_ep, t0 = 1, time.time()
    if a.resume and os.path.exists(ckpt):                                       # continue across Colab restarts
        c = torch.load(ckpt, map_location=a.device, weights_only=False)
        agent.online.load_state_dict(c["online"]); agent.target.load_state_dict(c["target"])
        start_ep = c["ep"] + 1; buf.extend(c["buf"])
        rng = np.random.default_rng(); rng.bit_generator.state = c["rng"]
        torch.set_rng_state(torch.from_numpy(c["torch"]))
        t0 = time.time() - c.get("elapsed", 0.0)
        print(f"resumed from episode {c['ep']} ({len(buf)} transitions)", flush=True)
    append = "a" if start_ep > 1 and os.path.exists(f"{a.out}/log.csv") else "w"
    f = open(f"{a.out}/log.csv", append, newline=""); w = csv.writer(f)
    if append == "w": w.writerow(["episode", "eps", "loss", "val_mean_size", "val_mean_return", "elapsed_s"])
    loss = float("nan")
    for ep in range(start_ep, a.episodes + 1):
        eps = max(a.eps_end, a.eps_start - (a.eps_start - a.eps_end) * ep / (a.eps_decay_frac * a.episodes))
        gd = sample_graph(a, rng)
        s = env.reset(gd, forced=forced_set(a, gd)) if a.env == "tdsp" else env.reset(gd)
        done = getattr(env, "done", False)
        while not done:
            v = agent.act(gd, s, env.candidates(), eps, rng); s2, r, done = env.step(v)
            if not (a.frozen_buffer and len(buf) >= a.buffer):
                buf.append((gd, s, s2, v, r, done, env.candidates()))
            s = s2
        if ep > a.train_start and len(buf) >= a.batch:
            loss = agent.update([buf[i] for i in rng.choice(len(buf), a.batch, replace=False)])
        if ep % a.target_every == 0:
            agent.soft_update()
        if ep % a.eval_every == 0:
            res = [run_policy(agent, env, g, a) for g in val]
            w.writerow([ep, round(eps, 3), loss, np.mean([x[0] for x in res]), np.mean([x[1] for x in res]), round(time.time() - t0, 1)]); f.flush()
            print(ep, f"eps={eps:.2f} loss={loss:.3f} val|D|={np.mean([x[0] for x in res]):.2f} ret={np.mean([x[1] for x in res]):.2f}", flush=True)
        if ep % a.ckpt_every == 0:
            torch.save({"ep": ep, "online": agent.online.state_dict(), "target": agent.target.state_dict(),
                        "buf": list(buf), "rng": rng.bit_generator.state,
                        "torch": torch.get_rng_state().numpy(), "elapsed": time.time() - t0}, ckpt)
    agent.save(f"{a.out}/model.pt")


if __name__ == "__main__":
    main()
