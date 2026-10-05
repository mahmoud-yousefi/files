"""Train on ER G(n,p) graphs (paper Sec. 4.2). Example:
   python train.py --n 20 --episodes 200000 --seed 0 --out runs/er20_s0
Writes: config.json, log.csv (loss, validation |D| and return), model.pt."""
import argparse, collections, csv, json, os, time
import numpy as np, torch
from graphs import er, make_graph_data
from env import DSPEnv
from agent import DDQNAgent


def run_policy(agent, env, gd):
    vf = env.reset(gd); done = False; ret = 0.0
    while not done:
        q = agent.q_values(gd, vf); cand = env.candidates()
        vf, r, done = env.step(int(cand[np.argmax(q[cand])])); ret += r
    return len(env.selected), ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--p", type=float, default=0.4)
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
    ap.add_argument("--reward", default="paper", choices=["paper", "gain"]); ap.add_argument("--agg", default="sum")
    ap.add_argument("--eval_every", type=int, default=1000); ap.add_argument("--val_graphs", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), "torch": torch.__version__}, open(f"{a.out}/config.json", "w"), indent=1)
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    env = DSPEnv(reward_mode=a.reward)
    agent = DDQNAgent(a.nf, 3, a.agg, a.lr, a.gamma, a.tau, a.device)
    vrng = np.random.default_rng(10_000 + a.seed)
    val = [make_graph_data(er(a.n, a.p, vrng)) for _ in range(a.val_graphs)]    # fixed validation set
    buf = collections.deque(maxlen=None if a.frozen_buffer else a.buffer)
    f = open(f"{a.out}/log.csv", "w", newline=""); w = csv.writer(f)
    w.writerow(["episode", "eps", "loss", "val_mean_size", "val_mean_return", "elapsed_s"]); t0 = time.time(); loss = float("nan")
    for ep in range(1, a.episodes + 1):
        eps = max(a.eps_end, a.eps_start - (a.eps_start - a.eps_end) * ep / (a.eps_decay_frac * a.episodes))
        gd = make_graph_data(er(a.n, a.p, rng)); vf = env.reset(gd); done = False
        while not done:
            v = agent.act(gd, vf, eps, rng); vf2, r, done = env.step(v)
            if not (a.frozen_buffer and len(buf) >= a.buffer):
                buf.append((gd, vf, vf2, v, r, done))
            vf = vf2
        if ep > a.train_start and len(buf) >= a.batch:
            loss = agent.update([buf[i] for i in rng.choice(len(buf), a.batch, replace=False)])
        if ep % a.target_every == 0:
            agent.soft_update()
        if ep % a.eval_every == 0:
            res = [run_policy(agent, env, g) for g in val]
            w.writerow([ep, round(eps, 3), loss, np.mean([x[0] for x in res]), np.mean([x[1] for x in res]), round(time.time() - t0, 1)]); f.flush()
            print(ep, f"eps={eps:.2f} loss={loss:.3f} val|D|={np.mean([x[0] for x in res]):.2f} ret={np.mean([x[1] for x in res]):.2f}", flush=True)
    agent.save(f"{a.out}/model.pt")


if __name__ == "__main__":
    main()
