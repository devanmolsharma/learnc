"""Train on short sequences (length 1-10), test on long ones (40-64): the C program vs a plain neural network."""
import time
import torch, torch.nn as nn, torch.nn.functional as F
from learnc import Program

W, T = torch.tensor([2.0, -1.0, 0.5]), 0.3                      # the hidden rule: 2a - b + 0.5c > 0.3


def data(n, lo, hi, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 64, 3, generator=g); length = torch.randint(lo, hi + 1, (n,), generator=g)
    valid = torch.arange(64)[None] < length[:, None]
    x = x * valid[..., None]                                     # positions past the length are zero
    return x, length, ((x @ W > T) & valid).sum(1)


xtr, ntr, ytr = data(3000, 1, 10, 0)
tests = {"lengths 1-10": data(2000, 1, 10, 1), "lengths 40-64": data(2000, 40, 64, 2)}

prog = Program("count.c", device="cpu"); t0 = time.time()
prog.fit(xtr, ntr, target=ytr, steps=300, lr=0.05, temperature=0.1)
print(f"C program: fitted in {time.time() - t0:.0f}s; learned w = {[round(v, 3) for v in prog.values['w'].tolist()]}, t = {prog.values['t'].item():.3f}"
      f"  (rule is scale-free: w/t = {[round(v, 2) for v in (prog.values['w'] / prog.values['t']).tolist()]} vs true {[round(v, 2) for v in (W / T).tolist()]})")
for name, (x, n, y) in tests.items(): print(f"   {name}: exact count {prog.accuracy(x, n, target=y):.4f}")

torch.manual_seed(0)                                             # baseline: an MLP that predicts the count (0..64)
net = nn.Sequential(nn.Linear(64 * 3 + 1, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 65))
feats = lambda x, n: torch.cat([x.flatten(1), n[:, None].float() / 64], 1)
opt = torch.optim.Adam(net.parameters(), 1e-3); t0 = time.time()
for step in range(3000):
    k = torch.randint(0, len(ytr), (128,))
    loss = F.cross_entropy(net(feats(xtr[k], ntr[k])), ytr[k]); opt.zero_grad(); loss.backward(); opt.step()
print(f"MLP (256-256, same data): trained in {time.time() - t0:.0f}s")
with torch.no_grad():
    for name, (x, n, y) in tests.items(): print(f"   {name}: exact count {(net(feats(x, n)).argmax(1) == y).float().mean():.4f}")
