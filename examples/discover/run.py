"""Can learnc rediscover Strassen (2x2 matmul with 7 products) from input/output examples only?"""
import sys, time
import torch
from learnc import Program

g = torch.Generator().manual_seed(0)
n = 3000
a, b = torch.randn(n, 4, generator=g), torch.randn(n, 4, generator=g)
j = torch.randint(0, 4, (n,), generator=g)
A, Bm = a.reshape(n, 2, 2), b.reshape(n, 2, 2)
target = (A @ Bm).reshape(n, 4)[torch.arange(n), j]
choices = sys.argv[1] if len(sys.argv) > 1 else "relax"
for seed in range(int(sys.argv[2]) if len(sys.argv) > 2 else 3):
    p = Program("matmul7.c", device="cpu"); t0 = time.time()
    p.fit(a, b, j, target=target, steps=800, lr=0.05, choices=choices, restarts=1, seed=seed)
    err = torch.mean((p(a, b, j) - target) ** 2).item()
    print(f"{choices} seed {seed}: mean squared error {err:.4g} ({time.time() - t0:.0f}s){'  <- EXACT' if err < 1e-8 else ''}", flush=True)
