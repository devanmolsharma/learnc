"""Recover a hidden formula, (x0 + x3) * (x1 - x2), from 2000 examples; with both ways of learning choices."""
import time
import torch
from learnc import Program
from learnc.bake import bake

g = torch.Generator().manual_seed(0)
x = torch.randn(2000, 5, generator=g)
y = (x[:, 0] + x[:, 3]) * (x[:, 1] - x[:, 2])
for choices in ("sample", "relax", "auto"):
    p = Program("formula.c", device="cpu"); t0 = time.time()
    p.fit(x, target=y, steps=400, lr=0.1, choices=choices, samples=16, restarts=4)
    d = p.describe(); err = torch.mean((p(x) - y) ** 2).item()
    lf = p.values["leaf"].tolist()
    print(f"{choices}: (x{lf[0]} {d['op'][0]} x{lf[1]}) {d['op'][2]} (x{lf[2]} {d['op'][1]} x{lf[3]})   "
          f"error {err:.3g} ({time.time() - t0:.0f}s){'  <- exact' if err < 1e-8 else ''}", flush=True)
bake(p, "formula.found.c")
print([l for l in open("formula.found.c").read().splitlines() if l.startswith("learned")])
