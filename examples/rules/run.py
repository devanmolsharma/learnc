"""Recover a hidden 2-of-3 rule over 8 features from labelled examples; show the learned rule as C."""
import time
import torch
from learnc import Program
from learnc.bake import bake

g = torch.Generator().manual_seed(0)
f = torch.rand(4000, 8, generator=g)
# hidden rule: f[2] > 0.6, f[5] < 0.3, f[7] > 0.45 -> at least two of them
truth = lambda f: (((f[:, 2] > 0.6).long() + (f[:, 5] < 0.3).long() + (f[:, 7] > 0.45).long()) >= 2).long()
y = truth(f)
prog = Program("approve.c", device="cpu"); t0 = time.time()
prog.fit(f, target=y, steps=600, lr=0.05, samples=16, temperature=0.05)
print(f"fitted in {time.time() - t0:.0f}s")
for k in range(3):
    fe, ab, th = prog.values["feature"][k].item(), prog.values["above"][k].item(), prog.values["threshold"][k].item()
    print(f"   test {k}: f[{fe}] {'>' if ab else '<'} {th:.3f}")
ft = torch.rand(4000, 8, generator=g)
print(f"accuracy on new examples: {prog.accuracy(ft, target=truth(ft)):.4f}")
bake(prog, "approve.trained.c")
print(open("approve.trained.c").read().split("learned_in(0, 7)")[1].split("int approve")[0])
