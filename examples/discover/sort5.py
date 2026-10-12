"""Discover a 9-comparator sorting network for 5 numbers from examples; prove it with the 0-1 principle, natively."""
import itertools, time
import torch
from learnc import Program
from learnc.native import Native

g = torch.Generator().manual_seed(0); n = 3000
x = torch.rand(n, 5, generator=g).argsort(1)                       # random orderings of 0..4
which = torch.randint(0, 5, (n,), generator=g)
target = x.sort(1).values[torch.arange(n), which]
p = Program("sort5.c", device="cpu"); t0 = time.time()
p.fit(x, which, target=target, steps=800, lr=0.1, choices="sample", samples=16, restarts=8, seed=0)
print(f"fitted in {time.time() - t0:.0f}s; steps: {list(zip(p.values['first'].tolist(), p.values['second'].tolist()))}")
from learnc.bake import bake
bake(p, "sort5.found.c"); nat = Native(Program("sort5.found.c"))
bits = torch.tensor(list(itertools.product([0, 1], repeat=5)))
ok01 = all(torch.equal(nat.run(bits, torch.full((32,), k)), bits.sort(1).values[:, k]) for k in range(5))
r = torch.randint(-10**6, 10**6, (5000, 5))
okr = all(torch.equal(nat.run(r, torch.full((5000,), k)), r.sort(1).values[:, k]) for k in range(5))
print(f"native build sorts all 32 zero/one inputs: {ok01} (0-1 principle: then it sorts every input); 5000 random inputs: {okr}")
