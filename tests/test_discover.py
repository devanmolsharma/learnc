"""Searching for exact programs: integer coefficients (continuous, then snapped) and sorting networks (sampled choices)."""
import itertools, shutil, os
import torch
from learnc import Program

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "examples", "discover")


def test_continuous_choices_snap_to_exact_integers(tmp_path):
    p = tmp_path / "lin.c"
    p.write_text("#include <learnc.h>\nlearned_in(-3, 3) int c[4];\n"
                 "int f(const int x[4]) { int s = 0; for (int i = 0; i < 4; i++) s += c[i] * x[i]; return s; }\n")
    x = torch.randint(-20, 20, (1000, 4)); y = x @ torch.tensor([3, -2, 0, 1])
    prog = Program(str(p), device="cpu").fit(x, target=y.float(), steps=400, lr=0.05, choices="continuous")
    assert prog.values["c"].tolist() == [3, -2, 0, 1]


def test_sorting_network_is_discovered_and_correct(tmp_path):
    src = tmp_path / "sort4.c"; shutil.copy(os.path.join(EXAMPLES, "sort4.c"), src)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(2000, 4, generator=g).argsort(1); which = torch.randint(0, 4, (2000,), generator=g)
    prog = Program(str(src), device="cpu")
    prog.fit(x, which, target=x.sort(1).values[torch.arange(2000), which], steps=300, lr=0.1, choices="sample", samples=16, restarts=6)
    bits = torch.tensor(list(itertools.product([0, 1], repeat=4)))
    for k in range(4):                                  # 0-1 principle: sorting all 0/1 inputs means sorting everything
        assert torch.equal(prog(bits, torch.full((16,), k)).cpu(), bits.sort(1).values[:, k])


def test_levenberg_marquardt_fits_exactly(tmp_path):
    p = tmp_path / "poly.c"
    p.write_text("#include <learnc.h>\nlearned float c[3];\nlearned float k;\n"
                 "float f(float x, float y) { return c[0] * x * x + c[1] * x * y + c[2] + k * y; }\n")
    g = torch.Generator().manual_seed(0); x, y = torch.randn(200, generator=g), torch.randn(200, generator=g)
    target = 1.5 * x * x - 2.0 * x * y + 0.25 + 3.0 * y
    prog = Program(str(p), device="cpu").fit(x, y, target=target, steps=30, batch=200, optimizer="lm")
    assert torch.mean((prog(x, y) - target) ** 2) < 1e-10
    assert torch.allclose(prog.values["c"], torch.tensor([1.5, -2.0, 0.25]), atol=1e-4)
