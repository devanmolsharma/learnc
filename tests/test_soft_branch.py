"""Branches whose condition depends on learned numbers: both sides run while training, and gradients reach the condition."""
import textwrap
import torch
from learnc import Program

SRC = """
    #include <learnc.h>
    learned float t;
    learned float gain = 1.0f;
    float f(float x) {
        float hi[4], lo[4];
        for (int i = 0; i < 4; i++) { hi[i] = 0; lo[i] = 0; }
        if (x * gain > t) {                       /* a real branch: each side writes its own array */
            for (int i = 0; i < 4; i++) hi[i] = x + i;
        } else {
            for (int i = 0; i < 4; i++) lo[i] = -x * (i + 1);
        }
        float s = 0;
        for (int i = 0; i < 4; i++) s += hi[i] + lo[i];
        return s;
    }
"""


def truth(x, t=0.4): return torch.where(x > t, 4 * x + 6, -10 * x)


def test_branch_stays_a_branch_in_ir(tmp_path):
    p = tmp_path / "br.c"; p.write_text(textwrap.dedent(SRC))
    prog = Program(str(p))
    f = prog.module.functions["@f"]
    conds = {i.name for b in f.blocks for i in b.insts if i.op == "fcmp"}
    assert any(b.term.op == "br" and b.term.args and b.term.args[0][1] in conds for b in f.blocks)


def test_threshold_learned_through_a_branch(tmp_path):
    p = tmp_path / "br.c"; p.write_text(textwrap.dedent(SRC))
    prog = Program(str(p), device="cpu")
    x = torch.rand(2000) * 2 - 1
    exact_before = prog(x)
    assert torch.allclose(exact_before, truth(x, 0.0))            # starts at t = 0
    prog.fit(x, target=truth(x), steps=400, lr=0.02, temperature=0.05)
    threshold = prog.values["t"].item() / prog.values["gain"].item()   # only the ratio matters (x * gain > t)
    assert abs(threshold - 0.4) < 0.02, prog.values
    assert ((prog(x) - truth(x)).abs() < 1e-4).float().mean() > 0.99   # the target jumps at t: count exact hits
