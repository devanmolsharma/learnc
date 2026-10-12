"""Fitting learned variables from examples, and baking them into a standalone C file."""
import ctypes, subprocess, textwrap
import numpy as np
import pytest
import torch
from learnc import Program
from learnc.bake import bake


def write(tmp_path, name, src):
    p = tmp_path / name; p.write_text(textwrap.dedent(src)); return str(p)


LINEAR = """
    #include <learnc.h>
    learned float w[4][6];
    learned float bias[4];
    int classify(const float x[6]) {
        int best = 0; float bs = -1e30f;
        for (int c = 0; c < 4; c++) {
            float s = bias[c];
            for (int i = 0; i < 6; i++) s += w[c][i] * x[i];
            if (s > bs) { bs = s; best = c; }
        }
        return best;
    }
"""


def linear_data(n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    pat = torch.randn(4, 6, generator=torch.Generator().manual_seed(7)) * 2
    y = torch.randint(0, 4, (n,), generator=g)
    return pat[y] + 0.5 * torch.randn(n, 6, generator=g), y


def test_fit_weights_through_argmax_loop(tmp_path):
    p = Program(write(tmp_path, "lin.c", LINEAR))
    x, y = linear_data()
    p.fit(x, target=y, steps=150, lr=0.05)
    xt, yt = linear_data(1000, seed=1)
    assert p.accuracy(xt, target=yt) > 0.95


def test_fit_operation_choices(tmp_path):
    p = Program(write(tmp_path, "ops.c", """
        #include <learnc.h>
        learned_in(0, 3) int op[3];
        static int apply(int o, int a, int b) {
            switch (o) { case 0: return a + b; case 1: return a - b; case 2: return a * b; default: return a > b ? a : b; }
        }
        int f(int a, int b, int c) { return apply(op[2], apply(op[0], a, b), apply(op[1], b, c)); }
    """))
    a, b, c = (torch.randint(-9, 10, (1000,), generator=torch.Generator().manual_seed(k)) for k in range(3))
    target = torch.maximum(a * b, b - c)                     # op = [2, 1, 3]
    p.fit(a, b, c, target=target, steps=200, lr=0.1)
    assert p.values["op"].tolist() == [2, 1, 3] and p.accuracy(a, b, c, target=target) == 1.0


def test_fit_threshold_and_mixed(tmp_path):
    p = Program(write(tmp_path, "rule.c", """
        #include <learnc.h>
        learned float t;
        learned float scale = 1.0f;
        learned_in(0, 1) int flip;
        int alarm(float temp, float humid) {
            int hot = temp * scale > t;
            return flip ? !hot : (hot && humid > 0.5f);
        }
    """))
    g = torch.Generator().manual_seed(0)
    temp, humid = torch.rand(3000, generator=g) * 2, torch.rand(3000, generator=g)
    y = ((temp > 1.3) & (humid > 0.5)).long()
    p.fit(temp, humid, target=y, steps=400, lr=0.05, temperature=0.05)
    assert p.values["flip"].item() == 0 and p.accuracy(temp, humid, target=y) > 0.97


def test_bake_gives_a_standalone_c_file(tmp_path):
    p = Program(write(tmp_path, "lin.c", LINEAR))
    x, y = linear_data()
    p.fit(x, target=y, steps=60, lr=0.05)
    out = bake(p, str(tmp_path / "lin.trained.c"))
    text = open(out).read()
    assert "learnc.h" not in text and "w[4][6] = {{" in text
    so = tmp_path / "lin.so"
    subprocess.run(["gcc", "-O2", "-shared", "-fPIC", out, "-o", str(so)], check=True)     # any compiler, no learnc
    lib = ctypes.CDLL(str(so)); lib.classify.restype = ctypes.c_int
    xt, _ = linear_data(300, seed=3); xs = np.ascontiguousarray(xt.numpy().astype(np.float32))
    native = [lib.classify(xs[i].ctypes.data_as(ctypes.c_void_p)) for i in range(len(xs))]
    assert native == p(xt).cpu().tolist()


def test_fit_without_learned_variables(tmp_path):
    p = Program(write(tmp_path, "plain.c", "int f(int a) { return a + 1; }\n"))
    with pytest.raises(ValueError, match="no learned variables"):
        p.fit(torch.arange(5), target=torch.arange(5) + 1, steps=1)
