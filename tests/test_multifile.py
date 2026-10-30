"""Learned tables in their own source file, fitted through a cost computed in C: objective='minimize', log space,
per-variable learning rates, penalty, validation, callback, native runs and writing the tables file back."""
import subprocess, textwrap
import torch
from learnc import Program


def write(tmp_path, name, src):
    p = tmp_path / name; p.write_text(textwrap.dedent(src)); return str(p)


HEADER = """
    #include <learnc.h>
    extern learned float freq[3];
    extern learned float gain[3];
"""

TABLES = """
    #include "tables.h"
    learned float freq[3] = {100.0f, 100.0f, 100.0f};
    learned float gain[3] = {0.0f, 0.0f, 0.0f};
"""

# a resonance per class: the cost is the squared log-spectral distance to a measured response at 8 points
MODEL = """
    #include <math.h>
    #include "tables.h"
    float distance(int k, const float target[8]) {
        float err = 0.0f;
        for (int i = 0; i < 8; i++) {
            float f = 250.0f * (i + 1), x = (f - freq[k]) / (0.2f * freq[k]);
            float d = gain[k] - logf(1.0f + x * x) - target[i];
            err += d * d;
        }
        return err / 8;
    }
"""

TRUE_F, TRUE_G = torch.tensor([400.0, 900.0, 1600.0]), torch.tensor([1.0, -0.5, 0.3])


def data(n, seed):
    g = torch.Generator().manual_seed(seed)
    k = torch.randint(0, 3, (n,), generator=g)
    f = 250.0 * torch.arange(1, 9)
    x = (f[None] - TRUE_F[k][:, None]) / (0.2 * TRUE_F[k][:, None])
    return k, TRUE_G[k][:, None] - torch.log1p(x * x) + 0.01 * torch.randn(n, 8, generator=g)


def program(tmp_path):
    write(tmp_path, "tables.h", HEADER)
    tables = write(tmp_path, "tables.c", TABLES)
    return Program(write(tmp_path, "model.c", MODEL), sources=[tables], flags=["-I", str(tmp_path)]), tables


def test_minimize_a_cost_with_tables_in_another_file(tmp_path):
    p, _ = program(tmp_path)
    assert set(p.learned) == {"freq", "gain"} and not p.native_only
    k, t = data(600, 0)
    vk, vt = data(200, 1)
    seen = []
    p.fit(k, t, target=None, objective="minimize", steps=600, batch=128, lr=0.03, space={"freq": "log"},
          lr_scale={"gain": 2.0}, penalty=lambda v: 1e-6 * (v["gain"] ** 2).sum(), validation=((vk, vt), None),
          eval_every=100, callback=lambda step, values, vl: seen.append((step, vl)))
    assert torch.allclose(p.values["freq"], TRUE_F, rtol=0.03)
    assert torch.allclose(p.values["gain"], TRUE_G, atol=0.05)
    assert [s for s, _ in seen] == [100, 200, 300, 400, 500, 600] and all(v is not None for _, v in seen)
    assert min(v for _, v in p.validation_history) < 0.01


def test_native_engine_sets_tables_of_another_file(tmp_path):
    p, _ = program(tmp_path)
    k, t = data(50, 2)
    values = {"freq": TRUE_F, "gain": TRUE_G}
    native = p.native().run(k, t, values=values)
    with torch.no_grad(): runner = p.run(k, t, learned_tensors={"@freq": TRUE_F, "@gain": TRUE_G})
    assert torch.allclose(native.float(), runner.float(), rtol=1e-4, atol=1e-6)


def test_write_sources_bakes_the_tables_file(tmp_path):
    p, tables = program(tmp_path)
    p.values = {"freq": TRUE_F, "gain": TRUE_G}
    out = p.write_sources(str(tmp_path / "out"))
    assert [o.split("/")[-1] for o in out] == ["tables.c"]
    text = open(out[0]).read()
    assert "400.0f, 900.0f, 1600.0f" in text and "learned float" in text
    exe = tmp_path / "check"
    main = write(tmp_path, "main.c", """
        #include <stdio.h>
        extern float freq[3];
        int main(void) { printf("%g\\n", freq[2]); return 0; }
    """)
    subprocess.run(["gcc", "-w", "-I", str(tmp_path), "-I", str(tmp_path / "out"), main, out[0], "-o", str(exe)], check=True)
    assert subprocess.run([str(exe)], capture_output=True, text=True).stdout.strip() == "1600"


def test_bounded_spaces_keep_values_in_range(tmp_path):
    p, _ = program(tmp_path)
    k, t = data(600, 3)
    p.fit(k, t, target=None, objective="minimize", steps=800, batch=128, lr=0.05,
          space={"freq": ("logrange", 50, 1000), "gain": ("range", [-0.2, -1.0, -1.0], [0.6, 1.0, 1.0])})
    assert (p.values["freq"] >= 50).all() and (p.values["freq"] <= 1000).all()
    assert p.values["freq"][2] > 850                            # the true 1600 is out of range: it heads for the bound
    assert torch.allclose(p.values["freq"][1], TRUE_F[1], rtol=0.05)     # within its bounds: found
    assert 0.5 < p.values["gain"][0] <= 0.6                     # true 1.0, bounded at 0.6


def test_loop_report_names_what_stops_vectorising(tmp_path):
    write(tmp_path, "tables.h", HEADER)
    tables = write(tmp_path, "tables.c", TABLES)
    src = MODEL.replace("float distance(", """
        __attribute__((noinline)) float bump(float x) { return logf(1.0f + x * x); }
        float slow(int k, const float target[8]) {
            float err = 0.0f;
            for (int i = 0; i < 8; i++) { float d = gain[k] - bump((250.0f * (i + 1) - freq[k]) / freq[k]) - target[i]; err += d * d; }
            return err;
        }
        float distance(""")
    fast = Program(write(tmp_path, "m1.c", MODEL), sources=[tables], flags=["-I", str(tmp_path)])
    assert [r[2] for r in fast.loop_report()] == ["all at once (8 iterations)"]
    slow = Program(write(tmp_path, "m2.c", src), sources=[tables], entry="slow", flags=["-I", str(tmp_path)])
    (fn, _, how), = slow.loop_report()
    assert fn == "slow" and how.startswith("step by step: a call to bump")


def test_array_whose_initializer_ends_in_zeros(tmp_path):
    # clang emits such a global as <{ [2 x float], [6 x float] zeroinitializer }>: still one array of 8
    p = Program(write(tmp_path, "z.c", """
        #include <learnc.h>
        learned float w[8] = {1.0f, 2.0f};
        float pick(int k) { return w[k]; }
    """))
    assert p.learned["w"].shape == (8,) and not p.flat               # a plain array: the fast runner
    k = torch.arange(8); y = torch.arange(8.0) * 0.5
    p.fit(k, target=y, steps=300, lr=0.2)
    assert p.values["w"].shape == (8,) and torch.allclose(p.values["w"], y, atol=0.05)
