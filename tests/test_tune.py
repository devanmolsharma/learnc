"""learnc.tune / `learnc tune`: constants of unmodified C code are tuned; the user's files stay untouched; the tuned
copy is plain C; changes that do not help are reverted."""
import subprocess, sys, textwrap
import numpy as np
import torch
import learnc

LIB = """
    #include <stdio.h>
    static const float kGain = 1.0f;          /* matters */
    static const int kWindow = 4;             /* matters (whole number) */
    static const float kUnused = 3.0f;        /* never read: must stay as it is */
    float smooth_peak(const float *x, int n) {
        float best = 0;
        for (int i = 0; i + kWindow <= n; i++) {
            float s = 0;
            for (int j = 0; j < kWindow; j++) s += x[i + j];
            if (s > best) best = s;
        }
        char buf[32]; snprintf(buf, sizeof buf, "%f", best * kGain);   /* library call: native engine */
        float r; sscanf(buf, "%f", &r);
        return r;
    }
"""
ENTRY = """
    float smooth_peak(const float *x, int n);
    float score(const float x[32]) { return smooth_peak(x, 32); }
"""


def files(tmp_path):
    lib = tmp_path / "peak.c"; lib.write_text(textwrap.dedent(LIB))
    entry = tmp_path / "score.c"; entry.write_text(textwrap.dedent(ENTRY))
    return str(entry), str(lib)


def data(entry, lib, n=600):
    x = torch.rand(n, 32, generator=torch.Generator().manual_seed(0))
    y = 1.25 * x.unfold(1, 7, 1).sum(-1).amax(1)                      # true: gain 1.25, window 7
    return x, y


def test_tune_function(tmp_path):
    entry, lib = files(tmp_path)
    before = open(lib).read()
    x, y = data(entry, lib)
    assert [c["name"] for c in learnc.candidates(lib)] == ["kGain", "kWindow", "kUnused"]
    r = learnc.tune(entry, [x], y, constants={lib: "all"}, sources=[lib], steps=150, lr=0.05, workers=2)
    assert open(lib).read() == before                                  # the user's file is untouched
    assert r.fit_report["validation_loss_final"] < 0.01 * r.fit_report["validation_loss_start"], r.report()
    assert int(r.values["kWindow"]) == 7 and abs(float(r.values["kGain"]) - 1.25) < 0.02, r.values
    assert float(r.values["kUnused"]) == 3.0                           # reverted: it did not help
    out = r.write(str(tmp_path / "tuned"))[0]
    text = open(out).read()
    assert "learned" not in text and "learnc" not in text
    subprocess.run(["gcc", "-c", out, "-o", str(tmp_path / "peak.o")], check=True)   # plain C


def test_tune_command_line(tmp_path):
    entry, lib = files(tmp_path)
    x, y = data(entry, lib)
    torch.save({"x": x, "output": y}, tmp_path / "data.pt")
    res = subprocess.run([sys.executable, "-m", "learnc.cli", "tune", entry, "--data", str(tmp_path / "data.pt"),
                          "--tune", f"{lib}:kGain,kWindow", "--source", lib, "--steps", "150", "--lr", "0.05",
                          "--workers", "2", "--out", str(tmp_path / "out")], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    assert "kWindow: 4  ->  7" in res.stdout, res.stdout[-2000:]
    assert (tmp_path / "out" / "peak.c").exists() and (tmp_path / "out" / "TUNING_REPORT.txt").exists()


HEADER = """
    #ifdef SCALE_IMPLEMENTATION
    float scale_sum(const float *x, int n) {
        static const float weights[4] = { 1.0f, 1.0f, 1.0f, 1.0f };   /* inside a function, in a header */
        float s = 0;
        for (int i = 0; i < n; i++) s += weights[i % 4] * x[i];
        return s;
    }
    #endif
"""


def test_tune_constant_inside_a_function_in_a_header(tmp_path):
    inc = tmp_path / "include"; inc.mkdir()
    header = inc / "scale.h"; header.write_text(textwrap.dedent(HEADER))
    entry = tmp_path / "score.c"
    entry.write_text('#include <stdio.h>\n#include "scale.h"\n'
                     'float score(const float x[8]) { char b[32]; snprintf(b, 32, "%d", 1); return scale_sum(x, 8); }\n')
    flags = ["-I", str(inc), "-DSCALE_IMPLEMENTATION"]
    found = {c["name"]: c for c in learnc.candidates(str(header), flags)}
    assert found["weights"]["function"][0] == "scale_sum"
    x = torch.rand(800, 8, generator=torch.Generator().manual_seed(1))
    w = torch.tensor([0.5, 2.0, -1.0, 3.0]).repeat(2)
    before = header.read_text()
    r = learnc.tune(str(entry), [x], (x * w).sum(1), constants={str(header): ["weights"]}, flags=flags,
                    steps=300, lr=0.05, keep_close=0, workers=2)
    assert header.read_text() == before
    assert np.allclose(r.values["weights"].numpy(), [0.5, 2.0, -1.0, 3.0], atol=0.02), r.values
    out = r.write(str(tmp_path / "tuned"))[0]
    text = open(out).read()
    assert "static const float weights[4] = { 0.5" in text and text.count("weights") == before.count("weights")


def test_build_cache_sees_header_edits(tmp_path):
    inc = tmp_path / "include"; inc.mkdir()
    (inc / "k.h").write_text("#define K 2.0f\n")
    src = tmp_path / "f.c"; src.write_text('#include "k.h"\nfloat f(float x) { return K * x; }\n')
    x = torch.ones(4)
    assert learnc.Program(str(src), device="cpu", flags=["-I", str(inc)]).native().run(x)[0].item() == 2.0
    (inc / "k.h").write_text("#define K 3.0f\n")
    assert learnc.Program(str(src), device="cpu", flags=["-I", str(inc)]).native().run(x)[0].item() == 3.0
