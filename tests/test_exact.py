"""The batched IR runner must compute exactly what the natively compiled program computes, example by example."""
import textwrap
import pytest
import torch
from learnc import Program, RunError, BuildError
from learnc.native import Native

N = 300

PROGRAMS = {
    "int_arithmetic": ("""
        int f(int a, int b) {
            int q = b != 0 ? a / b : 0, r = b != 0 ? a % b : 0;
            unsigned u = (unsigned)a * 2654435761u;
            short s = (short)(a * 1000);
            signed char c = (signed char)a;
            long long big = (long long)a * a * a;
            return q * 7 + r - (a << 3) + (a >> 2) + (int)(u >> 7) + s + c + (int)(big % 1000) + (a & b) - (a | b) + (a ^ b);
        }""", lambda: [torch.randint(-100000, 100000, (N,)), torch.randint(-50, 50, (N,))]),
    "unsigned_ops": ("""
        unsigned f(unsigned a, unsigned b) {
            unsigned x = a / (b | 1u) + a % (b | 1u);
            if (a > b) x += 7;               /* unsigned comparison */
            return x ^ (a >> 3) ^ (b << 2);
        }""", lambda: [torch.randint(0, 2**31 - 1, (N,)), torch.randint(0, 2**31 - 1, (N,))]),
    "floats_and_math": ("""
        #include <math.h>
        double f(float x, double y) {
            float a = sqrtf(fabsf(x)) + expf(-x * x) + tanhf(x) + floorf(x * 3.5f) + fmaxf(x, 0.25f);
            double b = log(1.0 + y * y) + pow(fabs(y), 1.5) + sin(y) * cos(y) + ceil(y);
            int k = (int)(x * 10);
            return a + b + k + (float)k / 3.0f;
        }""", lambda: [torch.randn(N) * 3, torch.randn(N, dtype=torch.float64) * 2]),
    "loops_data_dependent": ("""
        int f(int n, int m) {
            int s = 0, i = 0;
            while (i < n) { s += i * m; i++; if (s > 5000) break; }
            do { s -= 3; } while (s > 1000);
            for (int k = 0; k < 10; k++) { if (k == m % 10) continue; s += k; }
            return s;
        }""", lambda: [torch.randint(0, 60, (N,)), torch.randint(-20, 20, (N,))]),
    "recursion": ("""
        static int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
        static int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }
        int f(int a, int b) { return fib(a % 15) * 1000 + gcd(a, b); }""",
                  lambda: [torch.randint(1, 200, (N,)), torch.randint(1, 200, (N,))]),
    "arrays_and_tables": ("""
        static const int TABLE[10] = {5, 3, 8, 1, 9, 2, 7, 0, 4, 6};
        int f(const int x[8]) {
            int buf[8], hist[10] = {0};
            for (int i = 0; i < 8; i++) { buf[i] = TABLE[(x[i] % 10 + 10) % 10]; hist[buf[i]]++; }
            for (int i = 0; i < 7; i++)
                for (int j = 0; j < 7 - i; j++)
                    if (buf[j] > buf[j + 1]) { int t = buf[j]; buf[j] = buf[j + 1]; buf[j + 1] = t; }
            int best = 0;
            for (int k = 1; k < 10; k++) if (hist[k] > hist[best]) best = k;
            return buf[0] * 1000 + buf[7] * 100 + buf[3] * 10 + best;
        }""", lambda: [torch.randint(-100, 100, (N, 8))]),
    "two_d_and_pointers": ("""
        static float dot(const float *a, const float *b, int n) { float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
        static void scale(float *v, int n, float k) { for (float *p = v; p < v + n; p++) *p *= k; }
        float f(const float img[4][6]) {
            float row[6], col[4];
            for (int j = 0; j < 6; j++) row[j] = img[1][j];
            scale(row, 6, 0.5f);
            for (int i = 0; i < 4; i++) col[i] = img[i][2];
            return dot(row, img[3], 6) - dot(col, col, 4) + row[5];
        }""", lambda: [torch.randn(N, 4, 6)]),
    "structs": ("""
        struct P { int x, y; };
        static int area(struct P a, struct P b) { int w = b.x - a.x, h = b.y - a.y; return w * h; }
        int f(int a, int b) { struct P p = {a, b}, q = {a + 3, b * 2}; struct P box[2] = {p, q}; return area(box[0], box[1]) + box[1].y; }""",
                lambda: [torch.randint(-50, 50, (N,)), torch.randint(-50, 50, (N,))]),
    "chars_and_bools": ("""
        #include <stdbool.h>
        int f(const unsigned char s[12], bool flip) {
            int vowels = 0, upper = 0;
            for (int i = 0; i < 12 && s[i]; i++) {
                char c = s[i];
                bool v = c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
                vowels += v;
                upper += (c >= 'A' && c <= 'Z');
            }
            return flip ? upper * 100 + vowels : vowels * 100 + upper;
        }""", lambda: [torch.randint(0, 128, (N, 12)), torch.randint(0, 2, (N,))]),
    "switch_and_goto": ("""
        int f(int op, int a, int b) {
            int r;
            switch (op) {
                case 0: r = a + b; break;
                case 1: r = a - b; break;
                case 2: case 3: r = a * b; break;
                case 7: r = a > b ? a : b; break;
                default: r = -1;
            }
            if (r < 0) goto negative;
            return r;
        negative:
            return -r * 10;
        }""", lambda: [torch.randint(0, 9, (N,)), torch.randint(-30, 30, (N,)), torch.randint(-30, 30, (N,))]),
    "learned_globals": ("""
        #include <learnc.h>
        learned float w[3][5];
        learned float bias[3];
        learned_in(0, 3) int op[4];
        int f(const float x[5], int k) {
            int best = 0; float bs = -1e30f;
            for (int c = 0; c < 3; c++) {
                float s = bias[c];
                for (int i = 0; i < 5; i++) s += w[c][i] * x[i];
                if (s > bs) { bs = s; best = c; }
            }
            int v = k;
            for (int j = 0; j < 4; j++)
                switch (op[j]) { case 0: v += best; break; case 1: v -= 2; break; case 2: v *= 2; break; default: v = v > 5 ? v : 5; }
            return v;
        }""", lambda: [torch.randn(N, 5), torch.randint(-10, 10, (N,))]),
}

VALUES = {"learned_globals": lambda: {"w": torch.randn(3, 5), "bias": torch.randn(3), "op": torch.randint(0, 4, (4,))}}


def build(tmp_path, name, src, entry="f"):
    p = tmp_path / f"{name}.c"; p.write_text(textwrap.dedent(src)); return Program(str(p), entry=entry)


@pytest.mark.parametrize("name", list(PROGRAMS))
def test_runner_matches_native(tmp_path, name):
    src, gen = PROGRAMS[name]
    torch.manual_seed(0)
    prog = build(tmp_path, name, src)
    values = VALUES.get(name, lambda: None)()
    inputs = gen()
    got = prog(*inputs, values=values).cpu()
    want = Native(prog).run(*inputs, values=values)
    if want.is_floating_point():
        assert torch.allclose(got.double(), want.double(), rtol=1e-5, atol=1e-5), (got[:5], want[:5])
    else:
        bad = (got.long() != want.long()).sum().item()
        assert bad == 0, f"{bad}/{N} differ: runner {got[:8].tolist()} native {want[:8].tolist()}"


def test_string_function_runs_in_flat_memory(tmp_path):
    prog = build(tmp_path, "lib", """
        #include <string.h>
        int f(const char s[8]) { char buf[8]; for (int i = 0; i < 8; i++) buf[i] = s[i]; buf[3] = 0; return (int)strlen(buf); }""")
    x = torch.randint(1, 100, (4, 8)); x[1, 1] = 0
    assert prog.flat and not prog.native_only
    assert prog(x).tolist() == [3, 1, 3, 3] == Native(prog).run(x).tolist()


def test_unbuilt_library_function_falls_back_to_native(tmp_path):
    prog = build(tmp_path, "lib2", """
        #include <stdlib.h>
        int f(const char s[8]) { char buf[9]; for (int i = 0; i < 8; i++) buf[i] = s[i]; buf[8] = 0; return atoi(buf); }""")
    x = torch.tensor([[ord(c) for c in "12345678"], [ord(c) for c in "00000042"]])
    assert prog.native_only and ("atoi" in prog.native_reason or "strtol" in prog.native_reason)
    assert prog(x).tolist() == [12345678, 42]
    with pytest.raises(RunError, match="atoi|strtol"):
        prog.run(x)


def test_learned_int_needs_range(tmp_path):
    with pytest.raises(BuildError, match="learned_in"):
        build(tmp_path, "norange", """
            #include <learnc.h>
            learned int k;
            int f(int a) { return a + k; }""")


def test_library_source_files(tmp_path):
    (tmp_path / "mylib.c").write_text("int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }\n")
    p = tmp_path / "main.c"
    p.write_text("int clampi(int v, int lo, int hi);\nint f(int a) { return clampi(a, -5, 5) * 3; }\n")
    prog = Program(str(p), sources=[str(tmp_path / "mylib.c")], entry="f")
    x = torch.randint(-20, 20, (50,))
    assert torch.equal(prog(x).cpu(), (x.clamp(-5, 5) * 3)) and torch.equal(Native(prog).run(x), x.clamp(-5, 5) * 3)


LOOPS = """
    #include <learnc.h>
    learned float w[8][40];
    int f(const float x[40], const int k[40]) {
        float best = -1e30f; int arg = 0, total = 0, bits = 0;
        for (int c = 0; c < 8; c++) {
            float s = 0.0f, p = 1.0f;
            for (int i = 0; i < 40; i++) s += w[c][i] * x[i];            /* float sum: must stay bit-exact */
            for (int i = 0; i < 40; i++) p *= 1.0f + 0.01f * x[i];       /* float product */
            for (int i = 0; i < 40; i++) total += k[i] * (c + 1);         /* int sum */
            for (int i = 0; i < 40; i++) bits ^= k[i] << (i % 5);         /* xor */
            if (s + p > best) { best = s + p; arg = c; }
        }
        return arg * 100000 + (total % 1000) * 10 + (bits & 7);
    }
"""


def test_vectorized_loops_are_exact(tmp_path):
    prog = build(tmp_path, "loops", LOOPS)
    torch.manual_seed(1)
    values = {"w": torch.randn(8, 40)}
    x, k = torch.randn(N, 40), torch.randint(-1000, 1000, (N, 40))
    fast = prog(x, k, values=values).cpu()
    prog.vectorize = False
    slow = prog(x, k, values=values).cpu()
    native = Native(prog).run(x, k, values=values)
    assert torch.equal(fast, slow) and torch.equal(fast.long(), native.long())
    from learnc.vectorize import find_loops, plan_loop
    f = prog.module.functions["@f"]
    inner = [L for L in find_loops(f).values() if L.parent is not None]
    assert len(inner) == 4 and all(plan_loop(prog.module, f, L) for L in inner)      # all four inner loops qualify


NESTS = """
    #include <learnc.h>
    learned float w[6][16];
    float f(const float img[8][8], const int k[8]) {
        float pooled[16], h[6];
        for (int y = 0; y < 4; y++)                       /* nest with stores and a total running through 2 levels */
            for (int x = 0; x < 4; x++) {
                float s = 0;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) s += img[2 * y + dy][2 * x + dx];
                pooled[y * 4 + x] = s * 0.25f;
            }
        for (int j = 0; j < 6; j++) {                     /* matrix-vector product with stores */
            float s = 0;
            for (int i = 0; i < 16; i++) s += w[j][i] * pooled[i];
            h[j] = s > 0 ? s : 0;
        }
        int t = 0;
        for (int a = 0; a < 8; a++)                       /* integer total across two levels */
            for (int b = 0; b < 3; b++) t += k[a] * (b + 1) - a;
        float out = 0;
        for (int j = 0; j < 6; j++) out += h[j] * (j + 1);
        return out + t;
    }
"""


def test_vectorized_nests_are_exact(tmp_path):
    prog = build(tmp_path, "nests", NESTS)
    torch.manual_seed(2)
    values = {"w": torch.randn(6, 16)}
    img, k = torch.randn(N, 8, 8), torch.randint(-100, 100, (N, 8))
    fast = prog(img, k, values=values).cpu()
    prog.vectorize = False
    slow = prog(img, k, values=values).cpu()
    native = Native(prog).run(img, k, values=values)
    assert torch.equal(fast, slow) and torch.equal(fast, native.float())
    from learnc.vectorize import find_loops, plan_loop
    f = prog.module.functions["@f"]
    outer = [L for L in find_loops(f).values() if L.parent is None]
    assert len(outer) == 4 and all(plan_loop(prog.module, f, L) for L in outer)       # every nest qualifies
