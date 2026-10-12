"""Write the fitted values into a copy of the C file: a plain C program that any compiler builds, no learnc needed."""
import re
import torch
from .build import learned_declarations

HEADER = ("/* learned values filled in by learnc */\n#define learned\n#define learned_in(lo, hi)\n#define learned_of(...)\n"
          "#define learned_op(...) int\n#define learned_num\nenum { ADD, SUB, MUL, DIV, MAX, MIN, LEFT, RIGHT };\n"
          "static inline float learnc_apply(int op, float a, float b) {\n  switch (op) {\n    case ADD: return a + b;\n"
          "    case SUB: return a - b;\n    case MUL: return a * b;\n    case DIV: return b != 0 ? a / b : 0;\n"
          "    case MAX: return a > b ? a : b;\n    case MIN: return a < b ? a : b;\n    case LEFT: return a;\n"
          "    default: return b;\n  }\n}\n")


def _literal(v, fp_bits):
    if not fp_bits: return str(int(v))
    s = f"{float(v):.9g}" if fp_bits == 32 else repr(float(v))          # 9 / 17 digits: the value round-trips exactly
    if not any(ch in s for ch in ".enai"): s += ".0"
    return s + ("f" if fp_bits == 32 else "")


def _braces(flat, shape, fmt):
    if not shape: return fmt(flat[0])
    step = len(flat) // shape[0]
    return "{" + ", ".join(_braces(flat[i * step:(i + 1) * step], shape[1:], fmt) for i in range(shape[0])) + "}"


def bake(prog, out_path, values=None, source=None):
    """Write `source` (default: the entry file) with the fitted values of its learned variables to `out_path`."""
    values = values if values is not None else prog.values
    if values is None: raise ValueError("nothing to bake: fit the program first (or pass values)")
    source = source or prog.path
    src = open(source, "rb").read().decode()
    spots = learned_declarations(source, getattr(prog, "flags", ()))
    edits = []
    for name, L in prog.learned.items():
        if name not in spots:
            if source == prog.path and name in open(source).read(): raise ValueError(f"cannot find the declaration of {name} in {source}")
            continue                                                       # declared in another file
        flat = torch.as_tensor(values[name]).reshape(-1).tolist()
        bits = L.etype[1] if L.etype[0] == "fp" else 0
        fmt = (lambda v: L.describe(v)) if L.ops else (lambda v: _literal(v, bits))      # operators by name
        text = _braces(flat, L.shape, fmt)
        s, e = spots[name]
        edits.append((s, e, text if s != e else " = " + text))
    data = src.encode()
    for s, e, text in sorted(edits, reverse=True): data = data[:s] + text.encode() + data[e:]
    out = re.sub(r'#\s*include\s*[<"]learnc\.h[>"][^\n]*\n?', HEADER, data.decode())
    out = re.sub(r"\n/\* added by learnc: setters for the tunable constants \*/\n(void learnc_set_[^\n]*\n)*", "\n", out)
    open(out_path, "w").write(out)
    return out_path
