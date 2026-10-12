"""One call to tune the constants of existing C code from examples; your files are never modified.

    result = learnc.tune("entry.c", inputs, target,              # entry.c: a function scoring one example
                         constants={"vad_core.c": ["kLocalThresholdQ", "kGlobalThresholdQ"]},
                         sources=["vad_core.c", "vad_gmm.c"], flags=["-I", "include"])
    print(result.report())
    result.write("tuned/")                                        # tuned copies of the files, plain C

Steps: the named file-level constants are made tunable in working copies (learnc.lift), the program is built,
examples are split into training and validation, the native engine tunes with early stopping on validation and a
pull toward the original values (`keep_close`), and the report compares validation loss before and after.
"""
import os, tempfile
import numpy as np
import torch
from .lift import lift, apply
from .program import Program
from .bake import bake


def _split(inputs, target, fraction, seed):
    n = len(inputs[0])
    if not fraction: return inputs, target, None
    idx = np.random.default_rng(seed).permutation(n); k = max(1, int(round(n * fraction)))
    val, tr = idx[:k], idx[k:]
    pick = lambda x, i: [x[j] for j in i] if isinstance(x, (list, tuple)) else torch.as_tensor(x)[i]
    tgt = (lambda i: None) if target is None else (lambda i: torch.as_tensor(target)[i])
    return [pick(x, tr) for x in inputs], tgt(tr), ([pick(x, val) for x in inputs], tgt(val))


class TuneResult:
    def __init__(self, prog, lifted, originals, report, flags=()):
        self.program, self.lifted, self.originals, self.fit_report, self.flags = prog, lifted, originals, report, flags
        self.values = prog.values

    def report(self):
        r = self.fit_report
        a, b = r["validation_loss_start"], r["validation_loss_final"]
        change = f"{100 * (b / a - 1):+.1f}%" if a > 1e-9 and b >= 0 else f"{b - a:+.6g}"   # % only when meaningful
        lines = [f"validation loss: {a:.6g} with the original constants -> {b:.6g} tuned ({change})"]
        for name, L in self.program.learned.items():
            old = np.asarray(L.init).reshape(-1); new = np.asarray(self.values[name]).reshape(-1)
            fmt = (lambda v: f"{v:g}") if L.etype[0] == "fp" else (lambda v: f"{int(round(v))}")
            if np.allclose(old, new): lines.append(f"  {name}: unchanged"); continue
            lines.append(f"  {name}: {', '.join(fmt(v) for v in old)}  ->  {', '.join(fmt(v) for v in new)}")
        return "\n".join(lines)

    def write(self, out_dir):
        """Write tuned copies of every file whose constants were tuned: your file with only the changed values replaced
        (a minimal diff; any compiler builds it). Constants you marked `learned` yourself are baked instead."""
        os.makedirs(out_dir, exist_ok=True); written = []
        values = {k: v.numpy() for k, v in self.values.items()}
        originals = {k: L.init for k, L in self.program.learned.items()}
        for original, working in self.lifted.items():
            out = os.path.join(out_dir, os.path.basename(original))
            if working == original: bake(self.program, out, source=working)
            else: apply(original, values, out, self.flags, originals)
            written.append(out)
        with open(os.path.join(out_dir, "TUNING_REPORT.txt"), "w") as f: f.write(self.report() + "\n")
        return written


def tune(entry_file, inputs, target=None, *, entry=None, constants=None, sources=(), flags=(), objective=None,
         validation=0.2, keep_close=0.05, steps=200, batch=256, lr=0.02, restarts=2, workers=None, seed=0,
         work_dir=None, verbose=False, **fit_options):
    """Tune constants of existing C code. See the module docstring.

    entry_file: C file with the function that takes ONE example (numbers, arrays, strings) and returns a number.
    constants: {file: [constant names] or "all"}: constants to tune (made tunable in working copies). The file can
      be a source of the program or a header it includes (include it with -I, not from the entry file's folder).
      Constants already marked `learned` in the code are tuned too.
    objective: "match" (output should equal target) or "minimize" (output is a cost); default: "minimize" when no
      target is given, else "match".
    validation: fraction of examples held out to choose the final values (early stopping).
    keep_close: strength of the pull toward the original values (0: none)."""
    if not isinstance(inputs, (list, tuple)) or (inputs and isinstance(inputs[0], str)): inputs = [inputs]
    objective = objective or ("minimize" if target is None else "match")
    work_dir = work_dir or tempfile.mkdtemp(prefix="learnc-tune-")
    files = [os.path.abspath(entry_file)] + [os.path.abspath(s) for s in sources]
    lifted, flags = {}, list(flags)
    for f, names in (constants or {}).items():
        f = os.path.abspath(f)
        header = f.endswith(".h")
        if f not in files and not header: raise ValueError(f"{f} is not among the program's files ({files})")
        out = os.path.join(work_dir, os.path.basename(f))
        lift(f, names, out, flags)
        lifted[f] = out
        if header: flags = ["-I", work_dir] + flags                # #include "x.h" now finds the tunable copy first
        flags += ["-I", os.path.dirname(f)]                       # the copy still finds the original's headers
    files = [lifted.get(f, f) for f in files]
    prog = Program(files[0], sources=files[1:], entry=entry, device="cpu", flags=flags)
    if not prog.learned: raise ValueError("nothing to tune: name constants in `constants` or mark them `learned`")
    tr_in, tr_y, val = _split(inputs, target, validation, seed)
    prog.fit(*tr_in, target=tr_y, engine="native", objective=objective, validation=val, keep_close=keep_close,
             steps=steps, batch=batch, lr=lr, restarts=restarts, workers=workers, seed=seed, verbose=verbose, **fit_options)
    # constants marked `learned` directly in the user's own entry file are baked from there too
    if os.path.abspath(entry_file) not in lifted and any(n for n in prog.learned if n in open(entry_file).read()):
        lifted[os.path.abspath(entry_file)] = os.path.abspath(entry_file)
    return TuneResult(prog, lifted, files, prog.fit_report, flags)
