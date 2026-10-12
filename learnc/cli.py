"""learnc fit PROGRAM.c --data train.pt   |   learnc run PROGRAM.c --data test.pt   |   learnc show PROGRAM.c
learnc constants FILE.c   |   learnc tune ENTRY.c --data data.pt --tune FILE.c:NAME,NAME --source FILE.c --out tuned/

`constants` lists the file-level constants of existing C code that `tune` can tune; `tune` tunes them (your files are
never modified) and writes tuned copies plus TUNING_REPORT.txt to --out.

Data: a torch .pt file holding a dict with one tensor per parameter of the entry function (examples along the first
dimension) and `output` with the expected results.
`fit` writes PROGRAM.trained.c (your program with the learned values filled in: plain C, builds with any compiler)
and PROGRAM.values.pt (the values, to continue training or to `run` with).
"""
import argparse, os, shlex, time
import torch
from .bake import bake
from .program import Program


def _data(path, prog, need_output):
    d = torch.load(path, map_location="cpu", weights_only=True)
    names = [n for n, _ in prog.params]
    missing = [n for n in names if n not in d]
    if missing: raise SystemExit(f"{path}: no field for parameter(s) {missing} of {prog.entry}; fields: {sorted(d)}")
    if need_output and "output" not in d: raise SystemExit(f"{path}: needs an `output` field")
    return [d[n] for n in names], d.get("output")


def _report(prog, x, y):
    if y is None: return "no outputs to compare with"
    out = torch.cat([prog(*[a[i:i + 4096] for a in x]).cpu() for i in range(0, len(y), 4096)])
    if out.is_floating_point() or y.is_floating_point():
        return f"mean squared error {torch.mean((out.double() - y.double()) ** 2).item():.6f}"
    return f"accuracy {(out == y.to(out.dtype)).float().mean().item():.4f} on {len(y)} examples"


def _values_path(p): return os.path.splitext(p)[0] + ".values.pt"


def _join_cflags(argv):
    """`--cflags -DX` would look like an option to argparse: join it into `--cflags=-DX`."""
    out = []
    for a in argv:
        if out and out[-1] == "--cflags": out[-1] = f"--cflags={a}"
        else: out.append(a)
    return out


def main(argv=None):
    import sys
    argv = _join_cflags(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="learnc", description="Fit the `learned` variables of a C program to examples.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "run", "show"):
        p = sub.add_parser(name); p.add_argument("program")
        p.add_argument("--entry", help="entry function (default: the last function in the file)")
        p.add_argument("--source", action="append", default=[], help="source file of a library the program uses (repeatable)")
        p.add_argument("--device", help="cpu or cuda (default: cuda if available)")
        if name != "show": p.add_argument("--data", required=True)
        if name == "fit":
            p.add_argument("--val", help="validation data"); p.add_argument("--steps", type=int, default=1000)
            p.add_argument("--batch", type=int, default=128); p.add_argument("--lr", type=float, default=1e-2)
            p.add_argument("--samples", type=int, default=8, help="settings tried per step for whole-number learned variables")
            p.add_argument("--temperature", type=float, default=1.0, help="softness of comparisons that depend on learned numbers")
            p.add_argument("--resume", action="store_true", help="start from PROGRAM.values.pt")
            p.add_argument("--seed", type=int, default=0)
        if name == "run": p.add_argument("--values", help="values file (default PROGRAM.values.pt if it exists)")
    p = sub.add_parser("constants", help="list the constants of a C file that `tune` can tune"); p.add_argument("file")
    p.add_argument("--cflags", default="", help='compiler flags, e.g. "-I include -DNDEBUG"')
    p = sub.add_parser("tune", help="tune constants of existing C code (files are never modified)")
    p.add_argument("program", help="C file with the entry function that scores or computes ONE example")
    p.add_argument("--data", required=True); p.add_argument("--entry")
    p.add_argument("--tune", action="append", default=[], metavar="FILE.c:NAME,NAME|all",
                   help="constants to tune (repeatable); `learned` variables in the code are tuned too")
    p.add_argument("--source", action="append", default=[], help="other source file of the program (repeatable)")
    p.add_argument("--cflags", default="", help='compiler flags, e.g. "-I include -DNDEBUG"')
    p.add_argument("--minimize", action="store_true", help="the entry returns a cost to minimise (default when the data has no `output`)")
    p.add_argument("--validation", type=float, default=0.2, help="share of examples held out to pick the final values")
    p.add_argument("--keep-close", type=float, default=0.05, help="pull toward the original values (0: none)")
    p.add_argument("--steps", type=int, default=200); p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.02); p.add_argument("--workers", type=int)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default="tuned")
    a = ap.parse_args(argv)
    if a.cmd == "constants": return _constants(a)
    if a.cmd == "tune": return _tune(a)
    prog = Program(a.program, sources=a.source, entry=a.entry, device=a.device)
    if a.cmd == "show":
        print(f"entry: {prog.entry}({', '.join(n for n, _ in prog.params)})")
        for L in prog.learned.values(): print(f"  {L!r}")
        if not prog.learned: print("  (no learned variables)")
        return
    x, y = _data(a.data, prog, a.cmd == "fit")
    if a.cmd == "run":
        vp = a.values or _values_path(a.program)
        if prog.learned and os.path.exists(vp): prog.values = torch.load(vp, weights_only=True)
        print(f"{prog.entry}: {_report(prog, x, y)}"); return
    if a.resume and os.path.exists(_values_path(a.program)): prog.values = torch.load(_values_path(a.program), weights_only=True)
    print(f"{prog.entry}: {len(y)} examples; learning {', '.join(repr(L) for L in prog.learned.values())}", flush=True)
    t0 = time.time()
    prog.fit(*x, target=y, steps=a.steps, batch=a.batch, lr=a.lr, samples=a.samples, temperature=a.temperature, seed=a.seed, verbose=True)
    print(f"fitted in {time.time() - t0:.0f}s; training data: {_report(prog, x, y)}")
    if a.val:
        vx, vy = _data(a.val, prog, True); print(f"validation: {_report(prog, vx, vy)}")
    torch.save({k: v.cpu() for k, v in prog.values.items()}, _values_path(a.program))
    out = bake(prog, os.path.splitext(a.program)[0] + ".trained.c")
    print(f"wrote {out} and {_values_path(a.program)}")



def _constants(a):
    from .lift import candidates
    found = candidates(a.file, shlex.split(a.cflags))
    if not found: print(f"{a.file}: no file-level number constants with initial values"); return
    for c in found: print(f"  {c['name']:24s} {c['kind']:5s}  {' '.join(c['text'].split())[:90]}")
    print(f"tune them with: learnc tune ENTRY.c --tune {a.file}:{','.join(c['name'] for c in found[:3])},... --data DATA.pt")


def _tune(a):
    from .tune import tune
    flags = shlex.split(a.cflags)
    constants = {}
    for spec in a.tune:
        f, _, names = spec.rpartition(":")
        if not f: raise SystemExit(f"--tune {spec}: expected FILE.c:NAME,NAME or FILE.c:all")
        constants[f] = "all" if names == "all" else names.split(",")
    shape = Program(a.program, sources=a.source, entry=a.entry, device="cpu", flags=flags)   # parameter names
    d = torch.load(a.data, map_location="cpu", weights_only=True)
    names = [n for n, _ in shape.params]
    missing = [n for n in names if n not in d]
    if missing: raise SystemExit(f"{a.data}: no field for parameter(s) {missing} of {shape.entry}; fields: {sorted(d)}")
    target = None if a.minimize else d.get("output")
    t0 = time.time()
    r = tune(a.program, [d[n] for n in names], target, entry=shape.entry, constants=constants, sources=a.source,
             flags=flags, validation=a.validation, keep_close=a.keep_close, steps=a.steps, batch=a.batch, lr=a.lr,
             workers=a.workers, seed=a.seed, verbose=True)
    print(f"tuned in {time.time() - t0:.0f}s\n{r.report()}")
    print("wrote " + ", ".join(r.write(a.out)) + f" and {os.path.join(a.out, 'TUNING_REPORT.txt')}")


if __name__ == "__main__":
    main()
