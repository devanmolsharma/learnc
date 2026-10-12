"""Make constants of an existing C file tunable without editing it by hand.

    candidates(path)                     -> file-level numeric constants that can be tuned
    lift(path, names, out)               -> writes a copy of the file where those constants are `learned`
                                            (real numbers) or `learned_num` (whole numbers), starting from their
                                            original values; the original file is not changed.
    apply(path, values, out)             -> writes a copy of the ORIGINAL file with only those values replaced.
Covered: file-level variables of arithmetic type (or arrays of them) with an initial value, usually
`static const ...`. Not covered: #define constants and constants inside functions (they are not variables).
"""
import json, os, re
from .build import CLANG, INCLUDE, _run

INT_TYPES = {"char", "signed char", "unsigned char", "short", "unsigned short", "int", "unsigned int", "long",
             "unsigned long", "long long", "unsigned long long", "_Bool"}
INT_TYPES |= {f"{u}int{n}_t" for u in ("", "u") for n in (8, 16, 32, 64)} | {"size_t", "ssize_t", "ptrdiff_t",
              "intptr_t", "uintptr_t", "bool", "word16_t", "uint", "ushort", "uchar"}
FLOAT_TYPES = {"float", "double", "float32_t", "float64_t"}


def _base(qt):
    """'const short[3]' -> ('short', '[3]')"""
    t = re.sub(r"\b(const|volatile|static)\b", "", qt).strip()
    m = re.match(r"^([^\[]+?)\s*((\[\d+\])*)$", t)
    return (re.sub(r"\s+", " ", m.group(1)).strip(), m.group(2)) if m else (t, "")


def candidates(path, flags=()):
    """[{name, type, kind ('real' | 'whole'), offset, end, name_offset, value_text}] for liftable constants."""
    tree = json.loads(_run([CLANG, "-I", INCLUDE, *flags, "-Xclang", "-ast-dump=json", "-fsyntax-only", path]))
    src = open(path, "rb").read()
    out = []
    for d in tree.get("inner", []):
        if d.get("kind") != "VarDecl" or "inner" not in d: continue
        loc = d.get("loc", {}); loc = loc.get("expansionLoc", loc)
        if "includedFrom" in loc or loc.get("file", path) not in (path, None) and "file" in loc and os.path.abspath(loc["file"]) != os.path.abspath(path):
            continue
        if any(i.get("kind") == "AnnotateAttr" for i in d["inner"]): continue          # already learned
        if not any("Literal" in i.get("kind", "") or i.get("kind") in ("InitListExpr", "ImplicitCastExpr", "UnaryOperator") for i in d["inner"]):
            continue
        qt = d["type"].get("desugaredQualType", d["type"]["qualType"])
        base, dims = _base(qt)
        kind = "real" if base in FLOAT_TYPES else "whole" if base in INT_TYPES else None
        if kind is None or "*" in qt: continue
        b = d["range"]["begin"]; b = b.get("expansionLoc", b)
        e = d["range"]["end"]; e = e.get("expansionLoc", e)
        start, end = b["offset"], e["offset"] + e.get("tokLen", 0)
        out.append({"name": d["name"], "type": d["type"]["qualType"], "kind": kind, "offset": start, "end": end,
                    "name_offset": loc["offset"], "text": src[start:end].decode(errors="replace")})
    return out


def lift(path, names, out_path, flags=()):
    """Write a copy of `path` in which the named constants are tunable. Returns the list of lifted constants."""
    cands = {c["name"]: c for c in candidates(path, flags)}
    if names in (None, "all"): names = list(cands)
    missing = [n for n in names if n not in cands]
    if missing:
        raise ValueError(f"cannot tune {missing}: not file-level numeric constants with a value in {os.path.basename(path)}"
                         f" (liftable: {sorted(cands)})")
    src = open(path, "rb").read().decode()
    data = src.encode()
    for c in sorted((cands[n] for n in names), key=lambda c: -c["offset"]):
        decl = data[c["offset"]:c["name_offset"]].decode()               # everything before the variable's name
        prefix = re.sub(r"\bconst\b\s*", "", decl)
        macro = "learned " if c["kind"] == "real" else "learned_num "
        data = data[:c["offset"]] + (macro + prefix).encode() + data[c["name_offset"]:]
    text = data.decode()
    text = '#include <learnc.h>   /* added by learnc: tunable constants */\n#include <string.h>\n#line 1\n' + text
    # setters, so the tuner can change constants that are `static` (visible only inside this file)
    text += "\n/* added by learnc: setters for the tunable constants */\n" + "".join(
        f"void learnc_set_{n}(const void *v) {{ memcpy((void *)&{n}, v, sizeof({n})); }}\n" for n in names)
    open(out_path, "w").write(text)
    return [cands[n] for n in names]


def _literal(v, base):
    if base in FLOAT_TYPES:
        s = f"{float(v):.9g}"
        if not re.search(r"[.eEn]", s): s += ".0"                          # 2 -> 2.0 (n: nan/inf stay as they are)
        return s + ("f" if base in ("float", "float32_t") else "")
    return str(int(round(float(v))))


def _initializer(values, dims, base):
    """Nested C initializer for an array of shape dims (a scalar when dims is empty)."""
    if not dims: return _literal(values[0], base)
    step = len(values) // dims[0]
    return "{ " + ", ".join(_initializer(values[i * step:(i + 1) * step], dims[1:], base) for i in range(dims[0])) + " }"


def apply(path, values, out_path, flags=(), originals=None):
    """Write a copy of `path` (the ORIGINAL file) where only the initial values of the named constants differ:
    the smallest possible diff. values: {name: number or array}; originals: {name: original values}, to leave
    unchanged constants as written (default: compare with the numbers in the source text).
    Returns the names whose values changed."""
    import numpy as np
    cands = {c["name"]: c for c in candidates(path, flags)}
    data, changed = open(path, "rb").read(), []
    for c in sorted((cands[n] for n in values if n in cands), key=lambda c: -c["offset"]):
        base, dims = _base(c["type"])
        dims = [int(d) for d in re.findall(r"\[(\d+)\]", dims)]
        new = np.asarray(values[c["name"]], np.float64).reshape(-1)
        decl = data[c["offset"]:c["end"]].decode()
        eq = decl.index("=", c["name_offset"] - c["offset"])
        old_init = decl[eq + 1:].strip()
        init = _initializer(list(new), dims, base)
        same = (np.allclose(np.asarray(originals[c["name"]], np.float64).reshape(-1), new, rtol=1e-7, atol=0)
                if originals and c["name"] in originals else _same(old_init, new))
        if same: continue                                                  # unchanged: leave the text as it is
        data = data[:c["offset"]] + (decl[:eq + 1] + " " + init).encode() + data[c["end"]:]
        changed.append(c["name"])
    open(out_path, "wb").write(data)
    return changed[::-1]


def _same(old_init, new):
    """Does the original initializer text already hold these values? (numbers compared, not text)"""
    import numpy as np
    nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", re.sub(r"\b0[xX][0-9a-fA-F]+\b",
                      lambda m: str(int(m.group(0), 16)), old_init))
    if len(nums) != len(new): return False
    return np.allclose(np.array([float(n) for n in nums]), new, rtol=1e-7, atol=0)
