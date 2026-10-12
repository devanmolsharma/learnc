"""Make constants of an existing C file tunable without editing it by hand.

    candidates(path)                     -> numeric constants that can be tuned
    lift(path, names, out)               -> writes a copy of the file where those constants are `learned`
                                            (real numbers) or `learned_num` (whole numbers), starting from their
                                            original values; the original file is not changed.
    apply(path, values, out)             -> writes a copy of the ORIGINAL file with only those values replaced.
Covered: variables of arithmetic type (or arrays of them) with an initial value, usually `static const ...`, at file
level or `static` inside a function, in .c files and in headers. Not covered: #define constants and non-static
constants inside functions (they are not variables).
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


def _is_constant(d, path):
    """A variable of arithmetic type (or an array of them) with an initial value, declared in this file."""
    if d.get("kind") != "VarDecl" or "inner" not in d: return None
    loc = d.get("loc", {}); loc = loc.get("expansionLoc", loc)
    if "includedFrom" in loc or ("file" in loc and os.path.abspath(loc["file"]) != os.path.abspath(path)): return None
    if any(i.get("kind") == "AnnotateAttr" for i in d["inner"]): return None                # already learned
    if not any("Literal" in i.get("kind", "") or i.get("kind") in ("InitListExpr", "ImplicitCastExpr", "UnaryOperator")
               for i in d["inner"]): return None
    qt = d["type"].get("desugaredQualType", d["type"]["qualType"])
    base, _ = _base(qt)
    kind = "real" if base in FLOAT_TYPES else "whole" if base in INT_TYPES else None
    if kind is None or "*" in qt: return None
    return kind, loc


def _walk(node):
    for i in node.get("inner", []) or []:
        yield i
        yield from _walk(i)


def candidates(path, flags=()):
    """[{name, type, kind ('real' | 'whole'), offset, end, name_offset, text, function}] for liftable constants:
    file-level ones, and `static` ones inside functions (function: the function's name and where it starts).
    Headers (.h) are parsed as C, so pass the -D flags that enable their implementation part."""
    lang = ["-x", "c"] if path.endswith(".h") else []
    tree = json.loads(_run([CLANG, "-I", INCLUDE, *flags, *lang, "-Xclang", "-ast-dump=json", "-fsyntax-only", path]))
    src = open(path, "rb").read()
    out = []

    def add(d, kind, loc, function=None):
        b = d["range"]["begin"]; b = b.get("expansionLoc", b)
        e = d["range"]["end"]; e = e.get("expansionLoc", e)
        start, end = b["offset"], e["offset"] + e.get("tokLen", 0)
        out.append({"name": d["name"], "type": d["type"]["qualType"], "kind": kind, "offset": start, "end": end,
                    "name_offset": loc["offset"], "text": src[start:end].decode(errors="replace"), "function": function})

    for d in tree.get("inner", []):
        found = _is_constant(d, path)
        if found: add(d, *found); continue
        if d.get("kind") != "FunctionDecl" or not any(i.get("kind") == "CompoundStmt" for i in d.get("inner", [])): continue
        b = d["range"]["begin"]; b = b.get("expansionLoc", b)
        if b.get("file", path) != path and "file" in b and os.path.abspath(b["file"]) != os.path.abspath(path): continue
        for v in _walk(d):
            if v.get("kind") == "VarDecl" and v.get("storageClass") == "static":
                found = _is_constant(v, path)
                if found: add(v, *found, function=(d["name"], b["offset"]))
    names = [c["name"] for c in out]
    for c in out: c["ambiguous"] = names.count(c["name"]) > 1
    return out


def lift(path, names, out_path, flags=()):
    """Write a copy of `path` in which the named constants are tunable. Returns the list of lifted constants.
    A constant declared `static` inside a function is moved to just before that function (same name), so it can be
    set from outside."""
    all_cands = candidates(path, flags)
    cands = {c["name"]: c for c in all_cands if not c["ambiguous"]}
    if names in (None, "all"): names = list(cands)
    ambiguous = sorted({c["name"] for c in all_cands if c["ambiguous"]} & set(names))
    if ambiguous: raise ValueError(f"cannot tune {ambiguous}: several constants in {os.path.basename(path)} have that name")
    missing = [n for n in names if n not in cands]
    if missing:
        raise ValueError(f"cannot tune {missing}: not numeric constants with a value in {os.path.basename(path)}"
                         f" (liftable: {sorted(cands)})")
    data = open(path, "rb").read()
    edits = []                                                           # (offset, end, replacement)
    for c in (cands[n] for n in names):
        decl = data[c["offset"]:c["name_offset"]].decode()               # everything before the variable's name
        macro = "learned " if c["kind"] == "real" else "learned_num "
        prefix = macro + re.sub(r"\bconst\b\s*", "", decl)
        if c["function"] is None:
            edits.append((c["offset"], c["name_offset"], prefix))
        else:                                                            # move it out of the function
            if "static" not in prefix: prefix = "static " + prefix
            whole = prefix + data[c["name_offset"]:c["end"]].decode()
            stop = data.index(b";", c["end"]) + 1
            edits.append((c["offset"], stop, f"/* {c['name']}: moved before {c['function'][0]} by learnc */"))
            edits.append((c["function"][1], c["function"][1], whole + ";\n"))
    for a, b, text in sorted(edits, key=lambda e: (-e[0], -e[1])):
        data = data[:a] + text.encode() + data[b:]
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
    cands = {c["name"]: c for c in candidates(path, flags) if not c["ambiguous"]}
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
