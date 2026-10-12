"""Compiling a C file: LLVM IR for training, a native shared library for running, and the C signatures (from clang's AST)."""
import hashlib, json, os, re, shutil, subprocess, tempfile
from .ir import parse_module

CLANG = os.environ.get("LEARNC_CLANG") or shutil.which("clang") or "clang"
LLVM_LINK = next((p for p in (os.environ.get("LEARNC_LLVM_LINK"), "/usr/lib/llvm-18/bin/llvm-link", shutil.which("llvm-link"))
                  if p and os.path.exists(p)), "llvm-link")
INCLUDE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")
# -O1: registers in SSA form, simple loops. No vectors and no fused multiply-add, so the IR and the native code agree.
FLAGS = ["-O1", "-fno-unroll-loops", "-fno-vectorize", "-fno-slp-vectorize", "-ffp-contract=off", "-fno-math-errno",
         "-fno-builtin-memcpy", "-I", INCLUDE]


class BuildError(Exception):
    pass


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode: raise BuildError(f"{' '.join(cmd[:2])} ... failed:\n{p.stderr.strip()}")
    return p.stdout


def _dependencies(paths, extra):
    """Every file the sources read, headers included (clang -M), so editing a header rebuilds."""
    deps = set(paths)
    for p in paths:
        try: out = _run([CLANG, "-I", INCLUDE, *FLAGS, *extra, "-M", p])
        except BuildError: continue
        deps |= {d for d in out.replace("\\\n", " ").split()[1:] if os.path.exists(d)}
    return sorted(deps)


def cache_dir(paths, extra=()):
    files = _dependencies(paths, extra)
    h = hashlib.sha1(b"".join(open(p, "rb").read() + p.encode() for p in files)
                     + " ".join([*FLAGS, *extra]).encode()).hexdigest()[:12]
    d = os.path.join(tempfile.gettempdir(), "learnc-cache", h); os.makedirs(d, exist_ok=True); return d


def compile_ir(paths, extra=()):
    """C files (the program and the sources of any libraries it uses) -> one parsed LLVM IR module."""
    paths = [os.path.abspath(p) for p in paths]; d = cache_dir(paths, extra); outs = []
    for k, p in enumerate(paths):
        out = os.path.join(d, f"part{k}.ll"); _run([CLANG, *FLAGS, *extra, "-S", "-emit-llvm", p, "-o", out]); outs.append(out)
    final = outs[0]
    if len(outs) > 1:
        final = os.path.join(d, "linked.ll"); _run([LLVM_LINK, "-S", *outs, "-o", final])
    text = open(final).read()
    return parse_module(text), text


def compile_shared(paths, out, extra=()):
    _run([CLANG, *FLAGS, *extra, "-shared", "-fPIC", *paths, "-o", out, "-lm"])
    return out


# ---------------------------------------------------------------------------- C signatures
_SCALARS = {"char": ("int", 8, True), "signed char": ("int", 8, True), "unsigned char": ("int", 8, False),
            "short": ("int", 16, True), "unsigned short": ("int", 16, False), "int": ("int", 32, True),
            "unsigned int": ("int", 32, False), "long": ("int", 64, True), "unsigned long": ("int", 64, False),
            "long long": ("int", 64, True), "unsigned long long": ("int", 64, False), "_Bool": ("int", 1, False),
            "bool": ("int", 1, False), "float": ("fp", 32, True), "double": ("fp", 64, True), "void": ("void", 0, False)}


def c_type(qual):
    """'const float *' -> ('ptr', scalar) ; 'int' -> scalar. Scalars: (kind, bits, signed)."""
    q = re.sub(r"\b(const|volatile|restrict|__restrict|struct)\b", "", qual).strip()
    q = re.sub(r"\s+", " ", q)
    m = re.match(r"^(.*?)\s*\(\*\)(\[\d+\])+$", q)                         # float (*)[6]: a 2-D array parameter
    if m: q = m.group(1).strip() + " *"
    if q.endswith("*"):
        inner = q[:-1].strip()
        if inner.endswith("*"): raise BuildError(f"pointer-to-pointer parameters are not supported: {qual}")
        if inner not in _SCALARS: raise BuildError(f"unsupported parameter type {qual}")
        return ("ptr", _SCALARS[inner])
    m = re.match(r"(.*)\[(\d*)\]$", q)
    if m: return ("ptr", _SCALARS[m.group(1).strip()])
    if q not in _SCALARS: raise BuildError(f"unsupported type {qual}")
    return _SCALARS[q]


def signatures(path, extra=()):
    """name -> (return C type, [(param name, C type)]) for functions defined in the file itself."""
    out = _run([CLANG, "-I", INCLUDE, *extra, "-Xclang", "-ast-dump=json", "-fsyntax-only", path])
    tree = json.loads(out); sigs = {}
    for n in tree.get("inner", []):
        if n.get("kind") != "FunctionDecl" or "includedFrom" in n.get("loc", {}) or n.get("isImplicit"): continue
        if not any(i.get("kind") == "CompoundStmt" for i in n.get("inner", [])): continue
        t = n["type"]; ftype = t.get("desugaredQualType", t["qualType"])
        params = []
        for p in n.get("inner", []):
            if p.get("kind") == "ParmVarDecl":
                pt = p["type"]; params.append((p.get("name", f"arg{len(params)}"), pt.get("desugaredQualType", pt["qualType"])))
        sigs[n["name"]] = (ftype.split("(")[0].strip(), params)
    return sigs


def learned_declarations(path, extra=()):
    """name -> (start, end) byte offsets in the file where the declaration's initializer goes: an existing initializer
    is replaced; otherwise start == end, just after the declarator (before the `;`)."""
    out = _run([CLANG, "-I", INCLUDE, *extra, "-Xclang", "-ast-dump=json", "-fsyntax-only", path])
    tree = json.loads(out); found = {}
    def where(loc): return loc.get("expansionLoc", loc)
    for n in tree.get("inner", []):
        if n.get("kind") != "VarDecl" or not any(i.get("kind") == "AnnotateAttr" for i in n.get("inner", [])): continue
        if "includedFrom" in where(n.get("loc", {})): continue
        init = next((i for i in n.get("inner", []) if "Attr" not in i.get("kind", "")), None)
        if init is not None:
            b, e = where(init["range"]["begin"]), where(init["range"]["end"])
            found[n["name"]] = (b["offset"], e["offset"] + e["tokLen"])
        else:
            e = where(n["range"]["end"]); found[n["name"]] = (e["offset"] + e["tokLen"],) * 2
    return found
