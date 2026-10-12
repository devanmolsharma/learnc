"""Parser for the textual LLVM IR that clang emits (the subset produced for ordinary C at -O1).

Module      globals (name -> Global), functions (name -> Function), annotations (global name -> annotation string)
Function    params [(type, name)], ret type, blocks in reverse post-order (entry first)
Block       name, phis [Inst], insts [Inst], terminator Inst
Inst        op, name (result register or None), type, args (list), extra (dict: predicate, flags, targets...)
Values      ("reg", "%x") | ("glob", "@g") | ("const", python number, type) | ("null",) | ("undef", type)
            | ("cgep", source_type, base_value, [index values])  (constant getelementptr expressions)
Types       ("int", bits) | ("fp", bits) | ("ptr",) | ("void",) | ("array", n, T) | ("struct", (T, ...), packed)
"""
import re, struct
from dataclasses import dataclass, field


class IRError(Exception):
    pass


# ---------------------------------------------------------------------------- types
def size_of(t):
    k = t[0]
    if k == "int": return max(1, t[1] // 8) if t[1] > 1 else 1
    if k == "fp": return t[1] // 8
    if k == "ptr": return 8
    if k == "array": return t[1] * size_of(t[2])
    if k == "struct":
        off = 0
        for f in t[1]:
            if not t[2]: off = _align_up(off, align_of(f))
            off += size_of(f)
        return off if t[2] else _align_up(off, align_of(t))
    raise IRError(f"no size for type {t}")


def align_of(t):
    k = t[0]
    if k in ("int", "fp", "ptr"): return min(size_of(t), 8)
    if k == "array": return align_of(t[2])
    if k == "struct": return 1 if t[2] else max([align_of(f) for f in t[1]] or [1])
    raise IRError(f"no alignment for type {t}")


def _align_up(x, a): return (x + a - 1) // a * a


def field_offset(t, i):
    off = 0
    for k, f in enumerate(t[1]):
        if not t[2]: off = _align_up(off, align_of(f))
        if k == i: return off
        off += size_of(f)
    raise IRError(f"struct has no field {i}")


def scalar_of(t):
    """The single scalar type an aggregate is made of (arrays of arrays of float -> float)."""
    if t[0] in ("int", "fp", "ptr"): return t
    if t[0] == "array": return scalar_of(t[2])
    if t[0] == "struct":
        kinds = {scalar_of(f) for f in t[1]}
        if len(kinds) == 1 and size_of(t) == sum(size_of(f) for f in t[1]): return kinds.pop()
        raise IRError(f"structs mixing field types are not supported yet: {t}")
    raise IRError(f"no scalar type for {t}")


# ---------------------------------------------------------------------------- tokens
_TOKEN = re.compile(r'''\s*(?:
    (?P<str>c?"(?:[^"\\]|\\.)*")
  | (?P<local>%[-\w.$]+|%"[^"]*")
  | (?P<global>@[-\w.$]+|@"[^"]*")
  | (?P<meta>![-\w.]+)
  | (?P<attrgrp>\#\d+)
  | (?P<num>0x[KLMHR]?[0-9A-Fa-f]+|-?\d+\.\d*(?:[eE][-+]?\d+)?|-?\d+(?:[eE][-+]?\d+)?)
  | (?P<word>[A-Za-z_$][-\w.$]*)
  | (?P<punct>\.\.\.|[\[\]{}()<>,=*:])
)''', re.X)


def tokenize(s):
    out, pos = [], 0
    s = s.rstrip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            if s[pos:].strip() == "": break
            raise IRError(f"cannot read IR near: {s[pos:pos + 40]!r}")
        kind = m.lastgroup; out.append((kind, m.group(kind))); pos = m.end()
    return out


class Cursor:
    def __init__(self, toks): self.t, self.i = toks, 0
    def peek(self, k=0): return self.t[self.i + k][1] if self.i + k < len(self.t) else None
    def kind(self, k=0): return self.t[self.i + k][0] if self.i + k < len(self.t) else None
    def next(self): v = self.t[self.i][1]; self.i += 1; return v
    def eat(self, v):
        if self.peek() == v: self.i += 1; return True
        return False
    def expect(self, v):
        if self.next() != v: raise IRError(f"expected {v!r} in {' '.join(x[1] for x in self.t)}")
    def done(self): return self.i >= len(self.t)


_INT = re.compile(r"i(\d+)$")
NAMED_TYPES = {}                                              # %struct.x -> type, filled by parse_module
_SKIP_WORDS = {"noundef", "nonnull", "nocapture", "readonly", "writeonly", "readnone", "signext", "zeroext", "inreg",
               "noalias", "immarg", "returned", "nofree", "nest", "swiftself", "inbounds", "nuw", "nsw", "exact",
               "disjoint", "nneg", "fast", "nnan", "ninf", "nsz", "arcp", "contract", "afn", "reassoc", "tail",
               "musttail", "notail", "volatile", "fastcc", "ccc", "coldcc", "dereferenceable", "dereferenceable_or_null",
               "align", "range", "local_unnamed_addr", "unnamed_addr", "dso_local", "internal", "private"}
_PARAM_ATTR_WITH_ARG = {"dereferenceable", "dereferenceable_or_null", "align", "range"}


def parse_type(c):
    tok = c.next()
    m = _INT.match(tok)
    if m: t = ("int", int(m.group(1)))
    elif tok == "float": t = ("fp", 32)
    elif tok == "double": t = ("fp", 64)
    elif tok in ("half", "x86_fp80", "fp128", "bfloat"): raise IRError(f"type {tok} is not supported")
    elif tok == "ptr": t = ("ptr",)
    elif tok == "void": t = ("void",)
    elif tok == "[":
        n = int(c.next()); c.expect("x"); el = parse_type(c); c.expect("]"); t = ("array", n, el)
    elif tok == "{":
        fields = []
        if not c.eat("}"):
            while True:
                fields.append(parse_type(c))
                if c.eat("}"): break
                c.expect(",")
        t = ("struct", tuple(fields), False)
    elif tok == "<":
        if c.peek() == "{":
            c.next(); fields = []
            while True:
                fields.append(parse_type(c))
                if c.eat("}"): break
                c.expect(",")
            c.expect(">"); t = ("struct", tuple(fields), True)
        else: raise IRError("vector types are not supported (compile with -fno-vectorize -fno-slp-vectorize)")
    elif tok.startswith("%"):                                # a named type, e.g. %struct.node
        if tok not in NAMED_TYPES: raise IRError(f"unknown type {tok}")
        t = NAMED_TYPES[tok]
        if t is None: raise IRError(f"type {tok} is opaque (declared but never defined)")
    else: raise IRError(f"unknown type {tok!r}")
    return t


def _skip_attrs(c):
    while c.peek() in _SKIP_WORDS or c.kind() == "attrgrp":
        w = c.next()
        if w in _PARAM_ATTR_WITH_ARG:
            if c.eat("("):
                depth = 1
                while depth:
                    x = c.next(); depth += (x == "(") - (x == ")")
            elif c.kind() == "num": c.next()


def _float_const(tok, t):
    if tok.startswith("0x"):
        if tok[2] in "KLMHR": raise IRError(f"unsupported float constant {tok}")
        return struct.unpack(">d", bytes.fromhex(tok[2:].rjust(16, "0")))[0]
    return float(tok)


def parse_value(c, t):
    """A value of (already parsed) type t."""
    tok, kind = c.peek(), c.kind()
    if kind == "local": c.next(); return ("reg", tok)
    if kind == "global": c.next(); return ("glob", tok)
    if tok in ("true", "false"): c.next(); return ("const", 1 if tok == "true" else 0, t)
    if tok == "null": c.next(); return ("null",)
    if tok in ("undef", "poison"): c.next(); return ("undef", t)
    if tok == "zeroinitializer": c.next(); return ("zero", t)
    if kind == "num":
        c.next()
        return ("const", _float_const(tok, t) if t[0] == "fp" else int(tok), t)
    if tok == "getelementptr":
        c.next(); _skip_attrs(c); c.expect("("); src = parse_type(c); c.expect(",")
        bt = parse_type(c); base = parse_value(c, bt); idx = []
        while c.eat(","):
            it = parse_type(c); idx.append(parse_value(c, it))
        c.expect(")")
        return ("cgep", src, base, idx)
    raise IRError(f"cannot read value {tok!r}")


def parse_const_init(c, t):
    """A global initializer: nested lists of python numbers (or None for zero)."""
    tok = c.peek()
    if tok == "zeroinitializer": c.next(); return None
    if tok in ("undef", "poison"): c.next(); return None
    if tok and tok.startswith('c"'):
        c.next(); raw = tok[2:-1]
        data = re.sub(r"\\([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), raw)
        return [ord(ch) for ch in data]
    if tok in ("[", "{", "<"):
        close = {"[": "]", "{": "}", "<": ">"}[c.next()]
        if close == ">" and c.peek() == "{": c.next(); close = "}>"
        items = []
        while not c.eat(close if close != "}>" else "}"):
            et = parse_type(c); items.append(parse_const_init(c, et)); c.eat(",")
        if close == "}>": c.expect(">")
        return items
    v = parse_value(c, t)
    if v[0] == "const": return v[1]
    if v[0] in ("null", "zero", "undef"): return None
    raise IRError(f"unsupported global initializer {v}")


# ---------------------------------------------------------------------------- structures
@dataclass
class Inst:
    op: str
    name: str = None
    type: tuple = None
    args: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class Block:
    name: str
    phis: list = field(default_factory=list)
    insts: list = field(default_factory=list)
    term: Inst = None


@dataclass
class Function:
    name: str
    ret: tuple
    params: list
    blocks: list
    index: dict = field(default_factory=dict)       # block name -> position


@dataclass
class Global:
    name: str
    type: tuple
    init: object
    constant: bool


@dataclass
class Module:
    globals: dict
    functions: dict
    declarations: set
    annotations: dict
    unsupported: dict = field(default_factory=dict)      # function or global name -> why it could not be parsed


_BINOPS = {"add", "sub", "mul", "sdiv", "udiv", "srem", "urem", "shl", "lshr", "ashr", "and", "or", "xor",
           "fadd", "fsub", "fmul", "fdiv", "frem"}
_CASTS = {"trunc", "zext", "sext", "fptosi", "fptoui", "sitofp", "uitofp", "fpext", "fptrunc", "bitcast", "ptrtoint", "inttoptr", "addrspacecast"}


def _strip_meta(line):
    line = re.sub(r",\s*![-\w.]+\s+!\d+", "", line)               # , !tbaa !5   , !llvm.loop !9
    line = re.sub(r",\s*![-\w.]+\s+!\{[^}]*\}", "", line)
    line = re.sub(r",\s*align\s+\d+\s*$", "", line.rstrip())
    return line


def parse_inst(line):
    line = _strip_meta(line)
    c = Cursor(tokenize(line)); name = None
    if c.kind() == "local" and c.peek(1) == "=": name = c.next(); c.next()
    while c.peek() in ("tail", "musttail", "notail"): c.next()
    op = c.next()
    if op in _BINOPS:
        _skip_attrs(c); t = parse_type(c); a = parse_value(c, t); c.expect(","); b = parse_value(c, t)
        return Inst(op, name, t, [a, b])
    if op == "fneg":
        _skip_attrs(c); t = parse_type(c); return Inst(op, name, t, [parse_value(c, t)])
    if op in ("icmp", "fcmp"):
        _skip_attrs(c); pred = c.next(); t = parse_type(c); a = parse_value(c, t); c.expect(","); b = parse_value(c, t)
        return Inst(op, name, ("int", 1), [a, b], {"pred": pred, "argtype": t})
    if op in _CASTS:
        _skip_attrs(c); ft = parse_type(c); v = parse_value(c, ft); c.expect("to"); tt = parse_type(c)
        return Inst(op, name, tt, [v], {"from": ft})
    if op == "select":
        _skip_attrs(c); ct = parse_type(c); cv = parse_value(c, ct); c.expect(",")
        t = parse_type(c); a = parse_value(c, t); c.expect(","); parse_type(c); b = parse_value(c, t)
        return Inst(op, name, t, [cv, a, b])
    if op == "phi":
        _skip_attrs(c); t = parse_type(c); inc = []
        while c.eat("["):
            v = parse_value(c, t); c.expect(","); blk = c.next(); c.expect("]"); inc.append((v, blk)); c.eat(",")
        return Inst(op, name, t, [], {"incoming": inc})
    if op == "getelementptr":
        _skip_attrs(c); src = parse_type(c); c.expect(","); bt = parse_type(c); base = parse_value(c, bt); idx = []
        while c.eat(","):
            _skip_attrs(c); it = parse_type(c); idx.append(parse_value(c, it))
        return Inst(op, name, ("ptr",), [base] + idx, {"src": src})
    if op == "load":
        _skip_attrs(c); t = parse_type(c); c.expect(","); pt = parse_type(c); p = parse_value(c, pt)
        return Inst(op, name, t, [p])
    if op == "store":
        _skip_attrs(c); t = parse_type(c); v = parse_value(c, t); c.expect(","); pt = parse_type(c); p = parse_value(c, pt)
        return Inst(op, None, t, [v, p])
    if op == "alloca":
        _skip_attrs(c); t = parse_type(c); n = None
        if c.eat(","):
            if c.peek() != "align":
                nt = parse_type(c); n = parse_value(c, nt)
        return Inst(op, name, ("ptr",), [], {"alloc": t, "count": n})
    if op == "freeze":
        t = parse_type(c); return Inst(op, name, t, [parse_value(c, t)])
    if op == "call":
        _skip_attrs(c); rt = parse_type(c)
        if c.peek() == "(":                                   # variadic callee type, e.g. call i32 (ptr, ...) @printf(
            depth = 0
            while True:
                x = c.next(); depth += (x == "(") - (x == ")")
                if depth == 0: break
        fn = c.next(); fn_reg = None
        if fn.startswith("%"): fn_reg, fn = ("reg", fn), None          # a call through a function pointer
        c.expect("(")
        args = []
        while not c.eat(")"):
            at = parse_type(c); _skip_attrs(c); args.append((at, parse_value(c, at))); c.eat(",")
        return Inst(op, name, rt, [v for _, v in args], {"fn": fn, "fn_reg": fn_reg, "argtypes": [t for t, _ in args]})
    if op == "ret":
        if c.peek() == "void": return Inst("ret", None, ("void",), [])
        t = parse_type(c); return Inst("ret", None, t, [parse_value(c, t)])
    if op == "br":
        if c.peek() == "label": c.next(); return Inst("br", None, None, [], {"targets": [c.next()]})
        t = parse_type(c); cv = parse_value(c, t); c.expect(","); c.expect("label"); a = c.next(); c.expect(","); c.expect("label")
        return Inst("br", None, None, [cv], {"targets": [a, c.next()]})
    if op == "switch":
        t = parse_type(c); v = parse_value(c, t); c.expect(","); c.expect("label"); default = c.next(); c.expect("[")
        cases = []
        while not c.eat("]"):
            parse_type(c); k = int(c.next()); c.expect(","); c.expect("label"); cases.append((k, c.next()))
        return Inst("switch", None, t, [v], {"default": default, "cases": cases})
    if op == "unreachable": return Inst("unreachable")
    if op in ("extractvalue", "insertvalue", "extractelement", "insertelement", "shufflevector", "indirectbr",
              "invoke", "landingpad", "atomicrmw", "cmpxchg", "fence", "va_arg", "callbr"):
        raise IRError(f"instruction `{op}` is not supported yet")
    raise IRError(f"unknown instruction: {line.strip()}")


def _rpo(blocks, entry):
    succ = {}
    for b in blocks.values():
        t = b.term
        if t.op == "br": succ[b.name] = t.extra["targets"]
        elif t.op == "switch": succ[b.name] = [t.extra["default"]] + [x for _, x in t.extra["cases"]]
        else: succ[b.name] = []
    seen, order = set(), []
    stack = [(entry, iter(succ[entry]))]; seen.add(entry)
    while stack:
        node, it = stack[-1]
        nxt = next(it, None)
        if nxt is None: order.append(node); stack.pop(); continue
        if nxt not in seen: seen.add(nxt); stack.append((nxt, iter(succ[nxt])))
    return list(reversed(order))


def parse_function(header, body_lines):
    c = Cursor(tokenize(header))
    c.expect("define")
    while c.peek() in _SKIP_WORDS or c.peek() in ("hidden", "protected", "default", "weak", "linkonce_odr", "weak_odr", "available_externally", "external") or c.kind() == "attrgrp":
        c.next()
    ret = parse_type(c); _skip_attrs(c)
    name = c.next(); c.expect("(")
    params, k = [], 0
    while not c.eat(")"):
        if c.peek() == "...": raise IRError(f"{name}: variadic functions are not supported")
        t = parse_type(c); _skip_attrs(c)
        pname = c.next() if c.kind() == "local" else f"%{k}"
        params.append((t, pname)); k += 1; c.eat(",")
    blocks, cur, order = {}, None, []
    first = "%" + str(len(params))
    pending = []
    for raw in body_lines:
        line = raw.split(";")[0].rstrip() if '"' not in raw else raw.rstrip()
        if not line.strip(): continue
        m = re.match(r"^([-\w.$]+|\"[^\"]*\"):", line)
        if m:
            cur = Block("%" + m.group(1)); blocks[cur.name] = cur; order.append(cur.name); continue
        if cur is None: cur = Block(first); blocks[cur.name] = cur; order.append(cur.name)
        inst = parse_inst(line)
        if inst.op == "phi": cur.phis.append(inst)
        elif inst.op in ("ret", "br", "switch", "unreachable"): cur.term = inst
        else: cur.insts.append(inst)
    for b in blocks.values():
        if b.term is None: raise IRError(f"{name}: block {b.name} has no terminator")
    rpo = _rpo(blocks, order[0])
    f = Function(name, ret, params, [blocks[n] for n in rpo])
    f.index = {b.name: i for i, b in enumerate(f.blocks)}
    return f


def parse_module(text):
    lines = text.splitlines()
    NAMED_TYPES.clear()                                       # named types first: functions and globals refer to them
    pending = [l for l in lines if re.match(r'^%[-\w.$"]+ = type ', l)]
    for _ in range(len(pending) + 1):                         # a type may use one defined later: resolve in passes
        left = []
        for l in pending:
            name, body = l.split(" = type ", 1)
            if body.strip() == "opaque": NAMED_TYPES[name] = None; continue
            try: NAMED_TYPES[name] = parse_type(Cursor(tokenize(body)))
            except IRError: left.append(l)
        if not left or len(left) == len(pending): break
        pending = left
    # join multi-line switch statements
    joined, buf = [], None
    for l in lines:
        if buf is not None:
            buf += " " + l.strip()
            if l.strip().startswith("]"): joined.append(buf); buf = None
            continue
        if re.match(r"\s+switch ", l) and not l.rstrip().endswith("]"): buf = l.rstrip(); continue
        joined.append(l)
    globals_, functions, decls, ann, unsupported = {}, {}, set(), {}, {}
    i = 0
    while i < len(joined):
        l = joined[i]
        if l.startswith("define "):
            j = i + 1
            while joined[j].strip() != "}": j += 1
            try:
                f = parse_function(l.rstrip(" {"), joined[i + 1:j]); functions[f.name] = f
            except Exception as e:                       # keep going: the program can still run natively
                m = re.search(r"(@[-\w.$]+|@\"[^\"]*\")\(", l); unsupported[m.group(1) if m else l[:60]] = str(e)
            i = j + 1; continue
        if l.startswith("declare "):
            m = re.search(r"(@[-\w.$]+)\(", l); decls.add(m.group(1))
        elif l.startswith("@llvm.global.annotations"):
            for g, s in re.findall(r"ptr (@[-\w.$]+), ptr (@[-\w.$]+)", l): ann[g] = s
        elif l.startswith("@") and "=" in l:
            m = re.match(r"(@[-\w.$]+|@\"[^\"]*\")\s*=\s*(.*)", l)
            gname, rest = m.group(1), _strip_meta(m.group(2))
            if gname.startswith("@llvm."): i += 1; continue                 # compiler bookkeeping
            rest = re.sub(r',\s*section\s+"[^"]*"', "", rest)
            try:
                c = Cursor(tokenize(rest))
                while c.peek() not in ("global", "constant"): c.next()
                const = c.next() == "constant"; t = parse_type(c)
                init = parse_const_init(c, t) if not c.done() else None
                globals_[gname] = Global(gname, t, init, const)
            except Exception as e:
                unsupported[gname] = str(e)
        i += 1
    strings = {n: g.init for n, g in globals_.items()}
    annotations = {}
    for g, s in ann.items():
        txt = "".join(chr(x) for x in strings.get(s) or []).rstrip("\x00")
        annotations[g] = txt
    return Module({k: v for k, v in globals_.items() if not k.startswith("@llvm.")}, functions, decls, annotations, unsupported)
