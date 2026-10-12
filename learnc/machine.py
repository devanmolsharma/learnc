"""Batched executor for LLVM IR: runs one C function on B examples at once, with PyTorch tensors.

Control flow: every example (lane) has its own program counter. The machine repeatedly runs the earliest block (in
reverse post-order) that some lane is waiting at, for exactly those lanes; lanes that branch apart meet again where
their paths join, as in GPU (SIMT) execution. Loops and recursion are exact per lane.

Values: integers are int64 tensors holding the signed value wrapped to their bit width; i1 is bool; float/double are
float32/float64. A value is either a 0-d tensor (same for all lanes) or shape (B,).
Memory: objects of one scalar type. Per-lane objects (B, n) hold stack arrays and inputs; shared objects (n,) hold
globals (constant tables, learned variables). A pointer is (object id, byte offset).

Training (training=True) adds two relaxed kinds of value, created only by comparisons that depend on learned numbers:
  Soft  a condition that holds with probability p = sigmoid(difference / temperature)
  Mix   an integer that is one of several values per lane, with weights (e.g. "the index of the best score")
select on a Soft condition blends floats and mixes integers; comparisons of a Mix give a Soft; converting a Mix to
a float takes its expected value. Branches, loop conditions and stored integers use the most likely value.
"""
import torch
from .ir import field_offset, scalar_of, size_of

DTYPE = {32: torch.float32, 64: torch.float64}
MAX_PARTICLES = 32
JOINED = -2                       # "prev" marker: the lane's phis were already set (by a soft branch)


class RunError(Exception):
    pass


class Ptr:
    __slots__ = ("obj", "off")

    def __init__(self, obj, off): self.obj, self.off = obj, off


class Soft:
    __slots__ = ("p",)

    def __init__(self, p): self.p = p


class Mix:
    """Integer values V (P, B or 1) with weights W (P, B or 1); weights sum to 1 over P."""
    __slots__ = ("V", "W")

    def __init__(self, V, W): self.V, self.W = V, W


class Relaxed:
    """A learned whole-number variable while training: element e is values[k] with weight W[e, k]."""
    def __init__(self, W, values): self.W, self.values = W, values


class Obj:
    def __init__(self, data, etype, lane, name=None, writable=True):
        self.data, self.etype, self.lane, self.name, self.writable = data, etype, lane, name, writable
        self.relax = None
        self.esize = max(1, etype[1] // 8) if etype[0] == "int" else etype[1] // 8


def _flatten(init, t, out):
    if t[0] in ("int", "fp"): out.append(0 if init is None else init); return
    if t[0] == "array":
        for k in range(t[1]): _flatten(None if init is None else init[k], t[2], out)
        return
    if t[0] == "struct":
        for k, f in enumerate(t[1]): _flatten(None if init is None else init[k], f, out)
        return
    raise RunError(f"cannot initialise {t}")


def wrap(x, bits):
    if bits >= 64 or bits == 1: return x
    half = 1 << (bits - 1)
    return ((x + half) & ((1 << bits) - 1)) - half


def unsigned(x, bits):
    if bits >= 64: return x
    return x & ((1 << bits) - 1)


def _int_dtype(t): return torch.bool if t == ("int", 1) else torch.int64


# ---------------------------------------------------------------------------- relaxed values
def as_mix(v):
    if isinstance(v, Mix): return v
    t = torch.as_tensor(v)
    if t.dtype == torch.bool: t = t.long()
    V = t.reshape(1, -1) if t.dim() else t.reshape(1, 1)
    return Mix(V, torch.ones_like(V, dtype=torch.get_default_dtype()))


def prob(v):
    if isinstance(v, Soft): return v.p
    return torch.as_tensor(v).to(torch.get_default_dtype())


def hard(v):
    """The most likely value of a relaxed value (plain values unchanged)."""
    if isinstance(v, Soft): return v.p > 0.5
    if isinstance(v, Mix):
        B = max(v.V.shape[1], v.W.shape[1])
        k = v.W.expand(-1, B).argmax(0, keepdim=True)
        return v.V.expand(-1, B).gather(0, k).squeeze(0)
    return v


def expect(v):
    if isinstance(v, Mix): return (v.W * v.V.to(v.W.dtype)).sum(0)
    if isinstance(v, Soft): return v.p
    return v


def _prune(V, W):
    if V.shape[0] <= MAX_PARTICLES: return Mix(V, W)
    B = max(V.shape[1], W.shape[1]); V, W = V.expand(-1, B), W.expand(-1, B)
    k = W.topk(MAX_PARTICLES, 0).indices
    W = W.gather(0, k); return Mix(V.gather(0, k), W / W.sum(0, keepdim=True).clamp(min=1e-12))


def mix_concat(a, wa, b, wb):
    a, b = as_mix(a), as_mix(b)
    B = max(a.V.shape[1], b.V.shape[1], a.W.shape[1], b.W.shape[1], torch.as_tensor(wa).numel())
    V = torch.cat([a.V.expand(-1, B), b.V.expand(-1, B)])
    W = torch.cat([(wa * a.W).expand(-1, B), (wb * b.W).expand(-1, B)])
    return _prune(V, W)


def _pad(m, P):
    if m.V.shape[0] == P: return m
    n = P - m.V.shape[0]
    return Mix(torch.cat([m.V, torch.zeros(n, m.V.shape[1], dtype=m.V.dtype, device=m.V.device)]),
               torch.cat([m.W, torch.zeros(n, m.W.shape[1], dtype=m.W.dtype, device=m.W.device)]))


def blend(t, weights, values):
    """Combine the values that several weighted sides of a branch give one variable."""
    if any(isinstance(v, Ptr) for v in values):
        if len({v.obj for v in values}) > 1: raise RunError("a learned choice selects pointers into different objects")
        k = int(torch.stack([w.mean() for w in weights]).argmax()); return values[k]
    if t[0] == "fp": return sum(w.to(DTYPE[t[1]]) * expect(v) for w, v in zip(weights, values))
    if t == ("int", 1): return Soft(sum(w * prob(v) for w, v in zip(weights, values)))
    ms = [as_mix(v) for v in values]
    B = max([m.V.shape[1] for m in ms] + [m.W.shape[1] for m in ms] + [w.numel() for w in weights])
    return _prune(torch.cat([m.V.expand(-1, B) for m in ms]), torch.cat([(w * m.W).expand(-1, B) for w, m in zip(weights, ms)]))


def merge(mask, new, old):
    """Per lane: `new` where mask, else `old`, for every kind of value."""
    if isinstance(new, Ptr) or isinstance(old, Ptr):
        if new.obj != old.obj: raise RunError("a pointer points into different objects for different examples")
        if isinstance(new.off, Mix) or isinstance(old.off, Mix): return Ptr(new.obj, hard(merge(mask, new.off, old.off)))
        return Ptr(new.obj, torch.where(mask, new.off, old.off))
    if isinstance(new, Mix) or isinstance(old, Mix):
        a, b = as_mix(new), as_mix(old); P = max(a.V.shape[0], b.V.shape[0]); a, b = _pad(a, P), _pad(b, P)
        return Mix(torch.where(mask, a.V, b.V), torch.where(mask, a.W, b.W))
    if isinstance(new, Soft) or isinstance(old, Soft): return Soft(torch.where(mask, prob(new), prob(old)))
    return torch.where(mask, new, old)


def post_dominators(f):
    """block index -> index of its immediate post-dominator (None if paths from it end in different returns)."""
    if "_ipdom" in f.__dict__: return f._ipdom
    n = len(f.blocks); EXIT = n
    succ = []
    for b in f.blocks:
        t = b.term
        if t.op == "br": succ.append([f.index[x] for x in t.extra["targets"]])
        elif t.op == "switch": succ.append([f.index[t.extra["default"]]] + [f.index[x] for _, x in t.extra["cases"]])
        else: succ.append([EXIT])
    full = set(range(n + 1))
    pdom = {k: set(full) for k in range(n)}; pdom[EXIT] = {EXIT}
    changed = True
    while changed:
        changed = False
        for k in reversed(range(n)):
            new = {k} | set.intersection(*[pdom[s] for s in succ[k]])
            if new != pdom[k]: pdom[k], changed = new, True
    ipdom = {}
    for k in range(n):
        cands = pdom[k] - {k}
        # the immediate one is the candidate that every other candidate post-dominates... i.e. the closest
        best = next((c for c in cands if all(c == d or d in pdom[c] for d in cands)), None)
        ipdom[k] = None if best in (None, EXIT) else best
    f._ipdom = ipdom
    return ipdom


class Machine:
    def __init__(self, module, batch, device="cpu", learned=None, training=False, temperature=1.0, max_steps=1_000_000,
                 vectorize=True):
        """learned: optional dict global name -> tensor: shape (n,) shared by all lanes, or (B, n) one per lane."""
        self.m, self.B, self.dev, self.max_steps = module, batch, torch.device(device), max_steps
        self.training, self.temperature, self.vectorize = training, temperature, vectorize
        self.objs, self.gptr = [], {}
        self.lanes = torch.arange(batch, device=self.dev)
        learned = learned or {}
        for name, g in module.globals.items():
            try: et = scalar_of(g.type)
            except Exception: continue                                # unusual globals: only an error if used
            relax = None
            if name in learned and isinstance(learned[name], Relaxed):
                relax = learned[name]; data = torch.zeros(relax.W.shape[0], dtype=torch.int64, device=self.dev); lane = False
            elif name in learned:
                data = learned[name]; lane = data.dim() == 2
            else:
                flat = []; _flatten(g.init, g.type, flat); lane = False
                data = torch.tensor(flat, dtype=DTYPE[et[1]] if et[0] == "fp" else _int_dtype(et), device=self.dev)
            self.gptr[name] = self.new_obj(data, et, lane=lane, name=name, writable=not g.constant and name not in learned)
            self.objs[self.gptr[name]].relax = relax

    # ------------------------------------------------------------------ memory
    def input_pointer(self, data, etype, name):
        """An input array (B, n) as a pointer argument."""
        return Ptr(self.new_obj(data, etype, lane=True, name=name), torch.tensor(0, device=self.dev))

    def new_obj(self, data, etype, lane, name=None, writable=True):
        self.objs.append(Obj(data, etype, lane, name, writable)); return len(self.objs) - 1

    def _elem(self, p, t, off=None):
        if p.obj is None: raise RunError("load/store through a null pointer")
        o = self.objs[p.obj]
        want = scalar_of(t)
        if want != o.etype and not (want[0] == o.etype[0] == "int" and want[1] == o.etype[1]):
            raise RunError(f"memory of {o.name or 'an object'} holds {o.etype} but is accessed as {want} (type punning is not supported)")
        idx = torch.div(p.off if off is None else off, o.esize, rounding_mode="floor")
        return o, idx.clamp(0, o.data.shape[-1] - 1)                  # clamp: lanes that are not running may hold junk

    def _gather(self, o, idx):
        if o.lane:
            if idx.dim() == 0: return o.data[:, idx]
            if idx.dim() == 2: return o.data.gather(1, idx.T.expand(self.B, -1) if idx.shape[1] == 1 else idx.T).T
            return o.data[self.lanes, idx]
        return o.data[idx]

    def load(self, p, t):
        if isinstance(p.off, Mix):                                    # an address chosen by a learned comparison
            o, idx = self._elem(p, t, p.off.V)
            vals = self._gather(o, idx)
            return (p.off.W * vals).sum(0) if o.etype[0] == "fp" else Mix(vals, p.off.W)
        o, idx = self._elem(p, t)
        if o.relax is not None:                                       # a learned choice: all its values, weighted
            K = o.relax.values.numel(); Wsel = o.relax.W[idx].reshape(-1, K).T
            return Mix(o.relax.values.reshape(K, 1).to(self.dev), Wsel)
        return self._gather(o, idx)

    def store(self, p, v, t, mask):
        if isinstance(p.off, Mix): p = Ptr(p.obj, hard(p.off))
        v = hard(v)
        o, idx = self._elem(p, t)
        if not o.writable: raise RunError(f"the program writes to {o.name}, which is constant or learned")
        if not o.lane:
            o.data = o.data.unsqueeze(0).expand(self.B, -1).clone(); o.lane = True
        idx = idx.expand(self.B) if idx.dim() == 0 else idx
        old = o.data[self.lanes, idx]
        v = torch.as_tensor(v, device=self.dev).to(old.dtype)
        o.data = o.data.index_put((self.lanes, idx), torch.where(mask, v, old))

    # ------------------------------------------------------------------ values
    def const(self, v):
        k = v[0]
        if k == "const":
            t = v[2]
            if t[0] == "fp": return torch.tensor(v[1], dtype=DTYPE[t[1]], device=self.dev)
            if t == ("int", 1): return torch.tensor(bool(v[1]), device=self.dev)
            return torch.tensor(wrap(v[1], t[1]) if t[1] < 64 else v[1], dtype=torch.int64, device=self.dev)
        if k == "null": return Ptr(None, torch.tensor(0, device=self.dev))
        if k in ("undef", "zero"):
            t = v[1]
            if t[0] == "fp": return torch.tensor(0.0, dtype=DTYPE[t[1]], device=self.dev)
            if t[0] == "ptr": return Ptr(None, torch.tensor(0, device=self.dev))
            return torch.tensor(False if t == ("int", 1) else 0, device=self.dev)
        if k == "glob":
            if v[1] not in self.gptr: raise RunError(f"global {v[1]} is not supported (unusual type)")
            return Ptr(self.gptr[v[1]], torch.tensor(0, device=self.dev))
        if k == "cgep": return self.gep(self.value(v[2], {}), v[1], [self.value(i, {}) for i in v[3]])
        raise RunError(f"unknown value {v}")

    def value(self, v, regs):
        if v[0] == "reg":
            if v[1] not in regs: raise RunError(f"register {v[1]} read before it was set")
            return regs[v[1]]
        return self.const(v)

    def gep(self, base, src, idx):
        if not isinstance(base, Ptr): raise RunError("getelementptr on a non-pointer")
        off = base.off
        steps = [(idx[0], size_of(src))]
        t = src
        for i in idx[1:]:
            if t[0] == "array": steps.append((i, size_of(t[2]))); t = t[2]
            elif t[0] == "struct":
                k = int(i); steps.append((torch.tensor(field_offset(t, k), device=self.dev), 1)); t = t[1][k]
            else: raise RunError(f"getelementptr into {t}")
        for i, scale in steps:
            if isinstance(i, Mix) or isinstance(off, Mix):
                off = self.binop("add", ("int", 64), as_mix(off) if isinstance(off, Mix) else off,
                                 Mix(as_mix(i).V.long() * scale, as_mix(i).W) if isinstance(i, Mix) else i.long() * scale)
            else: off = off + hard(i).long() * scale
        return Ptr(base.obj, off)

    # ------------------------------------------------------------------ instructions
    def binop(self, op, t, a, b):
        if isinstance(a, Soft) or isinstance(b, Soft):
            if t == ("int", 1):
                pa, pb = prob(a), prob(b)
                if op in ("and", "mul"): return Soft(pa * pb)
                if op == "or": return Soft(pa + pb - pa * pb)
                if op in ("xor", "add", "sub"): return Soft(pa + pb - 2 * pa * pb)
            a, b = hard(a), hard(b)
        if isinstance(a, Mix) or isinstance(b, Mix):
            if t[0] == "fp": return self.binop(op, t, expect(a), expect(b))
            a, b = as_mix(a), as_mix(b)
            V = self.binop(op, t, a.V[:, None], b.V[None, :]).flatten(0, 1)
            W = (a.W[:, None] * b.W[None, :]).flatten(0, 1)
            return _prune(V, W)
        if t[0] == "fp":
            if op == "fadd": return a + b
            if op == "fsub": return a - b
            if op == "fmul": return a * b
            if op == "fdiv": return a / b
            if op == "frem": return torch.fmod(a, b)
        bits = t[1]
        if bits == 1:
            a, b = a.bool(), b.bool()
            if op in ("and", "mul"): return a & b
            if op == "or": return a | b
            if op in ("xor", "add", "sub"): return a ^ b
            raise RunError(f"{op} on i1")
        if op in ("sdiv", "udiv", "srem", "urem"):
            zero = b == 0; b1 = torch.where(zero, torch.ones_like(b), b)        # lanes not running may divide by 0
            if op == "sdiv": r = torch.div(a, b1, rounding_mode="trunc")
            elif op == "srem": r = torch.fmod(a, b1)
            elif op == "udiv": r = torch.div(unsigned(a, bits), unsigned(b1, bits), rounding_mode="floor")
            else: r = torch.remainder(unsigned(a, bits), unsigned(b1, bits))
            return wrap(torch.where(zero, torch.zeros_like(r), r), bits)
        if op == "add": return wrap(a + b, bits)
        if op == "sub": return wrap(a - b, bits)
        if op == "mul": return wrap(a * b, bits)
        if op == "and": return a & b
        if op == "or": return a | b
        if op == "xor": return a ^ b
        sh = torch.as_tensor(b).clamp(0, bits - 1)
        if op == "shl": return wrap(a << sh, bits)
        if op == "ashr": return a >> sh
        if op == "lshr": return wrap(unsigned(a, bits) >> sh, bits)
        raise RunError(f"unknown operation {op}")

    def icmp(self, pred, t, a, b):
        if isinstance(a, Ptr) or isinstance(b, Ptr):
            if a.obj != b.obj:
                if pred in ("eq", "ne"): return torch.tensor(pred == "ne", device=self.dev)
                raise RunError("comparing pointers into different objects")
            a, b = hard(a.off), hard(b.off); t = ("int", 64)
        if isinstance(a, Soft): a = self.cast("zext", ("int", 1), ("int", 8), a)      # a condition as a 0/1 number
        if isinstance(b, Soft): b = self.cast("zext", ("int", 1), ("int", 8), b)
        if isinstance(a, Mix) or isinstance(b, Mix):
            a, b = as_mix(a), as_mix(b)
            table = self.icmp(pred, t, a.V[:, None], b.V[None, :]).to(a.W.dtype)        # (Pa, Pb, B)
            return Soft((a.W[:, None] * table * b.W[None, :]).sum((0, 1)))
        bits = t[1]
        if pred in ("ult", "ule", "ugt", "uge"):
            a, b = unsigned(a.long(), bits), unsigned(b.long(), bits); pred = "s" + pred[1:]
        return {"eq": a == b, "ne": a != b, "slt": a < b, "sle": a <= b, "sgt": a > b, "sge": a >= b}[pred]

    def fcmp(self, pred, a, b):
        if pred == "true": return torch.ones_like(a, dtype=torch.bool)
        if pred == "false": return torch.zeros_like(a, dtype=torch.bool)
        rel = pred[1:]
        if self.training and rel in ("gt", "ge", "lt", "le") and (a.requires_grad or b.requires_grad):
            d = (a - b) if rel in ("gt", "ge") else (b - a)                # learned numbers decide: a soft comparison
            return Soft(torch.sigmoid(d / self.temperature))
        nan = torch.isnan(a) | torch.isnan(b)
        if pred == "ord": return ~nan
        if pred == "uno": return nan
        r = {"eq": a == b, "ne": a != b, "gt": a > b, "ge": a >= b, "lt": a < b, "le": a <= b}[rel]
        return (r & ~nan) if pred[0] == "o" else (r | nan)

    def cast(self, op, ft, tt, v):
        if isinstance(v, Soft):
            if op in ("zext", "sext"):
                one = -1 if op == "sext" else 1
                V = torch.tensor([[0], [one]], device=self.dev)
                return Mix(V, torch.stack([1 - v.p, v.p]).reshape(2, -1))
            if op in ("uitofp", "sitofp"): return (v.p * (-1 if op == "sitofp" else 1)).to(DTYPE[tt[1]])
            v = hard(v)
        if isinstance(v, Mix):
            if op in ("sitofp", "uitofp"): return expect(Mix(self.cast(op, ft, tt, v.V), v.W)).to(DTYPE[tt[1]])
            if op in ("trunc", "zext", "sext"): return _prune(self.cast(op, ft, tt, v.V), v.W) if tt != ("int", 1) else \
                Soft((v.W * (self.cast(op, ft, tt, v.V)).to(v.W.dtype)).sum(0))
            v = hard(v)
        if op in ("bitcast", "addrspacecast"):
            if ft[0] == "ptr" or ft == tt: return v
            raise RunError("bitcast between numbers is not supported")
        if op in ("ptrtoint", "inttoptr"): raise RunError(f"{op} is not supported")
        if op == "trunc":
            return (v & 1).bool() if tt == ("int", 1) else wrap(v, tt[1])
        if op in ("zext", "sext"):
            if ft == ("int", 1): return v.long() * (-1 if op == "sext" else 1)
            return v if op == "sext" else unsigned(v, ft[1])
        if op in ("fptosi", "fptoui"):
            r = torch.trunc(torch.nan_to_num(v.detach())).clamp(-2.0 ** 63, 2.0 ** 63 - 1).long()
            return (r != 0) if tt == ("int", 1) else wrap(r, tt[1])
        if op == "sitofp": return v.to(DTYPE[tt[1]])
        if op == "uitofp": return unsigned(v.long(), ft[1]).to(DTYPE[tt[1]])
        if op in ("fpext", "fptrunc"): return v.to(DTYPE[tt[1]])
        raise RunError(f"unknown cast {op}")

    def select(self, t, c, a, b):
        if isinstance(c, Soft):
            if isinstance(a, Ptr) or isinstance(b, Ptr): return self.select(t, hard(c), a, b)
            p = c.p
            if t[0] == "fp": return p.to(DTYPE[t[1]]) * expect(a) + (1 - p).to(DTYPE[t[1]]) * expect(b)
            if t == ("int", 1): return Soft(p * prob(a) + (1 - p) * prob(b))
            return mix_concat(a, p, b, 1 - p)
        c = hard(c)
        if isinstance(a, Ptr) or isinstance(b, Ptr):
            if a.obj != b.obj: raise RunError("choosing between pointers into different objects is not supported")
            return Ptr(a.obj, merge(c, a.off, b.off))
        return merge(c, a, b)

    def _set(self, regs, name, v, mask):
        regs[name] = merge(mask, v, regs[name]) if name in regs else v

    def call(self, inst, regs, mask):
        fn = inst.extra["fn"]
        if fn is None: raise RunError("calls through function pointers need the flat memory runner")
        args = [self.value(a, regs) for a in inst.args]
        if fn in self.m.functions: return self.run_function(self.m.functions[fn], args, mask)
        name = fn[1:]
        base = name.split(".")[1] if name.startswith("llvm.") else name
        if base in ("lifetime", "assume", "dbg", "experimental", "invariant", "sideeffect"): return None
        if base in ("memcpy", "memmove"):
            dst, src, n = args[0], args[1], int(hard(args[2]))
            o = self.objs[src.obj]; count = n // o.esize
            vals = [self.load(Ptr(src.obj, hard(src.off) + k * o.esize), o.etype) for k in range(count)]
            for k, v in enumerate(vals): self.store(Ptr(dst.obj, hard(dst.off) + k * o.esize), v, o.etype, mask)
            return None
        if base == "memset":
            dst, byte, n = args[0], int(hard(args[1])), int(hard(args[2])); o = self.objs[dst.obj]
            if byte != 0 and o.esize != 1: raise RunError("memset with a non-zero byte is only supported for char arrays")
            for k in range(n // o.esize): self.store(Ptr(dst.obj, hard(dst.off) + k * o.esize), torch.tensor(byte, device=self.dev), o.etype, mask)
            return None
        f = MATH.get(base) or MATH.get(name)
        if f is None: raise RunError(f"call to {fn[1:]}, which has no source: pass its source file, or avoid it")
        return f(*[expect(a) for a in args])

    # ------------------------------------------------------------------ functions
    def run_function(self, f, args, mask):
        regs = {name: a for (_, name), a in zip(f.params, args)}
        pc = torch.zeros(self.B, dtype=torch.long, device=self.dev); prev = torch.full((self.B,), -1, device=self.dev)
        ret, _ = self._blocks(f, regs, pc, prev, mask.clone())
        return ret

    def _blocks(self, f, regs, pc, prev, live, stop=None):
        """Run blocks for the `live` lanes until they return (or reach block index `stop`: they wait there).
        Returns (return value, prev) where prev tells, for lanes at `stop`, which block they came from."""
        B = self.B; ret = None; steps = 0; big = len(f.blocks)
        at_stop = torch.zeros(B, dtype=torch.bool, device=self.dev)
        while True:
            if stop is not None:
                arrived = live & (pc == stop); at_stop = at_stop | arrived; live = live & ~arrived
            if not bool(live.any()): break
            cur = int(torch.where(live, pc, torch.full_like(pc, big)).min())
            m = live & (pc == cur)
            blk = f.blocks[cur]
            steps += 1
            if steps > self.max_steps: raise RunError(f"{f.name}: more than {self.max_steps} block steps (endless loop?)")
            if self.vectorize:                                            # a loop nest: run all its iterations at once
                from .vectorize import try_nest
                done = try_nest(self, f, cur, regs, m, prev)
                if done is not None:
                    pc = torch.where(m, done[0], pc); prev = torch.where(m, done[1], prev); continue
            if blk.phis:
                fresh = m & (prev != JOINED)                              # JOINED: phis already set by a soft branch
                new = []
                for ph in blk.phis:
                    acc = None
                    for v, src in ph.extra["incoming"]:
                        if v[0] == "reg" and v[1] not in regs: continue
                        val = self.value(v, regs)
                        acc = val if acc is None else merge(prev == f.index[src], val, acc)
                    if acc is not None: new.append((ph.name, acc))       # None: only lanes joined by a soft branch
                for name, v in new: self._set(regs, name, v, fresh)
            for ins in blk.insts: self.execute(ins, regs, m)
            t = blk.term
            if t.op == "br":
                if len(t.extra["targets"]) == 1: nxt = torch.tensor(f.index[t.extra["targets"][0]], device=self.dev)
                else:
                    c = self.value(t.args[0], regs)
                    if isinstance(c, Soft):
                        j = self._soft_branch(f, cur, list(zip(t.extra["targets"], (c.p, 1 - c.p))), regs, m)
                        if j is not None:
                            pc = torch.where(m, j, pc); prev = torch.where(m, JOINED, prev); continue
                    nxt = torch.where(hard(c), f.index[t.extra["targets"][0]], f.index[t.extra["targets"][1]])
                pc = torch.where(m, nxt, pc); prev = torch.where(m, cur, prev)
            elif t.op == "switch":
                v = self.value(t.args[0], regs)
                if isinstance(v, Mix):                                    # a learned choice: every case, weighted
                    weights, total = {}, 0
                    for k, lbl in t.extra["cases"]:
                        pk = (v.W * (v.V == wrap(k, t.type[1])).to(v.W.dtype)).sum(0); total = total + pk
                        weights[lbl] = weights.get(lbl, 0) + pk
                    weights[t.extra["default"]] = weights.get(t.extra["default"], 0) + (1 - total).clamp(min=0)
                    j = self._soft_branch(f, cur, [(lbl, w.expand(B) if torch.is_tensor(w) else torch.full((B,), float(w), device=self.dev))
                                                   for lbl, w in weights.items()], regs, m)
                    if j is not None:
                        pc = torch.where(m, j, pc); prev = torch.where(m, JOINED, prev); continue
                v = hard(v)
                nxt = torch.full((B,), f.index[t.extra["default"]], device=self.dev)
                for k, lbl in t.extra["cases"]: nxt = torch.where(v == wrap(k, t.type[1]), f.index[lbl], nxt)
                pc = torch.where(m, nxt, pc); prev = torch.where(m, cur, prev)
            elif t.op == "ret":
                if t.args:
                    v = self.value(t.args[0], regs)
                    if isinstance(v, Ptr): raise RunError("returning a pointer is not supported")
                    ret = (v if isinstance(v, (Mix, Soft)) else v.expand(B).clone()) if ret is None else merge(m, v, ret)
                live = live & ~m
            elif t.op == "unreachable":
                live = live & ~m
        return ret, prev

    def _soft_branch(self, f, cur, targets, regs, m):
        """A branch decided by learned numbers: run every side (targets: [(block label, weight per lane)]) for the
        lanes in m and blend everything that differs where they meet again (the immediate post-dominator). Returns
        that block's index, or None if the sides never meet (then the branch is taken the hard way)."""
        j = post_dominators(f).get(cur)
        if j is None: return None
        snapshot = [(o.data, o.lane) for o in self.objs]
        sides, weights = [], [w for _, w in targets]
        for target, _ in targets:
            for o, (d, l) in zip(self.objs, snapshot): o.data, o.lane = d, l
            r = dict(regs)
            pc = torch.full((self.B,), f.index[target], device=self.dev); prev = torch.full((self.B,), cur, device=self.dev)
            _, prev = self._blocks(f, r, pc, prev, m.clone(), stop=j)
            vals = []
            for ph in f.blocks[j].phis:                               # what each phi of the meeting block receives
                acc = None
                for v, src in ph.extra["incoming"]:
                    if v[0] == "reg" and v[1] not in r: continue
                    val = self.value(v, r)
                    acc = val if acc is None else merge(prev == f.index[src], val, acc)
                vals.append(acc)
            sides.append((vals, [(o.data, o.lane) for o in self.objs]))
        for i, ph in enumerate(f.blocks[j].phis):
            self._set(regs, ph.name, blend(ph.type, weights, [vals[i] for vals, _ in sides]), m)
        for k in range(len(snapshot)):                               # memory that existed before, written on any side
            o = self.objs[k]; versions = [mem[k] for _, mem in sides]
            base = snapshot[k][0]
            if all(d is base for d, _ in versions): o.data, o.lane = base, snapshot[k][1]; continue
            if all(d is versions[0][0] for d, _ in versions): o.data, o.lane = versions[0]; continue
            ds = torch.stack([d if l else d.unsqueeze(0).expand(self.B, -1) for d, l in versions])      # (sides, B, n)
            w = torch.stack(weights).reshape(len(weights), -1, 1)
            if ds.is_floating_point(): o.data = (w.to(ds.dtype) * ds).sum(0)
            else: o.data = ds.gather(0, w.argmax(0, keepdim=True).expand(1, *ds.shape[1:])).squeeze(0)
            o.lane = True
        return j

    def execute(self, ins, regs, m):
        op = ins.op
        if op == "store":
            self.store(self.value(ins.args[1], regs), self.value(ins.args[0], regs), ins.type, m); return
        if op == "call":
            if not bool(m.any()): return
            r = self.call(ins, regs, m)
            if ins.name: self._set(regs, ins.name, r, m)
            return
        if op == "alloca":
            n = int(hard(self.value(ins.extra["count"], regs)).max()) if ins.extra["count"] is not None else 1
            t = ins.extra["alloc"]; et = scalar_of(t); count = size_of(t) * n // (max(1, et[1] // 8))
            data = torch.zeros(self.B, count, dtype=DTYPE[et[1]] if et[0] == "fp" else _int_dtype(et), device=self.dev)
            self._set(regs, ins.name, Ptr(self.new_obj(data, et, lane=True), torch.tensor(0, device=self.dev)), m); return
        a = [self.value(v, regs) for v in ins.args]
        if op in ("add", "sub", "mul", "sdiv", "udiv", "srem", "urem", "shl", "lshr", "ashr", "and", "or", "xor",
                  "fadd", "fsub", "fmul", "fdiv", "frem"): r = self.binop(op, ins.type, *a)
        elif op == "fneg": r = -expect(a[0])
        elif op == "icmp": r = self.icmp(ins.extra["pred"], ins.extra["argtype"], *a)
        elif op == "fcmp": r = self.fcmp(ins.extra["pred"], expect(a[0]), expect(a[1]))
        elif op == "select": r = self.select(ins.type, a[0], a[1], a[2])
        elif op == "freeze": r = a[0]
        elif op == "load": r = self.load(a[0], ins.type)
        elif op == "getelementptr": r = self.gep(a[0], ins.extra["src"], a[1:])
        elif op in ("trunc", "zext", "sext", "fptosi", "fptoui", "sitofp", "uitofp", "fpext", "fptrunc", "bitcast",
                    "ptrtoint", "inttoptr", "addrspacecast"): r = self.cast(op, ins.extra["from"], ins.type, a[0])
        else: raise RunError(f"instruction {op} is not supported")
        self._set(regs, ins.name, r, m)


MATH = {
    # llvm intrinsics (name after "llvm.")
    "smax": torch.maximum, "smin": torch.minimum,
    "umax": lambda a, b: torch.where(unsigned(a, 32) >= unsigned(b, 32), a, b),
    "umin": lambda a, b: torch.where(unsigned(a, 32) <= unsigned(b, 32), a, b),
    "abs": lambda a, *_: torch.abs(a), "fabs": torch.abs, "sqrt": torch.sqrt, "exp": torch.exp, "exp2": torch.exp2,
    "log": torch.log, "log2": torch.log2, "log10": torch.log10, "pow": torch.pow, "sin": torch.sin, "cos": torch.cos,
    "floor": torch.floor, "ceil": torch.ceil, "trunc": torch.trunc, "rint": torch.round, "nearbyint": torch.round,
    "round": lambda a: torch.sign(a) * torch.floor(torch.abs(a) + 0.5), "copysign": torch.copysign,
    "maxnum": torch.fmax, "minnum": torch.fmin, "maximum": torch.maximum, "minimum": torch.minimum,
    "fmuladd": lambda a, b, c: a * b + c, "fma": lambda a, b, c: a * b + c,
    # C library
    "expf": torch.exp, "logf": torch.log, "sqrtf": torch.sqrt, "fabsf": torch.abs, "tanhf": torch.tanh, "tanh": torch.tanh,
    "sinf": torch.sin, "cosf": torch.cos, "tanf": torch.tan, "tan": torch.tan, "powf": torch.pow,
    "floorf": torch.floor, "ceilf": torch.ceil, "truncf": torch.trunc,
    "roundf": lambda a: torch.sign(a) * torch.floor(torch.abs(a) + 0.5),
    "fmaxf": torch.fmax, "fmax": torch.fmax, "fminf": torch.fmin, "fmin": torch.fmin, "expm1f": torch.expm1, "expm1": torch.expm1,
    "log1pf": torch.log1p, "log1p": torch.log1p, "erff": torch.erf, "erf": torch.erf, "atan2f": torch.atan2, "atan2": torch.atan2,
    "atanf": torch.atan, "atan": torch.atan, "asinf": torch.asin, "asin": torch.asin, "acosf": torch.acos, "acos": torch.acos,
    "log2f": torch.log2, "log10f": torch.log10, "exp2f": torch.exp2, "sinhf": torch.sinh, "coshf": torch.cosh,
}
