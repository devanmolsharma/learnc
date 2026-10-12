"""Run whole loop nests at once: every loop level becomes a tensor dimension instead of block-by-block steps.

A loop nest qualifies (checked from the IR) when every loop in it has
  - a counter i = start, start + 1, ... and an exit test against a constant (so the trip count is known),
  - no branches except loop exits, no switch, no allocas, no calls except math functions,
and (checked while running) when
  - every other value carried around a loop is a running total, acc = acc OP term with OP in + * & | ^ (also fsub),
    possibly running through inner loops (acc flows into an inner loop's total and back): a running total may only
    be added to, never used otherwise inside the nest,
  - stores go to distinct elements, and nothing the nest stores is also loaded in it.
Anything else makes the machine run the nest block by block, so results never change. Outside training, float
totals are accumulated in C's order (bit-identical with the native program); while training they are summed directly.
Inside a nest every value has shape (lanes or 1, d1, ..., dD): loop level k iterates along dimension k.
"""
import torch
from .machine import MATH, Mix, Ptr, Soft

REDUCE = {"fadd", "add", "fmul", "mul", "and", "or", "xor", "fsub"}
ELEMENTWISE = {"add", "sub", "mul", "sdiv", "udiv", "srem", "urem", "shl", "lshr", "ashr", "and", "or", "xor", "fadd",
               "fsub", "fmul", "fdiv", "frem"}
CASTS = {"trunc", "zext", "sext", "fptosi", "fptoui", "sitofp", "uitofp", "fpext", "fptrunc", "bitcast"}
_ADD = {"fadd": "fadd", "fsub": "fadd", "add": "add", "fmul": "fmul", "mul": "mul", "and": "and", "or": "or", "xor": "xor"}


class NotVectorizable(Exception):
    pass


# ---------------------------------------------------------------------------- loop structure (static)
class Loop:
    def __init__(self, header, blocks):
        self.header, self.blocks, self.inner, self.parent, self.plan, self.planned = header, blocks, [], None, None, False


def _succ(f, b):
    t = f.blocks[b].term
    if t.op == "br": return [f.index[x] for x in t.extra["targets"]]
    if t.op == "switch": return [f.index[t.extra["default"]]] + [f.index[x] for _, x in t.extra["cases"]]
    return []


def find_loops(f):
    """header block index -> Loop (natural loops, with nesting)."""
    if "_loops" in f.__dict__: return f._loops
    n = len(f.blocks); preds = {i: [] for i in range(n)}
    for i in range(n):
        for s in _succ(f, i): preds[s].append(i)
    loops = {}
    for u in range(n):
        for v in _succ(f, u):
            if v <= u:                                                    # back edge u -> header v
                body, stack = {v, u}, ([u] if u != v else [])
                while stack:
                    x = stack.pop()
                    if x == v: continue                                       # never walk back past the header
                    for p in preds[x]:
                        if p not in body: body.add(p); stack.append(p)
                if v in loops: loops[v].blocks |= body
                else: loops[v] = Loop(v, body)
    for L in loops.values():
        outer = [M for M in loops.values() if M is not L and L.blocks < M.blocks]
        if outer: L.parent = min(outer, key=lambda M: len(M.blocks)); L.parent.inner.append(L)
    f._loops, f._preds = loops, preds
    return loops


class Plan:
    def __init__(self, **kw): self.__dict__.update(kw)


_TESTS = {"eq": lambda x, y: x == y, "ne": lambda x, y: x != y, "slt": lambda x, y: x < y, "sle": lambda x, y: x <= y,
          "sgt": lambda x, y: x > y, "sge": lambda x, y: x >= y, "ult": lambda x, y: x < y, "ule": lambda x, y: x <= y,
          "ugt": lambda x, y: x > y, "uge": lambda x, y: x >= y}


def plan_loop(module, f, L):
    """Static facts about a loop (counter, trip count, exit), or None if the loop (or one inside it) cannot qualify."""
    if L.planned: return L.plan
    L.planned = True
    preds = f._preds
    pre = [p for p in preds[L.header] if p not in L.blocks]
    latches = [p for p in preds[L.header] if p in L.blocks]
    exits = [(b, s) for b in L.blocks for s in _succ(f, b) if s not in L.blocks]
    if len(pre) != 1 or len(latches) != 1 or len(exits) != 1: return None
    exiting, exit_ = exits[0]
    t = f.blocks[exiting].term
    if t.op != "br" or len(t.args) != 1: return None
    inner = [plan_loop(module, f, M) for M in L.inner]
    if any(p is None for p in inner): return None
    def nested(M): return [M] + [x for K in M.inner for x in nested(K)]
    inner_exiting = {K.plan.exiting for M in L.inner for K in nested(M)}       # exits of every loop inside
    defs = {}
    for b in L.blocks:
        blk = f.blocks[b]
        if blk.term.op != "br": return None
        if len(blk.term.args) == 1 and b != exiting and b not in inner_exiting: return None     # a data-dependent branch
        for i in blk.insts + blk.phis:
            if i.name: defs[i.name] = i
            if i.op == "alloca": return None
            if i.op == "call":
                if i.extra["fn"] in module.functions: return None
                name = i.extra["fn"][1:]; base = name.split(".")[1] if name.startswith("llvm.") else name
                if base not in MATH and base != "lifetime": return None
    hdr = f.blocks[L.header]; ind = None
    pre_name, latch_name = f.blocks[pre[0]].name, f.blocks[latches[0]].name
    for ph in hdr.phis:
        inc = dict((src, v) for v, src in ph.extra["incoming"])
        if set(inc) != {pre_name, latch_name}: return None
        init, back = inc[pre_name], inc[latch_name]
        upd = defs.get(back[1]) if back[0] == "reg" else None
        if (ind is None and init[0] == "const" and upd is not None and upd.op == "add" and ("reg", ph.name) in upd.args
                and any(a[0] == "const" and a[1] == 1 for a in upd.args)):
            ind = (ph.name, init[1], upd.name)
    if ind is None: return None
    name, start, nxt = ind
    cond = defs.get(t.args[0][1]) if t.args[0][0] == "reg" else None
    if cond is None or cond.op != "icmp" or cond.extra["pred"] not in _TESTS: return None
    def val(a, i):
        if a == ("reg", name): return i
        if a == ("reg", nxt): return i + 1
        if a[0] == "const": return a[1]
        raise KeyError
    test = _TESTS[cond.extra["pred"]]
    stay = f.index[t.extra["targets"][0]] in L.blocks
    trips, i = 0, start
    try:
        while True:                                                       # simulate the counter: the trip count
            trips += 1
            if test(val(cond.args[0], i), val(cond.args[1], i)) != stay: break
            i += 1
            if trips > 1_000_000: return None
    except KeyError:
        return None
    L.plan = Plan(pre=pre[0], latch=latches[0], exiting=exiting, exit=exit_, ind=name, start=start, next=nxt,
                  trips=trips, reductions=[ph for ph in hdr.phis if ph.name != name])
    return L.plan


def depth(L): return 1 + max([depth(M) for M in L.inner] or [0])


# ---------------------------------------------------------------------------- running a nest
class Total:
    """A running total inside the nest: the value `phi` had when the loop started, plus the pieces added since.
    Each piece is (tensor, loop dimensions it still has to be folded over)."""
    def __init__(self, phi, op, pieces): self.phi, self.op, self.pieces = phi, op, pieces


class Nest:
    def __init__(self, mach, f, outer, regs, m):
        self.mach, self.f, self.outer, self.regs, self.m = mach, f, outer, regs, m
        self.D, self.B = depth(outer), mach.B
        self.stores, self.loaded = [], set()

    def shape(self, x):
        if isinstance(x, Ptr): return Ptr(x.obj, self.shape(x.off))
        if isinstance(x, (Mix, Total)): raise NotVectorizable
        if isinstance(x, Soft): return Soft(self.shape(x.p))
        x = torch.as_tensor(x, device=self.mach.dev)
        if x.dim() == 0: return x.reshape((1,) * (self.D + 1))
        if x.dim() == 1: return x.reshape((-1,) + (1,) * self.D)
        return x

    def value(self, v, env):
        if v[0] == "reg" and v[1] in env: return env[v[1]]
        return self.shape(self.mach.value(v, self.regs))

    def run(self):
        env = {}
        self.loop(self.outer, 1, env)
        mach = self.mach
        if self.loaded & {o for o, _, _ in self.stores}: raise NotVectorizable              # reads what it writes
        prepared = []
        for obj, idx, val in self.stores:                                                   # check everything first
            o = mach.objs[obj]
            if not o.writable: raise NotVectorizable
            full = torch.broadcast_shapes(idx.shape, val.shape)
            idx = idx.expand(full).expand(self.B, *full[1:]).reshape(self.B, -1)
            s = idx.sort(1).values
            if (s[:, 1:] == s[:, :-1]).any(): raise NotVectorizable                              # same element twice
            prepared.append((o, idx, val.expand(full).expand(self.B, *full[1:]).reshape(self.B, -1)))
        for o, idx, val in prepared:
            if not o.lane: o.data = o.data.unsqueeze(0).expand(self.B, -1).clone(); o.lane = True
            idx = idx.clamp(0, o.data.shape[1] - 1)
            old = o.data.gather(1, idx)
            o.data = o.data.scatter(1, idx, torch.where(self.m.reshape(-1, 1), val.to(o.data.dtype), old))
        for name, v in env.items():                                                         # values seen after the nest
            if isinstance(v, Total): raise NotVectorizable
            mach._set(self.regs, name, self.flat(v), self.m)

    def flat(self, v):
        if isinstance(v, Ptr): return Ptr(v.obj, self.flat(v.off))
        if isinstance(v, Soft): return Soft(self.flat(v.p))
        v = v.reshape(v.shape[0])
        return v.reshape(()) if v.shape[0] == 1 else v

    def loop(self, L, k, env):
        f, mach, P = self.f, self.mach, L.plan
        N = P.trips
        shp = [1] * (self.D + 1); shp[k] = N
        it = torch.arange(P.start, P.start + N, device=mach.dev).reshape(shp)
        env[P.ind] = it; env[P.next] = it + 1
        totals = {}
        pre_name, latch_name = f.blocks[P.pre].name, f.blocks[P.latch].name
        for ph in P.reductions:
            inc = dict((src, v) for v, src in ph.extra["incoming"])
            back = inc[latch_name]
            if back[0] != "reg": raise NotVectorizable
            totals[ph.name] = (self.value(inc[pre_name], env), back[1], ph)
            env[ph.name] = Total(ph.name, None, [])
        inner = {M.header: M for M in L.inner}
        skip = set()
        for b in sorted(L.blocks):
            if b in skip: continue
            if b in inner: self.loop(inner[b], k + 1, env); skip |= inner[b].blocks; continue
            blk = f.blocks[b]
            if b != L.header:
                for ph in blk.phis:
                    srcs = [v for v, s in ph.extra["incoming"] if f.index[s] in L.blocks]
                    if len(srcs) != 1: raise NotVectorizable
                    env[ph.name] = self.value(srcs[0], env)
            for ins in blk.insts: self.execute(ins, env)
        backs = set()
        for name, (init, back, ph) in totals.items():                                     # fold this loop's totals
            tot = env.get(back)
            if not isinstance(tot, Total) or tot.phi != name: raise NotVectorizable
            pieces = [(pc, dims + (k,)) for pc, dims in tot.pieces]
            if isinstance(init, Total): env[back] = Total(init.phi, tot.op, init.pieces + pieces)    # runs on outward
            else: env[back] = self.fold(init, tot.op or "fadd", pieces, ph.type)
            env.pop(name, None); backs.add(back)
        for b in L.blocks:                                                                # the rest: last iteration
            for ins in f.blocks[b].insts + f.blocks[b].phis:
                if ins.name in env and ins.name not in backs and not isinstance(env[ins.name], Total):
                    env[ins.name] = self.last(env[ins.name], k)
        one = [1] * (self.D + 1)
        env[P.ind] = torch.tensor(P.start + N - 1, device=mach.dev).reshape(one)
        env[P.next] = torch.tensor(P.start + N, device=mach.dev).reshape(one)

    def last(self, v, k):
        if isinstance(v, Ptr): return Ptr(v.obj, self.last(v.off, k))
        if isinstance(v, Soft): return Soft(self.last(v.p, k))
        return v.narrow(k, v.shape[k] - 1, 1) if v.shape[k] > 1 else v

    def fold(self, init, op, pieces, t):
        """init OP every element of every piece. Pieces were added in program order; each lists the loop dimensions
        to fold, innermost last. Outside training, float totals are added in exactly C's order."""
        mach, add = self.mach, _ADD[op]
        if not pieces: return init
        if mach.training or t[0] != "fp":
            acc = init
            for pc, dims in pieces:
                for d in sorted(set(dims), reverse=True):
                    if pc.shape[d] > 1: pc = _reduce(add, pc, d, mach, t)
                acc = mach.binop(add, t, acc, pc)
            return acc
        outer = min(min(dims) for _, dims in pieces)                                      # this loop's own dimension
        N = max(pc.shape[outer] for pc, _ in pieces)
        acc = init
        for i in range(N):
            for pc, dims in pieces:
                sl = pc.narrow(outer, i, 1) if pc.shape[outer] > 1 else pc
                rest = sorted(d for d in set(dims) if d != outer)
                for combo in _product([range(sl.shape[d]) for d in rest]):
                    x = sl
                    for d, j in zip(rest, combo):
                        if x.shape[d] > 1: x = x.narrow(d, j, 1)
                    acc = mach.binop(add, t, acc, x)
        return acc

    def execute(self, ins, env):
        mach, op = self.mach, ins.op
        if op == "call" and ins.extra["fn"].startswith("@llvm.lifetime"): return
        if op == "store":
            ptr, val = self.value(ins.args[1], env), self.value(ins.args[0], env)
            if not isinstance(ptr, Ptr) or isinstance(val, (Total, Mix, Soft)): raise NotVectorizable
            o, idx = mach._elem(ptr, ins.type)
            self.stores.append((ptr.obj, idx, val)); return
        a = [self.value(v, env) for v in ins.args]
        totals = [x for x in a if isinstance(x, Total)]
        if totals:                                                                       # a running total may only grow
            if op not in REDUCE or len(totals) != 1 or len(a) != 2: raise NotVectorizable
            tot = totals[0]; other = a[1] if a[0] is tot else a[0]
            if op == "fsub":
                if a[0] is not tot: raise NotVectorizable
                other = -other
            if tot.op is not None and _ADD[tot.op] != _ADD[op]: raise NotVectorizable
            if isinstance(other, (Mix, Ptr)): raise NotVectorizable
            env[ins.name] = Total(tot.phi, op, tot.pieces + [(other.p if isinstance(other, Soft) else other, ())]); return
        if op in ELEMENTWISE: r = mach.binop(op, ins.type, *a)
        elif op == "fneg": r = -a[0]
        elif op == "icmp": r = mach.icmp(ins.extra["pred"], ins.extra["argtype"], *a)
        elif op == "fcmp": r = mach.fcmp(ins.extra["pred"], *a)
        elif op == "select": r = mach.select(ins.type, *a)
        elif op == "freeze": r = a[0]
        elif op == "getelementptr": r = mach.gep(a[0], ins.extra["src"], a[1:])
        elif op == "load":
            p = a[0]
            if not isinstance(p, Ptr) or isinstance(p.off, Mix): raise NotVectorizable
            o, idx = mach._elem(p, ins.type)
            self.loaded.add(p.obj)
            if o.relax is not None:                     # a learned choice used only as a number: its expected value
                users = [u for b in self.outer.blocks for u in self.f.blocks[b].insts if ("reg", ins.name) in u.args]
                if not users or any(u.op not in ("sitofp", "uitofp") for u in users): raise NotVectorizable
                r = (o.relax.W @ o.relax.values.to(o.relax.W.dtype))[idx]
            elif o.lane:
                full = idx.expand(self.B, *idx.shape[1:])
                r = o.data.gather(1, full.reshape(self.B, -1)).reshape(full.shape)
            else: r = o.data[idx]
        elif op == "call":
            name = ins.extra["fn"][1:]; base = name.split(".")[1] if name.startswith("llvm.") else name
            r = MATH[base](*[x.p if isinstance(x, Soft) else x for x in a])
        elif op in CASTS: r = mach.cast(op, ins.extra["from"], ins.type, a[0])
        else: raise NotVectorizable
        if isinstance(r, Mix): raise NotVectorizable
        env[ins.name] = r


def _reduce(op, x, d, mach, t):
    if op in ("fadd", "add"): r = x.sum(d, keepdim=True)
    elif op in ("fmul", "mul"): r = x.prod(d, keepdim=True)
    else:
        logical = x.dtype == torch.bool
        fn = {"and": torch.logical_and if logical else torch.bitwise_and, "or": torch.logical_or if logical else torch.bitwise_or,
              "xor": torch.logical_xor if logical else torch.bitwise_xor}[op]
        r = x.narrow(d, 0, 1)
        for j in range(1, x.shape[d]): r = fn(r, x.narrow(d, j, 1))
    if op in ("add", "mul") and t[0] == "int": r = mach.binop("add", t, r, torch.zeros_like(r))     # wrap to the bit width
    return r


def _product(ranges):
    if not ranges: yield (); return
    for i in ranges[0]:
        for rest in _product(ranges[1:]): yield (i,) + rest


def try_nest(mach, f, cur, regs, m, prev):
    """If block `cur` heads a loop nest that qualifies and every lane is just entering it, run it all at once.
    Returns (exit block index, exiting block index), or None (then the nest runs block by block, unchanged)."""
    L = find_loops(f).get(cur)
    if L is None or plan_loop(mach.m, f, L) is None: return None
    if not bool((prev[m] == L.plan.pre).all()): return None
    snapshot = [(o.data, o.lane) for o in mach.objs]
    saved = dict(regs)
    try:
        Nest(mach, f, L, regs, m).run()
    except (NotVectorizable, IndexError, RuntimeError):
        for o, (d, l) in zip(mach.objs, snapshot): o.data, o.lane = d, l
        regs.clear(); regs.update(saved)
        return None
    return L.plan.exit, L.plan.exiting
