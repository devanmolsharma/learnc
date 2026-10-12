"""Flat virtual memory for the batched runner: every example (lane) gets its own byte-addressed memory, so programs
with pointers stored in memory (linked lists, trees), structs, malloc/free, pointer arithmetic and function pointers
run exactly, and float values keep their gradients through memory.

Address space (the same for every lane; contents differ per lane):
  0                       null
  DATA_BASE ...           globals (not learned), then inputs, then the stack, then the heap (malloc)
  LEARN_BASE + k*STRIDE   learned variable k: SHARED by all lanes (gradients flow; read-only to the program)
  FUNC_BASE + i           function i (function pointers)
Memory holds two parallel planes: floats (float64) and whole numbers / pointers (int64); a value lives at the byte
address where it was stored. Reading memory as a different type than it was written (type punning) is not supported.
"""
import torch
from .ir import align_of, field_offset, scalar_of, size_of
from .machine import DTYPE, MATH, Machine, Mix, Obj, Relaxed, RunError, Soft, as_mix, hard, merge, unsigned, wrap

DATA_BASE = 1 << 12
LEARN_BASE, LEARN_STRIDE = 1 << 44, 1 << 36
FUNC_BASE = 1 << 56
STACK_BYTES = 1 << 14
FLAT_LIBRARY = {"malloc", "calloc", "realloc", "free", "memcpy", "memmove", "memset", "strlen"}


def _align(x, a): return (x + a - 1) // a * a


def _layout(t, init, base, out):
    """(address, scalar type, value) for every scalar of a global's initial value."""
    k = t[0]
    if k in ("int", "fp", "ptr"):
        out.append((base, t, 0 if init is None else init)); return
    if k == "array":
        es = size_of(t[2])
        for i in range(t[1]): _layout(t[2], None if init is None else init[i], base + i * es, out)
        return
    if k == "struct":
        for i, f in enumerate(t[1]): _layout(f, None if init is None else init[i], base + field_offset(t, i), out)
        return
    raise RunError(f"cannot lay out global of type {t}")


class FlatMachine(Machine):
    def __init__(self, module, batch, device="cpu", learned=None, training=False, temperature=1.0, max_steps=1_000_000,
                 vectorize=False, heap_bytes=1 << 14):
        self.m, self.B, self.dev, self.max_steps = module, batch, torch.device(device), max_steps
        self.training, self.temperature, self.vectorize = training, temperature, False
        self.lanes = torch.arange(batch, device=self.dev)
        self._soft_depth = 0
        learned = learned or {}
        self.addr, self.learned_regions, inits = {}, [], []
        top = DATA_BASE
        for name, g in module.globals.items():
            if name in learned:                                        # shared, read-only region
                self.addr[name] = LEARN_BASE + len(self.learned_regions) * LEARN_STRIDE
                self.learned_regions.append((learned[name], scalar_of(g.type), name)); continue
            top = _align(top, max(1, align_of(g.type))); self.addr[name] = top
            _layout(g.type, g.init, top, inits); top += size_of(g.type)
        names = sorted(set(module.functions) | set(module.declarations))
        self.func_id = {n: i for i, n in enumerate(names)}; self.func_name = {i: n for n, i in self.func_id.items()}
        self.static_top = _align(top, 16)                              # inputs go here
        self.inits, self.heap_bytes = inits, heap_bytes
        self.mem_ready = False
        self.gptr = {}

    # ------------------------------------------------------------------ memory layout (built once inputs are placed)
    def input_pointer(self, data, etype, name):
        """Place one input array (B, n) and return its address (the same for every lane)."""
        es = max(1, etype[1] // 8) if etype[0] == "int" else etype[1] // 8
        a = _align(self.static_top, 16); self.static_top = a + es * data.shape[1]
        self.inits.append((a, ("input", es, etype), data))
        return torch.tensor(a, device=self.dev)

    def _build(self):
        self.stack_base = _align(self.static_top, 16); self.sp = self.stack_base
        self.heap_base = self.stack_base + STACK_BYTES
        M = self.heap_base + self.heap_bytes
        f = torch.zeros(self.B, M, dtype=torch.float64, device=self.dev)
        i = torch.zeros(self.B, M, dtype=torch.int64, device=self.dev)
        for a, t, v in self.inits:
            if t[0] == "input":
                _, es, et = t; idx = a + es * torch.arange(v.shape[1], device=self.dev)
                if et[0] == "fp": f[:, idx] = v.to(torch.float64)
                else: i[:, idx] = v.long()
            elif t[0] == "fp": f[:, a] = float(v)
            elif t[0] == "ptr": i[:, a] = 0 if v in (None, 0) else int(v)
            else: i[:, a] = int(v)
        self.objs = [Obj(f, ("fp", 64), True, "memory (floats)"), Obj(i, ("int", 64), True, "memory (integers)")]
        self.brk = torch.full((self.B,), self.heap_base, dtype=torch.int64, device=self.dev)
        self.mem_ready = True

    def _grow(self, need):
        M = self.objs[0].data.shape[1]
        if need <= M: return
        extra = max(need - M, M)
        for o in self.objs:
            o.data = torch.cat([o.data, torch.zeros(self.B, extra, dtype=o.data.dtype, device=self.dev)], 1)

    # ------------------------------------------------------------------ loads and stores
    def _addr(self, p):
        if isinstance(p, (Mix, Soft)): p = hard(p)
        if isinstance(p, tuple): raise RunError("unexpected pointer value")
        return torch.as_tensor(p, device=self.dev).long()

    def _learned_load(self, a, t):
        k = int(((a.flatten()[0] - LEARN_BASE) // LEARN_STRIDE).item())
        data, et, name = self.learned_regions[k]
        es = max(1, et[1] // 8) if et[0] == "int" else et[1] // 8
        idx = (a - LEARN_BASE - k * LEARN_STRIDE) // es
        if isinstance(data, Relaxed):                                  # a learned choice: all its values, weighted
            K = data.values.numel(); return Mix(data.values.reshape(K, 1).to(self.dev), data.W[idx].reshape(-1, K).T)
        idx = idx.clamp(0, data.shape[-1] - 1)
        return data[self.lanes, idx] if data.dim() == 2 else data[idx]

    def load(self, p, t):
        if not self.mem_ready: self._build()
        a = self._addr(p)
        learned = (a >= LEARN_BASE) & (a < FUNC_BASE)
        if bool(learned.all()): return self._learned_load(a, t)
        if bool(learned.any()): raise RunError("one load reads learned values for some examples and memory for others")
        o = self.objs[0] if t[0] == "fp" else self.objs[1]
        idx = a.clamp(0, o.data.shape[1] - 1)
        v = o.data[:, idx] if idx.dim() == 0 else o.data[self.lanes, idx]
        if t[0] == "fp": return v.to(DTYPE[t[1]])
        if t == ("int", 1): return v != 0
        if t[0] == "int": return wrap(v, t[1])
        return v                                                       # a pointer

    def store(self, p, v, t, mask):
        if not self.mem_ready: self._build()
        a = self._addr(p); v = hard(v)
        if bool(((a >= LEARN_BASE) & mask).any()): raise RunError("the program writes to a learned variable")
        o = self.objs[0] if t[0] == "fp" else self.objs[1]
        a = (a.expand(self.B) if a.dim() == 0 else a).clamp(0, o.data.shape[1] - 1)
        v = torch.as_tensor(v, device=self.dev)
        v = v.to(torch.float64) if t[0] == "fp" else v.long()
        old = o.data[self.lanes, a]
        new = torch.where(mask, v.expand_as(old) if v.dim() == 0 else v, old)
        if self._soft_depth: o.data = o.data.index_put((self.lanes, a), new)       # keep the snapshot intact
        else: o.data.index_put_((self.lanes, a), new)

    # ------------------------------------------------------------------ values that are pointers
    def const(self, v):
        k = v[0]
        if k == "null": return torch.tensor(0, device=self.dev)
        if k in ("undef", "zero") and v[1][0] == "ptr": return torch.tensor(0, device=self.dev)
        if k == "glob":
            if v[1] in self.addr: return torch.tensor(self.addr[v[1]], device=self.dev)
            if v[1] in self.func_id: return torch.tensor(FUNC_BASE + self.func_id[v[1]], device=self.dev)
            raise RunError(f"unknown global {v[1]}")
        return super().const(v)

    def gep(self, base, src, idx):
        off = self._addr(base)
        off = off + hard(idx[0]).long() * size_of(src)
        t = src
        for i in idx[1:]:
            if t[0] == "array": off = off + hard(i).long() * size_of(t[2]); t = t[2]
            elif t[0] == "struct": k = int(hard(i)); off = off + field_offset(t, k); t = t[1][k]
            else: raise RunError(f"getelementptr into {t}")
        return off

    def icmp(self, pred, t, a, b):
        if t == ("ptr",): t = ("int", 64)
        return super().icmp(pred, t, a, b)

    def cast(self, op, ft, tt, v):
        if op in ("ptrtoint", "inttoptr"): return hard(v).long() if tt[0] == "ptr" or tt[1] >= 64 else wrap(hard(v).long(), tt[1])
        if op == "bitcast" and ft[0] == "ptr": return v
        return super().cast(op, ft, tt, v)

    # ------------------------------------------------------------------ calls, stack, library
    def run_function(self, f, args, mask):
        if not self.mem_ready: self._build()
        sp = self.sp
        try: return super().run_function(f, args, mask)
        finally: self.sp = sp                                          # the callee's stack frame is gone

    def _soft_branch(self, f, cur, targets, regs, m):
        self._soft_depth += 1
        try: return super()._soft_branch(f, cur, targets, regs, m)
        finally: self._soft_depth -= 1

    def execute(self, ins, regs, m):
        if ins.op == "alloca":
            if not self.mem_ready: self._build()
            n = int(hard(self.value(ins.extra["count"], regs)).max()) if ins.extra["count"] is not None else 1
            t = ins.extra["alloc"]; a = _align(self.sp, max(8, align_of(t)))
            self.sp = a + size_of(t) * n
            if self.sp > self.heap_base: raise RunError(f"stack overflow: more than {STACK_BYTES} bytes of local arrays")
            self._set(regs, ins.name, torch.tensor(a, device=self.dev), m); return
        return super().execute(ins, regs, m)

    def call(self, inst, regs, mask):
        fn = inst.extra.get("fn")
        if fn is None:                                                  # through a function pointer
            target = self._addr(self.value(inst.extra["fn_reg"], regs)) - FUNC_BASE
            target = target.expand(self.B) if target.dim() == 0 else target
            out = None
            for fid in torch.unique(target[mask]).tolist():
                name = self.func_name.get(int(fid))
                if name is None: raise RunError("call through an invalid function pointer")
                sub = mask & (target == fid)
                fake = type(inst)(inst.op, inst.name, inst.type, inst.args, {**inst.extra, "fn": name})
                r = self.call(fake, regs, sub)
                if r is not None: out = r if out is None else merge(sub, r, out)
            return out
        if fn in self.m.functions: return super().call(inst, regs, mask)
        name = fn[1:]
        base = name.split(".")[1] if name.startswith("llvm.") else name
        args = [self.value(a, regs) for a in inst.args]
        if base in ("lifetime", "assume", "dbg", "experimental", "invariant", "sideeffect"): return None
        if base in ("malloc", "calloc", "realloc"):
            n = hard(args[0]).long() * (hard(args[1]).long() if base == "calloc" else 1) if base != "realloc" else hard(args[1]).long()
            n = n.expand(self.B) if n.dim() == 0 else n
            a = torch.div(self.brk + 15, 16, rounding_mode="floor") * 16
            new_brk = torch.where(mask, a + n, self.brk)
            self._grow(int(new_brk.max()) + 16); self.brk = new_brk
            if base == "realloc":
                old = self._addr(args[0]); self._copy(a, old.expand(self.B) if old.dim() == 0 else old, n, mask & (old != 0))
            return a
        if base == "free": return None
        if base in ("memcpy", "memmove"):
            dst, src, n = self._addr(args[0]), self._addr(args[1]), hard(args[2]).long()
            self._copy(dst.expand(self.B) if dst.dim() == 0 else dst, src.expand(self.B) if src.dim() == 0 else src,
                       n.expand(self.B) if n.dim() == 0 else n, mask); return None
        if base == "memset":
            dst, byte, n = self._addr(args[0]), hard(args[1]).long(), hard(args[2]).long()
            dst = dst.expand(self.B) if dst.dim() == 0 else dst; n = n.expand(self.B) if n.dim() == 0 else n
            N = int(n[mask].max()) if bool(mask.any()) else 0
            if N:
                off = torch.arange(N, device=self.dev); ok = mask[:, None] & (off[None] < n[:, None])
                addr = (dst[:, None] + off[None]).clamp(0, self.objs[0].data.shape[1] - 1)
                rows = self.lanes[:, None].expand_as(addr)
                for o, val in ((self.objs[1], wrap(byte, 8)), (self.objs[0], 0.0)):
                    cur = o.data[rows, addr]; new = torch.where(ok, torch.as_tensor(val, dtype=o.data.dtype, device=self.dev).expand_as(cur), cur)
                    if self._soft_depth: o.data = o.data.index_put((rows, addr), new)
                    else: o.data.index_put_((rows, addr), new)
            return None
        if base == "strlen":
            s = self._addr(args[0]); s = s.expand(self.B) if s.dim() == 0 else s
            n = torch.zeros(self.B, dtype=torch.int64, device=self.dev); done = ~mask
            while not bool(done.all()):
                c = self.objs[1].data[self.lanes, (s + n).clamp(0, self.objs[1].data.shape[1] - 1)]
                done = done | (c == 0); n = torch.where(done, n, n + 1)
                if int(n.max()) > self.objs[1].data.shape[1]: raise RunError("strlen ran off the end of memory")
            return n
        f = MATH.get(base) or MATH.get(name)
        if f is None: raise RunError(f"call to {name}, which has no source and is not built in: pass its source file")
        return f(*[a.p if isinstance(a, Soft) else (a.W * a.V).sum(0) if isinstance(a, Mix) else a for a in args])

    def _copy(self, dst, src, n, mask):
        N = int(n[mask].max()) if bool(mask.any()) else 0
        if not N: return
        off = torch.arange(N, device=self.dev); ok = mask[:, None] & (off[None] < n[:, None])
        M = self.objs[0].data.shape[1]
        sa, da = (src[:, None] + off[None]).clamp(0, M - 1), (dst[:, None] + off[None]).clamp(0, M - 1)
        rows = self.lanes[:, None].expand_as(sa)
        for o in self.objs:
            vals = o.data[rows, sa]; cur = o.data[rows, da]
            new = torch.where(ok, vals, cur)
            if self._soft_depth: o.data = o.data.index_put((rows, da), new)
            else: o.data.index_put_((rows, da), new)
