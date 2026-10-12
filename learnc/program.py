"""A C program with learned variables: run it on batches of examples through its LLVM IR, and fit the learned variables."""
import os
import torch
from .build import BuildError, c_type, compile_ir, signatures
from .ir import scalar_of, size_of
from .machine import DTYPE, Machine, Mix, Ptr, Relaxed, RunError, Soft, _flatten, unsigned, wrap


class Learned:
    """One `learned` global: name, element type, shape, range (whole numbers) and initial values."""
    def __init__(self, name, gtype, annotation, init):
        self.name, self.etype = name.lstrip("@"), scalar_of(gtype)
        self.count = size_of(gtype) // max(1, self.etype[1] // 8)
        self.shape = _shape(gtype)
        parts = annotation.split(":", 2)
        self.range, self.choices, self.ops = None, None, None
        self.numeric = len(parts) == 2 and parts[1] == "num"        # learned_num: a whole number tuned smoothly
        if self.numeric:
            if self.etype[0] != "int": raise BuildError(f"{self.name}: learned_num is for whole numbers; use `learned` for floats")
            self.init = init; return
        if len(parts) == 3 and parts[1] == "of":
            self.choices = sorted({int(v) for v in parts[2].split(",")})
        elif len(parts) == 3 and parts[1] == "op":
            self.ops = [v.strip() for v in parts[2].split(",")]
            bad = [o for o in self.ops if o not in OPERATORS]
            if bad: raise BuildError(f"{self.name}: unknown operator(s) {bad}; use {list(OPERATORS)}")
            self.choices = sorted(OPERATORS[o] for o in self.ops)
        elif len(parts) == 3:
            self.range = (int(parts[1]), int(parts[2])); self.choices = list(range(self.range[0], self.range[1] + 1))
        if self.etype[0] == "int" and self.choices is None:
            raise BuildError(f"learned whole-number variable {self.name} needs its values: learned_in(lo, hi) or learned_of(...)")
        if self.etype[0] == "fp" and self.choices is not None:
            raise BuildError(f"{self.name}: learned_in / learned_of / learned_op are for whole numbers; use `learned` for floats")
        self.init = init

    def describe(self, v):
        """A value of this variable for people (operator names for learned_op)."""
        if self.ops: return {c: n for n, c in OPERATORS.items()}[int(v)]
        return v

    def __repr__(self):
        if getattr(self, "numeric", False): return f"<learned {self.name}{list(self.shape) if self.shape else ''}: whole number, tuned smoothly>"
        kind = ("one of " + ", ".join(self.ops) if self.ops else f"one of {self.range[0]}..{self.range[1]}" if self.range
                else f"one of {self.choices}" if self.choices else "real")
        return f"<learned {self.name}{list(self.shape) if self.shape else ''}: {kind}>"


OPERATORS = {"ADD": 0, "SUB": 1, "MUL": 2, "DIV": 3, "MAX": 4, "MIN": 5, "LEFT": 6, "RIGHT": 7}     # as in learnc.h


def _shape(t): return (t[1],) + _shape(t[2]) if t[0] == "array" else ()


def log_likelihood(out, target):
    """Per lane: log P(output == target) for whole-number outputs, -(output - target)^2 for real outputs."""
    if isinstance(out, Mix):
        t = torch.as_tensor(target, device=out.V.device)
        hit = (out.V == t.long().reshape(1, -1)).to(out.W.dtype)
        return torch.log((out.W * hit).sum(0).clamp(min=1e-6))
    if isinstance(out, Soft):
        t = torch.as_tensor(target, device=out.p.device); p = out.p.clamp(1e-6, 1 - 1e-6)
        return torch.where(t.bool(), torch.log(p), torch.log(1 - p))
    t = torch.as_tensor(target, device=out.device)
    if t.is_floating_point() or out.is_floating_point(): return -(out.double() - t.double()) ** 2
    return torch.log((out.long() == t.long()).double().clamp(min=1e-4))     # no gradient: only a reward for choices


class _NoSignal(Exception):
    pass


class Program:
    """Program(path, sources=(), entry=None): compile a C file (plus the sources of libraries it uses).
    The entry function (default: the last one defined) takes one example; calls take a batch."""

    def __init__(self, path, sources=(), entry=None, device=None, flags=()):
        """flags: extra compiler flags for every file, e.g. ["-I", "third_party/include", "-DNDEBUG"]."""
        self.path, self.paths = os.path.abspath(path), [os.path.abspath(path)] + [os.path.abspath(s) for s in sources]
        self.flags = list(flags)
        self.module, self.ir_text = compile_ir(self.paths, self.flags)
        sigs = signatures(self.path, self.flags)
        if not sigs: raise BuildError(f"{path} defines no functions")
        self.entry = entry or list(sigs)[-1]
        if self.entry not in sigs: raise BuildError(f"{path} has no function {self.entry}; it defines {list(sigs)}")
        if "@" + self.entry not in self.module.functions and "@" + self.entry not in self.module.unsupported:
            raise BuildError(f"{self.entry} disappeared during compilation: make it non-static")
        self.native_reason, self.flat = self._runner_problem()   # native_reason None: the runner can execute it
        ret, params = sigs[self.entry]
        self.ret, self.params = c_type(ret), [(n, c_type(t)) for n, t in params]
        self.learned = {}
        for g, ann in self.module.annotations.items():
            if ann.startswith("learnc") and g in self.module.globals:
                gl = self.module.globals[g]; flat = []; _flatten(gl.init, gl.type, flat)
                L = Learned(g, gl.type, ann, flat); self.learned[L.name] = L
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.values, self.history = None, []
        self.vectorize = True                     # run simple loops all at once (see vectorize.py)
        self.lm_settings = {}                     # optimizer="lm": penalty (start), fade (per step), floor

    @property
    def native_only(self): return self.native_reason is not None

    def _runner_problem(self):
        """(problem, flat): problem is why the batched runner cannot run the entry function (None if it can); flat is
        True when it needs the flat virtual memory (pointers in memory, malloc, function pointers, string functions)."""
        from .machine import MATH
        from .flatmem import FLAT_LIBRARY
        flat, seen, todo = False, set(), ["@" + self.entry]
        while todo:
            f = todo.pop()
            if f in seen: continue
            seen.add(f)
            if f in self.module.unsupported:
                return f"{f[1:]} uses a feature the runner does not support ({self.module.unsupported[f]})", flat
            if f not in self.module.functions: continue
            for blk in self.module.functions[f].blocks:
                for ins in blk.insts:
                    if ins.op in ("ptrtoint", "inttoptr") or (ins.op in ("load", "store") and ins.type == ("ptr",)):
                        flat = True                              # pointers kept in memory or turned into numbers
                    if ins.op != "call": continue
                    callee = ins.extra["fn"]
                    if callee is None: flat = True; continue   # a call through a function pointer
                    if callee in self.module.functions or callee in self.module.unsupported: todo.append(callee); continue
                    base = callee[1:].split(".")[1] if callee.startswith("@llvm.") else callee[1:]
                    if base in FLAT_LIBRARY: flat = True; continue
                    if base not in MATH and base not in ("lifetime", "assume", "dbg", "experimental", "invariant",
                                                         "sideeffect", "memcpy", "memmove", "memset"):
                        return f"it calls {callee[1:]}, which has no source in the program", flat
        if any(g.type[0] == "struct" or (g.type[0] == "array" and g.type[2][0] == "struct") for g in self.module.globals.values()):
            flat = True
        if any(f in self.module.functions and any(ins.extra.get("fn") is None for b in self.module.functions[f].blocks
                                                  for ins in b.insts if ins.op == "call") for f in seen):
            flat = True
        return None, flat

    def native(self):
        """The natively compiled program (built once)."""
        if getattr(self, "_native", None) is None:
            from .native import Native
            self._native = Native(self)
        return self._native

    # ------------------------------------------------------------------ running
    def _lane_args(self, mach, inputs, B):
        if len(inputs) != len(self.params):
            raise TypeError(f"{self.entry} takes {len(self.params)} inputs ({[n for n, _ in self.params]}), got {len(inputs)}")
        args = []
        for (name, ct), x in zip(self.params, inputs):
            x = torch.as_tensor(x, device=self.device)
            kind, bits, _ = ct[1] if ct[0] == "ptr" else ct
            if kind == "fp": v = x.reshape(B, -1).to(DTYPE[bits])
            elif bits == 1: v = (x.reshape(B, -1) != 0)
            else: v = wrap(x.reshape(B, -1).long(), bits)
            if ct[0] == "ptr":
                et = ("fp", bits) if kind == "fp" else ("int", max(8, bits))
                args.append(mach.input_pointer(v.long() if bits == 1 else v, et, name))
            else: args.append(v.reshape(B))
        return args

    def _learned_tensors(self, values):
        out = {}
        for n, L in self.learned.items():
            v = (values or {}).get(n)
            dt = DTYPE[L.etype[1]] if L.etype[0] == "fp" else torch.int64
            out["@" + n] = (torch.as_tensor(v, device=self.device).reshape(-1) if v is not None
                            else torch.tensor(L.init, device=self.device)).to(dt)
        return out

    def run(self, *inputs, values=None, training=False, temperature=1.0, learned_tensors=None):
        """Run the entry function on a batch (the first dimension of every input). Returns the outputs; while training
        they may be relaxed values (Mix / Soft)."""
        B = len(inputs[0])
        if self.native_only: raise RunError(f"the batched runner cannot run {self.entry}: {self.native_reason}")
        M = Machine
        if self.flat:
            from .flatmem import FlatMachine as M
        mach = M(self.module, B, self.device, learned=learned_tensors if learned_tensors is not None else self._learned_tensors(values),
                 training=training, temperature=temperature, vectorize=self.vectorize)
        out = mach.run_function(self.module.functions["@" + self.entry], self._lane_args(mach, inputs, B),
                                torch.ones(B, dtype=torch.bool, device=self.device))
        if not training and self.ret[0] == "int" and not self.ret[2] and self.ret[1] > 1: out = unsigned(out, self.ret[1])
        return out

    def __call__(self, *inputs, values=None):
        """Run exactly (no relaxation) with the given values, else the fitted ones, else the initial ones."""
        values = values if values is not None else self.values
        if self.native_only:
            return self.native().run(*inputs, values={**{n: L.init for n, L in self.learned.items()}, **(values or {})})
        with torch.no_grad(): return self.run(*inputs, values=values)

    def accuracy(self, *inputs, target, chunk=2048):
        right, n = 0, len(target)
        for i in range(0, n, chunk):
            out = self(*[x[i:i + chunk] for x in inputs]).cpu().reshape(-1)
            right += (out == torch.as_tensor(target[i:i + chunk]).to(out.dtype)).sum().item()
        return right / n

    # ------------------------------------------------------------------ fitting
    def fit(self, *inputs, target, steps=1000, batch=128, lr=1e-2, temperature=1.0, choices="auto", samples=8,
            polish=3, restarts=None, choice_init=0.01, noise=1.0, sparsity=0.0, optimizer="adam", engine="auto", seed=0,
            gradient="auto", objective="match", sigma=0.05, pairs=16, validation=None, keep_close=0.0, workers=None,
            verbose=False):
        """Fit the learned variables so that the program's output matches `target` on the examples.

        Real-valued learned variables: gradient descent through the program; comparisons that depend on them are soft
        (sigmoid(difference / temperature)) while training.
        Whole-number learned variables (choices):
          choices="auto"   (default) restarts alternate between "relax" and "sample"; the best result is kept.
          choices="sample" `samples` complete settings are tried per step; better ones become more likely.
          choices="relax"  each one is a weighted mix of all its values (Gumbel-softmax, annealed): a learned index
                           blends the items it may pick, a switch runs every case and blends them.
          choices="continuous" each one is a real number in its range (read as an interpolation between the two nearest
                           whole values); in the second half of training a growing penalty pulls it to whole numbers.
                           Suits choices used as numbers (coefficients), e.g. searching for exact algorithms.
          Then `polish` rounds of exact local search: each choice in turn tries all its values (the others fixed),
          keeps the best, and the real numbers are re-fitted around the result.
        restarts: fit from this many seeds and keep the result that fits the examples best (default: 3 when there are
          learned choices, which can get stuck, else 1).
        choice_init: spread of the starting preferences of choices (larger: start from a random, decided program).
        noise: scale of the Gumbel noise on relaxed choices (0: a smooth, deterministic relaxation).
        optimizer: "adam" (mini-batches), "lbfgs" (full batch of up to `batch` examples, quasi-Newton) or "lm"
          (Levenberg-Marquardt on the per-example errors, with a fading penalty on large values): lbfgs and lm suit
          smooth fits that must become exact, e.g. real-valued coefficients of an algorithm; lm is the strongest.
        sparsity: cost per non-zero choice, so fits with fewer non-zero choices are preferred (simpler / cheaper
          programs). Whatever its value, a final step sets every choice to 0 that the fit does not need.
        engine: "runner" (exact gradients through the batched runner), "native" (gradients estimated from native runs:
          works for ANY program that compiles, including pointers, malloc, files and library calls; best for a modest
          number of learned values), or "auto" (default): the runner when it supports the program, else native.
        Native engine only: gradient "fd" (central differences: smooth outputs, few values), "es" (random
          perturbations, `pairs` per step, relative size `sigma`: outputs that jump, e.g. decisions or error counts),
          or "auto"; objective "match" (output should equal target) or "minimize" (the output itself is the cost to
          minimise, e.g. an error rate or a size; target is then ignored); validation (inputs, target): the values
          that score best on it are kept (early stopping); keep_close: pull toward the starting values; workers:
          parallel worker processes (default: CPU cores for perturbation gradients).
        The result is in self.values."""
        if engine == "native" or (engine == "auto" and self.native_only):
            from .native_fit import fit_native
            if verbose and self.native_only: print(f"  native engine: {self.native_reason}", flush=True)
            return fit_native(self, inputs, target, steps=steps, batch=batch, lr=lr, restarts=restarts or 1, seed=seed,
                              gradient=gradient, objective=objective, sigma=sigma, pairs=pairs, validation=validation,
                              keep_close=keep_close, workers=workers, verbose=verbose)
        has_choices = any(L.etype[0] == "int" for L in self.learned.values())
        restarts = restarts or (3 if has_choices else 1)
        start_values = self.values
        if restarts > 1:
            best = None
            for r in range(restarts):
                self.values = start_values
                method = ("relax", "sample")[r % 2] if choices == "auto" else choices
                self.fit(*inputs, target=target, steps=steps, batch=batch, lr=lr, temperature=temperature, choices=method,
                         samples=samples, polish=polish, restarts=1, choice_init=choice_init, noise=noise, sparsity=sparsity,
                         optimizer=optimizer, seed=seed + r,
                         verbose=verbose)
                score = self.score(*inputs, target=target)
                if verbose: print(f"  restart {r + 1}/{restarts} ({method}): score {score:.4f}", flush=True)
                if best is None or score > best[0]: best = (score, self.values, self.history)
            _, self.values, self.history = best
            return self
        if choices == "auto": choices = "relax"
        g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
        floats, logits, start = {}, {}, self.values or {}
        for n, L in self.learned.items():
            if L.etype[0] == "fp":
                v = torch.as_tensor(start[n]).reshape(-1).float() if n in start else torch.tensor(L.init, dtype=torch.float32)
                if n not in start and L.count > 1 and not v.any(): v = torch.randn(L.count, generator=g) * 0.1  # break symmetry
                floats[n] = torch.nn.Parameter(v.to(self.device))
            elif choices == "continuous":
                if L.range is None: raise ValueError(f"{n}: choices='continuous' needs a range (learned_in), not a list of values")
                lo, hi = L.range
                logits[n] = torch.nn.Parameter((lo + (hi - lo) * torch.rand(L.count, generator=g)).to(self.device))   # the value itself
            else:
                logits[n] = torch.nn.Parameter((torch.randn(L.count, len(L.choices), generator=g) * choice_init).to(self.device))
        if not floats and not logits: raise ValueError(f"{os.path.basename(self.path)} has no learned variables")
        X = [torch.as_tensor(x).to(self.device) for x in inputs]; Y = torch.as_tensor(target).to(self.device)
        self.history = []

        def phase(n_steps, fixed=None, mode=choices):
            params = list(floats.values()) + ([] if fixed is not None else list(logits.values()))
            if not params or n_steps == 0: return
            if optimizer == "lm":
                if logits and fixed is None: raise ValueError("optimizer='lm' needs a fit without learned choices (or fixed choices)")
                self._levenberg_marquardt(floats, fixed, X, Y, n_steps, batch, temperature, verbose); return
            if optimizer == "lbfgs":
                if logits and fixed is None: raise ValueError("optimizer='lbfgs' needs a fit without learned choices (or fixed choices)")
                k = torch.arange(min(batch, len(Y)), device=self.device); xs, ys = [x[k] for x in X], Y[k]
                opt = torch.optim.LBFGS(params, lr=1, max_iter=20, history_size=50, line_search_fn="strong_wolfe")
                def closure():
                    opt.zero_grad()
                    lt = {"@" + n: p.to(DTYPE[self.learned[n].etype[1]]) for n, p in floats.items()}
                    lt.update({"@" + n: v for n, v in (fixed or {}).items()})
                    loss = -log_likelihood(self.run(*xs, training=True, temperature=temperature, learned_tensors=lt), ys).float().mean()
                    loss.backward(); return loss
                for step in range(n_steps):
                    loss = float(opt.step(closure)); self.history.append(loss)
                    if verbose and (step + 1) % max(1, n_steps // 10) == 0: print(f"  step {step + 1}/{n_steps}  loss {loss:.3g}", flush=True)
                return
            opt = torch.optim.Adam(params, lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)
            S = samples if (logits and fixed is None and mode == "sample") else 1
            for step in range(n_steps):
                k = torch.randint(0, len(Y), (min(batch, len(Y)),), device=self.device); b = len(k)
                xs, ys = [x[k].repeat_interleave(S, 0) for x in X], Y[k].repeat_interleave(S, 0)  # lane = example * S + sample
                lt = {"@" + n: p.to(DTYPE[self.learned[n].etype[1]]) for n, p in floats.items()}
                logp = 0
                for n, lg in logits.items():
                    L = self.learned[n]
                    if fixed is not None: lt["@" + n] = fixed[n]
                    elif mode == "continuous":
                        vals = torch.arange(L.range[0], L.range[1] + 1, device=self.device, dtype=lg.dtype)
                        v = lg.clamp(L.range[0], L.range[1])
                        lt["@" + n] = Relaxed((1 - (v[:, None] - vals[None, :]).abs()).clamp(min=0), vals.long())   # hat functions
                    elif mode == "sample":
                        dist = torch.distributions.Categorical(logits=lg); z = dist.sample((S,))
                        logp = logp + dist.log_prob(z).sum(-1); lt["@" + n] = self._choices(n)[z].repeat(b, 1)
                    else:
                        tau = max(0.1, 1.0 - 0.9 * step / max(1, int(0.7 * n_steps)))                   # anneal 1 -> 0.1
                        gumbel = -noise * torch.log(-torch.log(torch.rand_like(lg).clamp(1e-9, 1 - 1e-9)))
                        lt["@" + n] = Relaxed(torch.softmax((lg + gumbel) / tau, -1), self._choices(n))
                ll = log_likelihood(self.run(*xs, training=True, temperature=temperature, learned_tensors=lt), ys).float()
                loss = -ll.mean()
                if mode == "continuous" and fixed is None and step > n_steps // 2:              # pull towards whole numbers
                    strength = 10.0 * ((step - n_steps // 2) / (n_steps - n_steps // 2)) ** 2
                    loss = loss + strength * sum(((lg - lg.round().detach()) ** 2).sum() for lg in logits.values())
                if sparsity and fixed is None:
                    for n, lg in logits.items():
                        if mode == "continuous": loss = loss + sparsity * lg.abs().sum()
                        elif 0 in self.learned[n].choices:
                            zero = self.learned[n].choices.index(0)
                            loss = loss + sparsity * (1 - torch.softmax(lg, -1)[:, zero]).sum()
                if S > 1:
                    R = ll.detach().reshape(b, S).mean(0); loss = loss - ((R - R.mean()) * logp).mean()
                if not loss.requires_grad:
                    if mode == "relax" and logits: raise _NoSignal()          # relaxing these choices gives no gradient
                    continue
                opt.zero_grad(); loss.backward(); opt.step(); sch.step(); self.history.append(loss.item())
                if verbose and (step + 1) % max(1, n_steps // 10) == 0:
                    print(f"  step {step + 1}/{n_steps}  loss {loss.item():.4f}", flush=True)

        if not logits: phase(steps)
        else:
            try: phase(steps - steps // 4)
            except _NoSignal:                                     # e.g. choices used as positions for stores
                if verbose: print("  relaxed choices give no gradient here: sampling them instead", flush=True)
                choices = "sample"; phase(steps - steps // 4, mode="sample")
            fixed = {n: (lg.detach().round().clamp(*self.learned[n].range).long() if choices == "continuous"
                         else self._choices(n)[lg.argmax(-1)]).detach() for n, lg in logits.items()}
            for _ in range(polish):
                if not self._polish(fixed, floats, X, Y, temperature): break
                phase(max(1, steps // (4 * polish)), fixed=fixed)          # re-fit the real numbers around the choices
            phase(steps // 4, fixed=fixed)
            self._simplify(fixed, floats, X, Y, temperature)
            self.values = {n: p.detach().reshape(self.learned[n].shape) for n, p in floats.items()}
            for n in logits: self.values[n] = fixed[n].reshape(self.learned[n].shape)
            return self
        self.values = {n: p.detach().reshape(self.learned[n].shape) for n, p in floats.items()}
        for n, lg in logits.items(): self.values[n] = self._choices(n)[lg.argmax(-1)].reshape(self.learned[n].shape)
        return self

    def _polish(self, fixed, floats, X, Y, temperature, n_eval=1024):
        """One round of exact local search over the learned choices. Changes `fixed` in place; True if anything changed."""
        k = torch.randperm(len(Y), device=self.device)[:n_eval]; xs, ys = [x[k] for x in X], Y[k]; b = len(k)
        lt0 = {"@" + n: p.detach().to(DTYPE[self.learned[n].etype[1]]) for n, p in floats.items()}
        changed = False
        with torch.no_grad():
            for n, cur in fixed.items():
                L = self.learned[n]; vals = self._choices(n); K = len(vals)
                for e in range(L.count):
                    cand = cur.reshape(1, -1).repeat(K, 1); cand[:, e] = vals                    # (K, count)
                    lt = dict(lt0); lt.update({"@" + m: v.reshape(1, -1).expand(K * b, -1) for m, v in fixed.items() if m != n})
                    lt["@" + n] = cand.repeat_interleave(b, 0)                                   # lane = candidate * b + example
                    out = self.run(*[x.repeat(K, *[1] * (x.dim() - 1)) for x in xs], training=True, temperature=temperature, learned_tensors=lt)
                    score = log_likelihood(out, ys.repeat(K)).float().reshape(K, b).mean(1)
                    best = int(score.argmax())
                    if vals[best] != cur[e] and score[best] > score[int((vals == cur[e]).nonzero()[0])] + 1e-6:
                        cur[e] = vals[best]; changed = True
        return changed

    def score(self, *inputs, target, n=4096):
        """How well the current values fit (mean log-likelihood of the targets on up to n examples, choices exact)."""
        k = torch.arange(min(n, len(target)))
        with torch.no_grad():
            out = self.run(*[torch.as_tensor(x)[k] for x in inputs], values=self.values, training=True)
            return log_likelihood(out, torch.as_tensor(target)[k].to(self.device)).float().mean().item()

    def _simplify(self, fixed, floats, X, Y, temperature, n_eval=1024, tol=1e-6):
        """Set each choice to 0 (where 0 is allowed) if the fit does not get worse. Changes `fixed` in place."""
        k = torch.randperm(len(Y), device=self.device)[:n_eval]; xs, ys = [x[k] for x in X], Y[k]
        lt0 = {"@" + n: p.detach().to(DTYPE[self.learned[n].etype[1]]) for n, p in floats.items()}
        def score():
            lt = dict(lt0); lt.update({"@" + n: v for n, v in fixed.items()})
            with torch.no_grad():
                return log_likelihood(self.run(*xs, training=True, temperature=temperature, learned_tensors=lt), ys).float().mean().item()
        base = score()
        for n, cur in fixed.items():
            if 0 not in self.learned[n].choices: continue
            for e in range(cur.numel()):
                if cur[e] == 0: continue
                old = cur[e].clone(); cur[e] = 0
                s = score()
                if s >= base - tol: base = max(base, s)
                else: cur[e] = old

    def _choices(self, n): return torch.tensor(self.learned[n].choices, device=self.device)

    def describe(self):
        """The fitted values, with operators shown by name."""
        out = {}
        for n, v in (self.values or {}).items():
            L = self.learned[n]
            out[n] = [L.describe(x) for x in v.reshape(-1).tolist()] if L.ops else v
        return out

    def _levenberg_marquardt(self, floats, fixed, X, Y, iters, batch, temperature, verbose, penalty=None, fade=None, floor=None):
        """Levenberg-Marquardt on the residuals output - target of up to `batch` examples. The Jacobian comes from one
        batched backward pass. A penalty on the size of the values starts at `penalty` and fades by `fade` per accepted
        step: it keeps the search away from 'almost' solutions with huge values."""
        k = torch.arange(min(batch, len(Y)), device=self.device); xs, ys = [x[k] for x in X], Y[k].double()
        names = list(floats); shapes = [floats[n].shape for n in names]; sizes = [floats[n].numel() for n in names]
        def residuals(flat):
            parts = torch.split(flat, sizes)
            lt = {"@" + n: p.reshape(s).to(DTYPE[self.learned[n].etype[1]]) for n, p, s in zip(names, parts, shapes)}
            lt.update({"@" + n: v for n, v in (fixed or {}).items()})
            out = self.run(*xs, training=True, temperature=temperature, learned_tensors=lt)
            if isinstance(out, (Mix, Soft)): raise ValueError("optimizer='lm' needs a program with a real-valued output")
            return out.double() - ys
        penalty = self.lm_settings.get("penalty", 1e-2) if penalty is None else penalty
        fade = self.lm_settings.get("fade", 0.9) if fade is None else fade
        floor = self.lm_settings.get("floor", 0.0) if floor is None else floor
        theta = torch.cat([floats[n].detach().reshape(-1).double() for n in names])
        lam, rho = 1e-3, penalty
        with torch.no_grad(): r = residuals(theta.float()); cost = (r ** 2).sum() + rho * (theta ** 2).sum()
        eye = torch.eye(len(theta), dtype=torch.float64, device=self.device)
        def jacobian(theta):
            """Every example gets its own copy of the values; one backward pass of the summed residuals then puts each
            example's gradient into its own copy: the whole Jacobian for about the cost of two forward passes."""
            copies = theta.float().expand(len(ys), -1).clone().requires_grad_()
            parts = torch.split(copies, sizes, dim=1)
            lt = {"@" + n: p.to(DTYPE[self.learned[n].etype[1]]) for n, p in zip(names, parts)}
            lt.update({"@" + n: v for n, v in (fixed or {}).items()})
            r = self.run(*xs, training=True, temperature=temperature, learned_tensors=lt).double() - ys
            r.sum().backward()
            return r.detach(), copies.grad.double()                                   # (examples,), (examples, values)
        for it in range(iters):
            r, J = jacobian(theta)
            A = J.T @ J; g = J.T @ r + rho * theta
            for _ in range(10):
                step = torch.linalg.solve(A + lam * torch.diag(torch.diag(A)) + (rho + 1e-12) * eye, -g)
                trial = theta + step
                with torch.no_grad(): rt = residuals(trial.float())
                new = (rt ** 2).sum() + rho * (trial ** 2).sum()
                if new < cost:
                    theta, lam = trial, max(lam / 3, 1e-9); rho = max(rho * fade, floor)
                    cost = (rt ** 2).sum() + rho * (trial ** 2).sum(); break
                lam *= 4
            err = float((rt ** 2).mean()); self.history.append(err)
            if verbose and (it + 1) % max(1, iters // 10) == 0: print(f"  lm {it + 1}/{iters}  error {err:.3g}  damping {lam:.1g}", flush=True)
            if err < 1e-20: break
        for n, p in zip(names, torch.split(theta, sizes)): floats[n].data.copy_(p.reshape(floats[n].shape).float())
